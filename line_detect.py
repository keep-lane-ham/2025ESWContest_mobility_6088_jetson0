#!/usr/bin/env python3
import cv2
import threading
import time
import numpy as np
import math
import zmq
import json

# ===== Jetson0 -> Raspberry Pi (lane status) ZMQ 설정 =====
RPI_HOST = "163.180.179.239"  # 라즈베리파이 IP
RPI_PORT = 5557  # 라즈베리파이에서 PULL로 받을 포트
RPI_ADDR = f"tcp://{RPI_HOST}:{RPI_PORT}"

class ZMQLaneEventSender:
    def __init__(self, addr):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUSH)
        self.sock.connect(addr)
        self.lock = threading.Lock()

    def send_status(self, state, extra=None):
        """
        state: "inside" | "left_exit" | "right_exit"
        """
        payload = {
            "type": "lane_status",
            "state": state,
            "ts_ms": int(time.time() * 1000)
        }
        if isinstance(extra, dict):
            payload.update(extra)
        with self.lock:
            self.sock.send_json(payload, flags=zmq.NOBLOCK)

# ============================================================
# (A) ZMQ로 Jetson1 에서 오는 차선 정보 수신
# ============================================================
class ZMQPointReceiver:
    def __init__(self, port=5555):
        self.port = port
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.PULL)
        self.socket.bind(f"tcp://0.0.0.0:{self.port}")
        print(f"[ZMQPointReceiver] Listening on tcp://0.0.0.0:{self.port} (PULL for lane data)")

        self.lock = threading.Lock()
        self.latest_points = []
        self.latest_segments = []
        self.running = True

        self.thread = threading.Thread(target=self._rx_loop, daemon=True)
        self.thread.start()

    def _rx_loop(self):
        while self.running:
            try:
                data = self.socket.recv_json()
                pts  = data.get("points", None)
                segs = data.get("segments", None)

                cleaned_pts = []
                if isinstance(pts, list):
                    for p in pts:
                        if ("x" in p) and ("y" in p):
                            cleaned_pts.append(p)

                cleaned_segs = []
                if isinstance(segs, list):
                    for s in segs:
                        contour_list = s.get("contour", None)
                        if isinstance(contour_list, list) and len(contour_list) >= 3:
                            cleaned_segs.append(s)

                if not cleaned_pts and not cleaned_segs:
                    continue

                with self.lock:
                    if cleaned_pts:
                        self.latest_points = cleaned_pts
                    if cleaned_segs:
                        self.latest_segments = cleaned_segs

            except Exception as e:
                print(f"[ZMQPointReceiver] recv error: {e}")
                time.sleep(0.01)

    def get_target_points(self, frame_w, frame_h):
        with self.lock:
            if self.latest_points:
                pts_out = []
                for p in self.latest_points:
                    tx = int(p.get("x", frame_w // 3))
                    ty = int(p.get("y", int(frame_h * 0.8)))
                    side = p.get("side", "unknown")
                    pts_out.append({"x": tx, "y": ty, "side": side})

                def sort_key(item):
                    if item["side"] == "left":
                        pri = 0
                    elif item["side"] == "right":
                        pri = 2
                    else:
                        pri = 1
                    return (pri, item["x"])

                pts_out.sort(key=sort_key)
                return pts_out

        tx_fb = frame_w // 3
        ty_fb = int(frame_h * 0.8)
        return [{"x": tx_fb, "y": ty_fb, "side": "fallback"}]

    def get_lane_polygons(self, frame_w, frame_h):
        polys_out = []
        with self.lock:
            segs_copy = list(self.latest_segments)

        for seg in segs_copy:
            side = seg.get("side", "unknown")
            contour_list = seg.get("contour", [])
            poly_pts = []
            for xy in contour_list:
                if isinstance(xy, (list, tuple)) and len(xy) == 2:
                    px, py = int(xy[0]), int(xy[1])
                    px = max(0, min(frame_w - 1, px))
                    py = max(0, min(frame_h - 1, py))
                    poly_pts.append([px, py])

            if len(poly_pts) >= 3:
                polys_out.append({
                    "side": side,
                    "pts": np.array(poly_pts, dtype=np.int32)
                })

        return polys_out

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.socket.close()
        self.ctx.term()

# ============================================================
# (A2) 라즈베리파이에서 오는 조향 인코더값 수신
# ============================================================
class ZMQSteeringReceiver:
    def __init__(self, port=5556):
        self.port = port
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.PULL)
        self.socket.bind(f"tcp://0.0.0.0:{self.port}")
        print(f"[ZMQSteeringReceiver] Listening on tcp://0.0.0.0:{self.port} (PULL for steering)")

        self.lock = threading.Lock()
        self.encoder_angle_deg = 90.0  # 합리적 초기값
        self.running = True
        self.thread = threading.Thread(target=self._rx_loop, daemon=True)
        self.thread.start()

    def _rx_loop(self):
        while self.running:
            try:
                data = self.socket.recv_json()
                enc_val = None
                if "encoder_deg" in data:
                    enc_val = data["encoder_deg"]
                elif "encoder" in data:
                    enc_val = data["encoder"]
                elif "enc" in data:
                    enc_val = data["enc"]

                if enc_val is not None:
                    with self.lock:
                        self.encoder_angle_deg = float(enc_val)

            except Exception as e:
                print(f"[ZMQSteeringReceiver] recv error: {e}")
                time.sleep(0.01)

    def get_signed_steering_angle_deg(self):
        with self.lock:
            enc = self.encoder_angle_deg
        steering_raw = 10.0 + (enc * (160.0 / 180.0))
        signed_steer = steering_raw - 90.0
        return signed_steer

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.socket.close()
        self.ctx.term()

# ============================================================
# (B) RTSP 비디오 프레임을 받아오는 스레드 (수정 금지)
# ============================================================
class VideoStream:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            print(f"Failed to open capture with pipeline: {pipeline}")
            self.running = False
            return
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()
            if not self.ret:
                print("VideoStream update: read failed, stopping.")
                self.running = False
                break

    def read(self):
        if self.running and self.ret and self.frame is not None:
            return True, self.frame.copy()
        return False, None

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        if self.cap.isOpened():
            self.cap.release()

def gstreamer_pipeline(rtsp_url):
    # [FIX] GStreamer가 UDP 대신 TCP를 사용하도록 `protocols=tcp` 추가
    return (
        f"rtspsrc location={rtsp_url} latency=0 protocols=tcp ! "
        "rtph264depay ! h264parse ! "
        "video/x-h264, stream-format=byte-stream ! "
        "nvv4l2decoder ! nvvidconv ! "
        "video/x-raw, format=BGRx ! videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=true sync=false"
    )

# ============================================================
# 유틸들
# ============================================================
def point_to_segment_distance(tx, ty, x1, y1, x2, y2):
    vx = x2 - x1
    vy = y2 - y1
    wx = tx - x1
    wy = ty - y1
    seg_len_sq = vx * vx + vy * vy
    if seg_len_sq == 0:
        dx = tx - x1
        dy = ty - y1
        return math.sqrt(dx*dx + dy*dy), (x1, y1)

    t = (vx * wx + vy * wy) / seg_len_sq
    if t < 0.0:
        cx, cy = x1, y1
    elif t > 1.0:
        cx, cy = x2, y2
    else:
        cx = x1 + t * vx
        cy = y1 + t * vy

    dx = tx - cx
    dy = ty - cy
    dist = math.sqrt(dx*dx + dy*dy)
    return dist, (cx, cy)

def match_point_to_hough(tx, ty, lines_gpu, frame_w, frame_h):
    # [OPTIMIZATION] GPU Hough 변환 결과를 CPU로 다운로드하여 처리
    if lines_gpu is None or lines_gpu.empty():
        return {"best_line": None, "near_pt": None, "slope": None}

    lines = lines_gpu.download()
    if lines is None or len(lines) == 0:
        return {"best_line": None, "near_pt": None, "slope": None}
    
    lines = lines[0] # [OPTIMIZATION] GpuHough returns lines in an extra dimension

    best_line, best_dist, best_near = None, None, None
    for line in lines:
        x1, y1, x2, y2 = line
        dist, near_pt = point_to_segment_distance(tx, ty, x1, y1, x2, y2)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_line = (x1, y1, x2, y2)
            best_near = near_pt

    slope_val = None
    if best_line is not None:
        x1, y1, x2, y2 = best_line
        dx, dy = x2 - x1, y2 - y1
        slope_val = (dy / dx) if dx != 0 else np.inf

    return {"best_line": best_line, "near_pt": best_near, "slope": slope_val}

def ray_to_border(px, py, slope, frame_w, frame_h):
    if slope is None:
        return (px, py)

    candidates = []
    if slope == np.inf or slope == -np.inf:
        cand_top = (px, 0)
        cand_top = (int(max(0, min(frame_w - 1, cand_top[0]))),
                    int(max(0, min(frame_h - 1, cand_top[1]))))
        candidates.append(cand_top)

        cand_bottom = (px, frame_h - 1)
        cand_bottom = (int(max(0, min(frame_w - 1, cand_bottom[0]))),
                       int(max(0, min(frame_h - 1, cand_bottom[1]))))
        candidates.append(cand_bottom)
    else:
        m = slope
        b = py - m * px

        if m != 0:
            x_top = -b / m
            if 0 <= x_top <= (frame_w - 1):
                candidates.append((x_top, 0))

        y_bot = frame_h - 1
        x_bot = (y_bot - b) / m if m != 0 else None
        if (x_bot is not None) and (0 <= x_bot <= (frame_w - 1)):
            candidates.append((x_bot, y_bot))

        if 0 <= b <= (frame_h - 1):
            candidates.append((0, b))

        y_right = m * (frame_w - 1) + b
        if 0 <= y_right <= (frame_h - 1):
            candidates.append((frame_w - 1, y_right))

    if not candidates:
        return (px, py)

    upward = []
    for (cx, cy) in candidates:
        if cy <= py:
            dist = math.hypot(cx - px, cy - py)
            upward.append((dist, (cx, cy)))

    if upward:
        upward.sort(key=lambda x: x[0])
        chosen = upward[0][1]
    else:
        all_list = []
        for (cx, cy) in candidates:
            all_list.append((math.hypot(cx - px, cy - py), (cx, cy)))
        all_list.sort(key=lambda x: x[0])
        chosen = all_list[0][1]

    cx, cy = chosen
    cx = int(max(0, min(frame_w - 1, cx)))
    cy = int(max(0, min(frame_h - 1, cy)))
    return (cx, cy)

def fill_polygon_alpha(img, pts_list, color_bgr=(255,200,100), alpha=0.4):
    overlay = img.copy()
    poly = np.array(pts_list, dtype=np.int32)
    cv2.fillPoly(overlay, [poly], color_bgr)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def _draw_poly_outline(dst, pts, color=(255,0,0), thick=2):
    pts2 = np.array(pts, dtype=np.int32).reshape(-1,1,2)
    cv2.polylines(dst, [pts2], isClosed=True, color=color, thickness=thick)

def _orient(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def _on_segment(a, b, p):
    return (min(a[0], b[0]) - 1e-6 <= p[0] <= max(a[0], b[0]) + 1e-6 and
            min(a[1], b[1]) - 1e-6 <= p[1] <= max(a[1], b[1]) + 1e-6)

def _segments_intersect(p1, p2, q1, q2):
    o1 = _orient(p1, p2, q1)
    o2 = _orient(p1, p2, q2)
    o3 = _orient(q1, q2, p1)
    o4 = _orient(q1, q2, p2)
    if (o1*o2 < 0) and (o3*o4 < 0):
        return True
    if abs(o1) < 1e-9 and _on_segment(p1, p2, q1): return True
    if abs(o2) < 1e-9 and _on_segment(p1, p2, q2): return True
    if abs(o3) < 1e-9 and _on_segment(q1, q2, p1): return True
    if abs(o4) < 1e-9 and _on_segment(q1, q2, p2): return True
    return False

def _segment_intersection_point(p1, p2, q1, q2):
    x1,y1 = p1; x2,y2 = p2; x3,y3 = q1; x4,y4 = q2
    den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(den) < 1e-12:
        return None
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / den
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / den
    P = (px, py)
    if _on_segment(p1, p2, P) and _on_segment(q1, q2, P):
        return P
    return None

def first_contact_on_trajectory(traj_pts, poly_pts):
    if traj_pts is None or len(traj_pts) < 2 or poly_pts is None or len(poly_pts) < 3:
        return False, None, None

    traj = np.asarray(traj_pts, dtype=np.float32)
    poly = np.asarray(poly_pts, dtype=np.float32)

    for i in range(len(traj)):
        p = (float(traj[i,0]), float(traj[i,1]))
        inside = cv2.pointPolygonTest(poly, p, False)
        if inside >= 0:
            return True, p, max(0, i-1)

    edges = []
    for i in range(len(poly)):
        q1 = (float(poly[i,0]), float(poly[i,1]))
        q2 = (float(poly[(i+1) % len(poly),0]), float(poly[(i+1) % len(poly),1]))
        edges.append((q1, q2))

    for i in range(len(traj)-1):
        p1 = (float(traj[i,0]),   float(traj[i,1]))
        p2 = (float(traj[i+1,0]), float(traj[i+1,1]))
        for (q1, q2) in edges:
            if _segments_intersect(p1, p2, q1, q2):
                ip = _segment_intersection_point(p1, p2, q1, q2)
                if ip is None:
                    ip = p2
                return True, (float(ip[0]), float(ip[1])), i

    return False, None, None

# ---- 최근접 폴리곤 점(에지에 대한 직교 투영) 찾기 ----
def nearest_point_on_polygon(px, py, poly_pts):
    poly = np.asarray(poly_pts, dtype=np.float32)
    best_d = None
    best = (None, None)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1) % n]
        vx, vy = x2 - x1, y2 - y1
        wx, wy = px - x1, py - y1
        seg_len_sq = vx*vx + vy*vy
        if seg_len_sq <= 1e-12:
            cx, cy = x1, y1
        else:
            t = (vx*wx + vy*wy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            cx = x1 + t * vx
            cy = y1 + t * vy
        d = (px - cx)*(px - cx) + (py - cy)*(py - cy)
        if (best_d is None) or (d < best_d):
            best_d = d
            best = (cx, cy)
    return best

# ---- 상태 판정: inside / left_exit / right_exit ----
# [OPTIMIZATION] 중복 계산을 피하기 위해 디버그용 최근접점을 반환
def compute_lane_state(traj_pts, poly_pts):
    """
    inside: traj 포인트가 폴리곤 내부/경계에 하나라도 존재
    아니면:
      1) 유도선 전체가 폴리곤 '아래쪽'(y > maxy + margin)에 있으면,
         최근 포인트들의 x 중앙값과 폴리곤 중심선을 비교해 left/right_exit 판정
      2) 그 외에는 최근접점 비교(p*.x vs q*.x)로 판정
    
    Returns:
      (state, dbg_p, dbg_q): 상태 문자열, traj 최근접점, poly 최근접점
    """
    if traj_pts is None or len(traj_pts) < 1 or poly_pts is None or len(poly_pts) < 3:
        return "inside", None, None  # 방어적 기본값

    poly = np.asarray(poly_pts, dtype=np.float32)
    traj = np.asarray(traj_pts, dtype=np.float32)

    # 내부/경계 포함 여부 빠른 검사
    for i in range(len(traj)):
        if cv2.pointPolygonTest(poly, (float(traj[i,0]), float(traj[i,1])), False) >= 0:
            return "inside", None, None # 내부에 있으므로 최근접점 불필요

    # 폴리곤 bbox/중심
    minx, miny = np.min(poly, axis=0)
    maxx, maxy = np.max(poly, axis=0)
    center_x = 0.5 * (float(minx) + float(maxx))

    # 유도선이 폴리곤 '아래쪽'에 완전히 있는 경우: 아래쪽 전용 좌/우 판정
    BELOW_Y_MARGIN = 8.0
    if np.all(traj[:, 1] > (maxy + BELOW_Y_MARGIN)):
        take = min(20, len(traj))
        px_rep = float(np.median(traj[-take:, 0]))  # 최근 포인트 x 중앙값
        X_MARGIN = 0.5
        if px_rep < (center_x - X_MARGIN):
            return "left_exit", None, None
        elif px_rep > (center_x + X_MARGIN):
            return "right_exit", None, None
        else:
            return "inside", None, None

    # 일반 케이스: 폴리곤에 가장 가까운 traj 포인트 p*와 폴리곤 최근접점 q* 비교
    y_lo, y_hi = miny - 5.0, maxy + 5.0
    cand_idx = [i for i in range(len(traj)) if (y_lo <= traj[i,1] <= y_hi)]

    best_i = None
    best_d2 = None
    best_q = None # [OPTIMIZATION] 디버그용 q 저장
    
    if cand_idx:
        for i in cand_idx:
            px, py = traj[i]
            qx, qy = nearest_point_on_polygon(px, py, poly_pts)
            d2 = (px - qx)*(px - qx) + (py - qy)*(py - qy)
            if (best_d2 is None) or (d2 < best_d2):
                best_d2 = d2
                best_i = i
                best_q = (qx, qy)
    else:
        for i in range(len(traj)):
            px, py = traj[i]
            qx, qy = nearest_point_on_polygon(px, py, poly_pts)
            d2 = (px - qx)*(px - qx) + (py - qy)*(py - qy)
            if (best_d2 is None) or (d2 < best_d2):
                best_d2 = d2
                best_i = i
                best_q = (qx, qy)

    if best_i is None:
        return "inside", None, None

    px, py = traj[best_i]
    qx, qy = best_q
    
    dbg_p = (px, py)
    dbg_q = (qx, qy)

    EPS = 0.5
    if px < qx - EPS:
        return "left_exit", dbg_p, dbg_q
    elif px > qx + EPS:
        return "right_exit", dbg_p, dbg_q
    else:
        return "inside", dbg_p, dbg_q

def draw_current_trajectory_ackermann(
    line_vis,
    steer_angle_signed_deg,
    frame_w,
    frame_h,
    car_u=None,
    car_v=None,
    wheelbase_px=200.0,
    step_px=5.0,
    max_forward_px=450.0,
    straight_extend_px=200.0,
    top_margin_px=10,
    color_bgr=(180, 0, 255),
    thickness=2
):
    # [OPTIMIZATION] 함수 내부 import 제거 (전역 사용)
    # import math
    # import numpy as np
    # import cv2

    if car_u is None:
        car_u = frame_w // 2
    if car_v is None:
        car_v = frame_h - 1

    delta_rad = np.deg2rad(steer_angle_signed_deg)
    tan_delta = math.tan(delta_rad)

    def _draw_poly(pts):
        if len(pts) >= 2:
            cv2.polylines(line_vis, [np.array(pts, dtype=np.int32)],
                          isClosed=False, color=color_bgr, thickness=thickness)

    if abs(tan_delta) < 1e-4:
        pts = []
        total_len = max_forward_px + straight_extend_px
        for s in np.arange(0, total_len + step_px, step_px):
            u_img = int(car_u)
            v_img = int(car_v - s)
            if v_img < top_margin_px:
                break
            pts.append([u_img, v_img])
        _draw_poly(pts)
        return np.array(pts, dtype=np.int32)

    kappa = -tan_delta / wheelbase_px
    kappa = max(min(kappa, 0.05), -0.05)

    pts = []
    x_end = 0.0
    y_end = 0.0
    for s in np.arange(0, max_forward_px + step_px, step_px):
        x_fwd = (1.0 / kappa) * math.sin(kappa * s)
        y_left = (1.0 / kappa) * (1.0 - math.cos(kappa * s))
        u_img = int(car_u - y_left)
        v_img = int(car_v - x_fwd)
        if v_img < top_margin_px:
            break
        pts.append([u_img, v_img])
        x_end, y_end = x_fwd, y_left

    theta_end = kappa * max_forward_px
    dx_ds = math.cos(theta_end)
    dy_ds = math.sin(theta_end)

    for s_ext in np.arange(step_px, straight_extend_px + step_px, step_px):
        x_ext = x_end + dx_ds * s_ext
        y_ext = y_end + dy_ds * s_ext
        u_img = int(car_u - y_ext)
        v_img = int(car_v - x_ext)
        if v_img < top_margin_px:
            break
        pts.append([u_img, v_img])

    _draw_poly(pts)
    return np.array(pts, dtype=np.int32)

# ============================================================
# (I) main 루프
# ============================================================
def main():
    RTSP_URL = "rtsp://163.180.179.239:8554/cam"

    # [FIX] GStreamer가 CUDA 객체와 충돌하지 않도록 가장 먼저 초기화
    video_stream = VideoStream(gstreamer_pipeline(RTSP_URL))
    if not video_stream.running:
        print("비디오 스트림 시작 실패.")
        return

    zmq_lane_rx = ZMQPointReceiver(port=5555)
    steering_rx = ZMQSteeringReceiver(port=5556)
    lane_sender = ZMQLaneEventSender(RPI_ADDR)

    last_status_t = 0.0
    STATUS_PERIOD = 0.2  # 초

    cv2.namedWindow("Lane Guidance Debug", cv2.WINDOW_AUTOSIZE)

    # [FIX] GStreamer 충돌을 피하기 위해 CUDA 객체들을 None으로 초기화
    # 첫 프레임 수신 후 '지연된 초기화' 수행
    gpu_frame = None
    gpu_hls = None
    gpu_l_norm = None
    gpu_bin_white = None
    gpu_lab = None
    gpu_b_norm = None
    gpu_bin_yellow = None
    gpu_mix = None
    gpu_lane_mask = None
    gpu_edges = None
    gpu_hough_lines = None # Hough 결과 저장용
    
    # [FIX] 메모리 누수 방지를 위해 모든 GpuMat 객체를 None으로 초기화
    gpu_hls_h = None
    gpu_hls_l = None
    gpu_hls_s = None
    gpu_lab_l = None
    gpu_lab_a = None
    gpu_lab_b = None

    canny_detector = None
    hough_detector = None

    smoothed_slope = None
    alpha_ewma = 0.2
    
    W_WHITE = 0.5
    W_YELLOW = 0.5
    BIN_AFTER_MIX_THRESH = 128

    try:
        while video_stream.running:
            ret, bgr_frame = video_stream.read()
            if not ret or bgr_frame is None:
                time.sleep(0.01)
                continue

            # [FIX] 첫 프레임 수신 후 CUDA 객체 "지연 초기화" (단 한번 실행됨)
            if canny_detector is None:
                print("--- Initializing CUDA objects (first frame received) ---")
                frame_h_init, frame_w_init = bgr_frame.shape[:2]
                
                # GpuMat 객체들 생성 (올바른 크기로)
                gpu_frame = cv2.cuda_GpuMat(frame_h_init, frame_w_init, cv2.CV_8UC3)
                gpu_hls = cv2.cuda_GpuMat(frame_h_init, frame_w_init, cv2.CV_8UC3)
                
                # [FIX] HLS 3채널 모두 생성
                gpu_hls_h = cv2.cuda_GpuMat(frame_h_init, frame_w_init, cv2.CV_8UC1)
                gpu_hls_l = cv2.cuda_GpuMat(frame_h_init, frame_w_init, cv2.CV_8UC1)
                gpu_hls_s = cv2.cuda_GpuMat(frame_h_init, frame_w_init, cv2.CV_8UC1)
                
                gpu_l_norm = cv2.cuda_GpuMat(frame_h_init, frame_w_init, cv2.CV_8UC1)
                gpu_bin_white = cv2.cuda_GpuMat(frame_h_init, frame_w_init, cv2.CV_8UC1)
                gpu_lab = cv2.cuda_GpuMat(frame_h_init, frame_w_init, cv2.CV_8UC3)

                # [FIX] LAB 3채널 모두 생성
                gpu_lab_l = cv2.cuda_GpuMat(frame_h_init, frame_w_init, cv2.CV_8UC1)
                gpu_lab_a = cv2.cuda_GpuMat(frame_h_init, frame_w_init, cv2.CV_8UC1)
                gpu_lab_b = cv2.cuda_GpuMat(frame_h_init, frame_w_init, cv2.CV_8UC1)

                gpu_b_norm = cv2.cuda_GpuMat(frame_h_init, frame_w_init, cv2.CV_8UC1)
                gpu_bin_yellow = cv2.cuda_GpuMat(frame_h_init, frame_w_init, cv2.CV_8UC1)
                gpu_mix = cv2.cuda_GpuMat(frame_h_init, frame_w_init, cv2.CV_8UC1)
                gpu_lane_mask = cv2.cuda_GpuMat(frame_h_init, frame_w_init, cv2.CV_8UC1)
                gpu_edges = cv2.cuda_GpuMat(frame_h_init, frame_w_init, cv2.CV_8UC1)
                gpu_hough_lines = cv2.cuda_GpuMat() # Hough 결과는 크기 지정 안 함

                # CUDA 연산자 생성
                canny_detector = cv2.cuda.createCannyEdgeDetector(50, 150)
                
                # [FIX] 'rho' 파라미터가 (float) 형식을 기대하므로 1.0f로 명시
                hough_detector = cv2.cuda.createHoughSegmentDetector(
                    rho=1.0,  # 1 -> 1.0
                    theta=np.pi / 180.0,
                    minLineLength=50,
                    maxLineGap=20
                )
                print("--- CUDA objects initialized ---")
                
            frame_h, frame_w = bgr_frame.shape[:2]

            # [DEBUG] ZMQ 및 연산 활성화
            targets = zmq_lane_rx.get_target_points(frame_w, frame_h)
            lane_polys = zmq_lane_rx.get_lane_polygons(frame_w, frame_h)
            steering_angle_deg = steering_rx.get_signed_steering_angle_deg()

            # [DEBUG] 1. GPU로 업로드
            gpu_frame.upload(bgr_frame)
            
            # [DEBUG] 3. 모든 CUDA 연산 활성화
            cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2HLS, dst=gpu_hls)
            # [FIX] 미리 할당된 GpuMat을 사용하여 메모리 누수 방지
            cv2.cuda.split(gpu_hls, [gpu_hls_h, gpu_hls_l, gpu_hls_s])
            cv2.cuda.normalize(gpu_hls_l, 0, 255, cv2.NORM_MINMAX, -1, dst=gpu_l_norm)
            cv2.cuda.threshold(gpu_l_norm, 100, 255, cv2.THRESH_BINARY, dst=gpu_bin_white)

            cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2LAB, dst=gpu_lab)
            # [FIX] 미리 할당된 GpuMat을 사용하여 메모리 누수 방지
            cv2.cuda.split(gpu_lab, [gpu_lab_l, gpu_lab_a, gpu_lab_b])
            cv2.cuda.normalize(gpu_lab_b, 0, 255, cv2.NORM_MINMAX, -1, dst=gpu_b_norm)
            cv2.cuda.threshold(gpu_b_norm, 75, 255, cv2.THRESH_BINARY, dst=gpu_bin_yellow)

            cv2.cuda.addWeighted(gpu_bin_white, W_WHITE, gpu_bin_yellow, W_YELLOW, 0, dst=gpu_mix)
            cv2.cuda.threshold(gpu_mix, BIN_AFTER_MIX_THRESH, 255, cv2.THRESH_BINARY, dst=gpu_lane_mask)

            # [FIX] Canny `detect`는 dst 키워드 인자를 사용하지 않음
            canny_detector.detect(gpu_lane_mask, gpu_edges)
            
            # [OPTIMIZATION] GPU Hough 변환 실행
            hough_detector.detect(gpu_edges, gpu_hough_lines)
            
            # (시각화 준비) 원본 프레임 복사
            line_vis = bgr_frame.copy()
            last_poly_pts = None

            if len(lane_polys) == 1:
                single_pts = lane_polys[0]["pts"]
                if len(single_pts) >= 3:
                    fill_polygon_alpha(line_vis, single_pts, color_bgr=(200,180,80), alpha=0.4)
                    _draw_poly_outline(line_vis, single_pts, color=(255,0,0), thick=2)
                    last_poly_pts = single_pts

            elif len(lane_polys) >= 2:
                all_pts_list = []
                for lp in lane_polys:
                    if "pts" in lp and isinstance(lp["pts"], np.ndarray) and len(lp["pts"]) >= 3:
                        all_pts_list.append(lp["pts"])
                if len(all_pts_list) >= 2:
                    merged_pts = np.vstack(all_pts_list)
                    hull = cv2.convexHull(merged_pts.reshape(-1,1,2))
                    hull_pts = hull.reshape(-1,2).astype(np.int32)
                    fill_polygon_alpha(line_vis, hull_pts, color_bgr=(200,180,80), alpha=0.45)
                    _draw_poly_outline(line_vis, hull_pts, color=(255,0,0), thick=2)
                    last_poly_pts = hull_pts

            # (디버그) Hough/점 시각화 (CPU 연산)
            matches = []
            
            # [OPTIMIZATION] GPU Hough 결과를 다운로드하여 매칭
            lines_downloaded = gpu_hough_lines.download()
            lines = None
            if lines_downloaded is not None and len(lines_downloaded) > 0:
                lines = lines_downloaded[0]
            
            for t in targets:
                tx, ty = t["x"], t["y"]
                # CPU 기반 매칭 함수 (원본)
                mres = match_point_to_hough(tx, ty, lines, frame_w, frame_h)
                matches.append({
                    "pt": (tx, ty),
                    "side": t.get("side","unknown"),
                    "slope": mres["slope"],
                    "best_line": mres["best_line"],
                    "near_pt": mres["near_pt"]
                })

            # slope EWMA 유지(디버그용)
            slope_now = matches[0]["slope"] if len(matches) >= 1 else None
            if slope_now is not None:
                if smoothed_slope is None:
                    smoothed_slope = slope_now
                else:
                    smoothed_slope = alpha_ewma * slope_now + (1 - alpha_ewma) * smoothed_slope

            for idx, m in enumerate(matches):
                (tx, ty) = m["pt"]
                slope_val = m["slope"]
                best_line = m["best_line"]
                near_pt = m["near_pt"]
                cv2.circle(line_vis, (tx, ty), 6, (0, 0, 255), -1)
                if best_line is not None:
                    x1, y1, x2, y2 = best_line
                    color_seg = (0,255,0) if idx == 0 else (0,255,255)
                    cv2.line(line_vis, (x1, y1), (x2, y2), color_seg, 3)
                if near_pt is not None:
                    nx, ny = map(int, near_pt)
                    cv2.circle(line_vis, (nx, ny), 5, (255, 0, 0), -1)

            # 유도선(현재 조향각) 궤적 생성
            traj_pts = draw_current_trajectory_ackermann(
                line_vis,
                steer_angle_signed_deg=steering_angle_deg,
                frame_w=frame_w, frame_h=frame_h,
                wheelbase_px=200.0, step_px=5.0,
                max_forward_px=450.0, straight_extend_px=200.0,
                top_margin_px=10, color_bgr=(180, 0, 255), thickness=2,
            )

            # ====== 상태 계산: inside / left_exit / right_exit ======
            state_for_status = "inside"
            dbg_p, dbg_q = None, None # 디버그용 점 초기화
            
            if last_poly_pts is not None and traj_pts is not None and len(traj_pts) >= 2:
                # [OPTIMIZATION] 중복 계산 방지
                state_for_status, dbg_p, dbg_q = compute_lane_state(traj_pts, last_poly_pts)

                # (디버그 표시) 최근접점 화살표
                if dbg_p is not None and dbg_q is not None:
                    try:
                        px, py = map(int, dbg_p)
                        qx, qy = map(int, dbg_q)
                        cv2.circle(line_vis, (px, py), 6, (0, 0, 255), -1)
                        cv2.circle(line_vis, (qx, qy), 6, (0, 255, 255), -1)
                        cv2.line(line_vis, (px, py), (qx, qy), (0, 255, 255), 2)
                    except Exception as e:
                        print(f"Debug draw error: {e}")
                        pass # 디버그 표시에 실패해도 메인 루프는 계속되어야 함

            # 상태 전송(주기)
            now = time.time()
            if (now - last_status_t) >= STATUS_PERIOD:
                try:
                    lane_sender.send_status(state_for_status)
                except Exception as e:
                    cv2.putText(line_vis, f"StatusErr:{e}", (10, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                last_status_t = now

            # 오버레이
            cv2.putText(line_vis, f"lane_state: {state_for_status}",
                            (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # [DEBUG] 5. 최종 시각화
            cv2.imshow("Lane Guidance Debug", line_vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("사용자에 의해 중단됨")
    finally:
        video_stream.stop()
        zmq_lane_rx.stop()
        steering_rx.stop()
        cv2.destroyAllWindows()
        print("프로그램 종료.")

if __name__ == "__main__":
    main()
