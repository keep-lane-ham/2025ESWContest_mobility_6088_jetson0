#!/usr/bin/env python3
import cv2
import threading
import time
import numpy as np
import math
import zmq
import json

# ============================================================
# (A) ZMQ로 Jetson1 에서 오는 차선 정보 수신
# ============================================================
class ZMQPointReceiver:
    def __init__(self, port=5555):
        self.port = port
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.PULL)
        # Jetson0이 bind (PULL), Jetson1이 connect (PUSH) 한다는 기존 구조 유지
        self.socket.bind(f"tcp://0.0.0.0:{self.port}")
        print(f"[ZMQPointReceiver] Listening on tcp://0.0.0.0:{self.port} (PULL for lane data)")

        self.lock = threading.Lock()
        self.latest_points = []      # [{"side":..,"x":..,"y":..}, ...]
        self.latest_segments = []    # [{"side":..,"contour":[[x,y],...]} , ...]
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
                        if (
                            isinstance(contour_list, list)
                            and len(contour_list) >= 3  # polygon 최소 3점
                        ):
                            cleaned_segs.append(s)

                # 최소한 뭔가 유효하면 갱신
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
        """
        조향/기울기/레이 가이드에 쓰일 기준점들(좌우 lane 한 점씩)
        fallback 포함.
        return:
          [
            {"x":..., "y":..., "side":"left"/"right"/"unknown"},
            {... up to 2 ...}
          ]
        """
        with self.lock:
            if self.latest_points:
                pts_out = []
                for p in self.latest_points:
                    tx = int(p.get("x", frame_w // 3))
                    ty = int(p.get("y", int(frame_h * 0.8)))
                    side = p.get("side", "unknown")
                    pts_out.append({"x": tx, "y": ty, "side": side})

                # left -> unknown -> right 정렬 (또는 x 좌표 보조)
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

        # fallback 한 점
        tx_fb = frame_w // 3
        ty_fb = int(frame_h * 0.8)
        return [{"x": tx_fb, "y": ty_fb, "side": "fallback"}]

    def get_lane_polygons(self, frame_w, frame_h):
        """
        세그멘테이션 기반 lane polygon들을 가져온다.
        return:
          [
            {"side":"left"/"right"/"unknown",
             "pts": np.array([[x0,y0],[x1,y1],...], dtype=np.int32)}
          ]
        (좌표를 frame boundary 안으로 클램프해서 돌려준다)
        """
        polys_out = []
        with self.lock:
            segs_copy = list(self.latest_segments)

        for seg in segs_copy:
            side = seg.get("side", "unknown")
            contour_list = seg.get("contour", [])
            poly_pts = []
            for xy in contour_list:
                if (
                    isinstance(xy, (list, tuple))
                    and len(xy) == 2
                ):
                    px, py = int(xy[0]), int(xy[1])
                    # 프레임 범위로 클램프
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
# (A2) 라즈베리파이(163.180.179.239)에서 오는 조향 인코더값 수신
#
# 라즈베리파이는 PUSH.connect() 로 send_json({"encoder_deg": ...})
# Jetson0은 여기서 PULL.bind() 해서 받는다.
#
# 매핑 규칙:
#   encoder 0   -> steering_raw 10 deg
#   encoder 180 -> steering_raw 170 deg
# -> steering_raw = 10 + enc*(160/180)
#
# 우리는 HUD에 사용할 "signed 조향각"을
#   signed = steering_raw - 90
#
# signed < 0 이면 왼쪽 조향, signed > 0 이면 오른쪽 조향.
# ============================================================
class ZMQSteeringReceiver:
    def __init__(self, port=5556):
        self.port = port
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.PULL)
        # Jetson0이 bind, 라즈베리파이가 connect (PUSH) 한다고 가정
        self.socket.bind(f"tcp://0.0.0.0:{self.port}")
        print(f"[ZMQSteeringReceiver] Listening on tcp://0.0.0.0:{self.port} (PULL for steering)")

        self.lock = threading.Lock()
        self.encoder_angle_deg = 90.0  # 합리적 초기값(중립 근처)
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
        """
        라즈베리에서 받은 encoder 값을 실제 조향각(차 바퀴 각도 근사)으로 선형 매핑 후,
        중립(90deg) 기준으로 좌우 부호를 줘서 (-=왼쪽 조향, +=오른쪽 조향) 반환.
        """
        with self.lock:
            enc = self.encoder_angle_deg

        # encoder 0 -> 10 deg, encoder 180 -> 170 deg
        steering_raw = 10.0 + (enc * (160.0 / 180.0))  # 10 + enc*0.888...
        signed_steer = steering_raw - 90.0             # <0: 왼쪽, >0: 오른쪽
        return signed_steer

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.socket.close()
        self.ctx.term()


# ============================================================
# (B) RTSP 비디오 프레임을 받아오는 스레드
# ============================================================
class VideoStream:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            self.running = False
            return
        self.ret, self.frame = self.cap.read()
        sheld = True
        self.running = True
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()
            if not self.ret:
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
    return (
        f"rtspsrc location={rtsp_url} latency=0 ! "
        "rtph264depay ! h264parse ! "
        "video/x-h264, stream-format=byte-stream ! "
        "nvv4l2decoder ! nvvidconv ! "
        "video/x-raw, format=BGRx ! videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=true sync=false"
    )


# ============================================================
# (C) (옛날) 현재 조향각 더미 함수
# ============================================================
def get_current_steering_angle():
    return 0.0


# ============================================================
# (D) 점-선분 최소거리 계산
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


# ============================================================
# (E) 각 점에 대해 Hough 선분들 중 가장 가까운 선/기울기 찾기
# ============================================================
def match_point_to_hough(tx, ty, lines):
    if lines is None or len(lines) == 0:
        return {"best_line": None, "near_pt": None, "slope": None}

    best_line, best_dist, best_near = None, None, None
    for line in lines:
        x1, y1, x2, y2 = line[0]
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

    return {
        "best_line": best_line,
        "near_pt": best_near,
        "slope": slope_val
    }


# ============================================================
# (F) (px,py)에서 slope 방향으로 뻗었을 때 화면 경계와 만나는 점(반직선 끝점)
# (연산만, 더 이상 그리지 않음)
# ============================================================
def ray_to_border(px, py, slope, frame_w, frame_h):
    if slope is None:
        return (px, py)

    candidates = []

    if slope == np.inf or slope == -np.inf:
        # 수직선: x=px
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
        b = py - m * px  # y = m*x + b

        # top: y=0 -> x = -b/m
        if m != 0:
            x_top = -b / m
            if 0 <= x_top <= (frame_w - 1):
                candidates.append((x_top, 0))

        # bottom: y=frame_h-1 -> x = (y-b)/m
        y_bot = frame_h - 1
        x_bot = (y_bot - b) / m if m != 0 else None
        if (x_bot is not None) and (0 <= x_bot <= (frame_w - 1)):
            candidates.append((x_bot, y_bot))

        # left: x=0 -> y = b
        if 0 <= b <= (frame_h - 1):
            candidates.append((0, b))

        # right: x=frame_w-1 -> y = m*(frame_w-1)+b
        y_right = m * (frame_w - 1) + b
        if 0 <= y_right <= (frame_h - 1):
            candidates.append((frame_w - 1, y_right))

    if not candidates:
        return (px, py)

    # y <= py (앞/위쪽) 우선
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


# ============================================================
# (G) 반투명 영역(폴리곤들) 칠하기
# ============================================================
def fill_polygon_alpha(img, pts_list, color_bgr=(255,200,100), alpha=0.4):
    overlay = img.copy()
    poly = np.array(pts_list, dtype=np.int32)
    cv2.fillPoly(overlay, [poly], color_bgr)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


# ============================================================
# (H) 현재 조향각 기반 예상 궤적(arc) 그리기  [Ackermann 기반]
# ============================================================
def draw_current_trajectory_ackermann(
    line_vis,
    steer_angle_signed_deg,
    frame_w,
    frame_h,
    car_u=None,
    car_v=None,
    wheelbase_px=200.0,
    step_px=5.0,
    max_forward_px=300.0,
    color_bgr=(180, 0, 255),
    thickness=2
):
    if car_u is None:
        car_u = frame_w // 2
    if car_v is None:
        car_v = frame_h - 1

    delta_rad = np.deg2rad(steer_angle_signed_deg)
    tan_delta = math.tan(delta_rad)

    # 거의 직진이면 직선
    if abs(tan_delta) < 1e-4:
        pts = []
        for s in np.arange(0, max_forward_px + step_px, step_px):
            u_img = int(car_u)
            v_img = int(car_v - s)
            if v_img < 0:
                break
            pts.append([u_img, v_img])
        if len(pts) >= 2:
            pts_np = np.array(pts, dtype=np.int32)
            cv2.polylines(line_vis, [pts_np], isClosed=False, color=color_bgr, thickness=thickness)
        return

    kappa = -tan_delta / wheelbase_px  # 부호 뒤집어서 왼조향(음수) -> 왼쪽 회전
    kappa = max(min(kappa, 0.05), -0.05)  # 안정화용 클램프

    pts = []
    for s in np.arange(0, max_forward_px + step_px, step_px):
        x_fwd = (1.0 / kappa) * math.sin(kappa * s)
        y_left = (1.0 / kappa) * (1.0 - math.cos(kappa * s))

        u_img = int(car_u - y_left)  # 왼쪽(+y_left)일수록 u 작아짐
        v_img = int(car_v - x_fwd)   # 앞으로 갈수록 위쪽

        if v_img < 0:
            break
        pts.append([u_img, v_img])

    if len(pts) >= 2:
        pts_np = np.array(pts, dtype=np.int32)
        cv2.polylines(line_vis, [pts_np], isClosed=False, color=color_bgr, thickness=thickness)


# ============================================================
# (I) main 루프
# ============================================================
def main():
    RTSP_URL = "rtsp://163.180.179.239:8554/cam"

    # 비디오 스트림
    video_stream = VideoStream(gstreamer_pipeline(RTSP_URL))
    if not video_stream.running:
        print("비디오 스트림 시작 실패.")
        return

    # Jetson1(세그멘테이션)에서 lane contour/points 수신
    zmq_lane_rx = ZMQPointReceiver(port=5555)

    # 라즈베리파이(엔코더)에서 조향각용 encoder 값 수신
    steering_rx = ZMQSteeringReceiver(port=5556)

    cv2.namedWindow("Lane Guidance Debug", cv2.WINDOW_AUTOSIZE)

    # CUDA 준비 (전처리+엣지)
    gpu_frame = cv2.cuda_GpuMat()
    canny_detector = cv2.cuda.createCannyEdgeDetector(50, 150)

    # 기울기 추적(EWMA)
    smoothed_slope = None
    alpha_ewma = 0.2

    # 조향각 vs 요구 조향각 비교 파라미터
    K_GAIN = 25.0
    TURN_MARGIN = 5.0

    # 색 추출 파이프라인 파라미터 (HLS+LAB 결합)
    W_WHITE = 0.5
    W_YELLOW = 0.5
    BIN_AFTER_MIX_THRESH = 128

    try:
        while video_stream.running:
            ret, bgr_frame = video_stream.read()
            if not ret or bgr_frame is None:
                time.sleep(0.01)
                continue

            frame_h, frame_w = bgr_frame.shape[:2]

            # -------- 1) ZMQ 데이터 가져오기 --------
            # steering 판단용 기준점들 (Jetson1)
            targets = zmq_lane_rx.get_target_points(frame_w, frame_h)

            # segmentation 기반 lane polygons (Jetson1)
            lane_polys = zmq_lane_rx.get_lane_polygons(frame_w, frame_h)

            # 라즈베리파이에서 받은 현재 조향각(부호 있는 값, deg)
            steering_angle_deg = steering_rx.get_signed_steering_angle_deg()

            # -------- 2) CUDA 기반 차선 edge 생성 (HLS+LAB 결합) --------
            gpu_frame.upload(bgr_frame)

            # HLS -> 흰 차선 강조
            gpu_hls = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2HLS)
            _, gpu_l, _ = cv2.cuda.split(gpu_hls)
            gpu_l_norm = cv2.cuda.normalize(gpu_l, 0, 255, cv2.NORM_MINMAX, dtype=-1)
            _, gpu_bin_white = cv2.cuda.threshold(gpu_l_norm, 100, 255, cv2.THRESH_BINARY)

            # LAB -> 노란 차선 강조 (b 채널이 노란 계열에서 큼)
            gpu_lab = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2LAB)
            _, _, gpu_b = cv2.cuda.split(gpu_lab)
            gpu_b_norm = cv2.cuda.normalize(gpu_b, 0, 255, cv2.NORM_MINMAX, dtype=-1)
            _, gpu_bin_yellow = cv2.cuda.threshold(gpu_b_norm, 75, 255, cv2.THRESH_BINARY)

            # 선형 결합 후 다시 이진화
            gpu_mix = cv2.cuda.addWeighted(gpu_bin_white, W_WHITE, gpu_bin_yellow, W_YELLOW, 0)
            _, gpu_lane_mask = cv2.cuda.threshold(gpu_mix, BIN_AFTER_MIX_THRESH, 255, cv2.THRESH_BINARY)

            # Canny -> Hough 입력
            gpu_edges = canny_detector.detect(gpu_lane_mask)
            edges_bgr = cv2.cuda.cvtColor(gpu_edges, cv2.COLOR_GRAY2BGR).download()
            edges_gray = cv2.cvtColor(edges_bgr, cv2.COLOR_BGR2GRAY)

            lines = cv2.HoughLinesP(
                edges_gray,
                rho=1,
                theta=np.pi / 180.0,
                threshold=50,
                minLineLength=50,
                maxLineGap=20
            )

            # -------- 3) 디버그 캔버스 준비 --------
            line_vis = bgr_frame.copy()

            # -------- 4) 세그멘테이션 기반 차선 영역 칠하기 --------
            if len(lane_polys) == 1:
                single_pts = lane_polys[0]["pts"]
                if len(single_pts) >= 3:
                    fill_polygon_alpha(
                        line_vis,
                        single_pts,
                        color_bgr=(200,180,80),   # 연한 청록/회색 느낌
                        alpha=0.4
                    )

            elif len(lane_polys) >= 2:
                all_pts_list = []
                for lp in lane_polys:
                    if "pts" in lp and isinstance(lp["pts"], np.ndarray) and len(lp["pts"]) >= 3:
                        all_pts_list.append(lp["pts"])

                if len(all_pts_list) >= 2:
                    merged_pts = np.vstack(all_pts_list)  # shape: (N,2)
                    hull = cv2.convexHull(merged_pts.reshape(-1,1,2))
                    hull_pts = hull.reshape(-1,2).astype(np.int32)

                    fill_polygon_alpha(
                        line_vis,
                        hull_pts,
                        color_bgr=(200,180,80),
                        alpha=0.45
                    )
                else:
                    # 유효한 polygon이 사실상 하나뿐인 경우 fallback
                    for lp in lane_polys:
                        if "pts" in lp and isinstance(lp["pts"], np.ndarray) and len(lp["pts"]) >= 3:
                            fill_polygon_alpha(
                                line_vis,
                                lp["pts"],
                                color_bgr=(200,180,80),
                                alpha=0.4
                            )
                            break

            # -------- 5) 기존 기울기/레이 기반 가이드 계산 & 시각화 --------
            matches = []
            for t in targets:
                tx, ty = t["x"], t["y"]
                mres = match_point_to_hough(tx, ty, lines)
                matches.append({
                    "pt": (tx, ty),
                    "side": t.get("side","unknown"),
                    "slope": mres["slope"],
                    "best_line": mres["best_line"],
                    "near_pt": mres["near_pt"]
                })

            # slope fallback: 한쪽 slope 없으면 다른쪽 slope 복사
            if len(matches) == 2:
                s0 = matches[0]["slope"]
                s1 = matches[1]["slope"]

                def slope_is_valid(s):
                    if s is None:
                        return False
                    if s in [np.inf, -np.inf]:
                        return True
                    if isinstance(s, float) and np.isnan(s):
                        return False
                    return True

                if (not slope_is_valid(s0)) and slope_is_valid(s1):
                    matches[0]["slope"] = s1
                if (not slope_is_valid(s1)) and slope_is_valid(s0):
                    matches[1]["slope"] = s0

            # slope_now: 첫 번째 대상 기준
            slope_now = matches[0]["slope"] if len(matches) >= 1 else None

            # EWMA 업데이트로 smoothed_slope 추적
            if slope_now is not None:
                if smoothed_slope is None:
                    smoothed_slope = slope_now
                else:
                    smoothed_slope = alpha_ewma * slope_now + (1 - alpha_ewma) * smoothed_slope

            # 디버그 요소들 (점, Hough 최적선, 근접점)  <-- 레이 시각화만 제거
            for idx, m in enumerate(matches):
                (tx, ty) = m["pt"]
                slope_val = m["slope"]
                best_line = m["best_line"]
                near_pt = m["near_pt"]

                # 기준점 (빨강)
                cv2.circle(line_vis, (tx, ty), 6, (0, 0, 255), -1)

                # Hough에서 가장 가까운 실제 선분 (초록/노랑)
                if best_line is not None:
                    x1, y1, x2, y2 = best_line
                    color_seg = (0,255,0) if idx == 0 else (0,255,255)
                    cv2.line(line_vis, (x1, y1), (x2, y2), color_seg, 3)

                # 그 선분상의 최근접 투영점(파랑)
                if near_pt is not None:
                    nx, ny = map(int, near_pt)
                    cv2.circle(line_vis, (nx, ny), 5, (255, 0, 0), -1)

                # 이전에는 여기서 레이(보라색) 그렸음.
                # 지금은 경계 교차점만 계산하고 화면에는 안 그림.
                if slope_val is not None:
                    _bx, _by = ray_to_border(tx, ty, slope_val, frame_w, frame_h)
                    # no cv2.line(...)

            # -------- 6) 조향각 비교 / 텍스트 오버레이 / 궤적 HUD --------
            if (smoothed_slope is None) or (smoothed_slope in [np.inf, -np.inf]):
                required_turn = None
                turn_ok = True
            else:
                # slope -> 요구 조향각(deg) 단순 매핑
                required_turn = K_GAIN * (-smoothed_slope)

                # 현재 조향각 대비 충분한가 체크
                if abs(steering_angle_deg) + TURN_MARGIN >= abs(required_turn):
                    turn_ok = True
                else:
                    turn_ok = False

            # 현재 조향각 유지 시 예상 주행 궤적을 Ackermann 모델로 HUD에 그리기
            draw_current_trajectory_ackermann(
                line_vis,
                steer_angle_signed_deg=steering_angle_deg,
                frame_w=frame_w,
                frame_h=frame_h,
                wheelbase_px=200.0,
                step_px=5.0,
                max_forward_px=300.0,
                color_bgr=(180, 0, 255),
                thickness=2,
            )

            txt1 = (
                f"slope_now: {slope_now:.3f}"
                if (slope_now is not None and slope_now not in [np.inf, -np.inf])
                else "slope_now: None/inf"
            )
            txt2 = (
                f"slope_smooth: {smoothed_slope:.3f}"
                if (smoothed_slope is not None and smoothed_slope not in [np.inf, -np.inf])
                else "slope_smooth: None/inf"
            )
            txt3 = f"steer_deg(signed): {steering_angle_deg:.2f}"
            txt4 = (
                "required_turn: None"
                if required_turn is None
                else f"required_turn: {required_turn:.2f} deg"
            )
            txt5 = (
                "status: N/A"
                if required_turn is None
                else ("status: OK" if turn_ok else "status: TURN MORE")
            )

            cv2.putText(line_vis, txt1, (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(line_vis, txt2, (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(line_vis, txt3, (10, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.putText(line_vis, txt4, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            status_color = (0,255,0) if (required_turn is None or turn_ok) else (0,0,255)
            cv2.putText(line_vis, txt5, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            # -------- 7) 최종 디버그 뷰 --------
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
