import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO
from jetcam.csi_camera import CSICamera
from tars_config import get_roi_slice, LANE_WIDTH_PX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS

# def get_roi_slice(H_img):
#     y0 = H_img // 2 
#     return slice(y0, H_img)
    
class LaneCenterTracker:

    # def __init__(self, lane_width_px: float, poly_deg: int = 2):
    #     self.lane_width_px = lane_width_px      # 고정 차로 폭(픽셀)
    #     self.poly_deg = poly_deg
    #     self.left_coef  = None                  # np.ndarray | None
    #     self.right_coef = None
    #     self.center_px  = None                  # float | None
    #     self.left_y_range = None
    #     self.right_y_range = None
    
    def __init__(self, lane_width_px: float = LANE_WIDTH_PX, poly_deg: int = 2):
        # lane_width_px 기본값을 tars_config.py의 값으로 설정
        self.lane_width_px = lane_width_px
        self.poly_deg = poly_deg
        self.left_coef  = None
        self.right_coef = None
        self.center_px  = None
        self.left_y_range = None
        self.right_y_range = None

    # ──────────────────────────────────────────────────────────
    # 헬퍼: 단일 mask → (poly_coef, x_bottom)
    # ──────────────────────────────────────────────────────────
    def _mask_to_poly_bottom(self, mask_bin: np.ndarray, thr=0.5):
        """mask_bin : 2-D uint8 (0/1)"""
        ys, xs = np.nonzero(mask_bin)
        if xs.size < self.poly_deg + 1:           # 데이터 부족
            return None, None, None
        # (1) 다항식(x) = f(y)  ← y가 세로(row)
        coef = np.polyfit(ys, xs, self.poly_deg)
        # (2) 이미지 하단(row = H-1)에서 x 좌표 예측
        y_bottom = mask_bin.shape[0] - 1
        x_bottom = np.polyval(coef, y_bottom)
        y_min, y_max = int(np.min(ys)), int(np.max(ys))
        return coef, float(x_bottom), (y_min, y_max)

    # ──────────────────────────────────────────────────────────
    # 메인 업데이트
    # ──────────────────────────────────────────────────────────
    def update(self, results, *, roi=None, thr=0.5):
        """
        results : ultralytics.engine.results.Results (단일 이미지)
        returns : center_x (float) or None
        """
        r = results
        H_img, W_img = r.orig_shape[:2]
        roi = roi or get_roi_slice(H_img)  # 설정 모듈의 함수 사용
        
        if r.masks is None or len(r.masks.data) == 0:     # CASE 0
            return self.center_px                         # 그대로 유지

        # (A) 모든 mask → 원본 해상도(픽셀)로 upsample
        masks = r.masks.data.cpu().numpy()                # (N,160,160) etc.
        full_bin = [cv2.resize((m > thr).astype(np.uint8),
                                (W_img, H_img),
                                cv2.INTER_NEAREST)[roi]
                    for m in masks]    

        # (B) 픽셀 수 기준 내림차순 정렬
        pix_counts = [m.sum() for m in full_bin]
        idx_sorted = np.argsort(pix_counts)[::-1]

        # ──────────────────────────────────────────
        # CASE 2 : 두 개 이상 검출
        # ──────────────────────────────────────────
        if len(idx_sorted) >= 2:
            idx_top2 = idx_sorted[:2]
            infos = [self._mask_to_poly_bottom(full_bin[i]) for i in idx_top2]
            # infos = [(coef, x_bottom), ...]  길이가 2
            if any(c is None for c, _, _ in infos):
                return self.center_px                    # 실패 → 이전 값
            # 왼쪽/오른쪽 분리
            (coef1, x1, y_range1), (coef2, x2, y_range2) = infos
            if x1 < x2:
                self.left_coef, self.right_coef = coef1, coef2
                self.left_y_range, self.right_y_range = y_range1, y_range2
                x_left, x_right = x1, x2
            else:
                self.left_coef, self.right_coef = coef2, coef1
                self.left_y_range, self.right_y_range = y_range2, y_range1
                x_left, x_right = x2, x1
            self.center_px = (x_left + x_right) / 2.0
            return self.center_px

        # ──────────────────────────────────────────
        # CASE 1 : 한 개만 검출
        # ──────────────────────────────────────────
        idx = idx_sorted[0]
        coef_new, x_new, y_range_new = self._mask_to_poly_bottom(full_bin[idx])
        if coef_new is None:
            return self.center_px

        # (1) 새 차선이 이전 왼/오 중 어느 쪽과 가까운지 계산
        dist_left  = abs(x_new - np.polyval(self.left_coef,  H_img-1)) \
                     if self.left_coef is not None else np.inf
        dist_right = abs(x_new - np.polyval(self.right_coef, H_img-1)) \
                     if self.right_coef is not None else np.inf

        if dist_left < dist_right:          # 왼쪽 차선으로 간주
            self.left_coef = coef_new
            self.left_y_range = y_range_new
            x_left  = x_new
            x_right = x_left + self.lane_width_px
            self.right_coef = None          # 추후 갱신 예정
            self.right_y_range = None
        else:                               # 오른쪽 차선
            self.right_coef = coef_new
            self.right_y_range = y_range_new
            x_right = x_new
            x_left  = x_right - self.lane_width_px
            self.left_coef = None
            self.left_y_range = None

        self.center_px = (x_left + x_right) / 2.0
        return self.center_px
        
        
def draw_polyline(img, coef, color, thickness=3, n_pts=50, y_range=None):
    """
    img      : BGR 영상 (in-place 로 그려짐)
    coef     : np.ndarray (x = f(y) 다항식 계수)
    color    : (B,G,R)
    """
    if coef is None:
        return

    H, W = img.shape[:2]
    if y_range is not None:
        y_min, y_max = y_range
    else:
        y_min, y_max = 0, H-1
    ys = np.linspace(y_min, y_max, n_pts)
    xs = np.polyval(coef, ys)

    # 이미지 경계 바깥은 버림
    pts = np.stack([xs, ys], axis=-1)
    pts = pts[(pts[:,0] >= 0) & (pts[:,0] < W)]   # x in [0, W)
    if len(pts) < 2:              # 선 최소 2점
        return

    cv2.polylines(
        img,
        [pts.astype(int)],
        isClosed=False,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA
    )

def draw_transparent_box(frame, x1, y1, x2, y2, color, alpha=0.3):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

def draw_middle_curve(frame, left_curve, right_curve, thickness=5):
    if left_curve is None or right_curve is None:
        return
    if len(left_curve) != len(right_curve):
        return
    mid_pts = []
    for (lx, ly), (rx, ry) in zip(left_curve, right_curve):
        if ly == ry:
            x = int((lx + rx) / 2)
            mid_pts.append((x, ly))
    for i in range(1, len(mid_pts)):
        cv2.line(frame, mid_pts[i-1], mid_pts[i], (0,255,0), thickness)
    # 디버깅: 중간 곡선 포인트에 작은 원 표시
    for pt in mid_pts:
        cv2.circle(frame, pt, 5, (0,255,0), -1)

def main():
    device     = 0 if torch.cuda.is_available() else "cpu"
    model      = YOLO("lane.pt")
    model.fuse()

    print("Initializing camera...")
    # WIDTH, HEIGHT = 1280, 720
    # FPS = 30
    # camera = CSICamera(width=WIDTH, height=HEIGHT, capture_fps=FPS)
    camera = CSICamera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT, capture_fps=CAMERA_FPS)
    camera.running = True

    print("Waiting for camera to be ready...")
    while camera.value is None:
        time.sleep(0.1)
    print("✅ Camera ready!")

    prev_t = time.time()
    # tracker = LaneCenterTracker(lane_width_px=700, poly_deg=1)
    tracker = LaneCenterTracker(lane_width_px=LANE_WIDTH_PX, poly_deg=POLY_DEG)
    while True:
        frame = camera.value

        if frame is None:
            time.sleep(0.01)
            continue

        H_img, W_img = frame.shape[:2]
        # roi = slice(H_img // 2, H_img)
        roi = get_roi_slice(H_img)  # 설정 모듈의 함수 사용
        # 3-A) YOLO 추론
        results = model.predict(frame, device = device)

        # 결과
        r = results[0]
        center_x = tracker.update(r, roi=roi)
        boxes   = r.boxes
        clss    = boxes.cls.cpu().numpy()
        confs   = boxes.conf.cpu().numpy()
        xyxy    = boxes.xyxy.cpu().numpy()
        frame_roi = frame[roi]
        draw_polyline(frame_roi, tracker.left_coef,  (255,0,0), y_range=tracker.left_y_range)
        draw_polyline(frame_roi, tracker.right_coef, (0,255,0), y_range=tracker.right_y_range)
        if center_x is not None:
            cv2.circle(frame, (int(center_x), frame.shape[0]-1), 6,
                       (0,0,255), -1, cv2.LINE_AA)
        # 3-B) 박스 그리기
        for (x1, y1, x2, y2), cls_id, conf in zip(xyxy, clss, confs):
            label = f"{model.names[int(cls_id)]} {conf:.2f}"
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(frame, p1, p2, (0, 255, 255), 2)
            cv2.putText(frame, label, (p1[0], p1[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("YOLO-Detect", frame)
        cv2.imshow("Lane-ROI", frame_roi)
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break

    # Release camera resources
    # cap.release()
    print("Releasing camera resources...")
    camera.running = False
    # Attempt to explicitly release the underlying capture object
    if hasattr(camera, 'cap') and hasattr(camera.cap, 'release'):
        print("Attempting to release camera.cap...")
        camera.cap.release()
        print("✅ camera.cap released.")
    # Explicitly delete the camera object to ensure resource release
    del camera
    cv2.destroyAllWindows()
    print("Camera stopped.")

if __name__ == "__main__":
    main()
