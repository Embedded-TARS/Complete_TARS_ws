# 01_tars_perception.py

from __future__ import annotations
from typing import Optional, Tuple, List
import cv2
from ultralytics import YOLO
import torch
import time
import numpy as np

class LaneDetectionModel:
    """차선 감지를 위한 YOLO 모델을 로드하고 관리합니다."""
    
    def __init__(self, model_path="lane.pt", lane_class_id=12):
        self.lane_class_id = lane_class_id
        self.device = 0 if torch.cuda.is_available() else "cpu"
        print(f"YOLO 모델을 {self.device}에 로드하는 중...")
        self.model = YOLO(model_path).to(self.device)
        print(f"✅ YOLO 모델이 {self.device}에 로드되었습니다.")
    
    def predict(self, frame):
        return self.model.predict(frame, device=self.device, conf=0.5, iou=0.45)

# 이미지 하단 절반을 ROI (관심 영역)로 설정하는 함수
def get_roi_slice(H_img):
    y0 = int(H_img * 3 / 4)
    return slice(y0, H_img)

# 다항식 계수를 이용하여 이미지에 차선을 그리는 함수
def draw_polyline(img, coef, color, thickness=5, n_pts=50, y_range_roi=None, roi_offset_y=0):
    if coef is None:
        return

    H_img, W_img = img.shape[:2]
    
    if y_range_roi is not None:
        y_min_roi, y_max_roi = y_range_roi
        y_min_abs = y_min_roi + roi_offset_y
        y_max_abs = y_max_roi + roi_offset_y
    else:
        y_min_abs, y_max_abs = roi_offset_y, H_img - 1

    ys_abs = np.linspace(y_max_abs, y_min_abs, n_pts).astype(int)
    ys_roi = ys_abs - roi_offset_y
    xs_roi = np.polyval(coef, ys_roi)
    
    pts = np.stack([xs_roi, ys_abs], axis=-1)
    pts = pts[(pts[:,0] >= 0) & (pts[:,0] < W_img) & (pts[:,1] >=0) & (pts[:,1] < H_img)]

    if len(pts) < 2:
        return

    cv2.polylines(
        img,
        [pts.astype(int)],
        isClosed=False,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA
    )

class LanePerception:
    """차선 중앙을 추적하고 필요에 따라 EMA(지수 이동 평균)로 스무딩합니다."""
    
    def __init__(self, *, lane_width_px: float, ema_alpha: Optional[float] = 0.5,
                 poly_deg: Optional[int] = 2):
        self.lane_width_px = lane_width_px
        self.ema_alpha = ema_alpha
        self.poly_deg = poly_deg

        self.center_raw: Optional[float] = None
        self.center_px: Optional[float] = None

        self.left_x_prev = self.left_y_prev = None
        self.right_x_prev = self.right_y_prev = None

        self.left_coef: Optional[np.ndarray] = None
        self.right_coef: Optional[np.ndarray] = None
        self.left_mask: Optional[np.ndarray] = None
        self.right_mask: Optional[np.ndarray] = None
        
        # 성능 모니터링 변수 추가
        self.last_update_time = time.time()
        self.fps = 0
        
        # 디버깅 텍스트 색상 정의
        self.text_color = (0, 200, 255)  # 주황색
        self.fps_color = (0, 255, 0)     # 녹색
        self.time_color = (255, 255, 0)  # 노랑색
        self.lane_status_color = (255, 255, 255)  # 흰색

    @staticmethod
    def _mask_bottom_x(mask_bin: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        ys, xs = np.nonzero(mask_bin)
        if xs.size == 0:
            return None, None
        y_max = ys.max()
        x_mean = xs[ys == y_max].mean()
        return float(y_max), float(x_mean)

    def update(self, results, *, roi: Optional[slice] = None, thr: float = 0.5) -> Optional[float]:
        start_time = time.time()
        
        r = results
        H_img, W_img = r.orig_shape[:2]
        roi = roi or slice(0, H_img)

        if r.masks is None or len(r.masks.data) == 0:
            return self.center_px

        masks_np = r.masks.data.cpu().numpy()
        full = [cv2.resize((m > thr).astype(np.uint8), (W_img, H_img), cv2.INTER_NEAREST)[roi]
                 for m in masks_np]
        
        counts = np.array([m.sum() for m in full])
        if counts.max() == 0:
            return self.center_px
        
        order = counts.argsort()[::-1]

        def _fit(mask_bin: np.ndarray) -> Optional[np.ndarray]:
            if self.poly_deg is None:
                return None
            ys, xs = np.nonzero(mask_bin)
            if xs.size < self.poly_deg + 1:
                return None

            # Add check for unique y-coordinates
            if np.unique(ys).size < self.poly_deg + 1:
                # print(f"Debug: Not enough unique y-coordinates for polyfit (needed: {self.poly_deg + 1}, got: {np.unique(ys).size})") # Optional debug print
                return None

            return np.polyfit(ys, xs, self.poly_deg)

        # 두 개의 가장 큰 마스크를 차선으로 시도
        if len(order) >= 2:
            i1, i2 = order[:2]
            y1, x1 = self._mask_bottom_x(full[i1])
            y2, x2 = self._mask_bottom_x(full[i2])
            
            if None not in (x1, x2):
                if x1 < x2:
                    x_left, y_left, x_right, y_right = x1, y1, x2, y2
                    self.left_coef, self.left_mask  = _fit(full[i1]), full[i1]
                    self.right_coef, self.right_mask = _fit(full[i2]), full[i2]
                else:
                    x_left, y_left, x_right, y_right = x2, y2, x1, y1
                    self.left_coef, self.left_mask  = _fit(full[i2]), full[i2]
                    self.right_coef, self.right_mask = _fit(full[i1]), full[i1]
                    
                self.left_x_prev, self.left_y_prev = x_left, y_left + roi.start
                self.right_x_prev, self.right_y_prev = x_right, y_right + roi.start
                
                self.center_raw = (x_left + x_right) / 2.0
                
                # FPS 계산
                current_time = time.time()
                elapsed = current_time - self.last_update_time
                self.last_update_time = current_time
                self.fps = 1.0 / elapsed if elapsed > 0 else 0
                
                return self._apply_ema()
                
        # 하나의 마스크만 사용 (싱글 차선)
        idx = order[0]
        yb, xb = self._mask_bottom_x(full[idx])
        
        if xb is None:
            return self.center_px
            
        xb_g = xb
        yb_g = yb + roi.start
        
        dl = abs(xb_g - self.left_x_prev) if self.left_x_prev is not None else np.inf
        dr = abs(xb_g - self.right_x_prev) if self.right_x_prev is not None else np.inf
        
        if dl < dr:
            self.left_x_prev, self.left_y_prev = xb_g, yb_g
            x_left, x_right = xb_g, xb_g + self.lane_width_px
            if self.poly_deg is not None:
                self.left_coef, self.left_mask = _fit(full[idx]), full[idx]
                self.right_coef, self.right_mask = None, None
        else:
            self.right_x_prev, self.right_y_prev = xb_g, yb_g
            x_right, x_left = xb_g, xb_g - self.lane_width_px
            if self.poly_deg is not None:
                self.left_coef, self.left_mask = None, None
                self.right_coef, self.right_mask = _fit(full[idx]), full[idx]
                
        self.center_raw = (x_left + x_right) / 2.0
        
        # FPS 계산
        current_time = time.time()
        elapsed = current_time - self.last_update_time
        self.last_update_time = current_time
        self.fps = 1.0 / elapsed if elapsed > 0 else 0
        
        return self._apply_ema()

    def _apply_ema(self) -> Optional[float]:
        if self.center_raw is None:
            return self.center_px

        if self.ema_alpha is None or self.center_px is None:
            self.center_px = self.center_raw
        else:
            a = self.ema_alpha
            self.center_px = a * self.center_raw + (1 - a) * self.center_px

        return self.center_px
    
    # 차선 인식 시각화 함수 추가
    def visualize_lanes(self, frame, deviation=0.0, steering=0.0, roi=None):
        """
        차선 인식 결과를 시각화합니다.
        
        Parameters
        ----------
        frame : np.ndarray
            원본 이미지 프레임
        deviation : float
            계산된 차선 중앙과 이미지 중앙 간의 편차
        steering : float
            계산된 조향 각도
            
        Returns
        -------
        np.ndarray
            차선 시각화가 추가된 이미지 프레임
        """
        # 원본 프레임 복사
        frame_with_lanes = frame.copy()

        y_offset = roi.start or 0
        
        # 왼쪽 차선 그리기
        if self.left_coef is not None and self.left_mask is not None:
            draw_polyline_masked(
                frame_with_lanes, 
                self.left_coef, 
                self.left_mask, 
                (0, 255, 0),  # 녹색
                thickness=3,
                y_offset=y_offset
            )
        
        # 오른쪽 차선 그리기
        if self.right_coef is not None and self.right_mask is not None:
            draw_polyline_masked(
                frame_with_lanes, 
                self.right_coef, 
                self.right_mask, 
                (0, 0, 255),  # 빨간색
                thickness=3,
                y_offset=y_offset
            )
        
        # 차선 중앙 표시 (인식된 경우)
        if self.center_px is not None:
            img_center_x = frame.shape[1] // 2 - 11
            
            # 차선 중앙점 좌표
            center_point = (int(self.center_px), frame.shape[0] - 30)
            # 이미지 중앙점 좌표
            img_center_point = (img_center_x, frame.shape[0] - 30)
            
            # 차선 중앙 점 그리기
            cv2.circle(frame_with_lanes, center_point, 5, (0, 255, 255), -1)  # 노란색 원
            
            # 이미지 중앙 점 그리기
            cv2.circle(frame_with_lanes, img_center_point, 5, (255, 0, 255), -1)  # 핑크색 원
            
            # 두 점을 연결하는 선
            cv2.line(frame_with_lanes, center_point, img_center_point, (255, 255, 255), 2)
        
        # 디버깅 정보 표시
        cv2.putText(frame_with_lanes, f"dev:{deviation:.2f} steer:{steering:.2f}", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.text_color, 2)
        cv2.putText(frame_with_lanes, f"{self.fps:.1f} FPS", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.fps_color, 2)
        
        # Lane status display
        lane_status = "Both lanes detected"
        if self.left_coef is not None and self.right_coef is None:
            lane_status = "Only left lane detected"
        elif self.left_coef is None and self.right_coef is not None:
            lane_status = "Only right lane detected"
        elif self.left_coef is None and self.right_coef is None:
            lane_status = "Lane not detected"
            
        print(f"Lane Status: {lane_status}")

        cv2.putText(frame_with_lanes, lane_status, 
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.lane_status_color, 2)
        
        return frame_with_lanes

# 마스크 영역에 맞춰 다항식으로 차선을 그리는 함수
def draw_polyline_masked(img: np.ndarray,
                         coef: Optional[np.ndarray],
                         mask_bin: Optional[np.ndarray],
                         color: Tuple[int, int, int],
                         *,
                         n_pts: int = 50,
                         thickness: int = 3,
                         y_offset=0) -> None:
    if coef is None or mask_bin is None:
        return
        
    H, W = img.shape[:2]
    
    valid_rows = np.where(mask_bin.sum(axis=1) > 0)[0]
    if valid_rows.size == 0:
        return
        
    ys = np.linspace(valid_rows.min(), valid_rows.max(), n_pts)
    xs = np.polyval(coef, ys)
    pts = np.stack([xs, ys + y_offset], axis=-1)
    
    in_img = (pts[:, 0] >= 0) & (pts[:, 0] < W)
    pts = pts[in_img]
    
    if len(pts) < 2:
        return
        
    cv2.polylines(img, [pts.astype(np.int32)], False, color, thickness, cv2.LINE_AA)