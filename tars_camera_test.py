import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO
from jetcam.csi_camera import CSICamera
from tars_config import (
    get_roi_slice, LANE_WIDTH_PX, CAMERA_WIDTH, CAMERA_HEIGHT, 
    CAMERA_FPS, POLY_DEG, EMA_ALPHA, ROI_RATIO, MIDDLE_LANE_WIDTH_PX
)

# def get_roi_slice(H_img):
#     y0 = H_img // 2 
#     return slice(y0, H_img)
    
class LanePerception:
    """차선 중앙을 추적하고 필요에 따라 EMA(지수 이동 평균)로 스무딩합니다."""
    
    def __init__(self, *, lane_width_px: float = LANE_WIDTH_PX, 
                 middle_lane_width_px: float = MIDDLE_LANE_WIDTH_PX,
                 ema_alpha: float = EMA_ALPHA,
                 poly_deg: int = POLY_DEG):
        print(f"\nLanePerception 초기화:")
        print(f"lane_width_px: {lane_width_px}")
        print(f"middle_lane_width_px: {middle_lane_width_px}")
        print(f"ema_alpha: {ema_alpha}")
        print(f"poly_deg: {poly_deg}")
        
        self.lane_width_px = lane_width_px
        self.middle_lane_width_px = middle_lane_width_px
        self.ema_alpha = ema_alpha
        self.poly_deg = poly_deg

        self.center_raw: float = None
        self.center_px: float = None
        self.center_coef: np.ndarray = None  # 중앙 차선의 다항식 계수

        self.left_x_prev = self.left_y_prev = None
        self.right_x_prev = self.right_y_prev = None

        self.left_coef: np.ndarray = None
        self.right_coef: np.ndarray = None
        self.left_mask: np.ndarray = None
        self.right_mask: np.ndarray = None
        
        # Add previous coefficients for continuity
        self.left_coef_prev: np.ndarray = None
        self.right_coef_prev: np.ndarray = None
        
        # 성능 모니터링 변수 추가
        self.last_update_time = time.time()
        self.fps = 0
        
        # 디버깅 텍스트 색상 정의
        self.text_color = (0, 200, 255)  # 주황색
        self.fps_color = (0, 255, 0)     # 녹색
        self.time_color = (255, 255, 0)  # 노랑색
        self.lane_status_color = (255, 255, 255)  # 흰색
        
        # 디버깅 정보 추가
        self.left_confidence = 0.0
        self.right_confidence = 0.0
        self.mask_debug = None

    def _get_dynamic_lane_width(self, y: float, H: int) -> float:
        """y좌표에 따라 동적으로 차선 폭을 계산합니다."""
        # y가 0에 가까울수록 middle_lane_width_px에 가깝게, H에 가까울수록 lane_width_px에 가깝게
        ratio = y / H
        return self.middle_lane_width_px * (1 - ratio) + self.lane_width_px * ratio

    def _calculate_center_coefficients(self):
        """왼쪽과 오른쪽 차선의 계수를 사용하여 중앙 차선의 계수를 계산합니다."""
        if self.left_coef is not None and self.right_coef is not None:
            # 두 차선의 계수의 평균을 계산
            self.center_coef = (self.left_coef + self.right_coef) / 2
        elif self.left_coef is not None:
            # 왼쪽 차선만 있는 경우, lane_width_px만큼 오른쪽으로 이동
            self.center_coef = self.left_coef.copy()
            self.center_coef[-1] += self.lane_width_px / 2
        elif self.right_coef is not None:
            # 오른쪽 차선만 있는 경우, lane_width_px만큼 왼쪽으로 이동
            self.center_coef = self.right_coef.copy()
            self.center_coef[-1] -= self.lane_width_px / 2

    @staticmethod
    def _mask_bottom_x(mask_bin: np.ndarray) -> tuple[float, float]:
        ys, xs = np.nonzero(mask_bin)
        if xs.size == 0:
            return None, None
        y_max = ys.max()
        x_mean = xs[ys == y_max].mean()
        return float(y_max), float(x_mean)

    def update(self, results, *, roi: slice = None, thr: float = 0.5) -> float:
        start_time = time.time()
        
        r = results
        H_img, W_img = r.orig_shape[:2]
        roi = roi or slice(0, H_img)
        
        print(f"\n프레임 크기: {W_img}x{H_img}")
        print(f"ROI 영역: {roi}")

        if r.masks is None or len(r.masks.data) == 0:
            print("⚠️ 마스크가 감지되지 않았습니다")
            return self.center_px

        masks_np = r.masks.data.cpu().numpy()
        full = [cv2.resize((m > thr).astype(np.uint8), (W_img, H_img), cv2.INTER_NEAREST)[roi]
                 for m in masks_np]
        
        counts = np.array([m.sum() for m in full])
        if counts.max() == 0:
            print("⚠️ 마스크에 유효한 픽셀이 없습니다")
            return self.center_px
        
        print(f"✅ {len(counts)}개의 마스크가 감지되었습니다")
        print(f"마스크 픽셀 수: {counts}")
        
        order = counts.argsort()[::-1]
        
        # 디버깅: 마스크 시각화를 위한 이미지 생성
        self.mask_debug = np.zeros((H_img, W_img, 3), dtype=np.uint8)
        for i, mask in enumerate(full):
            color = (0, 255, 0) if i == order[0] else (0, 0, 255) if i == order[1] else (255, 0, 0)
            self.mask_debug[roi][mask > 0] = color

        def _fit(mask_bin: np.ndarray, is_left: bool = True) -> np.ndarray:
            if self.poly_deg is None:
                return None
                
            H, W = mask_bin.shape
            
            # Define number of horizontal strips
            n_strips = 30 # Increased number of strips for better sampling
            strip_height = H // n_strips
            
            sample_ys = []
            sample_xs = []
            
            # Process each horizontal strip from bottom to top
            for i in range(n_strips - 1, -1, -1):
                y_start = i * strip_height
                y_end = min(y_start + strip_height, H)
                
                # Get pixels within the current strip
                strip_mask = mask_bin[y_start:y_end, :]
                
                # Find non-zero pixels in the strip
                strip_ys, strip_xs = np.nonzero(strip_mask)
                
                if len(strip_xs) > 5: # Require a minimum number of points in a strip
                    # Use the median x-coordinate as the representative point for the strip
                    # And use the median y-coordinate relative to the full mask height
                    representative_x = np.median(strip_xs)
                    representative_y = y_start + np.median(strip_ys) # Global y-coordinate
                    
                    sample_xs.append(representative_x)
                    sample_ys.append(representative_y)
            
            if len(sample_xs) < self.poly_deg + 1:
                return None

            try:
                sample_xs = np.array(sample_xs)
                sample_ys = np.array(sample_ys)
                
                # Sort by y-coordinate before fitting
                sort_idx = np.argsort(sample_ys)
                sample_ys = sample_ys[sort_idx]
                sample_xs = sample_xs[sort_idx]
                
                # --- User's Proposed Logic for Straight Lane Handling ---
                # 1. 직선 피팅으로 기본 직선성 평가
                linear_coef = np.polyfit(sample_ys, sample_xs, 1)
                linear_predict = np.polyval(linear_coef, sample_ys)
                linear_error = np.mean(np.abs(sample_xs - linear_predict))
                
                # 2. 3차 다항식 피팅 수행
                poly_coef = np.polyfit(sample_ys, sample_xs, self.poly_deg)
                poly_predict = np.polyval(poly_coef, sample_ys)
                poly_error = np.mean(np.abs(sample_xs - poly_predict))
                
                # 3. 차선 직선성 판단 및 계수 조정
                # 직선 피팅과 다항식 피팅의 오차 비교
                error_ratio = poly_error / (linear_error + 1e-6)  # 0에 의한 나눗셈 방지
                
                # 오른쪽 차선에 대한 특별 처리 (이미지에서 문제가 된 부분)
                if not is_left:
                    print(f"오른쪽 차선 분석: 직선 오차={linear_error:.2f}, 다항식 오차={poly_error:.2f}, 비율={error_ratio:.2f}")
                    
                    # 오른쪽 차선이 충분히 직선적인 경우 (더 엄격한 기준 적용)
                    # 직선 오차가 작거나 다항식이 크게 개선하지 않는 경우 (error_ratio가 높음)
                    if linear_error < 10.0 or error_ratio > 0.7: 
                        print("오른쪽 차선이 직선으로 판단됨: 계수 조정 적용")
                        
                        # 원래 3차 계수 저장 (디버깅용)
                        # orig_coef = poly_coef.copy()
                        
                        # 방법 1: 고차항 계수를 0에 가깝게 설정 (직선에 가깝게)
                        poly_coef[0] = 0.0  # 3차항 제거
                        poly_coef[1] = 0.0  # 2차항 제거
                        
                        # 또는 방법 2: 선형 계수를 기반으로 다항식 재구성 (주석 처리)
                        # poly_coef = np.zeros_like(poly_coef)
                        # poly_coef[-1] = linear_coef[1]  # 상수항
                        # poly_coef[-2] = linear_coef[0]  # 1차항
                        # poly_coef[0] = 0.0  # 3차항 제거
                        # poly_coef[1] = 0.0  # 2차항 제거
                        
                        # print(f"원래 계수: {orig_coef}")
                        # print(f"조정된 계수: {poly_coef}")
                else:
                    # 왼쪽 차선은 덜 엄격하게 처리 (곡선 허용)
                    if linear_error < 5.0 and error_ratio > 0.9:
                        # 매우 직선적인 경우만 고차항 감소
                        poly_coef[0] *= 0.5
                        poly_coef[1] *= 0.7
                # --- End User's Proposed Logic ---

                
                # Validate the fit (using the potentially adjusted poly_coef)
                # Generate y values covering the entire height of the mask
                y_full_range = np.linspace(0, H, 100)
                x_fit_full = np.polyval(poly_coef, y_full_range)
                
                # Check if the curve is reasonable across the full height
                if np.any(np.abs(np.diff(x_fit_full)) > 50): # Increased tolerance for full range
                    print("Validation failed: Rapid change in full range fit.")
                    return None
                    
                # Check if the curve stays within reasonable horizontal bounds across the full height
                if np.any(x_fit_full < -W_img*0.5) or np.any(x_fit_full > W_img*1.5): # Allow some extrapolation but not extreme
                     print("Validation failed: Fit out of horizontal bounds.")
                     return None
                
                # --- Check fit consistency within the sampled range ---
                # Check the fit only over the sampled y-range to ensure it matches the visible part well
                # y_sampled_range = np.linspace(sample_ys.min(), sample_ys.max(), 50) # This range is not needed for residual check
                
                # Compare fitted points with sampled points within the visible range
                # This helps ensure the polynomial doesn't diverge wildly within the visible area
                predicted_xs_at_samples = np.polyval(poly_coef, sample_ys)
                residuals = np.abs(sample_xs - predicted_xs_at_samples)
                if np.any(residuals > 20): # Residuals check (20 pixels tolerance)
                     print("Validation failed: High residuals within sampled range.")
                     return None
                
                # Additional check for partial lanes: prevent strong convergence towards center at invisible end
                # This check needs to be carefully applied after coefficients are potentially adjusted
                # Let's re-evaluate this check based on the adjusted coefficients
                # if len(sample_ys) < H * 0.7: # If less than 70% of height is sampled - This condition might be too strict
                # Let's check extrapolation above the highest sampled point if it's not near the top of the mask
                if sample_ys.min() > H * 0.1: # If the highest sampled point is below the top 10% of the mask
                     y_extrapolate = np.linspace(max(0, sample_ys.min() - 50), sample_ys.min(), 20) # Check extrapolation above sampled points (up to 50 pixels)
                     # Ensure y_extrapolate has at least 2 points
                     if len(y_extrapolate) > 1:
                        x_extrapolate = np.polyval(poly_coef, y_extrapolate)

                        # Calculate slope near the top of the sampled points
                        if len(sample_ys) > 1:
                           # Slope of the fitted curve at the highest sampled sampled point
                           fitted_slope_at_top_sample = np.polyval(np.polyder(poly_coef), sample_ys.min())

                           # Check for excessive change in x or slope in the extrapolated part
                           # A simplified check: is the extrapolated segment moving sharply towards the center?
                           # This depends on the expected lane behavior - typically lanes diverge or stay parallel.
                           # If left lane, x should not decrease significantly when going up (towards smaller y)
                           # If right lane, x should not increase significantly when going up (towards smaller y)

                           # Check for large absolute slope in the extrapolated region
                           extrapolated_slopes = np.abs(np.diff(x_extrapolate) / np.diff(y_extrapolate + 1e-6))
                           if np.any(extrapolated_slopes > np.abs(fitted_slope_at_top_sample) * 3 + 1.0): # Check if extrapolated slope is much larger
                                # This check is heuristic and might need tuning
                                print("Validation failed: Excessive slope in extrapolated region.")
                                # return None # Temporarily disabling this aggressive check

                           # Check if the end point of extrapolation is moving towards center too much
                           if is_left and x_extrapolate[0] > sample_xs[0] + 40: # Left lane extrapolating too far right
                                print("Validation failed: Left lane extrapolating too far right.")
                                return None
                           if not is_left and x_extrapolate[0] < sample_xs[0] - 40: # Right lane extrapolating too far left
                                print("Validation failed: Right lane extrapolating too far left.")
                                return None

                return poly_coef # Return the calculated or adjusted coefficient
            except np.linalg.LinAlgError:
                print("Fitting failed due to LinAlgError.")
                return None

        # 두 개의 가장 큰 마스크를 차선으로 시도
        if len(order) >= 2:
            i1, i2 = order[:2]
            y1, x1 = self._mask_bottom_x(full[i1])
            y2, x2 = self._mask_bottom_x(full[i2])
            
            if None not in (x1, x2):
                if x1 < x2:
                    x_left, y_left, x_right, y_right = x1, y1, x2, y2
                    self.left_coef, self.left_mask  = _fit(full[i1], is_left=True), full[i1]
                    self.right_coef, self.right_mask = _fit(full[i2], is_left=False), full[i2]
                else:
                    x_left, y_left, x_right, y_right = x2, y2, x1, y1
                    self.left_coef, self.left_mask  = _fit(full[i2], is_left=True), full[i2]
                    self.right_coef, self.right_mask = _fit(full[i1], is_left=False), full[i1]
                    
                self.left_x_prev, self.left_y_prev = x_left, y_left + roi.start
                self.right_x_prev, self.right_y_prev = x_right, y_right + roi.start
                
                self.center_raw = (x_left + x_right) / 2.0
                
                # 중앙 차선 계수 계산
                self._calculate_center_coefficients()
                
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
            # If _mask_bottom_x fails, no valid points found in the mask
            # Fallback to previous center or None
            return self.center_px

        xb_g = xb
        yb_g = yb + roi.start # Global y-coordinate of the bottom point

        dl = abs(xb_g - self.left_x_prev) if self.left_x_prev is not None else np.inf
        dr = abs(xb_g - self.right_x_prev) if self.right_x_prev is not None else np.inf

        # Determine if the detected lane is likely left or right based on distance to previous bottom points
        is_likely_left = dl < dr

        # Fit the single detected lane using the improved _fit function
        detected_coef = _fit(full[idx], is_left=is_likely_left)
        detected_mask = full[idx]

        # --- Handle single lane detection and potential estimation ---
        estimated_left_coef = None
        estimated_right_coef = None

        if detected_coef is not None:
             if is_likely_left:
                 self.left_coef, self.left_mask = detected_coef, detected_mask
                 # Try to estimate right lane based on previous right lane if available
                 if self.right_coef_prev is not None:
                     # Calculate a potential horizontal shift using a point near the bottom of the detected mask
                     y_bottom_sample = max(0, len(full[idx]) - 10) # Sample near the bottom
                     if y_bottom_sample < len(full[idx]):
                         x_current_at_bottom = np.polyval(detected_coef, y_bottom_sample)
                         if self.left_coef_prev is not None:
                              x_prev_left_at_bottom = np.polyval(self.left_coef_prev, y_bottom_sample)
                              shift = x_current_at_bottom - x_prev_left_at_bottom
                              estimated_right_coef = self.right_coef_prev.copy()
                              estimated_right_coef[-1] += shift # Apply shift
                              # Basic validation for estimated lane (check if it's within plausible bounds)
                              y_check = np.linspace(0, len(full[idx]), 10)
                              x_check = np.polyval(estimated_right_coef, y_check)
                              if np.any(x_check < -full[idx].shape[1]*0.2) or np.any(x_check > full[idx].shape[1]*1.2): # Wider bounds for estimated lane
                                   estimated_right_coef = None # Invalidate if out of bounds
                         else:
                             # If previous left is not available, use dynamic lane width
                             estimated_right_coef = detected_coef.copy()
                             # Use dynamic lane width based on y-coordinate
                             dynamic_width = self._get_dynamic_lane_width(y_bottom_sample, len(full[idx]))
                             estimated_right_coef[-1] += dynamic_width
                 # Assign estimated coefficient, allowing it to be None if estimation failed
                 self.right_coef = estimated_right_coef
                 self.left_coef_prev = self.left_coef
                 self.right_coef_prev = self.right_coef # Update previous with potentially None right coef

             else: # is_likely_right
                 self.right_coef, self.right_mask = detected_coef, detected_mask
                 # Try to estimate left lane based on previous left lane if available
                 if self.left_coef_prev is not None:
                      y_bottom_sample = max(0, len(full[idx]) - 10) # Sample near the bottom
                      if y_bottom_sample < len(full[idx]):
                         x_current_at_bottom = np.polyval(detected_coef, y_bottom_sample)
                         if self.right_coef_prev is not None:
                             x_prev_right_at_bottom = np.polyval(self.right_coef_prev, y_bottom_sample)
                             shift = x_current_at_bottom - x_prev_right_at_bottom
                             estimated_left_coef = self.left_coef_prev.copy()
                             estimated_left_coef[-1] += shift # Apply shift
                             # Basic validation for estimated lane
                             y_check = np.linspace(0, len(full[idx]), 10)
                             x_check = np.polyval(estimated_left_coef, y_check)
                             if np.any(x_check < -full[idx].shape[1]*0.2) or np.any(x_check > full[idx].shape[1]*1.2):
                                  estimated_left_coef = None
                         else:
                             # If previous right is not available, use dynamic lane width
                             estimated_left_coef = detected_coef.copy()
                             # Use dynamic lane width based on y-coordinate
                             dynamic_width = self._get_dynamic_lane_width(y_bottom_sample, len(full[idx]))
                             estimated_left_coef[-1] -= dynamic_width

                 # Assign estimated coefficient
                 self.left_coef = estimated_left_coef
                 self.left_coef_prev = self.left_coef # Update previous with potentially None left coef
                 self.right_coef_prev = self.right_coef

        else: # If detected_coef is None (fitting failed for the single detected mask)
            # Revert to previous coefficients if available to maintain continuity
            self.left_coef = self.left_coef_prev
            self.right_coef = self.right_coef_prev
            # Clear masks as the current frame didn't provide a valid fit
            self.left_mask = None
            self.right_mask = None

        # Recalculate center line based on potentially updated or estimated lane coefficients
        self._calculate_center_coefficients()

        # Update previous bottom points (optional, could keep using the bottom of current mask if available)
        if self.left_coef is not None:
             y_bottom_left = len(full[idx]) - 1 # Assuming bottom of mask corresponds to bottom of ROI/image
             self.left_x_prev, self.left_y_prev = np.polyval(self.left_coef, y_bottom_left), y_bottom_left + roi.start
        elif self.left_x_prev is not None: # If left coef is None but prev exists, keep prev bottom point
             pass # Keep previous bottom point
        else: # If no left coef and no prev, reset bottom point
             self.left_x_prev, self.left_y_prev = None, None

        if self.right_coef is not None:
             y_bottom_right = len(full[idx]) - 1
             self.right_x_prev, self.right_y_prev = np.polyval(self.right_coef, y_bottom_right), y_bottom_right + roi.start
        elif self.right_x_prev is not None: # If right coef is None but prev exists, keep prev bottom point
             pass # Keep previous bottom point
        else: # If no right coef and no prev, reset bottom point
             self.right_x_prev, self.right_y_prev = None, None

        # FPS 계산
        current_time = time.time()
        elapsed = current_time - self.last_update_time
        self.last_update_time = current_time
        self.fps = 1.0 / elapsed if elapsed > 0 else 0

        return self._apply_ema()

    def _apply_ema(self) -> float:
        if self.center_raw is None:
            return self.center_px

        if self.ema_alpha is None or self.center_px is None:
            self.center_px = self.center_raw
        else:
            a = self.ema_alpha
            self.center_px = a * self.center_raw + (1 - a) * self.center_px

        return self.center_px
    
    def visualize_lanes(self, frame, deviation=0.0, steering=0.0, roi=None):
        """
        차선 인식 결과를 시각화합니다.
        """
        frame_with_lanes = frame.copy()

        y_offset = roi.start or 0
        H_img, W_img = frame.shape[:2]
        
        # ROI 영역 시각화
        cv2.line(frame_with_lanes, (0, y_offset), (W_img, y_offset), (0, 255, 0), 2)
        cv2.putText(frame_with_lanes, f"ROI (y={y_offset})", (10, y_offset-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
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
            # 왼쪽 차선 좌표 디버깅 정보
            if self.left_x_prev is not None and self.left_y_prev is not None:
                cv2.putText(frame_with_lanes, f"Left: ({self.left_x_prev:.1f}, {self.left_y_prev:.1f})", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.text_color, 2)
            cv2.putText(frame_with_lanes, "Left lane", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
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
            # 오른쪽 차선 좌표 디버깅 정보
            if self.right_x_prev is not None and self.right_y_prev is not None:
                cv2.putText(frame_with_lanes, f"Right: ({self.right_x_prev:.1f}, {self.right_y_prev:.1f})", 
                           (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.text_color, 2)
            cv2.putText(frame_with_lanes, "Right lane", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 중앙 차선 그리기
        if self.center_coef is not None:
            # 중앙 차선의 마스크 생성 (왼쪽과 오른쪽 차선의 마스크를 결합)
            center_mask = None # Initialize center_mask
            valid_mask_shape = None

            # Determine a valid mask shape from left or right mask
            if self.left_mask is not None and isinstance(self.left_mask, np.ndarray) and self.left_mask.ndim == 2 and self.left_mask.shape[0] > 0 and self.left_mask.shape[1] > 0:
                 valid_mask_shape = self.left_mask.shape
            elif self.right_mask is not None and isinstance(self.right_mask, np.ndarray) and self.right_mask.ndim == 2 and self.right_mask.shape[0] > 0 and self.right_mask.shape[1] > 0:
                 valid_mask_shape = self.right_mask.shape

            # Create center_mask only if a valid shape is found
            if valid_mask_shape is not None:
                center_mask = np.zeros(valid_mask_shape, dtype=np.uint8) # Create zero mask with valid shape
                if self.left_mask is not None and isinstance(self.left_mask, np.ndarray):
                    # Ensure masks have the same shape before combining
                    if self.left_mask.shape == valid_mask_shape:
                        center_mask = np.logical_or(center_mask, self.left_mask).astype(np.uint8)

                if self.right_mask is not None and isinstance(self.right_mask, np.ndarray):
                     if self.right_mask.shape == valid_mask_shape:
                        center_mask = np.logical_or(center_mask, self.right_mask).astype(np.uint8)

            # Only attempt to draw if center_mask was successfully created and has content
            if center_mask is not None and np.sum(center_mask) > 0:
                draw_polyline_masked(
                    frame_with_lanes,
                    self.center_coef,
                    center_mask,
                    (255, 255, 0),  # 노란색
                    thickness=3,
                    y_offset=y_offset
                )
            
            # 중앙 차선 다항식 계수 출력
            coef_str = "Center: "
            for i, coef in enumerate(self.center_coef):
                if i == 0:
                    coef_str += f"{coef:.3f}x^{len(self.center_coef)-1}"
                else:
                    coef_str += f" + {coef:.3f}x^{len(self.center_coef)-1-i}"
            cv2.putText(frame_with_lanes, coef_str, 
                       (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.text_color, 2)
        
        # 차선 중앙 표시 (인식된 경우)
        if self.center_px is not None:
            img_center_x = frame.shape[1] // 2
            
            # 차선 중앙점 좌표
            center_point = (int(self.center_px), frame.shape[0] - 30)
            # 이미지 중앙점 좌표
            img_center_point = (img_center_x, frame.shape[0] - 30)
            
            # 차선 중앙 점 그리기
            cv2.circle(frame_with_lanes, center_point, 8, (0, 255, 255), -1)  # 노란색 원
            cv2.putText(frame_with_lanes, f"Lane center: {self.center_px:.1f}", 
                       (int(self.center_px) + 10, frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # 이미지 중앙 점 그리기
            cv2.circle(frame_with_lanes, img_center_point, 8, (255, 0, 255), -1)  # 핑크색 원
            cv2.putText(frame_with_lanes, f"Image center: {img_center_x}", 
                       (img_center_x + 10, frame.shape[0] - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # 두 점을 연결하는 선
            cv2.line(frame_with_lanes, center_point, img_center_point, (255, 255, 255), 2)
            
            # 중앙점 좌표 디버깅 정보
            cv2.putText(frame_with_lanes, f"Center: {self.center_px:.1f}", 
                       (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.text_color, 2)
        
        # 디버깅 정보 표시
        cv2.putText(frame_with_lanes, f"dev:{deviation:.2f} steer:{steering:.2f}", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.text_color, 2)
        cv2.putText(frame_with_lanes, f"{self.fps:.1f} FPS", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.fps_color, 2)
        
        # Lane status display
        lane_status = "상태: "
        if self.left_coef is not None and self.right_coef is not None:
            lane_status += "양쪽 차선 감지됨"
        elif self.left_coef is not None:
            lane_status += "왼쪽 차선만 감지됨"
        elif self.right_coef is not None:
            lane_status += "오른쪽 차선만 감지됨"
        else:
            lane_status += "차선 감지 안됨"
            
        print(f"Lane Status: {lane_status}")

        cv2.putText(frame_with_lanes, lane_status, 
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.lane_status_color, 2)
        
        # 마스크 디버깅 이미지 표시
        if self.mask_debug is not None:
            cv2.imshow("Lane Masks", self.mask_debug)
        
        return frame_with_lanes

def draw_polyline_masked(img: np.ndarray,
                         coef: np.ndarray,
                         mask_bin: np.ndarray,
                         color: tuple[int, int, int],
                         *,
                         n_pts: int = 50,
                         thickness: int = 3,
                         y_offset=0) -> None:
    if coef is None or mask_bin is None:
        return
    
    # Add check for valid mask_bin shape and dimension
    if not isinstance(mask_bin, np.ndarray) or mask_bin.ndim != 2 or mask_bin.shape[0] == 0 or mask_bin.shape[1] == 0:
        # print("Debug: Invalid mask_bin provided to draw_polyline_masked")
        return
        
    H, W = img.shape[:2]
    
    # 마스크에서 유효한 y 좌표 찾기
    valid_rows = np.where(mask_bin.sum(axis=1) > 0)[0]
    if valid_rows.size == 0:
        return
        
    # y 좌표 범위 설정
    y_min = valid_rows.min()
    y_max = valid_rows.max()
    
    # y 좌표를 균일하게 분포
    ys = np.linspace(y_min, y_max, n_pts)
    
    # 다항식으로 x 좌표 계산
    xs = np.polyval(coef, ys)
    
    # 좌표 쌍 생성 및 y_offset 적용
    pts = np.stack([xs, ys + y_offset], axis=-1)
    
    # 이미지 경계 내의 점만 선택
    in_img = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
    pts = pts[in_img]
    
    if len(pts) < 2:
        return
        
    # 차선 그리기
    cv2.polylines(img, [pts.astype(np.int32)], False, color, thickness, cv2.LINE_AA)
    
    # 디버깅: 차선 포인트 표시
    for pt in pts:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 2, color, -1)

class LaneDetectionModel:
    """차선 감지를 위한 YOLO 모델을 로드하고 관리합니다."""
    
    def __init__(self, model_path="lane.pt", lane_class_id=12):
        self.lane_class_id = lane_class_id
        self.device = 0 if torch.cuda.is_available() else "cpu"
        print(f"YOLO 모델을 {self.device}에 로드하는 중...")
        try:
            self.model = YOLO(model_path).to(self.device)
            print(f"✅ YOLO 모델이 {self.device}에 로드되었습니다.")
            print("\n모델 정보:")
            print(f"클래스 수: {len(self.model.names)}")
            print("클래스 목록:")
            for idx, name in self.model.names.items():
                print(f"  {idx}: {name}")
            print(f"\n차선 클래스 ID: {self.lane_class_id}")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            raise
    
    def predict(self, frame):
        results = self.model.predict(frame, device=self.device, conf=0.5, iou=0.45)
        # 디버깅: 예측 결과 출력
        r = results[0]
        print("\nYOLO Detection Results:")
        print(f"Number of detections: {len(r.boxes)}")
        if len(r.boxes) > 0:
            print("Detected classes:", [self.model.names[int(cls)] for cls in r.boxes.cls])
            print("Confidence scores:", r.boxes.conf.cpu().numpy())
            print("Bounding boxes:", r.boxes.xyxy.cpu().numpy())
        if r.masks is not None:
            print(f"Number of masks: {len(r.masks.data)}")
            print(f"Mask shapes: {[m.shape for m in r.masks.data]}")
        else:
            print("No masks detected")
        return results

def camera_test_main():
    # YOLO 모델 초기화
    print("Initializing YOLO model...")
    try:
        lane_model = LaneDetectionModel(model_path="lane.pt", lane_class_id=12)
    except Exception as e:
        print(f"Failed to initialize YOLO model: {e}")
        return

    print("Initializing camera...")
    try:
        camera = CSICamera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT, capture_fps=CAMERA_FPS)
        camera.running = True
        print(f"Camera initialized with settings: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}fps")
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return

    print("Waiting for camera to be ready...")
    wait_start = time.time()
    while camera.value is None:
        if time.time() - wait_start > 5.0:  # 5초 타임아웃
            print("Camera initialization timeout!")
            return
        print("Waiting for camera frame...")
        time.sleep(0.1)
    print("✅ Camera ready!")

    prev_t = time.time()
    lane_perception = LanePerception()
    
    print("Starting main loop...")
    frame_count = 0
    while True:
        frame = camera.value

        if frame is None:
            print("No frame received from camera")
            time.sleep(0.01)
            continue

        frame_count += 1
        if frame_count % 30 == 0:  # 30프레임마다 상태 출력
            print(f"\nProcessing frame {frame_count}")
            print(f"Frame shape: {frame.shape}")

        H_img, W_img = frame.shape[:2]
        roi = get_roi_slice(H_img)
        
        # YOLO 추론
        results = lane_model.predict(frame)

        # 결과 처리
        r = results[0]
        center_x = lane_perception.update(r, roi=roi)
        
        # 차선 시각화
        frame_with_lanes = lane_perception.visualize_lanes(frame, roi=roi)
        
        # ROI 영역 시각화
        roi_vis = frame.copy()
        y_start = roi.start
        cv2.line(roi_vis, (0, y_start), (W_img, y_start), (0, 255, 0), 2)
        cv2.putText(roi_vis, f"ROI (y={y_start})", (10, y_start-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # ROI 영역에서의 처리 결과 시각화
        frame_roi = frame[roi].copy()
        
        # FPS 계산 및 표시
        current_time = time.time()
        fps = 1.0 / (current_time - prev_t) if current_time - prev_t > 0 else 0
        prev_t = current_time
        cv2.putText(frame_with_lanes, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 박스 그리기
        boxes = r.boxes
        if len(boxes) > 0:
            clss = boxes.cls.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            
            for (x1, y1, x2, y2), cls_id, conf in zip(xyxy, clss, confs):
                label = f"{lane_model.model.names[int(cls_id)]} {conf:.2f}"
                p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.rectangle(frame_with_lanes, p1, p2, (0, 255, 255), 2)
                cv2.putText(frame_with_lanes, label, (p1[0], p1[1]-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Lane Detection", frame_with_lanes)
        cv2.imshow("Lane-ROI", frame_roi)
        cv2.imshow("ROI Visualization", roi_vis)
        
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            print("Quit signal received")
            break

    print("Releasing camera resources...")
    camera.running = False
    if hasattr(camera, 'cap') and hasattr(camera.cap, 'release'):
        print("Attempting to release camera.cap...")
        camera.cap.release()
        print("✅ camera.cap released.")
    del camera
    cv2.destroyAllWindows()
    print("Camera stopped.")

if __name__ == "__main__":
    camera_test_main()
