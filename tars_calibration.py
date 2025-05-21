import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO
from jetcam.csi_camera import CSICamera
from tars_config import (
    get_roi_slice, CAMERA_WIDTH, CAMERA_HEIGHT, 
    CAMERA_FPS, ROI_RATIO, POLY_DEG
)

class LaneCalibration:
    """차선 폭을 측정하고 칼리브레이션하는 클래스"""
    
    def __init__(self):
        self.bottom_lane_widths = []  # 하단부 측정된 차선 폭 저장
        self.middle_lane_widths = []  # 중간부 측정된 차선 폭 저장
        self.left_x_prev = None
        self.right_x_prev = None
        self.calibration_complete = False
        self.measurement_count = 0
        self.required_measurements = 10  # 필요한 측정 횟수
        
        # 디버깅 텍스트 색상 정의
        self.text_color = (0, 200, 255)  # 주황색
        self.status_color = (0, 255, 0)  # 녹색
        self.point_color = (0, 0, 255)  # 빨간색
        self.curve_color = (255, 0, 0)  # 파란색
        
        # 차선 곡선 저장
        self.left_curve = None
        self.right_curve = None
        
    def update(self, results, roi: slice = None, thr: float = 0.5) -> bool:
        """차선 감지 결과를 처리하고 차선 폭을 측정합니다."""
        r = results
        H_img, W_img = r.orig_shape[:2]
        roi = roi or slice(0, H_img)
        
        if r.masks is None or len(r.masks.data) == 0:
            print("⚠️ 마스크가 감지되지 않았습니다")
            return False

        masks_np = r.masks.data.cpu().numpy()
        full = [cv2.resize((m > thr).astype(np.uint8), (W_img, H_img), cv2.INTER_NEAREST)[roi]
                 for m in masks_np]
        
        counts = np.array([m.sum() for m in full])
        if counts.max() == 0:
            print("⚠️ 마스크에 유효한 픽셀이 없습니다")
            return False
        
        order = counts.argsort()[::-1]
        
        # 두 개의 가장 큰 마스크를 차선으로 시도
        if len(order) >= 2:
            i1, i2 = order[:2]
            y1_bottom, x1_bottom = self._mask_bottom_x(full[i1])
            y2_bottom, x2_bottom = self._mask_bottom_x(full[i2])
            
            # 중간 지점 측정
            y1_middle, x1_middle = self._mask_middle_x(full[i1])
            y2_middle, x2_middle = self._mask_middle_x(full[i2])
            
            if None not in (x1_bottom, x2_bottom, x1_middle, x2_middle):
                # 하단부 차선 폭 계산
                if x1_bottom < x2_bottom:
                    x_left_bottom, x_right_bottom = x1_bottom, x2_bottom
                    left_mask, right_mask = full[i1], full[i2]
                else:
                    x_left_bottom, x_right_bottom = x2_bottom, x1_bottom
                    left_mask, right_mask = full[i2], full[i1]
                
                # 중간부 차선 폭 계산
                if x1_middle < x2_middle:
                    x_left_middle, x_right_middle = x1_middle, x2_middle
                else:
                    x_left_middle, x_right_middle = x2_middle, x1_middle
                
                bottom_lane_width = abs(x_right_bottom - x_left_bottom)
                middle_lane_width = abs(x_right_middle - x_left_middle)
                
                # 차선 곡선 피팅
                self.left_curve = self._fit_lane_curve(left_mask)
                self.right_curve = self._fit_lane_curve(right_mask)
                
                # 이전 측정값과 비교하여 유효한 측정인지 확인
                if self._is_valid_measurement(bottom_lane_width, middle_lane_width):
                    self.bottom_lane_widths.append(bottom_lane_width)
                    self.middle_lane_widths.append(middle_lane_width)
                    self.measurement_count += 1
                    print(f"✅ 차선 폭 측정: 하단 {bottom_lane_width:.1f}px, 중간 {middle_lane_width:.1f}px ({self.measurement_count}/{self.required_measurements})")
                    
                    if self.measurement_count >= self.required_measurements:
                        self.calibration_complete = True
                        return True
                
                self.left_x_prev, self.right_x_prev = x_left_bottom, x_right_bottom
                
        return False
    
    def _fit_lane_curve(self, mask: np.ndarray) -> np.ndarray:
        """마스크에서 차선 곡선을 피팅합니다."""
        ys, xs = np.nonzero(mask)
        if len(ys) < POLY_DEG + 1:
            return None
        
        # y값을 기준으로 정렬
        sort_idx = np.argsort(ys)
        ys = ys[sort_idx]
        xs = xs[sort_idx]
        
        # 다항식 피팅
        coeffs = np.polyfit(ys, xs, POLY_DEG)
        return coeffs
    
    def _draw_lane_curve(self, frame: np.ndarray, coeffs: np.ndarray, color: tuple, roi_offset: int = 0) -> np.ndarray:
        """차선 곡선을 그립니다."""
        if coeffs is None:
            return frame
        
        h, w = frame.shape[:2]
        ys = np.linspace(0, h-1, h)
        xs = np.polyval(coeffs, ys)
        
        # ROI offset 적용
        ys = ys + roi_offset
        
        # 유효한 범위 내의 점만 그리기
        valid_points = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        points = np.column_stack((xs[valid_points], ys[valid_points])).astype(np.int32)
        
        # 곡선 그리기
        for i in range(len(points)-1):
            cv2.line(frame, tuple(points[i]), tuple(points[i+1]), color, 2)
        
        return frame
    
    def _is_valid_measurement(self, bottom_width: float, middle_width: float) -> bool:
        """측정된 차선 폭이 유효한지 확인합니다."""
        if not self.bottom_lane_widths:
            return True
            
        # 이전 측정값들과 비교하여 큰 차이가 나지 않는지 확인
        mean_bottom = np.mean(self.bottom_lane_widths)
        std_bottom = np.std(self.bottom_lane_widths)
        
        # 평균에서 표준편차의 2배 이상 벗어나면 유효하지 않은 측정으로 간주
        return abs(bottom_width - mean_bottom) <= 2 * std_bottom
    
    @staticmethod
    def _mask_bottom_x(mask_bin: np.ndarray) -> tuple[float, float]:
        """마스크의 하단부에서 x 좌표를 계산합니다."""
        ys, xs = np.nonzero(mask_bin)
        if xs.size == 0:
            return None, None
        y_max = ys.max()
        x_mean = xs[ys == y_max].mean()
        return float(y_max), float(x_mean)
    
    @staticmethod
    def _mask_middle_x(mask_bin: np.ndarray) -> tuple[float, float]:
        """마스크의 중간부에서 x 좌표를 계산합니다."""
        ys, xs = np.nonzero(mask_bin)
        if xs.size == 0:
            return None, None
        y_min, y_max = ys.min(), ys.max()
        y_middle = (y_min + y_max) // 2
        x_mean = xs[ys == y_middle].mean()
        return float(y_middle), float(x_mean)
    
    def get_calibrated_widths(self) -> tuple[float, float]:
        """칼리브레이션된 차선 폭을 반환합니다."""
        if not self.calibration_complete:
            return None, None
        return np.mean(self.bottom_lane_widths), np.mean(self.middle_lane_widths)
    
    def visualize(self, frame, roi: slice = None):
        """칼리브레이션 상태를 시각화합니다."""
        frame_with_info = frame.copy()
        
        # ROI 영역 표시
        roi_offset = 0
        if roi is not None:
            roi_offset = roi.start
            y_start = roi.start
            cv2.line(frame_with_info, (0, y_start), (frame.shape[1], y_start), 
                    (0, 255, 0), 2)
            cv2.putText(frame_with_info, f"ROI (y={y_start})", (10, y_start-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 칼리브레이션 상태 표시
        status = "Calibration Complete" if self.calibration_complete else f"Measuring... ({self.measurement_count}/{self.required_measurements})"
        cv2.putText(frame_with_info, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.status_color, 2)
        
        # 측정된 차선 폭 표시
        if self.bottom_lane_widths:
            bottom_mean = np.mean(self.bottom_lane_widths)
            bottom_std = np.std(self.bottom_lane_widths)
            middle_mean = np.mean(self.middle_lane_widths)
            middle_std = np.std(self.middle_lane_widths)
            
            cv2.putText(frame_with_info, f"Bottom Width: {bottom_mean:.1f} ± {bottom_std:.1f}px", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.text_color, 2)
            cv2.putText(frame_with_info, f"Middle Width: {middle_mean:.1f} ± {middle_std:.1f}px", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.text_color, 2)
            
            # 현재 인식된 차선 폭 표시
            if self.left_x_prev is not None and self.right_x_prev is not None:
                current_width = abs(self.right_x_prev - self.left_x_prev)
                cv2.putText(frame_with_info, f"Current Width: {current_width:.1f}px", 
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.text_color, 2)
                
                # 다항식 차수 표시
                cv2.putText(frame_with_info, f"Polynomial Degree: {POLY_DEG}", 
                           (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.text_color, 2)
                
                # 차선 곡선 그리기 (ROI offset 적용)
                if self.left_curve is not None:
                    frame_with_info = self._draw_lane_curve(frame_with_info, self.left_curve, self.curve_color, roi_offset)
                if self.right_curve is not None:
                    frame_with_info = self._draw_lane_curve(frame_with_info, self.right_curve, self.curve_color, roi_offset)
                
                # 하단부 측정선 표시
                bottom_y = frame.shape[0] - 1
                cv2.line(frame_with_info, 
                        (int(self.left_x_prev), bottom_y),
                        (int(self.right_x_prev), bottom_y),
                        self.point_color, 2)
                
                # 중간부 측정선 표시
                if roi is not None:
                    middle_y = (roi.start + roi.stop) // 2
                    cv2.line(frame_with_info,
                            (int(self.left_x_prev), middle_y),
                            (int(self.right_x_prev), middle_y),
                            self.point_color, 2)
                
                # 차선 양 끝점 표시
                if self.left_curve is not None and self.right_curve is not None:
                    # 하단 끝점
                    bottom_y = frame.shape[0] - 1
                    left_bottom_x = int(np.polyval(self.left_curve, bottom_y - roi_offset))
                    right_bottom_x = int(np.polyval(self.right_curve, bottom_y - roi_offset))
                    cv2.circle(frame_with_info, (left_bottom_x, bottom_y), 5, self.point_color, -1)
                    cv2.circle(frame_with_info, (right_bottom_x, bottom_y), 5, self.point_color, -1)
                    
                    # 상단 끝점
                    top_y = roi_offset
                    left_top_x = int(np.polyval(self.left_curve, 0))
                    right_top_x = int(np.polyval(self.right_curve, 0))
                    cv2.circle(frame_with_info, (left_top_x, top_y), 5, self.point_color, -1)
                    cv2.circle(frame_with_info, (right_top_x, top_y), 5, self.point_color, -1)
        
        return frame_with_info

    def reset(self):
        """칼리브레이션을 초기화합니다."""
        self.bottom_lane_widths = []
        self.middle_lane_widths = []
        self.left_x_prev = None
        self.right_x_prev = None
        self.calibration_complete = False
        self.measurement_count = 0
        self.left_curve = None
        self.right_curve = None
        print("\n🔄 칼리브레이션이 초기화되었습니다.")

def update_config_file(bottom_width: float, middle_width: float):
    """설정 파일을 업데이트합니다."""
    try:
        with open('tars_config.py', 'r') as f:
            lines = f.readlines()
        
        # LANE_WIDTH_PX와 MIDDLE_LANE_WIDTH_PX 값을 찾아 업데이트
        bottom_found = False
        middle_found = False
        
        for i, line in enumerate(lines):
            if line.strip().startswith('LANE_WIDTH_PX'):
                lines[i] = f'LANE_WIDTH_PX = {int(bottom_width)}  # Calibrated bottom lane width\n'
                bottom_found = True
            elif line.strip().startswith('MIDDLE_LANE_WIDTH_PX'):
                lines[i] = f'MIDDLE_LANE_WIDTH_PX = {int(middle_width)}  # Calibrated middle lane width\n'
                middle_found = True
        
        # MIDDLE_LANE_WIDTH_PX가 없으면 추가
        if not middle_found:
            for i, line in enumerate(lines):
                if line.strip().startswith('LANE_WIDTH_PX'):
                    lines.insert(i + 1, f'MIDDLE_LANE_WIDTH_PX = {int(middle_width)}  # Calibrated middle lane width\n')
                    break
        
        with open('tars_config.py', 'w') as f:
            f.writelines(lines)
        
        print(f"✅ 설정 파일이 업데이트되었습니다.")
        print(f"   - 하단 차선 폭: {int(bottom_width)}px")
        print(f"   - 중간 차선 폭: {int(middle_width)}px")
    except Exception as e:
        print(f"❌ 설정 파일 업데이트 실패: {e}")

def main():
    print("🚗 차선 폭 칼리브레이션을 시작합니다...")
    print("차량을 직선 차로 중앙에 위치시켜주세요.")
    print("'c' 키를 눌러 칼리브레이션을 다시 시작할 수 있습니다.")
    print("'q' 키를 눌러 종료하세요.")
    
    # YOLO 모델 초기화
    print("\nYOLO 모델을 로드하는 중...")
    try:
        model = YOLO("lane.pt")
        print("✅ YOLO 모델이 로드되었습니다.")
    except Exception as e:
        print(f"❌ YOLO 모델 로드 실패: {e}")
        return

    # 카메라 초기화
    print("\n카메라를 초기화하는 중...")
    try:
        camera = CSICamera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT, capture_fps=CAMERA_FPS)
        camera.running = True
        print(f"✅ 카메라가 초기화되었습니다. ({CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}fps)")
    except Exception as e:
        print(f"❌ 카메라 초기화 실패: {e}")
        return

    # 카메라 준비 대기
    print("\n카메라 준비 중...")
    wait_start = time.time()
    while camera.value is None:
        if time.time() - wait_start > 5.0:
            print("❌ 카메라 초기화 타임아웃!")
            return
        time.sleep(0.1)
    print("✅ 카메라가 준비되었습니다.")

    # 칼리브레이션 객체 초기화
    calibration = LaneCalibration()
    
    print("\n칼리브레이션을 시작합니다...")
    while True:
        frame = camera.value
        if frame is None:
            continue

        H_img = frame.shape[0]
        roi = get_roi_slice(H_img)
        
        # YOLO 추론
        results = model.predict(frame, conf=0.5, iou=0.45)
        
        # 칼리브레이션 업데이트
        calibration.update(results[0], roi=roi)
        
        # 결과 시각화
        frame_with_info = calibration.visualize(frame, roi)
        cv2.imshow("Lane Calibration", frame_with_info)
        
        # 칼리브레이션이 완료되면 설정 파일 업데이트
        if calibration.calibration_complete:
            bottom_width, middle_width = calibration.get_calibrated_widths()
            if bottom_width is not None and middle_width is not None:
                update_config_file(bottom_width, middle_width)
                print("\n칼리브레이션이 완료되었습니다.")
                print("'c' 키를 눌러 다시 시작하거나 'q' 키를 눌러 종료하세요.")
        
        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            calibration.reset()
            print("\n🔄 칼리브레이션을 다시 시작합니다...")

    # 리소스 정리
    print("\n리소스를 정리하는 중...")
    camera.running = False
    if hasattr(camera, 'cap') and hasattr(camera.cap, 'release'):
        camera.cap.release()
    cv2.destroyAllWindows()
    print("✅ 프로그램이 종료되었습니다.")

if __name__ == "__main__":
    main()
