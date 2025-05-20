import cv2
import numpy as np
import time
import argparse
import os
import sys
import traceback

try:
    import torch
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ ultralytics 또는 torch를 가져올 수 없습니다.")
    print("pip install ultralytics torch 명령으로 설치하세요.")

# 설정값
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_FPS = 30
ROI_RATIO = 0.5  # ROI 시작 위치 (이미지 높이의 비율)
LANE_WIDTH_PX = 700  # 예상 차선 폭(픽셀)
POLY_DEG = 2  # 차선 곡선 피팅에 사용할 다항식 차수
EMA_ALPHA = 0.8  # 차선 중앙 위치 스무딩 계수

def get_roi_slice(img_height):
    """이미지 높이에 기반하여 ROI 슬라이스를 계산합니다."""
    roi_start = int(img_height * ROI_RATIO)
    return slice(roi_start, img_height)

class LanePerception:
    """차선 중앙을 추적하고 필요에 따라 EMA(지수 이동 평균)로 스무딩합니다."""
    
    def __init__(self, lane_width_px=LANE_WIDTH_PX, ema_alpha=EMA_ALPHA, poly_deg=POLY_DEG):
        self.lane_width_px = lane_width_px
        self.ema_alpha = ema_alpha
        self.poly_deg = poly_deg

        self.center_raw = None
        self.center_px = None

        self.left_x_prev = self.left_y_prev = None
        self.right_x_prev = self.right_y_prev = None

        self.left_coef = None
        self.right_coef = None
        self.left_mask = None
        self.right_mask = None
        
        # 성능 모니터링 변수
        self.last_update_time = time.time()
        self.fps = 0
        
        # 디버깅 텍스트 색상 정의
        self.text_color = (0, 200, 255)  # 주황색
        self.fps_color = (0, 255, 0)     # 녹색
        self.time_color = (255, 255, 0)  # 노랑색
        self.lane_status_color = (255, 255, 255)  # 흰색

    @staticmethod
    def _mask_bottom_x(mask_bin):
        """마스크의 하단 중심 x 좌표를 찾습니다."""
        if mask_bin is None or mask_bin.size == 0:
            return None, None
            
        ys, xs = np.nonzero(mask_bin)
        if xs.size == 0:
            return None, None
        y_max = ys.max()
        x_mean = xs[ys == y_max].mean()
        return float(y_max), float(x_mean)

    def update(self, results, roi=None, thr=0.5):
        """
        YOLO 결과를 기반으로 차선 중앙을 업데이트합니다.
        """
        start_time = time.time()
        
        if results is None:
            return self.center_px
            
        r = results[0]  # 첫 번째 결과만 사용
        
        try:
            H_img, W_img = r.orig_shape[:2]
            roi = roi or slice(0, H_img)

            if r.masks is None or len(r.masks.data) == 0:
                return self.center_px

            # 마스크 데이터가 있는 경우에만 처리
            masks_np = r.masks.data.cpu().numpy()
            
            # 마스크 리사이즈 및 ROI 적용
            full = []
            for m in masks_np:
                try:
                    # 이진 마스크로 변환
                    binary_mask = (m > thr).astype(np.uint8)
                    
                    # 원본 크기로 리사이즈
                    resized = cv2.resize(binary_mask, (W_img, H_img), cv2.INTER_NEAREST)
                    
                    # ROI 적용
                    roi_mask = resized[roi]
                    
                    full.append(roi_mask)
                except Exception as e:
                    print(f"마스크 처리 중 오류: {e}")
                    continue
            
            if not full:  # 마스크가 없으면 종료
                return self.center_px
            
            counts = np.array([m.sum() for m in full])
            if counts.max() == 0:
                return self.center_px
            
            order = counts.argsort()[::-1]

            def _fit(mask_bin):
                """마스크로부터 다항식 계수를 계산합니다."""
                if self.poly_deg is None or mask_bin is None:
                    return None
                    
                ys, xs = np.nonzero(mask_bin)
                if xs.size < self.poly_deg + 1:
                    return None

                # 고유한 y 좌표 체크
                if np.unique(ys).size < self.poly_deg + 1:
                    return None

                try:
                    return np.polyfit(ys, xs, self.poly_deg)
                except Exception:
                    # 다항식 피팅 오류 - 더 낮은 차수로 시도
                    try:
                        return np.polyfit(ys, xs, 1)  # 1차 다항식으로 시도
                    except Exception:
                        return None

            # 두 개의 가장 큰 마스크를 차선으로 처리
            if len(order) >= 2:
                try:
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
                except Exception as e:
                    print(f"두 차선 처리 중 오류: {e}")
                    # 오류 발생 시 싱글 차선 로직으로 진행
                    
            # 하나의 마스크만 사용 (싱글 차선)
            try:
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
            except Exception as e:
                print(f"싱글 차선 처리 중 오류: {e}")
                return self.center_px
            
            # FPS 계산
            current_time = time.time()
            elapsed = current_time - self.last_update_time
            self.last_update_time = current_time
            self.fps = 1.0 / elapsed if elapsed > 0 else 0
            
            return self._apply_ema()
        except Exception as e:
            print(f"차선 업데이트 중 오류: {e}")
            return self.center_px

    def _apply_ema(self):
        """중앙 위치에 EMA 스무딩을 적용합니다."""
        if self.center_raw is None:
            return self.center_px

        if self.ema_alpha is None or self.center_px is None:
            self.center_px = self.center_raw
        else:
            a = self.ema_alpha
            self.center_px = a * self.center_raw + (1 - a) * self.center_px

        return self.center_px
    
    def draw_polyline_masked(self, img, coef, mask_bin, color, thickness=3, y_offset=0):
        """
        마스크 영역에 맞춰 다항식으로 차선을 그립니다.
        """
        if coef is None or mask_bin is None:
            return
            
        H, W = img.shape[:2]
        
        # 유효한 행 찾기
        valid_rows = np.where(np.sum(mask_bin, axis=1) > 0)[0]
        if len(valid_rows) == 0:
            return
            
        # 다항식으로 곡선 그리기
        try:
            y_min, y_max = np.min(valid_rows), np.max(valid_rows)
            y_points = np.linspace(y_min, y_max, 20)
            x_points = np.polyval(coef, y_points)
            
            # 이미지 내부의 점만 유지
            valid_points = []
            for x, y in zip(x_points, y_points):
                x_int, y_int = int(x), int(y + y_offset)
                if 0 <= x_int < W and 0 <= y_int < H:
                    valid_points.append((x_int, y_int))
            
            if len(valid_points) >= 2:
                # 점들을 연결하는 선 그리기
                for i in range(1, len(valid_points)):
                    cv2.line(img, valid_points[i-1], valid_points[i], color, thickness, cv2.LINE_AA)
        except Exception as e:
            print(f"곡선 그리기 오류: {e}")
    
    def visualize_lanes(self, frame, deviation=0.0, steering=0.0, roi=None):
        """
        차선 인식 결과를 시각화합니다.
        """
        # 원본 프레임 복사
        frame_with_lanes = frame.copy()

        y_offset = roi.start if roi else 0
        
        # 왼쪽 차선 그리기
        if self.left_coef is not None and self.left_mask is not None:
            try:
                self.draw_polyline_masked(
                    frame_with_lanes, 
                    self.left_coef, 
                    self.left_mask, 
                    (0, 255, 0),  # 녹색
                    thickness=3,
                    y_offset=y_offset
                )
            except Exception as e:
                print(f"왼쪽 차선 그리기 오류: {e}")
        
        # 오른쪽 차선 그리기
        if self.right_coef is not None and self.right_mask is not None:
            try:
                self.draw_polyline_masked(
                    frame_with_lanes, 
                    self.right_coef, 
                    self.right_mask, 
                    (0, 0, 255),  # 빨간색
                    thickness=3,
                    y_offset=y_offset
                )
            except Exception as e:
                print(f"오른쪽 차선 그리기 오류: {e}")
        
        # 차선 중앙 표시 (인식된 경우)
        if self.center_px is not None:
            try:
                img_center_x = frame.shape[1] // 2
                
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
            except Exception as e:
                print(f"중앙점 그리기 오류: {e}")
        
        # 디버깅 정보 표시
        try:
            cv2.putText(frame_with_lanes, f"dev:{deviation:.2f} steer:{steering:.2f}", 
                      (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.text_color, 2)
            cv2.putText(frame_with_lanes, f"{self.fps:.1f} FPS", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.fps_color, 2)
        except Exception as e:
            print(f"텍스트 그리기 오류: {e}")
        
        # Lane status display
        try:
            lane_status = "Both lanes detected"
            if self.left_coef is not None and self.right_coef is None:
                lane_status = "Only left lane detected"
            elif self.left_coef is None and self.right_coef is not None:
                lane_status = "Only right lane detected"
            elif self.left_coef is None and self.right_coef is None:
                lane_status = "Lane not detected"
                
            cv2.putText(frame_with_lanes, lane_status, 
                      (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.lane_status_color, 2)
        except Exception as e:
            print(f"상태 표시 오류: {e}")
        
        return frame_with_lanes

class LanePlanner:
    """검출된 차선 중앙을 기반으로 조향각과 속도를 계획합니다."""
    
    def __init__(self):
        self.MAX_STEER = 0.5
        self.MAX_SPEED = 0.5
        self.MIN_SPEED = 0.5
        self.STRAIGHT_SPEED = 0.33
        self.TURN_THRESHOLD = 0.15

    def plan(self, lane_center_x, image_center_x):
        """차선 중앙 위치를 기반으로 조향각과 속도를 계산합니다."""
        deviation = 0.0
        steering = 0.0
        linear_speed = 0.0

        # 차선 중앙이 검출되지 않은 경우
        if lane_center_x is None:
            return 0.0, 0.0, 0.0

        # 이미지 중앙으로부터의 편차 계산
        deviation = (lane_center_x - image_center_x) / image_center_x

        # 편차를 적절한 범위로 제한
        deviation = np.clip(deviation, -1.0, 1.0)

        # 편차에 비례한 조향각 계산
        steering = self.MAX_STEER * deviation
        steering = np.clip(steering, -self.MAX_STEER, self.MAX_STEER)

        # 조향각에 따른 선형 속도 결정
        if abs(steering) > 0.08:  # 작은 오차 허용 범위
            linear_speed = self.MIN_SPEED  # 회전 시 속도
        else:
            linear_speed = self.STRAIGHT_SPEED  # 직진 시 속도

        # 선형 속도 제한
        linear_speed = np.clip(linear_speed, -self.MAX_SPEED, self.MAX_SPEED)

        return linear_speed, steering, deviation

def main():
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='Mac용 차선 인식 테스트')
    parser.add_argument('--source', type=str, default='0', 
                        help='영상 소스 (0: 웹캠, 파일명: 비디오 파일)')
    parser.add_argument('--model', type=str, default='lane.pt',
                        help='YOLO 모델 경로')
    parser.add_argument('--width', type=int, default=DEFAULT_CAMERA_WIDTH,
                        help='카메라/비디오 너비')
    parser.add_argument('--height', type=int, default=DEFAULT_CAMERA_HEIGHT,
                        help='카메라/비디오 높이')
    parser.add_argument('--skip', type=int, default=2,
                        help='몇 프레임마다 추론할지 (기본값: 2)')
    args = parser.parse_args()
    
    if not YOLO_AVAILABLE:
        print("❌ YOLO를 사용할 수 없습니다. 설치 명령: pip install ultralytics torch")
        return
    
    # 비디오 소스 설정
    try:
        if args.source.isdigit():
            print(f"웹캠 {args.source}을(를) 사용합니다.")
            cap = cv2.VideoCapture(int(args.source))
        else:
            print(f"비디오 파일 {args.source}을(를) 사용합니다.")
            cap = cv2.VideoCapture(args.source)
        
        # 해상도 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        
        if not cap.isOpened():
            raise ValueError("비디오 소스를 열 수 없습니다.")
            
        # 실제 적용된 해상도 확인
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"카메라 해상도: {int(actual_width)}x{int(actual_height)}")
        
    except Exception as e:
        print(f"비디오 소스 초기화 오류: {e}")
        return
    
    # YOLO 모델 로드
    try:
        print(f"YOLO 모델을 로드하는 중: {args.model}")
        if not os.path.exists(args.model):
            print(f"❌ 모델 파일을 찾을 수 없음: {args.model}")
            return
            
        device = "cpu"  # Mac에서는 CPU 사용
        model = YOLO(args.model).to(device)
        print(f"✅ YOLO 모델 로드 완료 (device: {device})")
        
        # 마스크가 있는지 확인하기 위한 테스트 추론
        print("🔍 테스트 추론 실행 중...")
        ret, test_frame = cap.read()
        if not ret:
            print("❌ 테스트 프레임을 읽을 수 없음")
            return
            
        # 30초 제한으로 테스트 추론
        results = model.predict(test_frame, conf=0.25, verbose=False)
        print(f"✅ 테스트 추론 완료!")
        
        # 마스크 확인
        if results[0].masks is None:
            print("⚠️ 경고: 모델이 마스크를 반환하지 않습니다. 차선 감지가 동작하지 않을 수 있습니다.")
        
    except Exception as e:
        print(f"❌ 모델 로드 또는 테스트 추론 실패: {e}")
        print("모델 파일이 올바른지 확인하세요.")
        traceback.print_exc()
        return
    
    # 모듈 초기화
    perception = LanePerception()
    planner = LanePlanner()
    
    print("🚗 차선 인식 테스트 시작 - q 키를 눌러 종료")
    
    frame_count = 0
    start_time = time.time()
    last_results = None
    
    try:
        while True:
            # 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                print("❌ 프레임 읽기 실패")
                if not args.source.isdigit():
                    print("비디오 파일을 처음으로 되감습니다.")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            
            # 프레임 카운트 증가
            frame_count += 1
            
            # ROI 계산
            roi = get_roi_slice(frame.shape[0])
            
            # 몇 프레임마다 추론
            if frame_count % args.skip == 0:
                try:
                    # YOLO 추론
                    results = model.predict(frame, conf=0.25, verbose=False)
                    last_results = results
                except Exception as e:
                    print(f"❌ 추론 중 오류: {e}")
                    # 오류 발생 시 이전 결과 사용
            
            # 차선 감지
            if last_results is not None:
                # 이미지 중앙 x 좌표 계산
                img_center_x = frame.shape[1] // 2
                
                # 차선 중앙 업데이트
                lane_center_x = perception.update(last_results, roi=roi)
                
                # 속도 및 조향각 계획
                linear_speed, steering, deviation = planner.plan(lane_center_x, img_center_x)
                
                # 시각화
                frame_with_lanes = perception.visualize_lanes(frame, deviation, steering, roi)
                
                # 결과 이미지 출력
                cv2.imshow("Lane Detection", frame_with_lanes)
                
                # ROI 영역 표시
                roi_y = roi.start
                roi_frame = frame.copy()
                cv2.line(roi_frame, (0, roi_y), (roi_frame.shape[1], roi_y), (255, 0, 0), 2)
                cv2.imshow("ROI", roi_frame[roi])
            else:
                # 결과가 없는 경우 원본 프레임만 표시
                cv2.imshow("Lane Detection", frame)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n테스트가 Ctrl+C로 중단되었습니다.")
    except Exception as e:
        print(f"\n테스트 중 오류 발생: {e}")
        traceback.print_exc()
    finally:
        # 정리
        cap.release()
        cv2.destroyAllWindows()
        print("🚗 테스트 종료")

if __name__ == "__main__":
    main()