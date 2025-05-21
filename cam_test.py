import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO
from jetcam.csi_camera import CSICamera
from tars_config import (
    get_roi_slice, LANE_WIDTH_PX, CAMERA_WIDTH, CAMERA_HEIGHT, 
    CAMERA_FPS, POLY_DEG, EMA_ALPHA, ROI_RATIO
)

class LaneCenterTracker:
    def __init__(self, lane_width_px: float = LANE_WIDTH_PX, poly_deg: int = POLY_DEG):
        print(f"\nLaneCenterTracker 초기화:")
        print(f"lane_width_px: {lane_width_px}")
        print(f"poly_deg: {poly_deg}")
        
        self.lane_width_px = lane_width_px
        self.poly_deg = poly_deg
        self.left_coef = None
        self.right_coef = None
        self.center_px = None
        self.left_y_range = None
        self.right_y_range = None
        
        # 디버깅을 위한 변수 추가
        self.debug_masks = None

    def _mask_to_poly_bottom(self, mask_bin: np.ndarray, thr=0.5):
        """mask_bin : 2-D uint8 (0/1)"""
        ys, xs = np.nonzero(mask_bin)
        if xs.size < self.poly_deg + 1:           # 데이터 부족
            return None, None, None
            
        # 유니크한 y값이 poly_deg+1보다 적으면 다항식 피팅 불가
        if np.unique(ys).size < self.poly_deg + 1:
            print(f"경고: 유니크한 y값이 부족함 (필요: {self.poly_deg + 1}, 실제: {np.unique(ys).size})")
            return None, None, None
            
        # (1) 다항식(x) = f(y)  ← y가 세로(row)
        coef = np.polyfit(ys, xs, self.poly_deg)
        # (2) 이미지 하단(row = H-1)에서 x 좌표 예측
        y_bottom = mask_bin.shape[0] - 1
        x_bottom = np.polyval(coef, y_bottom)
        y_min, y_max = int(np.min(ys)), int(np.max(ys))
        return coef, float(x_bottom), (y_min, y_max)

    def update(self, results, *, roi=None, thr=0.5):
        """
        results : ultralytics.engine.results.Results (단일 이미지)
        returns : center_x (float) or None
        """
        r = results
        H_img, W_img = r.orig_shape[:2]
        roi = roi or get_roi_slice(H_img)  # 설정 모듈의 함수 사용
        
        print(f"\n이미지 크기: {W_img}x{H_img}, ROI: y={roi.start}~{roi.stop}")
        
        if r.masks is None or len(r.masks.data) == 0:     # CASE 0
            print("⚠️ 마스크가 감지되지 않음")
            return self.center_px                         # 그대로 유지

        # (A) 모든 mask → 원본 해상도(픽셀)로 upsample
        masks = r.masks.data.cpu().numpy()                # (N,160,160) etc.
        print(f"✅ {len(masks)}개의 마스크 감지됨")
        
        full_bin = [cv2.resize((m > thr).astype(np.uint8),
                                (W_img, H_img),
                                cv2.INTER_NEAREST)[roi]
                    for m in masks]    

        # 디버깅용 마스크 시각화 이미지 생성
        self.debug_masks = np.zeros((roi.stop - roi.start, W_img, 3), dtype=np.uint8)

        # (B) 픽셀 수 기준 내림차순 정렬
        pix_counts = [m.sum() for m in full_bin]
        idx_sorted = np.argsort(pix_counts)[::-1]
        
        print(f"마스크 픽셀 수: {pix_counts}")
        print(f"정렬된 인덱스: {idx_sorted}")
        
        # 마스크 시각화 (색상으로 구분)
        for i, idx in enumerate(idx_sorted):
            if i == 0:  # 첫 번째 마스크 (가장 큰 것) - 빨간색
                self.debug_masks[full_bin[idx] > 0] = (0, 0, 255)
            elif i == 1:  # 두 번째 마스크 - 초록색
                self.debug_masks[full_bin[idx] > 0] = (0, 255, 0)
            else:  # 그 외 마스크 - 파란색
                self.debug_masks[full_bin[idx] > 0] = (255, 0, 0)

        # ──────────────────────────────────────────
        # CASE 2 : 두 개 이상 검출
        # ──────────────────────────────────────────
        if len(idx_sorted) >= 2:
            idx_top2 = idx_sorted[:2]
            infos = [self._mask_to_poly_bottom(full_bin[i]) for i in idx_top2]
            # infos = [(coef, x_bottom), ...]  길이가 2
            if any(c is None for c, _, _ in infos):
                print("⚠️ 다항식 피팅 실패")
                return self.center_px                    # 실패 → 이전 값
                
            # 왼쪽/오른쪽 분리
            (coef1, x1, y_range1), (coef2, x2, y_range2) = infos
            print(f"두 차선 감지됨: x1={x1:.1f}, x2={x2:.1f}")
            
            if x1 < x2:
                self.left_coef, self.right_coef = coef1, coef2
                self.left_y_range, self.right_y_range = y_range1, y_range2
                x_left, x_right = x1, x2
            else:
                self.left_coef, self.right_coef = coef2, coef1
                self.left_y_range, self.right_y_range = y_range2, y_range1
                x_left, x_right = x2, x1
            self.center_px = (x_left + x_right) / 2.0
            print(f"차선 중앙: {self.center_px:.1f}")
            return self.center_px

        # ──────────────────────────────────────────
        # CASE 1 : 한 개만 검출
        # ──────────────────────────────────────────
        idx = idx_sorted[0]
        coef_new, x_new, y_range_new = self._mask_to_poly_bottom(full_bin[idx])
        if coef_new is None:
            print("⚠️ 다항식 피팅 실패 (단일 마스크)")
            return self.center_px

        print(f"단일 차선 감지됨: x={x_new:.1f}")
        
        # (1) 새 차선이 이전 왼/오 중 어느 쪽과 가까운지 계산
        dist_left = abs(x_new - np.polyval(self.left_coef, H_img-1)) if self.left_coef is not None else np.inf
        dist_right = abs(x_new - np.polyval(self.right_coef, H_img-1)) if self.right_coef is not None else np.inf

        if dist_left < dist_right:          # 왼쪽 차선으로 간주
            print(f"왼쪽 차선으로 판단됨 (거리: left={dist_left:.1f}, right={dist_right:.1f})")
            self.left_coef = coef_new
            self.left_y_range = y_range_new
            x_left = x_new
            x_right = x_left + self.lane_width_px
            self.right_coef = None          # 추후 갱신 예정
            self.right_y_range = None
        else:                               # 오른쪽 차선
            print(f"오른쪽 차선으로 판단됨 (거리: left={dist_left:.1f}, right={dist_right:.1f})")
            self.right_coef = coef_new
            self.right_y_range = y_range_new
            x_right = x_new
            x_left = x_right - self.lane_width_px
            self.left_coef = None
            self.left_y_range = None

        self.center_px = (x_left + x_right) / 2.0
        print(f"추정된 차선 중앙: {self.center_px:.1f}")
        return self.center_px
        
def draw_polyline(img, coef, color, thickness=3, n_pts=50, y_range=None, roi_offset_y=0):
    """
    img      : BGR 영상 (in-place 로 그려짐)
    coef     : np.ndarray (x = f(y) 다항식 계수)
    color    : (B,G,R)
    roi_offset_y : ROI의 y축 오프셋
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
    pts = np.stack([xs, ys + roi_offset_y], axis=-1)  # ROI offset 적용
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
    
    # 점 시각화 추가
    for pt in pts:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 2, color, -1)

def draw_transparent_box(frame, x1, y1, x2, y2, color, alpha=0.3):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"디바이스: {device}")
    try:
        print("YOLO 모델 로딩 중...")
        model = YOLO("lane.pt")
        model.fuse()
        print("✅ YOLO 모델 로드 성공")
        print(f"클래스 이름: {model.names}")
    except Exception as e:
        print(f"❌ YOLO 모델 로드 실패: {e}")
        print(f"현재 작업 디렉토리: {os.getcwd()}")
        print("lane.pt 파일이 있는지 확인하세요")
        return

    print(f"카메라 초기화 중... ({CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}fps)")
    try:
        camera = CSICamera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT, capture_fps=CAMERA_FPS)
        camera.running = True
    except Exception as e:
        print(f"❌ 카메라 초기화 실패: {e}")
        return

    print("카메라가 준비될 때까지 대기 중...")
    wait_start = time.time()
    while camera.value is None:
        if time.time() - wait_start > 5.0:  # 5초 타임아웃
            print("⚠️ 카메라 초기화 타임아웃!")
            return
        time.sleep(0.1)
    print("✅ 카메라 준비됨")

    prev_t = time.time()
    tracker = LaneCenterTracker(lane_width_px=LANE_WIDTH_PX, poly_deg=POLY_DEG)
    
    frame_count = 0
    while True:
        frame = camera.value

        if frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1
        current_time = time.time()
        fps = 1.0 / (current_time - prev_t) if current_time - prev_t > 0 else 0
        prev_t = current_time
        
        if frame_count % 30 == 0:  # 30 프레임마다 정보 출력
            print(f"\n프레임 #{frame_count} | FPS: {fps:.1f}")

        H_img, W_img = frame.shape[:2]
        roi = get_roi_slice(H_img)  # 설정 모듈의 함수 사용
        
        # ROI 표시 이미지
        roi_vis = frame.copy()
        cv2.line(roi_vis, (0, roi.start), (W_img, roi.start), (0, 255, 0), 2)
        cv2.putText(roi_vis, f"ROI (y={roi.start})", (10, roi.start-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 3-A) YOLO 추론
        try:
            results = model.predict(frame, device=device, conf=0.3)  # 낮은 신뢰도로 설정
        except Exception as e:
            print(f"❌ YOLO 예측 오류: {e}")
            continue

        # 결과
        r = results[0]
        center_x = tracker.update(r, roi=roi)
        
        # ROI 영역에서의 처리 결과 시각화
        frame_roi = frame[roi].copy()
        
        # 원래 이미지에 차선 그리기
        display_frame = frame.copy()
        
        # 왼쪽/오른쪽 차선 그리기
        if tracker.left_coef is not None:
            draw_polyline(display_frame, tracker.left_coef, (0, 255, 0), y_range=tracker.left_y_range, roi_offset_y=roi.start)
            cv2.putText(display_frame, "Left lane", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if tracker.right_coef is not None:
            draw_polyline(display_frame, tracker.right_coef, (0, 0, 255), y_range=tracker.right_y_range, roi_offset_y=roi.start)
            cv2.putText(display_frame, "Right lane", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 차선 중앙 표시
        if center_x is not None:
            img_center_x = W_img // 2
            
            # 차선 중앙점 그리기
            cv2.circle(display_frame, (int(center_x), H_img-30), 8, (0, 255, 255), -1)
            cv2.putText(display_frame, f"Lane center: {center_x:.1f}", (int(center_x) + 10, H_img-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # 이미지 중앙점 그리기
            cv2.circle(display_frame, (img_center_x, H_img-30), 8, (255, 0, 255), -1)
            cv2.putText(display_frame, f"Image center: {img_center_x}", (img_center_x + 10, H_img-60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # 두 점을 연결하는 선
            cv2.line(display_frame, (int(center_x), H_img-30), (img_center_x, H_img-30), (255, 255, 255), 2)
            
            # 편차 계산 및 표시
            deviation = (center_x - img_center_x) / img_center_x
            deviation_text = f"Deviation: {deviation:.2f}"
            cv2.putText(display_frame, deviation_text, (50, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        
        # 다른 정보 표시
        lane_status = "상태: "
        if tracker.left_coef is not None and tracker.right_coef is not None:
            lane_status += "양쪽 차선 감지됨"
        elif tracker.left_coef is not None:
            lane_status += "왼쪽 차선만 감지됨"
        elif tracker.right_coef is not None:
            lane_status += "오른쪽 차선만 감지됨"
        else:
            lane_status += "차선 감지 안됨"
            
        cv2.putText(display_frame, lane_status, (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 3-B) 박스 그리기
        boxes = r.boxes
        if len(boxes) > 0:
            clss = boxes.cls.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            
            for (x1, y1, x2, y2), cls_id, conf in zip(xyxy, clss, confs):
                label = f"{model.names[int(cls_id)]} {conf:.2f}"
                p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.rectangle(display_frame, p1, p2, (0, 255, 255), 2)
                cv2.putText(display_frame, label, (p1[0], p1[1]-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 마스크 디버그 이미지 표시
        if tracker.debug_masks is not None:
            cv2.imshow("Masks Debug", tracker.debug_masks)

        cv2.imshow("YOLO-Detect", display_frame)
        cv2.imshow("Lane-ROI", frame_roi)
        cv2.imshow("ROI Visualization", roi_vis)
        
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            print("종료 신호 받음")
            break

    # Release camera resources
    print("카메라 리소스 정리 중...")
    camera.running = False
    # Attempt to explicitly release the underlying capture object
    if hasattr(camera, 'cap') and hasattr(camera.cap, 'release'):
        print("camera.cap 해제 시도...")
        camera.cap.release()
        print("✅ camera.cap 해제됨")
    # Explicitly delete the camera object to ensure resource release
    del camera
    cv2.destroyAllWindows()
    print("카메라 정지됨")

if __name__ == "__main__":
    import os
    main()