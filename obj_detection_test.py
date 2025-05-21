import cv2
import time
import torch
import numpy as np
from jetcam.csi_camera import CSICamera
from ultralytics import YOLO
from tars_config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS

# 클래스별 색상 정의 (BGR 형식)
CLASS_COLORS = {
    'person': (0, 255, 0),      # 초록색
    'car': (255, 0, 0),         # 파란색
    'truck': (0, 0, 255),       # 빨간색
    'bus': (255, 255, 0),       # 청록색
    'motorcycle': (255, 0, 255), # 보라색
    'bicycle': (0, 255, 255),   # 노란색
    'stop sign': (128, 0, 128), # 보라색
    'traffic light': (0, 128, 128), # 갈색
    'fire hydrant': (128, 128, 0), # 청록색
    'bench': (128, 0, 0),       # 진한 파란색
    'bird': (0, 128, 0),        # 진한 초록색
    'cat': (0, 0, 128),         # 진한 빨간색
    'dog': (128, 128, 128),     # 회색
    'horse': (64, 64, 64),      # 어두운 회색
    'sheep': (192, 192, 192),   # 밝은 회색
    'cow': (64, 0, 0),          # 어두운 파란색
    'elephant': (0, 64, 0),     # 어두운 초록색
    'bear': (0, 0, 64),         # 어두운 빨간색
    'zebra': (64, 64, 0),       # 어두운 청록색
    'giraffe': (0, 64, 64),     # 어두운 갈색
}

class ObjectDetectionModel:
    def __init__(self, model_path="obj.pt", conf_threshold=0.3):
        """
        YOLO 모델을 초기화합니다.
        
        Args:
            model_path (str): YOLO 모델 파일 경로
            conf_threshold (float): 신뢰도 임계값 (0.0 ~ 1.0)
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.fuse()  # 모델 최적화
        
    def detect(self, frame):
        """
        이미지에서 물체를 감지합니다.
        
        Args:
            frame: 입력 이미지 (numpy array)
            
        Returns:
            results: YOLO 감지 결과
        """
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            device=self.device
        )
        return results

def get_color_for_class(class_name):
    """클래스 이름에 따른 색상 반환"""
    return CLASS_COLORS.get(class_name.lower(), (0, 255, 255))  # 기본값은 노란색

def draw_detection_box(frame, x1, y1, x2, y2, label, conf, color):
    """물체 감지 결과를 프레임에 그리는 함수"""
    # 박스 그리기
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    
    # 라벨 텍스트 준비
    text = f"{label} {conf:.2f}"
    
    # 텍스트 배경 크기 계산
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    
    # 텍스트 배경 그리기
    cv2.rectangle(frame, 
                 (int(x1), int(y1) - text_height - 10),
                 (int(x1) + text_width, int(y1)),
                 color, -1)
    
    # 텍스트 그리기 (검은색)
    cv2.putText(frame, text,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"디바이스: {device}")
    
    try:
        print("YOLO 모델 로딩 중...")
        detector = ObjectDetectionModel(model_path="obj.pt")
        print("✅ YOLO 모델 로드 성공")
    except Exception as e:
        print(f"❌ YOLO 모델 로드 실패: {e}")
        print(f"현재 작업 디렉토리: {os.getcwd()}")
        print("obj.pt 파일이 있는지 확인하세요")
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

        # 물체 감지 수행
        try:
            results = detector.detect(frame)
        except Exception as e:
            print(f"❌ YOLO 예측 오류: {e}")
            continue

        # 결과 시각화
        display_frame = frame.copy()
        
        # FPS 표시
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (50, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 감지된 물체 표시
        boxes = results[0].boxes
        if len(boxes) > 0:
            clss = boxes.cls.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            
            for (x1, y1, x2, y2), cls_id, conf in zip(xyxy, clss, confs):
                label = results[0].names[int(cls_id)]
                color = get_color_for_class(label)
                draw_detection_box(display_frame, x1, y1, x2, y2, label, conf, color)

        # 화면에 표시
        cv2.imshow("Object Detection", display_frame)
        
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            print("종료 신호 받음")
            break

    # 카메라 리소스 정리
    print("카메라 리소스 정리 중...")
    camera.running = False
    if hasattr(camera, 'cap') and hasattr(camera.cap, 'release'):
        print("camera.cap 해제 시도...")
        camera.cap.release()
        print("✅ camera.cap 해제됨")
    del camera
    cv2.destroyAllWindows()
    print("카메라 정지됨")

if __name__ == "__main__":
    import os
    main() 