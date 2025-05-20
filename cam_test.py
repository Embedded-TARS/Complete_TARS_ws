import cv2
import numpy as np
import time
import argparse
import traceback
import os

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
    """
    이미지 높이에 기반하여 ROI 슬라이스를 계산합니다.
    """
    roi_start = int(img_height * ROI_RATIO)
    return slice(roi_start, img_height)

def debug_camera():
    """카메라만 테스트하는 함수"""
    print("📷 간단한 카메라 테스트 시작")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return
    
    # 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_CAMERA_HEIGHT)
    
    # 실제 적용된 해상도 확인
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"카메라 해상도: {int(actual_width)}x{int(actual_height)}")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 프레임을 읽을 수 없습니다.")
                break
            
            cv2.imshow("카메라 테스트", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"카메라 테스트 중 오류: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("📷 카메라 테스트 종료")

def debug_yolo_inference(model_path="lane.pt"):
    """YOLO 모델 추론 테스트"""
    if not YOLO_AVAILABLE:
        print("❌ YOLO를 사용할 수 없습니다.")
        return
    
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일이 존재하지 않습니다: {model_path}")
        return
    
    print(f"🔍 YOLO 모델 로드 중: {model_path}")
    try:
        device = "cpu"  # Mac에서는 CPU 사용
        model = YOLO(model_path).to(device)
        print(f"✅ 모델 로드 성공 (device: {device})")
        
        # 테스트 이미지로 추론
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ 카메라를 열 수 없습니다.")
            return
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("❌ 테스트 이미지를 얻을 수 없습니다.")
            return
        
        print("🔍 테스트 추론 실행 중...")
        results = model.predict(frame, conf=0.25)
        print(f"✅ 추론 성공! 결과: {len(results)} 항목")
        
        # 결과 시각화
        result_img = results[0].plot()
        cv2.imshow("YOLO Test", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"❌ YOLO 테스트 중 오류: {e}")
        traceback.print_exc()

def debug_step_by_step(model_path="lane.pt"):
    """단계별 디버깅 함수"""
    if not YOLO_AVAILABLE:
        print("❌ YOLO를 사용할 수 없습니다.")
        return
    
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일이 존재하지 않습니다: {model_path}")
        return
    
    # 1. 카메라 초기화
    print("1️⃣ 카메라 초기화 중...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_CAMERA_HEIGHT)
    
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"✅ 카메라 초기화 완료: {int(actual_width)}x{int(actual_height)}")
    
    # 2. 모델 로드
    print("2️⃣ YOLO 모델 로드 중...")
    try:
        device = "cpu"  # Mac에서는 CPU 사용
        model = YOLO(model_path).to(device)
        print(f"✅ 모델 로드 완료 (device: {device})")
    except Exception as e:
        print(f"❌ 모델 로드 중 오류: {e}")
        cap.release()
        return
    
    # 3. 단일 프레임 테스트
    print("3️⃣ 단일 프레임 테스트 중...")
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임을 읽을 수 없습니다.")
        cap.release()
        return
    
    cv2.imshow("테스트 프레임", frame)
    cv2.waitKey(1)
    print("✅ 프레임 읽기 성공")
    
    # 4. 단일 추론 테스트
    print("4️⃣ 단일 추론 테스트 중...")
    try:
        # 낮은 신뢰도 임계값으로 시작
        results = model.predict(frame, conf=0.1)
        print(f"✅ 추론 성공! 검출된 객체: {len(results[0].boxes)}")
        
        # 결과 시각화
        result_img = results[0].plot()
        cv2.imshow("추론 결과", result_img)
        cv2.waitKey(0)
    except Exception as e:
        print(f"❌ 추론 중 오류: {e}")
        traceback.print_exc()
    
    # 5. 루프 테스트 (optional)
    print("5️⃣ 루프 테스트를 시작하려면 아무 키나 누르세요 (q: 건너뛰기)")
    if cv2.waitKey(0) & 0xFF == ord('q'):
        print("루프 테스트 건너뜀")
    else:
        print("루프 테스트 시작 (q: 종료)")
        try:
            for _ in range(10):  # 10 프레임만 테스트
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 추론
                results = model.predict(frame, conf=0.1)
                result_img = results[0].plot()
                
                cv2.imshow("루프 테스트", result_img)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
            print("✅ 루프 테스트 완료")
        except Exception as e:
            print(f"❌ 루프 테스트 중 오류: {e}")
            traceback.print_exc()
    
    cap.release()
    cv2.destroyAllWindows()
    print("🎯 디버깅 완료!")

def main():
    parser = argparse.ArgumentParser(description='Mac용 차선 인식 디버거')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['camera', 'yolo', 'step', 'all'],
                        help='디버깅 모드 (camera: 카메라만, yolo: YOLO만, step: 단계별, all: 전체)')
    parser.add_argument('--model', type=str, default='lane.pt',
                        help='YOLO 모델 경로')
    args = parser.parse_args()
    
    print("🛠️ Mac용 차선 인식 디버거 시작")
    
    if args.mode == 'camera' or args.mode == 'all':
        debug_camera()
    
    if args.mode == 'yolo' or args.mode == 'all':
        debug_yolo_inference(args.model)
    
    if args.mode == 'step' or args.mode == 'all':
        debug_step_by_step(args.model)
    
    print("🏁 디버깅 종료")

if __name__ == "__main__":
    main()