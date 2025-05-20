# 00_tars_main.py

"""
메인 엔트리 포인트. 모드 스위칭에 집중하고 나머지 기능은 각 모듈에 위임합니다.
"""

# 필요한 모듈 임포트
from tars_perception import LanePerception, LaneDetectionModel
from tars_planning import LanePlanner
from tars_control import RobotController
from tars_camera import CameraManager
from base_ctrl_js import BaseController
import cv2
import glob
import time
import sys
import pygame
from tars_manual_ctrl import PygameKeyboardController
from tars_manual_ctrl import TerminalKeyboardController
import pathlib

# 베이스 컨트롤러 초기화
available_ports = glob.glob('/dev/ttyUSB*')
if available_ports:
    port = available_ports[0]
    print(f"시리얼 포트 감지됨: {port}")
else:
    print("시리얼 포트를 찾을 수 없습니다. 가상 모드로 실행합니다.")
    port = "VIRTUAL"

# BaseController 인스턴스 생성
base = BaseController(port, 115200)

# 자율주행 메인 루프
def main():
    # 자율주행 모듈 및 카메라 초기화
    lane_model = LaneDetectionModel(model_path="lane.pt", lane_class_id=12)
    perception = LanePerception(lane_width_px=700, ema_alpha=0.8)
    planner = LanePlanner()
    controller = RobotController(base)
    camera_manager = CameraManager.get_instance()
    camera_manager.initialize_camera(width=640, height=480, capture_fps=30)

    print("🚗 자율주행 모드 시작 - q 키를 눌러 종료")
    
    try:
        while True:
            frame = camera_manager.get_frame()
            if frame is None:
                print("❌ 프레임 수신 실패")
                time.sleep(0.1)
                continue

            # 이미지 중앙 x 좌표 계산
            img_center_x = (frame.shape[1] // 2) - 11

            # Perception: YOLO 추론 및 차선 감지
            results = lane_model.predict(frame)
            lane_center_x = perception.update(results[0])

            # Planning: 속도 및 스티어링 결정
            linear_speed, steering, deviation = planner.plan(lane_center_x, img_center_x)

            # Control: 로봇에 제어 명령 전송
            controller.send_control(linear_speed, steering)

            # 차선 인식 시각화를 위해 perception 모듈에 위임
            frame_with_lanes = perception.visualize_lanes(frame, deviation, steering)
            
            # 결과 이미지 출력
            cv2.imshow("YOLO-AutoDrive", frame_with_lanes)

            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break

    except KeyboardInterrupt:
        print("\n자율주행 모드가 Ctrl+C로 중단되었습니다.")
    except Exception as e:
        print(f"\n자율주행 모드 오류: {e}")
    finally:
        # 자율주행 종료 시 정리 작업
        camera_manager.release_camera()
        cv2.destroyAllWindows()
        base.base_velocity_ctrl(0, 0)
        if hasattr(base, 'gimbal_dev_close'):
            pass
        
        # Add a shutdown call for the base controller if implemented
        if hasattr(base, 'shutdown'):
            base.shutdown()
            
        print("🚗 자율주행 종료")

# 메인 메뉴 출력 함수
def print_menu():
    print("\n===== 모드 선택 =====")
    print("a: 자율주행 (lane tracking)")
    print("mp: 메뉴얼 (Pygame)")
    print("mt: 메뉴얼 (Terminal) - 비디오 녹화 포함")
    print("c: 카메라 테스트")
    print("cc: 카메라 사진 캡쳐")
    print("q: 정지 및 메뉴로")
    print("x: 종료")
    print("====================")

# 메인 메뉴 루프 - 간소화
def main_menu():
    while True:
        print_menu()
        mode = input("모드 선택 (a/mp/mt/c/cc/q/x): ").strip().lower()

        if mode == 'a':
            main()  # 자율주행 모드 실행
        elif mode == 'mp':
            controller = PygameKeyboardController(base)
            result = controller.run()
            if result == 'quit':
                break
        elif mode == 'mt':
            controller = TerminalKeyboardController(base)
            result = controller.run()
            if result == 'quit':
                break
        elif mode == 'c':
            run_camera_test()
        elif mode == 'cc':
            capture_photo()
        elif mode == 'q':
            base.base_velocity_ctrl(0, 0)
            print("정지 및 메뉴로 돌아갑니다.")
        elif mode == 'x':
            base.base_velocity_ctrl(0, 0)
            print("프로그램 종료!")
            break
        else:
            print("잘못된 입력입니다. 다시 선택하세요.")

# 카메라 테스트 함수
def run_camera_test():
    camera_manager = CameraManager.get_instance()
    camera_manager.initialize_camera()
    
    print("카메라 테스트 중... 'q' 키를 눌러 종료하세요.")
    
    try:
        while True:
            frame = camera_manager.get_frame()
            if frame is not None:
                cv2.imshow("Camera Test", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("프레임을 읽을 수 없습니다.")
                time.sleep(0.1)
    finally:
        camera_manager.release_camera()
        cv2.destroyAllWindows()

# 사진 캡쳐 함수
def capture_photo():
    camera_manager = CameraManager.get_instance()
    camera_manager.initialize_camera()
    
    print("카메라 준비 중...")
    time.sleep(1)
    
    # Add directory creation logic
    OUTDIR = "./captures"
    path = pathlib.Path(OUTDIR).resolve()
    path.mkdir(exist_ok=True, parents=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Update file_path to include the directory
    file_path = str(path / f"capture_{timestamp}.jpg")

    frame = camera_manager.get_frame()
    if frame is not None:
        cv2.imwrite(file_path, frame)
        print(f"✅ 이미지가 저장되었습니다: {file_path}")

        # Remove imshow and related calls
        # cv2.imshow("Captured Image", frame)
        # print("아무 키나 눌러 계속하세요...")
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        print("❌ 프레임을 캡쳐할 수 없습니다.")
    
    camera_manager.release_camera()

# 스크립트 직접 실행 시 main_menu() 호출
if __name__ == "__main__":
    main_menu()