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
import termios
import tty
import select
from tars_config import (  # 설정 모듈 임포트
    get_roi_slice, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, 
    LANE_WIDTH_PX, EMA_ALPHA, CAPTURE_DIR
)

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

# 터미널 설정을 위한 함수들
def set_terminal_mode():
    """터미널을 raw 모드로 설정"""
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin.fileno())
    return old_settings

def restore_terminal_mode(old_settings):
    """터미널 설정을 원래대로 복구"""
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

def is_key_pressed():
    """키 입력이 있는지 확인"""
    return select.select([sys.stdin], [], [], 0)[0]

def get_key():
    """키 입력을 읽음"""
    return sys.stdin.read(1)

def main():
    # 자율주행 모듈 및 카메라 초기화
    lane_model = LaneDetectionModel(model_path="lane.pt", lane_class_id=12)
    perception = LanePerception(lane_width_px=LANE_WIDTH_PX, ema_alpha=EMA_ALPHA)
    planner = LanePlanner()
    controller = RobotController(base)
    camera_manager = CameraManager.get_instance()
    camera_manager.initialize_camera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT, capture_fps=CAMERA_FPS)

    print("🚗 자율주행 모드 시작 - q 키를 눌러 종료, 스페이스바로 일시정지/재시작")
    
    is_paused = False  # 일시정지 상태를 추적하는 변수
    
    # 터미널 설정 변경
    old_terminal_settings = set_terminal_mode()
    
    try:
        while True:
            frame = camera_manager.get_frame()
            if frame is None:
                print("❌ 프레임 수신 실패")
                time.sleep(0.1)
                continue

            # 이미지 중앙 x 좌표 계산
            img_center_x = (frame.shape[1] // 2) - 11

            roi = get_roi_slice(frame.shape[0]) 

            # Perception: YOLO 추론 및 차선 감지
            results = lane_model.predict(frame)
            lane_center_x = perception.update(results[0], roi = roi)
            
            # 출력 정리
            sys.stdout.write('\033[2J\033[H')  # 화면 클리어 및 커서를 맨 위로
            sys.stdout.flush()
            
            status = [
                "=== 자율주행 상태 ===",
                f"차선 중심점: {lane_center_x:.2f}" if lane_center_x is not None else "차선 감지: ❌ (차선을 찾을 수 없음)",
                f"이미지 중심점: {img_center_x}",
                f"상태: {'일시정지' if is_paused else '주행중'}",
                "==================="
            ]
            
            # 한 번에 모든 상태 출력
            sys.stdout.write('\n'.join(status) + '\n')
            sys.stdout.flush()

            # Planning: 속도 및 스티어링 결정
            if lane_center_x is not None:
                linear_speed, steering, deviation = planner.plan(lane_center_x, img_center_x)
            else:
                # 차선이 감지되지 않았을 때는 천천히 직진
                linear_speed = 0.3  # 낮은 속도
                steering = 0.0      # 직진
                deviation = 0.0

            # Control: 로봇에 제어 명령 전송 (일시정지 상태가 아닐 때만)
            if not is_paused:
                controller.send_control(linear_speed, steering)
            else:
                controller.send_control(0, 0)  # 일시정지 상태일 때는 정지

            # 차선 인식 시각화를 위해 perception 모듈에 위임
            frame_with_lanes = perception.visualize_lanes(frame, deviation, steering, roi)
            
            # 일시정지 상태 표시
            if is_paused:
                cv2.putText(frame_with_lanes, "PAUSED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 결과 이미지 출력
            # cv2.imshow("YOLO-AutoDrive", frame_with_lanes)

            # 키 입력 처리 (터미널과 OpenCV 모두)
            if is_key_pressed():
                key = get_key()
                if key == 'q':
                    break
                elif key == ' ':  # 스페이스바
                    is_paused = not is_paused
                    sys.stdout.write("\n⏸️ 일시정지\n" if is_paused else "\n▶️ 재시작\n")
                    sys.stdout.flush()
            
            # OpenCV 창의 키 입력도 처리
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            elif key == 32:  # 스페이스바
                is_paused = not is_paused
                sys.stdout.write("\n⏸️ 일시정지\n" if is_paused else "\n▶️ 재시작\n")
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n자율주행 모드가 Ctrl+C로 중단되었습니다.")
    except Exception as e:
        print(f"\n자율주행 모드 오류: {e}")
    finally:
        # 터미널 설정 복구
        restore_terminal_mode(old_terminal_settings)
        
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
    print("cal: 차선 폭 칼리브레이션")
    print("q: 정지 및 메뉴로")
    print("x: 종료")
    print("====================")

# 메인 메뉴 루프 - 간소화
def main_menu():
    while True:
        print_menu()
        mode = input("모드 선택 (a/mp/mt/c/cc/cal/q/x): ").strip().lower()

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
        elif mode == 'cal':
            run_calibration()
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
    try:
        import tars_camera_test
        tars_camera_test.camera_test_main()
    except Exception as e:
        import traceback
        print(f"카메라 테스트 실행 중 오류 발생: {e}")
        print("상세 오류 정보:")
        traceback.print_exc()
        print("\n기본 카메라 테스트로 대체합니다.")
        
        # 기본 카메라 테스트로 대체
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

# 칼리브레이션 실행 함수
def run_calibration():
    try:
        import tars_calibration
        tars_calibration.main()
    except Exception as e:
        import traceback
        print(f"칼리브레이션 실행 중 오류 발생: {e}")
        print("상세 오류 정보:")
        traceback.print_exc()
        print("\n칼리브레이션이 실패했습니다.")

# 스크립트 직접 실행 시 main_menu() 호출
if __name__ == "__main__":
    main_menu()
