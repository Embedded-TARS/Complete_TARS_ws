# 00_tars_main.py

"""
메인 엔트리 포인트. 모드 스위칭에 집중하고 나머지 기능은 각 모듈에 위임합니다.
"""

# 필요한 모듈 임포트
from tars_perception import LanePerception, LaneDetectionModel
from tars_planning import EnhancedLanePlanner, CLASS_INFO
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
import json
import socket
import threading

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
    """터미널을 cbreak 모드로 설정 (raw 모드보다 덜 제한적)"""
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())  # raw 대신 cbreak 사용
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

def clear_screen():
    """화면을 깨끗하게 클리어"""
    print("\033[H\033[2J", end="", flush=True)

def print_status_clean(status_lines):
    """상태를 깨끗하게 출력"""
    # 커서를 맨 위로 이동하고 화면 클리어
    print("\033[H", end="")
    
    # 각 줄을 출력하고 줄 끝까지 클리어
    for line in status_lines:
        print(f"{line}\033[K")  # \033[K는 커서부터 줄 끝까지 클리어
    
    # 추가 빈 줄들도 클리어 (이전 출력이 더 길었을 경우를 대비)
    for _ in range(5):
        print("\033[K")
    
    # 커서를 다시 상태 출력 아래로 이동
    print(f"\033[{len(status_lines)+1}H", end="", flush=True)

class CommandHandler:
    def __init__(self, host='0.0.0.0', port=5000):
        self.host = host
        self.port = port
        self.server_socket = None
        self.current_command = None
        self.command_lock = threading.Lock()
        self.is_running = False
        self.server_thread = None

    def start_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.is_running = True
        self.server_thread = threading.Thread(target=self._listen_for_commands)
        self.server_thread.daemon = True
        self.server_thread.start()
        print(f"명령 서버가 시작되었습니다. 포트: {self.port}")

    def stop_server(self):
        print("서버 종료 중...")
        self.is_running = False
        if self.server_socket:
            try:
                # 서버 소켓을 닫기 전에 타임아웃 설정
                self.server_socket.settimeout(1.0)
                # 서버 소켓을 닫음
                self.server_socket.close()
            except Exception as e:
                print(f"서버 소켓 종료 중 오류: {e}")
        
        if self.server_thread and self.server_thread.is_alive():
            try:
                # 스레드가 종료될 때까지 최대 2초 대기
                self.server_thread.join(timeout=2.0)
            except Exception as e:
                print(f"서버 스레드 종료 중 오류: {e}")
        
        print("서버가 종료되었습니다.")

    def _listen_for_commands(self):
        while self.is_running:
            try:
                client_socket, addr = self.server_socket.accept()
                print(f"클라이언트 연결됨: {addr}")
                data = client_socket.recv(1024).decode('utf-8')
                try:
                    command = json.loads(data)
                    with self.command_lock:
                        self.current_command = command
                    print(f"새로운 명령 수신: {command}")
                except json.JSONDecodeError:
                    print("잘못된 JSON 형식")
                client_socket.close()
            except Exception as e:
                print(f"명령 수신 중 오류: {e}")

    def get_current_command(self):
        with self.command_lock:
            return self.current_command

    def clear_current_command(self):
        with self.command_lock:
            self.current_command = None

def main(display_mode=True):
    # 명령 핸들러 초기화 및 시작
    command_handler = CommandHandler()
    command_handler.start_server()

    # 자율주행 모듈 및 카메라 초기화
    lane_model = LaneDetectionModel(model_path="lane.pt", lane_class_id=12)
    perception = LanePerception(lane_width_px=LANE_WIDTH_PX, ema_alpha=EMA_ALPHA)
    planner = EnhancedLanePlanner(model_path="obj.pt")
    controller = RobotController(base)
    camera_manager = CameraManager.get_instance()
    camera_manager.initialize_camera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT, capture_fps=CAMERA_FPS)

    print("🚗 자율주행 모드 시작 - q 키를 눌러 종료, 스페이스바로 일시정지/재시작")
    print("JSON 명령을 기다리는 중...")
    
    is_paused = False
    old_terminal_settings = set_terminal_mode()
    clear_screen()
    
    try:
        while True:
            # 현재 명령 확인
            current_command = command_handler.get_current_command()
            
            if current_command is None:
                # 명령이 없으면 대기하면서 키 입력 체크
                status = [
                    "=== 자율주행 상태 ===",
                    "상태: 명령 대기 중...",
                    "속도: 0.00 m/s",
                    "조향: 0.00 rad/s",
                    "주행: 대기",
                    "현재 명령: 없음",
                    "JSON 명령을 기다리는 중... (종료하려면 'q' 키를 누르세요)"
                ]
                print_status_clean(status)
                
                if is_key_pressed():
                    key = get_key()
                    if key == 'q':
                        print("\n자율주행 모드를 종료하고 메인 메뉴로 돌아갑니다...")
                        return 'menu'
                time.sleep(0.1)
                continue

            frame = camera_manager.get_frame()
            if frame is None:
                print("❌ 프레임 수신 실패")
                time.sleep(0.1)
                continue

            img_center_x = (frame.shape[1] // 2)
            roi = get_roi_slice(frame.shape[0]) 

            results = lane_model.predict(frame)
            lane_center_x = perception.update(results[0], roi=roi)

            if lane_center_x is not None:
                linear_speed, steering, deviation, state, detected_objects = planner.plan_with_objects(frame, lane_center_x, img_center_x)
            else:
                linear_speed = 0.3
                steering = 0.0
                deviation = 0.0
                state = "no_lane_detected"
                detected_objects = []

            status = [
                "=== 자율주행 상태 ===",
                f"상태: {state}",
                f"속도: {linear_speed:.2f} m/s",
                f"조향: {steering:.2f} rad/s",
                f"주행: {'일시정지' if is_paused else '주행중'}",
                f"현재 명령: {current_command}"
            ]
            
            if detected_objects:
                status.append(f"객체: {len(detected_objects)}개")
                for obj in detected_objects:
                    if obj['confidence'] > 0.5:
                        status.append(f"- {CLASS_INFO[obj['class']]['name']} ({obj['confidence']:.0%})")
            
            print_status_clean(status)

            if not is_paused:
                controller.send_control(linear_speed, steering)
            else:
                controller.send_control(0, 0)

            if display_mode:
                frame_with_lanes = perception.visualize_lanes(frame, deviation, steering, roi)
                frame_with_objects = planner.visualize_detections(frame_with_lanes, detected_objects)
                
                if is_paused:
                    cv2.putText(frame_with_objects, "PAUSED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if is_key_pressed():
                key = get_key()
                if key == 'q':
                    print("\n자율주행 모드를 종료하고 메인 메뉴로 돌아갑니다...")
                    return 'menu'
                elif key == 'm':
                    print("\n메인 메뉴로 돌아갑니다...")
                    return 'menu'
                elif key == ' ':
                    is_paused = not is_paused
                    print(f"\n{'⏸️  일시정지 상태' if is_paused else '▶️  주행 재시작'}")
            
            if display_mode:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q')):
                    break
                elif key == 32:
                    is_paused = not is_paused
                    print(f"\n{'⏸️  일시정지 상태' if is_paused else '▶️  주행 재시작'}")

            # 목적지 도착 확인 로직 (예시)
            if current_command.get('destination_reached', False):
                print("목적지에 도착했습니다. 다음 명령을 기다립니다...")
                command_handler.clear_current_command()

    except KeyboardInterrupt:
        print("\n\n자율주행 모드가 Ctrl+C로 중단되었습니다.")
        return 'menu'
    except Exception as e:
        print(f"\n\n자율주행 모드 오류: {e}")
        print("상세 오류 정보:")
        import traceback
        traceback.print_exc()
        return 'menu'
    finally:
        restore_terminal_mode(old_terminal_settings)
        clear_screen()
        print("🚗 자율주행 종료\n")
        print("메인 메뉴로 돌아갑니다...")
        
        try:
            command_handler.stop_server()
        except Exception as e:
            print(f"서버 종료 중 오류 발생: {e}")
        
        camera_manager.release_camera()
        if display_mode:
            cv2.destroyAllWindows()
        base.base_velocity_ctrl(0, 0)
        if hasattr(base, 'gimbal_dev_close'):
            pass
        if hasattr(base, 'shutdown'):
            base.shutdown()
        
        return 'menu'

# 메인 메뉴 출력 함수
def print_menu():
    print("\n===== 모드 선택 =====")
    print("a: 자율주행 (lane tracking) - 화면 표시")
    print("an: 자율주행 (lane tracking) - 화면 미표시")
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
        mode = input("모드 선택 (a/an/mp/mt/c/cc/cal/q/x): ").strip().lower()

        if mode == 'a':
            result = main(display_mode=True)  # 자율주행 모드 실행 (화면 표시)
            if result == 'menu':
                continue
        elif mode == 'an':
            result = main(display_mode=False)  # 자율주행 모드 실행 (화면 미표시)
            if result == 'menu':
                continue
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
                    # cv2.imshow("Camera Test", frame)
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