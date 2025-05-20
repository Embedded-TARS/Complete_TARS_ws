
# 필요한 모듈 임포트
from tars_perception import LanePerception, LaneDetectionModel
from sign_detect import LanePlanner, SignDetectionModel, decide_action
from tars_control import RobotController
from tars_camera import CameraManager
from base_ctrl_js import BaseController
# from tars_pure_pursuit import PurePursuitController
import cv2
import glob
import time
import sys
# import pygame
from tars_manual_ctrl import PygameKeyboardController, TerminalKeyboardController
import pathlib

# 베이스 컨트롤러 초기화
available_ports = glob.glob('/dev/ttyUSB*')
if available_ports:
    port = available_ports[0]
    print(f"\u0001f50c 시리얼 포트 감지됨: {port}")
else:
    print("\u26a0\ufe0f 시리얼 포트를 찾을 수 없습니다. 가상 모드로 실행합니다.")
    port = "VIRTUAL"

base = BaseController(port, 115200)

# 자율주행 메인 루프
def main():
    lane_model = LaneDetectionModel(model_path="lane.pt", lane_class_id=12)
    sign_model = SignDetectionModel(model_path="detect.pt")
    perception = LanePerception(lane_width_px=700, ema_alpha=0.8)
    planner = LanePlanner()
    controller = RobotController(base)
    # camera_manager = CameraManager.get_instance()
    # camera_manager.initialize_camera(width=640, height=480, capture_fps=30)
    # pure_pursuit = PurePursuitController(lookahead_distance=1.0, wheelbase=0.5)

    prev_speed = 0.0
    prev_steering = 0.0

    print("\ud83d\ude97 자율주행 모드 시작 - q 키를 눌러 종료")

    cap = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L2)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\u274c 프레임 수신 실패")
                time.sleep(0.1)
                continue

            img_center_x = frame.shape[1] // 2
            y = frame.shape[0]
            roi = slice(y * 2 // 4, y)

            results = lane_model.predict(frame)
            lane_center_x = perception.update(results[0], roi=roi)

            signs = sign_model.detect_signs(frame)
            linear_speed, steering_angle = decide_action(
                signs, lane_center_x, img_center_x, prev_speed, prev_steering
            )

            controller.send_control(linear_speed, steering_angle)

            prev_speed = linear_speed
            prev_steering = steering_angle

            frame_with_lanes = perception.visualize_lanes(frame, 0.0, steering_angle, roi)
            # cv2.imshow("YOLO-AutoDrive", frame_with_lanes)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break

    except KeyboardInterrupt:
        print("\n\u26d4\ufe0f 자율주행 모드가 Ctrl+C로 중단되었습니다.")
    except Exception as e:
        print(f"\n\u26a0\ufe0f 자율주행 모드 오류: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        base.base_velocity_ctrl(0, 0)
        if hasattr(base, 'shutdown'):
            base.shutdown()
        print("\ud83d\ude97 자율주행 종료")


def print_menu():
    print("\n===== 모드 선택 =====")
    print("a: 자유주회 (lane tracking)")
    print("mp: 메뉴얼 (Pygame)")
    print("mt: 메뉴얼 (Terminal) - 비디오 노크타이드 포함")
    print("c: 카메라 테스트")
    print("cc: 카메라 사진 캐프")
    print("q: 정지 및 메뉴로")
    print("x: 종료")
    print("====================")

def main_menu():
    while True:
        print_menu()
        mode = input("모드 선택 (a/mp/mt/c/cc/q/x): ").strip().lower()

        if mode == 'a':
            main()
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
            print("정지 및 메뉴로 돌아가는다.")
        elif mode == 'x':
            base.base_velocity_ctrl(0, 0)
            print("프로그램 종료!")
            break
        else:
            print("잘못된 입력입니다. 다시 선택하세요.")

def run_camera_test():
    camera_manager = CameraManager.get_instance()
    camera_manager.initialize_camera()

    print("카메라 테스트 중... 'q' 키를 누르면 종료합니다.")

    try:
        while True:
            frame = camera_manager.get_frame()
            if frame is not None:
                cv2.imshow("Camera Test", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("프렌을 읽을 수 없습니다.")
                time.sleep(0.1)
    finally:
        camera_manager.release_camera()
        cv2.destroyAllWindows()

def capture_photo():
    camera_manager = CameraManager.get_instance()
    camera_manager.initialize_camera()

    print("카메라 준비 중...")
    time.sleep(1)

    OUTDIR = "./captures"
    path = pathlib.Path(OUTDIR).resolve()
    path.mkdir(exist_ok=True, parents=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_path = str(path / f"capture_{timestamp}.jpg")

    frame = camera_manager.get_frame()
    if frame is not None:
        cv2.imwrite(file_path, frame)
        print(f"\u2705 이미지가 저장되었습니다: {file_path}")
    else:
        print("\u274c 프렌을 캐프할 수 없습니다.")

    camera_manager.release_camera()

if __name__ == "__main__":
    main_menu()
