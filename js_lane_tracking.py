import cv2
import time
import glob
import numpy as np
import torch
from base_ctrl_js import BaseController
from ultralytics import YOLO
import sys

# === 메뉴얼 컨트롤러 (키보드) ===
import pygame

class RobotKeyboardController:
    def __init__(self, base):
        self.base = base
        pygame.init()
        pygame.display.set_caption("로봇 키보드 제어")
        self.screen = pygame.display.set_mode((600, 400))
        self.font = pygame.font.Font(None, 36)
        self.linear_speed = 0.0
        self.angular_speed = 0.0
        self.running = True
        self.last_update_time = time.time()
        self.light_on = False
        self.MAX_STEER = 0.5
        self.MAX_SPEED = 0.5
        self.STEP_STEER = 0.4
        self.STEP_SPEED = 0.02
        self.UPDATE_INTERVAL = 0.05

    def update_robot(self):
        self.base.base_velocity_ctrl(self.linear_speed, self.angular_speed)

    def handle_key_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_l:
                    self.light_on = not self.light_on
                    if self.light_on:
                        self.base.lights_ctrl(255, 255)
                    else:
                        self.base.lights_ctrl(0, 0)
                elif event.key == pygame.K_SPACE:
                    self.linear_speed = 0.0
                    self.angular_speed = 0.0
                    self.update_robot()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            self.linear_speed = min(self.linear_speed + self.STEP_SPEED, self.MAX_SPEED)
        elif keys[pygame.K_DOWN]:
            self.linear_speed = max(self.linear_speed - self.STEP_SPEED, -self.MAX_SPEED)
        else:
            if self.linear_speed > 0:
                self.linear_speed = max(0, self.linear_speed - self.STEP_SPEED)
            elif self.linear_speed < 0:
                self.linear_speed = min(0, self.linear_speed + self.STEP_SPEED)
        if keys[pygame.K_LEFT]:
            self.angular_speed = max(self.angular_speed - self.STEP_STEER, -self.MAX_STEER)
        elif keys[pygame.K_RIGHT]:
            self.angular_speed = min(self.angular_speed + self.STEP_STEER, self.MAX_STEER)
        else:
            if self.angular_speed > 0:
                self.angular_speed = max(0, self.angular_speed - self.STEP_STEER)
            elif self.angular_speed < 0:
                self.angular_speed = min(0, self.angular_speed + self.STEP_STEER)

    def update_display(self):
        self.screen.fill((0, 0, 0))
        speed_text = self.font.render(f"Speed: {self.linear_speed:.2f} m/s", True, (255, 255, 255))
        steer_text = self.font.render(f"Steering: {self.angular_speed:.2f} rad/s", True, (255, 255, 255))
        light_text = self.font.render(f"Light: {'ON' if self.light_on else 'OFF'}", True, (255, 255, 255))
        self.screen.blit(speed_text, (50, 50))
        self.screen.blit(steer_text, (50, 100))
        self.screen.blit(light_text, (50, 150))
        help_text1 = self.font.render("Arrow keys: Move and Turn", True, (200, 200, 200))
        help_text2 = self.font.render("L: Toggle Light", True, (200, 200, 200))
        help_text3 = self.font.render("Spacebar: Emergency Stop", True, (200, 200, 200))
        help_text4 = self.font.render("ESC: Exit", True, (200, 200, 200))
        self.screen.blit(help_text1, (50, 250))
        self.screen.blit(help_text2, (50, 290))
        self.screen.blit(help_text3, (50, 330))
        self.screen.blit(help_text4, (50, 370))
        pygame.display.flip()

    def run(self):
        try:
            print("메뉴얼 모드: 방향키로 로봇을 제어하세요! (ESC: 종료, q: 메뉴로)")
            while self.running:
                self.handle_key_events()
                if time.time() - self.last_update_time >= self.UPDATE_INTERVAL:
                    self.update_robot()
                    self.last_update_time = time.time()
                self.update_display()
                pygame.time.delay(10)
                # q키로 메뉴 복귀
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                        self.running = False
                        return 'menu'
        finally:
            self.base.base_velocity_ctrl(0, 0)
            pygame.quit()

# ──────────────────────────────────────────────
# 베이스 컨트롤러 초기화
# ──────────────────────────────────────────────
available_ports = glob.glob('/dev/ttyUSB*')
if available_ports:
    port = available_ports[0]
    print(f"시리얼 포트 감지됨: {port}")
else:
    print("시리얼 포트를 찾을 수 없습니다. 가상 모드로 실행합니다.")
    port = "VIRTUAL"
base = BaseController(port, 115200)

# ──────────────────────────────────────────────
# 카메라 열기 (Jetson용 GStreamer 파이프라인 예시)
# ──────────────────────────────────────────────
def gstreamer_pipeline(sensor_id=0, width=1280, height=720, framerate=30):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, format=NV12, framerate={framerate}/1 ! "
        f"nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink"
    )

def autonomous_mode(base):
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO("lane.pt").to(device)
    lane_class_id = 12
    cap = cv2.VideoCapture(gstreamer_pipeline(sensor_id=0), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("❌ 카메라 열기 실패")
        return
    print("✅ 카메라 연결됨 (자율주행)")
    MAX_STEER = 0.5
    MAX_SPEED = 0.5
    MIN_SPEED = 0.5
    STRAIGHT_SPEED = 0.3
    TURN_THRESHOLD = 0.15
    prev_t = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 프레임 수신 실패")
            break
        img_center_x = frame.shape[1] / 2
        left_lanes = []
        right_lanes = []
        results = model.predict(frame, device=device, conf=0.5, iou=0.45, stream=True)
        for r in results:
            boxes = r.boxes
            if boxes is None or boxes.xyxy is None:
                continue
            clss = boxes.cls.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            for (x1, y1, x2, y2), cls_id in zip(xyxy, clss):
                if int(cls_id) == lane_class_id:
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    if cx < img_center_x:
                        left_lanes.append((cx, cy))
                    else:
                        right_lanes.append((cx, cy))
        if left_lanes and right_lanes:
            left_mean = np.mean(left_lanes, axis=0)
            right_mean = np.mean(right_lanes, axis=0)
            center_point = (left_mean + right_mean) / 2
            center_x = center_point[0]
            deviation = (center_x - img_center_x) / img_center_x
            deviation = np.clip(deviation, -1.0, 1.0)
            steering = -MAX_STEER * deviation
            if abs(steering) < 0.4 and abs(deviation) > TURN_THRESHOLD:
                steering = -0.4 if deviation > 0 else 0.4
            steering = np.clip(steering, -MAX_STEER, MAX_STEER)
            if abs(deviation) < TURN_THRESHOLD:
                linear_speed = STRAIGHT_SPEED
            else:
                linear_speed = MIN_SPEED
            linear_speed = np.clip(linear_speed, -MAX_SPEED, MAX_SPEED)
            base.base_velocity_ctrl(linear_speed, steering)
            print(f"전송: v:{linear_speed:.2f}, s:{steering:.2f}, dev:{deviation:.2f}")
        elif left_lanes:
            left_mean = np.mean(left_lanes, axis=0)
            deviation = (left_mean[0] - img_center_x) / img_center_x - 0.3
            deviation = np.clip(deviation, -1.0, 1.0)
            steering = -MAX_STEER * deviation
            if abs(steering) < 0.4:
                steering = -0.4 if deviation > 0 else 0.4
            steering = np.clip(steering, -MAX_STEER, MAX_STEER)
            linear_speed = MIN_SPEED
            base.base_velocity_ctrl(linear_speed, steering)
            print(f"왼쪽만: v:{linear_speed:.2f}, s:{steering:.2f}, dev:{deviation:.2f}")
        elif right_lanes:
            right_mean = np.mean(right_lanes, axis=0)
            deviation = (right_mean[0] - img_center_x) / img_center_x + 0.3
            deviation = np.clip(deviation, -1.0, 1.0)
            steering = -MAX_STEER * deviation
            if abs(steering) < 0.4:
                steering = -0.4 if deviation > 0 else 0.4
            steering = np.clip(steering, -MAX_STEER, MAX_STEER)
            linear_speed = MIN_SPEED
            base.base_velocity_ctrl(linear_speed, steering)
            print(f"오른쪽만: v:{linear_speed:.2f}, s:{steering:.2f}, dev:{deviation:.2f}")
        curr_t = time.time()
        fps = 1.0 / (curr_t - prev_t)
        prev_t = curr_t
        steer_dir = "직진"
        if 'steering' in locals():
            if steering > 0.1:
                steer_dir = "우회전"
            elif steering < -0.1:
                steer_dir = "좌회전"
        cv2.putText(frame, f"dev:{deviation:.2f} steer:{steering:.2f} [{steer_dir}]", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
        cv2.putText(frame, f"{fps:.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("YOLO-AutoDrive", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
    cap.release()
    cv2.destroyAllWindows()
    base.base_velocity_ctrl(0, 0)
    if hasattr(base, 'gimbal_dev_close'):
        base.gimbal_dev_close()
    print("🚗 자율주행 종료")

# ──────────────────────────────────────────────
# 모드 선택 메뉴
# ──────────────────────────────────────────────
def print_menu():
    print("\n===== 모드 선택 =====")
    print("a: 자율주행 (lane tracking)")
    print("m: 메뉴얼 (키보드)")
    print("q: 정지 및 메뉴로")
    print("x: 종료")
    print("====================")

def main():
    while True:
        print_menu()
        mode = input("모드 선택 (a/m/q/x): ").strip().lower()
        if mode == 'a':
            autonomous_mode(base)
        elif mode == 'm':
            controller = RobotKeyboardController(base)
            result = controller.run()
            if result == 'menu':
                continue
        elif mode == 'q':
            base.base_velocity_ctrl(0, 0)
            print("정지 및 메뉴로 돌아갑니다.")
            continue
        elif mode == 'x':
            base.base_velocity_ctrl(0, 0)
            print("프로그램 종료!")
            break
        else:
            print("잘못된 입력입니다. 다시 선택하세요.")

if __name__ == "__main__":
    main()
