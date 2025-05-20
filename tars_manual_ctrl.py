# tars_manual_ctrl.py

import pygame
import time
import sys
import threading
import pathlib # Import pathlib
import cv2 # Import cv2
from tars_camera import CameraManager  # 변경된 부분: CameraManager 사용

from base_ctrl_js import BaseController

# Video Capture Configuration
OUTDIR = "./videos"
WIDTH, HEIGHT = 640, 480
FPS = 30

class BaseManualController:
    def __init__(self, base):
        self.base = base
        self.running = True
        self.linear_speed = 0.0
        self.angular_speed = 0.0
        self.MAX_STEER = 0.5
        self.MAX_SPEED = 0.5
        self.STEP_STEER = 0.05 # Smaller steps for smoother control
        self.STEP_SPEED = 0.02
        self.UPDATE_INTERVAL = 0.05
        self.light_on = False
        self.last_update_time = time.time()

    def update_robot(self):
        # Implement velocity sending and light control in subclasses
        pass

    def run(self):
        # Main loop handled by subclasses
        pass

class PygameKeyboardController(BaseManualController):
    def __init__(self, base):
        super().__init__(base)
        pygame.init()
        pygame.display.set_caption("로봇 키보드 제어 (Pygame)")
        self.screen = pygame.display.set_mode((600, 400))
        self.font = pygame.font.Font(None, 36)

    def update_robot(self):
        self.base.base_velocity_ctrl(self.linear_speed, self.angular_speed)
        if self.light_on:
            self.base.lights_ctrl(255, 255)
        else:
            self.base.lights_ctrl(0, 0)

    def handle_key_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return 'quit'
                elif event.key == pygame.K_l:
                    self.light_on = not self.light_on
                elif event.key == pygame.K_SPACE:
                    self.linear_speed = 0.0
                    self.angular_speed = 0.0
                    self.update_robot() # Immediate stop
                elif event.key == pygame.K_q: # Add 'q' to return to menu
                    self.running = False
                    return 'menu'

        keys = pygame.key.get_pressed()
        # Reset speeds if no keys are pressed (for gradual stop)
        linear_target = 0.0
        angular_target = 0.0

        if keys[pygame.K_UP]:
            linear_target = self.MAX_SPEED
        elif keys[pygame.K_DOWN]:
            linear_target = -self.MAX_SPEED

        if keys[pygame.K_LEFT]:
            angular_target = -self.MAX_STEER # Pygame left should turn left (reversed)
        elif keys[pygame.K_RIGHT]:
            angular_target = self.MAX_STEER # Pygame right should turn right (reversed)
            
        # Gradual change in speed and steer
        if self.linear_speed < linear_target:
            self.linear_speed = min(self.linear_speed + self.STEP_SPEED, linear_target)
        elif self.linear_speed > linear_target:
             self.linear_speed = max(self.linear_speed - self.STEP_SPEED, linear_target)

        if self.angular_speed < angular_target:
             self.angular_speed = min(self.angular_speed + self.STEP_STEER, angular_target)
        elif self.angular_speed > angular_target:
             self.angular_speed = max(self.angular_speed - self.STEP_STEER, angular_target)

        return None # Indicate continue

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
        help_text4 = self.font.render("Q: Menu, ESC: Exit", True, (200, 200, 200))
        self.screen.blit(help_text1, (50, 250))
        self.screen.blit(help_text2, (50, 290))
        self.screen.blit(help_text3, (50, 330))
        self.screen.blit(help_text4, (50, 370))
        pygame.display.flip()

    def run(self):
        print("메뉴얼 모드 (Pygame): 방향키로 로봇을 제어하세요! (Q: 메뉴로, ESC: 종료)")
        while self.running:
            action = self.handle_key_events()
            if action:
                break # Exit loop if 'menu' or 'quit' is returned

            current_time = time.time()
            if current_time - self.last_update_time >= self.UPDATE_INTERVAL:
                self.update_robot()
                self.last_update_time = current_time

            self.update_display()
            pygame.time.delay(10) # Small delay to reduce CPU usage

        self.base.base_velocity_ctrl(0, 0) # Stop robot on exit
        return action # Return the action ('menu' or 'quit')

class TerminalKeyboardController(BaseManualController):
    def __init__(self, base):
        super().__init__(base)
        self._input_thread = None
        self._input_char = None
        self._stop_event = threading.Event()
        # Adjust speeds for terminal control
        self.MAX_STEER = 0.2 # User requested >= 0.2 for rotation power
        self.MAX_SPEED = 0.4 # User requested default speed 0.3
        self.STEP_STEER = 0.1 # Terminal control is step-based
        self.STEP_SPEED = 0.1 # Terminal control is step-based

        # Video capture attributes
        self.camera_manager = None
        self.video_writer = None
        self.video_filepath = None
        
        # 비디오 녹화 상태
        self.is_recording = False
        
        # v 키 감지 변수 추가
        self.v_key_pressed = False

    def _read_input(self):
        # Read characters from stdin without waiting for newline
        import tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while not self._stop_event.is_set():
                ch = sys.stdin.read(1)
                
                # v 키가 입력되면 v_key_pressed 플래그 설정
                if ch == 'v':
                    self.v_key_pressed = True
                else:
                    self._input_char = ch
                    
                # Use a small sleep to prevent busy-waiting
                time.sleep(0.005)
        except Exception as e:
            print(f"Error reading input: {e}")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def update_robot(self):
        self.base.base_velocity_ctrl(self.linear_speed, self.angular_speed)
        if self.light_on:
            self.base.lights_ctrl(255, 255)
        else:
            self.base.lights_ctrl(0, 0)

    def handle_input(self):
        action = None
        if self._input_char is not None:
            ch = self._input_char
            self._input_char = None # Consume the character

            # Control mappings
            if ch == '\x1b': # ESC key
                 self.running = False
                 action = 'quit'
            elif ch == 'q':
                self.running = False
                action = 'menu'
            elif ch == ' ': # Emergency Stop
                self.linear_speed = 0.0
                self.angular_speed = 0.0
                self.update_robot() # Immediate stop
            elif ch == 'w': # Forward
                self.linear_speed = 0.3 # Base speed
            elif ch == 's': # Backward
                self.linear_speed = -0.3 # Base speed
            elif ch == 'a': # Turn Left (counter-clockwise) - Reversed
                 self.angular_speed = -self.MAX_STEER
                 if self.linear_speed != 0.0:
                     self.linear_speed = 0.5 if self.linear_speed > 0 else -0.5 # Boost speed while turning
            elif ch == 'd': # Turn Right (clockwise) - Reversed
                 self.angular_speed = self.MAX_STEER
                 if self.linear_speed != 0.0:
                     self.linear_speed = 0.5 if self.linear_speed > 0 else -0.5 # Boost speed while turning
            elif ch == 'l': # Toggle Light
                self.light_on = not self.light_on
                # print(f"Light: {'ON' if self.light_on else 'OFF'}") # Optional feedback
            # Add keys to stop movement when released (or use a different logic)
            # For simple step-based control, we just set speed and it holds
            # If 'w'/'s'/'a'/'d' are not pressed, speed stays, need stop command
            # Let's add explicit stop keys or timed decay
            elif ch == 'W': # Stop forward/backward
                self.linear_speed = 0.0
            elif ch == 'A' or ch == 'D': # Stop turning
                self.angular_speed = 0.0

            self.print_status()
        return action

    def toggle_recording(self):
        """비디오 녹화 시작/중지 토글 함수"""
        if not self.is_recording:
            # Start Recording
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            path = pathlib.Path(OUTDIR).resolve()
            path.mkdir(exist_ok=True, parents=True)
            self.video_filepath = str(path / f"{timestamp}.mp4")

            print(f"\n카메라 초기화 중 - 녹화 파일: {self.video_filepath}...")
            
            # CameraManager를 통해 카메라 초기화
            self.camera_manager = CameraManager.get_instance()
            camera = self.camera_manager.initialize_camera(width=WIDTH, height=HEIGHT, capture_fps=FPS)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.video_filepath, fourcc, FPS, (WIDTH, HEIGHT))

            if not self.video_writer.isOpened():
                print(f"❌ VideoWriter open failed for {self.video_filepath}")
                # Clean up camera if writer fails
                self.camera_manager.release_camera()
                self.camera_manager = None
                self.video_writer = None
                self.is_recording = False
                print("녹화 시작 실패!")
            else:
                self.is_recording = True
                print(f"✅ 녹화가 시작되었습니다: {self.video_filepath}. 'v'를 다시 누르면 중지됩니다.")
        else:
            # Stop Recording
            print("\n녹화 중지 중...")
            self.is_recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
                print("✅ VideoWriter 해제 완료.")
            if self.camera_manager is not None:
                self.camera_manager.release_camera()
                self.camera_manager = None
                print("✅ 카메라 해제 완료.")
            print("녹화가 중지되었습니다.")

    def print_status(self):
        # Simple status print for terminal mode
        direction_linear = "Stopped"
        if self.linear_speed > 0:
            direction_linear = "Forward"
        elif self.linear_speed < 0:
            direction_linear = "Backward"

        direction_angular = "Straight"
        if self.angular_speed > 0:
            direction_angular = "Left"
        elif self.angular_speed < 0:
            direction_angular = "Right"

        print(f"\rDirection: {direction_linear}, {direction_angular} | Speed: {self.linear_speed:.2f} m/s | Steering: {self.angular_speed:.2f} rad/s | Light: {'ON' if self.light_on else 'OFF'} | Recording: {'YES' if self.is_recording else 'NO'}  ", end='', flush=True)

    def run(self):
        print("메뉴얼 모드 (터미널): w/s (앞/뒤), a/d (왼/오), W/A/D (정지), space (긴급 정지), l (전등), q (메뉴), ESC (종료)")
        print("Press 'v' to start/stop video recording.") # Instruction for video recording

        action = None # Initialize action

        # Start the input reading thread
        self._stop_event.clear()
        self._input_thread = threading.Thread(target=self._read_input)
        self._input_thread.daemon = True # Allow the main program to exit even if thread is alive
        self._input_thread.start()

        try:
            # cv2.namedWindow("Camera Feed", cv2.WINDOW_AUTOSIZE) # Create OpenCV window

            self.print_status() # Initial status
            while self.running:
                # Handle user input
                input_action = self.handle_input()
                if input_action:
                    action = input_action # Set action if input handler returns one
                
                # v 키 처리 (별도 플래그 사용)
                if self.v_key_pressed:
                    self.v_key_pressed = False  # 플래그 초기화
                    self.toggle_recording()     # 녹화 토글 함수 호출
                    self.print_status()         # 상태 업데이트 표시

                if action is not None: # Break loop if handle_input returned 'menu' or 'quit'
                    break

                # Capture and write frame if recording is active
                if self.is_recording and self.camera_manager is not None:
                    frame = self.camera_manager.get_frame()
                    if frame is not None:
                        # cv2.imshow("Camera Feed", frame) # Display the frame
                        cv2.waitKey(1) # Process OpenCV window events
                        self.video_writer.write(frame)

                # Update robot at intervals even if no key is pressed (maintains last command)
                current_time = time.time()
                if current_time - self.last_update_time >= self.UPDATE_INTERVAL:
                    self.update_robot()
                    self.last_update_time = current_time

                time.sleep(0.01) # Small loop delay

        except KeyboardInterrupt: # Handle Ctrl+C
            print("\nCtrl+C detected. Returning to menu.")
            action = 'menu' # Set action to return to menu
        except Exception as e:
            print(f"\nError in terminal control loop: {e}")
            action = 'quit' # Exit on unexpected error
        finally:
            print("\nTerminal control cleanup...")
            self.base.base_velocity_ctrl(0, 0) # Stop robot on exit

            # Ensure recording is stopped and resources released on exit
            if self.is_recording and self.video_writer is not None:
                 print("녹화 중지 중...")
                 self.video_writer.release()
                 self.video_writer = None
                 print("✅ VideoWriter 해제 완료.")

            if self.camera_manager is not None:
                 print("카메라 해제 중...")
                 self.camera_manager.release_camera()
                 self.camera_manager = None
                 print("✅ 카메라 해제 완료.")

            self._stop_event.set() # Signal input thread to stop
            if self._input_thread and self._input_thread.is_alive():
                self._input_thread.join(1.0) # Wait for input thread to finish
                if self._input_thread.is_alive():
                     print("경고: 입력 스레드가 1초 내에 종료되지 않았습니다.")

            cv2.destroyAllWindows() # Ensure all OpenCV windows are closed
            print("터미널 모드가 종료되었습니다.") 
            return action # Return the action (menu, quit, or None from Exception)
