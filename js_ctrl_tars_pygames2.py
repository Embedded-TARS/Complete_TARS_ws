import pygame
import sys
import time
from base_ctrl_js import BaseController

# === 상수 정의 ===
MAX_STEER = 0.5     # 최대 조향 각속도 (rad/s)
MAX_SPEED = 0.5     # 최대 선형 속도 (m/s)
STEP_STEER = 0.4    # 조향 증가 단계
STEP_SPEED = 0.02   # 속도 증가 단계
UPDATE_INTERVAL = 0.05  # 명령 업데이트 간격 (초)

class RobotKeyboardController:
    def __init__(self, port='/dev/ttyUSB0', baud=115200):
        # 로봇 컨트롤러 초기화
        print(f"로봇 컨트롤러 초기화 중... ({port}, {baud})")
        self.base = BaseController(port, baud)
        
        # pygame 초기화
        pygame.init()
        pygame.display.set_caption("로봇 키보드 제어")
        self.screen = pygame.display.set_mode((600, 400))
        self.font = pygame.font.Font(None, 36)
        
        # 제어 변수 초기화
        self.linear_speed = 0.0  # 선형 속도 (m/s)
        self.angular_speed = 0.0  # 각속도 (rad/s)
        self.running = True
        self.last_update_time = time.time()
        
        # 베이스 라이트 상태
        self.light_on = False
        
        print("초기화 완료!")
        
    def update_robot(self):
        """로봇에 속도 명령 전송"""
        self.base.base_velocity_ctrl(self.linear_speed, self.angular_speed)
        
    def handle_key_events(self):
        """키보드 이벤트 처리"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    
                # 조명 제어
                elif event.key == pygame.K_l:
                    self.light_on = not self.light_on
                    if self.light_on:
                        self.base.lights_ctrl(255, 255)
                    else:
                        self.base.lights_ctrl(0, 0)
                
                # 비상 정지
                elif event.key == pygame.K_SPACE:
                    self.linear_speed = 0.0
                    self.angular_speed = 0.0
                    self.update_robot()
                    
        # 키 상태 가져오기 (누르고 있는 키 감지)
        keys = pygame.key.get_pressed()
        
        # 속도 제어
        if keys[pygame.K_UP]:
            self.linear_speed = min(self.linear_speed + STEP_SPEED, MAX_SPEED)
        elif keys[pygame.K_DOWN]:
            self.linear_speed = max(self.linear_speed - STEP_SPEED, -MAX_SPEED)
        else:
            # 키를 누르지 않으면 속도 감소
            if self.linear_speed > 0:
                self.linear_speed = max(0, self.linear_speed - STEP_SPEED)
            elif self.linear_speed < 0:
                self.linear_speed = min(0, self.linear_speed + STEP_SPEED)
        
        # 조향 제어
        if keys[pygame.K_LEFT]:
            self.angular_speed = max(self.angular_speed - STEP_STEER, -MAX_STEER)
        elif keys[pygame.K_RIGHT]:
            self.angular_speed = min(self.angular_speed + STEP_STEER, MAX_STEER)
        else:
            # 키를 누르지 않으면 조향 복원
            if self.angular_speed > 0:
                self.angular_speed = max(0, self.angular_speed - STEP_STEER)
            elif self.angular_speed < 0:
                self.angular_speed = min(0, self.angular_speed + STEP_STEER)

    
    def update_display(self):
        """화면 업데이트"""
        self.screen.fill((0, 0, 0))
        
        # 상태 정보 표시
        speed_text = self.font.render(f"Speed: {self.linear_speed:.2f} m/s", True, (255, 255, 255))
        steer_text = self.font.render(f"Steering: {self.angular_speed:.2f} rad/s", True, (255, 255, 255))
        light_text = self.font.render(f"Light: {'ON' if self.light_on else 'OFF'}", True, (255, 255, 255))
        
        self.screen.blit(speed_text, (50, 50))
        self.screen.blit(steer_text, (50, 100))
        self.screen.blit(light_text, (50, 150))
        
        # Control help display
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
        """메인 루프"""
        try:
            print("프로그램 시작. 방향키로 로봇을 제어하세요!")
            while self.running:
                current_time = time.time()
                
                # 키 이벤트 처리
                self.handle_key_events()
                
                # 일정 간격으로 로봇 명령 업데이트
                if current_time - self.last_update_time >= UPDATE_INTERVAL:
                    self.update_robot()
                    self.last_update_time = current_time
                
                # 화면 업데이트
                self.update_display()
                
                # CPU 사용량 줄이기
                pygame.time.delay(10)
                
        except KeyboardInterrupt:
            print("키보드 인터럽트로 프로그램 종료")
        except Exception as e:
            print(f"오류 발생: {e}")
        finally:
            # 종료 시 로봇 정지 및 pygame 종료
            try:
                self.base.base_velocity_ctrl(0, 0)
                pygame.quit()
            except:
                pass
            print("프로그램이 종료되었습니다.")

if __name__ == "__main__":
    import os
    import glob
    
    # 사용 가능한 시리얼 포트 찾기
    available_ports = glob.glob('/dev/ttyUSB*')
    
    if available_ports:
        port = available_ports[0]
        print(f"시리얼 포트 감지됨: {port}")
    else:
        print("시리얼 포트를 찾을 수 없습니다. 가상 모드로 실행합니다.")
        port = "VIRTUAL"
    
    try:
        # 기본 포트 및 속도 설정, 필요시 변경
        controller = RobotKeyboardController(port, 115200)
        controller.run()
    except Exception as e:
        print(f"프로그램 실행 중 오류 발생: {e}")