import time
from base_ctrl_js import BaseController
import glob
import threading
import queue
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import whisper
import warnings

# 경고 메시지 무시
warnings.filterwarnings("ignore")

SAMPLE_RATE = 16000
CHANNELS = 1
OUTPUT_FILE = "recorded_audio.wav"
recording = []

# Whisper 모델을 전역 변수로 한 번만 로드
print("음성 인식 모델을 로드하는 중...")
whisper_model = whisper.load_model("tiny.en")

def audio_callback(indata, frames, time_info, status):
    recording.append(indata.copy())

def record_audio():
    global recording
    recording = []
    print("🎤 음성을 녹음 중입니다... (Enter를 눌러 녹음을 종료하세요)")
    
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback):
        input()  # Enter 키를 누를 때까지 대기
    
    audio_data = np.concatenate(recording, axis=0)
    wav.write(OUTPUT_FILE, SAMPLE_RATE, audio_data)

def transcribe_audio():
    result = whisper_model.transcribe(OUTPUT_FILE)
    return result["text"].strip().lower()

class TarsAIDriver:
    def __init__(self):
        # 시리얼 포트 찾기 및 BaseController 초기화
        try:
            available_ports = glob.glob('/dev/ttyUSB*')
            if available_ports:
                port = available_ports[0]
                print(f"사용 가능한 시리얼 포트: {port}")
                self.base = BaseController(port, 115200)
            else:
                print("시리얼 포트를 찾을 수 없습니다. 가상 모드로 실행합니다.")
                self.base = BaseController("VIRTUAL", 115200)
        except Exception as e:
            print(f"초기화 오류: {e}")
            self.base = BaseController("VIRTUAL", 115200)

        # 제어 관련 변수
        self.running = True
        self.command_queue = queue.Queue()
        self.last_update_time = time.time()
        self.UPDATE_INTERVAL = 0.1  # 100ms 간격으로 제어 명령 전송

        # 기본 제어 파라미터
        self.linear_speed = 0.0
        self.angular_speed = 0.0
        self.MAX_SPEED = 0.5
        self.MAX_STEER = 0.3

        # 음성 명령 매핑
        self.command_mapping = {
            'go': (0.3, 0.0),      # 전진
            'back': (-0.3, 0.0),   # 후진
            'left': (0.5, 0.3),    # 좌회전
            'right': (0.5, -0.3),  # 우회전
            'stop': (0.0, 0.0)     # 정지
        }

    def update_robot(self):
        """로봇의 속도와 방향을 업데이트"""
        self.base.base_velocity_ctrl(self.linear_speed, self.angular_speed)

    def set_velocity(self, linear_x, angular_z):
        """선형 속도와 각속도 설정"""
        # 속도 제한 적용
        self.linear_speed = np.clip(linear_x, -self.MAX_SPEED, self.MAX_SPEED)
        self.angular_speed = np.clip(angular_z, -self.MAX_STEER, self.MAX_STEER)

    def stop(self):
        """로봇 정지"""
        self.linear_speed = 0.0
        self.angular_speed = 0.0
        self.update_robot()

    def process_voice_command(self, command):
        """음성 명령 처리"""
        # 명령어 텍스트 정리 (소문자 변환, 특수문자 제거)
        command = command.lower().strip('!.,?')
        
        # 명령어 매핑에서 가장 잘 매칭되는 명령 찾기
        best_match = None
        for cmd in self.command_mapping.keys():
            if cmd in command:
                best_match = cmd
                break
        
        if best_match:
            linear, angular = self.command_mapping[best_match]
            self.set_velocity(linear, angular)
            print(f"명령 실행: {best_match} (선속도: {linear}, 각속도: {angular})")
        else:
            print(f"알 수 없는 명령: {command}")
            print("사용 가능한 명령어: go, back, left, right, stop")

    def run(self):
        """메인 제어 루프"""
        print("음성 제어 모드 시작...")
        print("사용 가능한 명령어: go, back, left, right, stop")
        print("Enter를 눌러 음성 명령을 입력하세요. 종료하려면 'quit'를 입력하세요.")
        
        try:
            while self.running:
                user_input = input("\n🔘 [Enter]를 눌러 음성 입력을 시작하세요 (또는 'quit' 입력):").strip()
                
                if user_input.lower() == 'quit':
                    break
                
                if not user_input:
                    record_audio()
                    command = transcribe_audio()
                    print(f"\n👤 음성 입력: {command}")
                    self.process_voice_command(command)
                
                # 주기적으로 로봇 상태 업데이트
                current_time = time.time()
                if current_time - self.last_update_time >= self.UPDATE_INTERVAL:
                    self.update_robot()
                    self.last_update_time = current_time
                
                time.sleep(0.01)  # CPU 사용량 감소를 위한 짧은 대기

        except KeyboardInterrupt:
            print("\n사용자에 의해 중단됨")
        except Exception as e:
            print(f"\n오류 발생: {e}")
        finally:
            self.stop()
            print("음성 제어 모드 종료")

if __name__ == "__main__":
    driver = TarsAIDriver()
    driver.run() 