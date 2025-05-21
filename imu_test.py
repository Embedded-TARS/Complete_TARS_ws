import time
import sys
import glob
import termios
import tty
import select
import json

# tars_main.py에서 터미널 설정 함수 가져오기
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

# base_ctrl_js 모듈에서 BaseController 임포트
try:
    from base_ctrl_js import BaseController
except ImportError:
    print("Error: base_ctrl_js.py not found. Make sure it is in the same directory.")
    sys.exit(1)

def main():
    # 시리얼 포트 자동 감지 및 BaseController 초기화
    port = "VIRTUAL"
    available_ports = glob.glob('/dev/ttyUSB*')
    if available_ports:
        port = available_ports[0]
        print(f"시리얼 포트 감지됨: {port}")
    else:
        print("시리얼 포트를 찾을 수 없습니다. 실제 로봇 연결이 필요합니다.")
        sys.exit(1) # 가상 모드에서는 실제 IMU 데이터 테스트 불가

    try:
        # BaseController 초기화 (이때 시리얼 포트가 열립니다)
        base = BaseController(port, 115200)
        
        # 가상 모드가 아닌지 다시 확인 (안전을 위해)
        if base.virtual_mode:
            print("가상 모드에서는 실제 IMU 데이터를 읽을 수 없습니다.")
            sys.exit(1)

        print(f"시리얼 포트 {port}를 통해 IMU 데이터 읽기 시작 - q 키를 눌러 종료")
        print("로봇에게 {'T': 126} 명령을 주기적으로 보냅니다.")

    except Exception as e:
        print(f"BaseController 초기화 또는 시리얼 포트 열기 오류: {e}")
        sys.exit(1)

    # 터미널 설정 변경
    old_terminal_settings = set_terminal_mode()

    # IMU 데이터 요청 명령
    imu_request_command = {"T": 126}
    last_request_time = time.time()
    request_interval = 0.05 # IMU 데이터 요청 간격 (초) - 20Hz

    try:
        while True:
            # 주기적으로 IMU 데이터 요청 명령 전송
            current_time = time.time()
            if current_time - last_request_time > request_interval:
                # JSON dumps 후 바이트로 인코딩하여 전송 (base.base_json_ctrl 사용)
                base.base_json_ctrl(imu_request_command)
                last_request_time = current_time
                # print(f"Sent: {imu_request_command}") # 디버깅용: 보낸 명령 확인
                
            # BaseController의 피드백 데이터를 주기적으로 읽어서 IMU 데이터 업데이트
            # base.feedback_data() 호출 시 T=1002 데이터가 base.imu_data에 저장됨
            base.feedback_data()

            # 최신 IMU 데이터 가져오기
            imu_data = base.get_latest_imu_data()

            # 데이터가 유효한 경우 (T=1002) 출력
            if imu_data and imu_data.get('T') == 1002:
                # 화면 지우고 커서 이동하여 항상 같은 위치에 출력
                sys.stdout.write('\033[2J\033[H')
                sys.stdout.flush()

                print("==== IMU Data ====")
                print(f"Timestamp (T): {imu_data.get('T')}") # 1002
                # Roll, Pitch, Yaw (각도, 단위: 도)
                print(f"Roll (r): {imu_data.get('r', 0):.2f}°")
                print(f"Pitch (p): {imu_data.get('p', 0):.2f}°")
                print(f"Yaw (y): {imu_data.get('y', 0):.2f}°")
                print("------------------")
                # Linear Acceleration (가속도, 단위: m/s^2)
                print(f"Accel X (ax): {imu_data.get('ax', 0):.2f} m/s²")
                print(f"Accel Y (ay): {imu_data.get('ay', 0):.2f} m/s²")
                print(f"Accel Z (az): {imu_data.get('az', 0):.2f} m/s²")
                print("------------------")
                # Angular Velocity (각속도, 단위: deg/s 또는 rad/s 확인 필요 - Control Panel 예시 기반 °/s로 가정)
                print(f"Gyro X (gx): {imu_data.get('gx', 0):.2f}°/s")
                print(f"Gyro Y (gy): {imu_data.get('gy', 0):.2f}°/s")
                print(f"Yaw Rate (gz): {imu_data.get('gz', 0):.2f}°/s")
                print("------------------")
                # Temperature (단위: 섭씨)
                print(f"Temperature (temp): {imu_data.get('temp', 0):.2f} °C")
                print("==================")

                # 추가 정보 (예: 속도 등 - T=1003 데이터에서 가져올 수 있다면 출력 가능)
                # base_data = base.base_data # 최신 1003 타입 데이터 (feedback_data가 반환하는 데이터)
                # if base_data and base_data.get('T') == 1003:
                #     print(f"Base Data: {base_data}")
                #     # 예시 데이터 형식에 따라 속도 키가 다를 수 있음. 확인 후 사용하세요.
                #     # print(f"Base Speed L: {base_data.get('L', 0):.2f}, R: {base_data.get('R', 0):.2f}") # L, R 사용 예시
                #     print("==================")

            else:
                # IMU 데이터가 아직 수신되지 않았거나 유효하지 않음
                sys.stdout.write('\033[2J\033[H') # 화면 지우기
                sys.stdout.flush()
                print("Waiting for IMU data (T=1002)... Ensure robot is connected and sending data for T=126 requests.")
                # print(f"Latest received non-IMU data: {base.base_data}") # 디버깅용: 다른 데이터가 오는지 확인

            # 짧은 지연
            time.sleep(0.01) # 데이터 처리 및 요청 간격 고려하여 지연 조정

            # 키 입력 확인
            if is_key_pressed():
                key = get_key()
                if key == 'q':
                    break

    except KeyboardInterrupt:
        print("\nIMU 데이터 읽기 중단 (Ctrl+C)")
    except Exception as e:
        print(f"\n오류 발생: {e}")
    finally:
        # 종료 시 터미널 설정 복구
        restore_terminal_mode(old_terminal_settings)
        # 시리얼 포트 닫기
        if 'base' in locals() and hasattr(base, 'ser') and base.ser.isOpen():
             base.ser.close()
        print("IMU 데이터 읽기 종료.")

if __name__ == "__main__":
    main() 