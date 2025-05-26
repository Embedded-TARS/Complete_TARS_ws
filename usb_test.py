import serial
import time
import glob
import subprocess

def get_usb_info():
    """USB 장치 정보 가져오기"""
    try:
        # lsusb 명령어 실행
        result = subprocess.run(['lsusb'], capture_output=True, text=True)
        print("\n=== USB 장치 목록 ===")
        print(result.stdout)
        
        # dmesg에서 USB 관련 정보 가져오기
        result = subprocess.run(['dmesg | grep -i usb'], shell=True, capture_output=True, text=True)
        print("\n=== USB 연결 로그 ===")
        print(result.stdout)
    except Exception as e:
        print(f"USB 정보 가져오기 실패: {e}")

def test_serial_port(port):
    """시리얼 포트 테스트"""
    try:
        print(f"\n=== {port} 포트 테스트 ===")
        ser = serial.Serial(port, 115200, timeout=1)
        
        # 포트 정보 출력
        print(f"포트: {ser.port}")
        print(f"Baud rate: {ser.baudrate}")
        print(f"바이트 크기: {ser.bytesize}")
        print(f"패리티: {ser.parity}")
        print(f"정지 비트: {ser.stopbits}")
        
        # 간단한 테스트 명령 전송
        test_commands = [
            {"T": 0},  # 정지
            {"T": 13, "X": 0.1, "Z": 0},  # 천천히 전진
            {"T": 13, "X": 0, "Z": 0},  # 정지
        ]
        
        for cmd in test_commands:
            print(f"\n명령 전송: {cmd}")
            ser.write((str(cmd) + '\n').encode('utf-8'))
            time.sleep(1)
            
            try:
                response = ser.readline().decode('utf-8').strip()
                print(f"응답: {response}")
            except:
                print("응답 없음")
        
        ser.close()
        return True
    except Exception as e:
        print(f"포트 테스트 실패: {e}")
        return False

def main():
    print("USB 포트 테스트 시작")
    
    # USB 정보 출력
    get_usb_info()
    
    # 사용 가능한 시리얼 포트 찾기
    ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
    
    if not ports:
        print("\n시리얼 포트를 찾을 수 없습니다.")
        print("다음을 확인해주세요:")
        print("1. USB 케이블이 제대로 연결되어 있는지")
        print("2. 로봇의 전원이 켜져있는지")
        print("3. 다른 USB 포트를 시도해보세요")
    else:
        print("\n발견된 시리얼 포트:")
        for port in ports:
            print(f"- {port}")
        
        print("\n각 포트 테스트 시작...")
        for port in ports:
            if test_serial_port(port):
                print(f"\n{port} 포트 테스트 성공!")
            else:
                print(f"\n{port} 포트 테스트 실패!")

if __name__ == "__main__":
    main() 