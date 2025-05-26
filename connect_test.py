import serial
import time
import glob

def find_serial_ports():
    """사용 가능한 시리얼 포트 찾기"""
    ports = []
    # Linux에서 일반적으로 사용되는 시리얼 포트 패턴들
    patterns = ['/dev/ttyUSB*', '/dev/ttyACM*', '/dev/ttyS*']
    
    for pattern in patterns:
        ports.extend(glob.glob(pattern))
    
    return ports

def test_connection(port):
    try:
        # 시리얼 포트 열기
        print(f"시리얼 포트 {port} 연결 시도 중...")
        ser = serial.Serial(port, 115200, timeout=1)
        print("시리얼 포트 연결 성공!")

        # 연결 상태 확인
        print("\n포트 정보:")
        print(f"포트: {ser.port}")
        print(f"Baud rate: {ser.baudrate}")
        print(f"바이트 크기: {ser.bytesize}")
        print(f"패리티: {ser.parity}")
        print(f"정지 비트: {ser.stopbits}")
        print(f"타임아웃: {ser.timeout}")

        # 간단한 테스트 메시지 전송
        print("\n테스트 메시지 전송 중...")
        test_message = "TEST\n"
        ser.write(test_message.encode('utf-8'))
        print(f"전송된 메시지: {test_message.strip()}")

        # 응답 대기
        print("\n응답 대기 중...")
        try:
            response = ser.readline().decode('utf-8').strip()
            print(f"수신된 응답: {response}")
        except:
            print("응답 없음")

        # 연결 종료
        ser.close()
        print("\n시리얼 포트 연결 종료")

    except serial.SerialException as e:
        print(f"시리얼 포트 오류: {e}")
    except Exception as e:
        print(f"예상치 못한 오류: {e}")

if __name__ == "__main__":
    # 사용 가능한 시리얼 포트 찾기
    available_ports = find_serial_ports()
    
    if not available_ports:
        print("사용 가능한 시리얼 포트를 찾을 수 없습니다.")
        print("다음을 확인해주세요:")
        print("1. USB 케이블이 제대로 연결되어 있는지")
        print("2. 로봇의 전원이 켜져있는지")
        print("3. 다른 USB 포트를 시도해보세요")
        print("\n시스템 정보 확인:")
        print("dmesg | grep tty 명령어로 시리얼 포트 인식 여부를 확인해보세요")
    else:
        print("발견된 시리얼 포트:")
        for port in available_ports:
            print(f"- {port}")
        
        # 첫 번째 발견된 포트로 테스트
        print(f"\n{available_ports[0]} 포트로 테스트를 시작합니다...")
        test_connection(available_ports[0]) 