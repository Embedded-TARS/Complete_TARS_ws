import serial
import json
import time

def test_serial_communication():
    try:
        # 시리얼 포트 열기
        ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
        print("시리얼 포트 연결 성공")

        # 테스트 명령 전송
        test_commands = [
            {"T": 13, "X": 0.1, "Z": 0},  # 천천히 전진
            {"T": 13, "X": 0, "Z": 0},    # 정지
            {"T": 13, "X": -0.1, "Z": 0}, # 천천히 후진
            {"T": 13, "X": 0, "Z": 0.1},  # 천천히 좌회전
            {"T": 13, "X": 0, "Z": -0.1}, # 천천히 우회전
        ]

        for cmd in test_commands:
            print(f"\n명령 전송: {cmd}")
            ser.write((json.dumps(cmd) + '\n').encode('utf-8'))
            
            # 응답 대기
            response = ser.readline().decode('utf-8').strip()
            print(f"응답: {response}")
            
            time.sleep(2)  # 2초 대기

        ser.close()
        print("\n테스트 완료")

    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    test_serial_communication() 