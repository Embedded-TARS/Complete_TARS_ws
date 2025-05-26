import serial
import time

# 시리얼 포트 설정
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)

# 명령 전송 (종료 문자 포함 여러 방식 시도)
cmds = [b'M', b'M\n', b'M\r', b'M\r\n']

for cmd in cmds:
    print(f"보내는 명령: {cmd}")
    ser.write(cmd)
    ser.flush()
    time.sleep(0.1)

    response = ser.readline()
    print(f"응답: {response}")
    time.sleep(0.5)

ser.close()
