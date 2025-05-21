import serial
import json
import queue
import threading
import os
import time
import glob
import numpy as np
import yaml

thisPath = os.path.dirname(os.path.abspath(__file__))

# config.yaml 파일 로드
def load_config():
    try:
        with open(thisPath + '/config.yaml', 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
            print("config.yaml 파일 로드 성공")
            return config
    except Exception as e:
        print(f"config.yaml 로드 오류: {e}")
        # 오류 발생 시 대체 설정 제공
        return {
            'base_config': {
                'use_lidar': False,
                'extra_sensor': False,
                'robot_name': 'UGV Rover'
            },
            'cmd_config': {
                'cmd_set_servo_id': 501,
                'cmd_servo_torque': 210,
                'cmd_set_servo_mid': 502,
                'cmd_movition_ctrl': 1
            },
            'args_config': {
                'max_speed': 1.3,
                'slow_speed': 0.2
            }
        }

# 설정 로드
f = load_config()

class ReadLine:
    def __init__(self, s):
        self.buf = bytearray()
        self.s = s

        self.ANGLE_PER_FRAME = 12
        self.HEADER = 0x54
        self.lidar_angles = []
        self.lidar_distances = []
        self.lidar_angles_show = []
        self.lidar_distances_show = []
        self.last_start_angle = 0
        self.breath_light_flag = True

    def readline(self):
        i = self.buf.find(b"\n")
        if i >= 0:
            r = self.buf[:i+1]
            self.buf = self.buf[i+1:]
            return r
        while True:
            i = max(1, min(512, self.s.in_waiting))
            data = self.s.read(i)
            i = data.find(b"\n")
            if i >= 0:
                r = self.buf + data[:i+1]
                self.buf[0:] = data[i+1:]
                return r
            else:
                self.buf.extend(data)

    def clear_buffer(self):
        self.s.reset_input_buffer()

    def parse_lidar_frame(self, data):
        start_angle = (data[5] << 8 | data[4]) * 0.01
        for i in range(0, self.ANGLE_PER_FRAME):
            offset = 6 + i * 3
            distance = data[offset+1] << 8 | data[offset]
            confidence = data[offset+2]
            self.lidar_angles.append(np.radians(start_angle + i * 0.83333 + 180))
            self.lidar_distances.append(distance)
        return start_angle

    def lidar_data_recv(self):
        if self.lidar_ser == None:
            return
        try:
            while True:
                self.header = self.lidar_ser.read(1)
                if self.header == b'\x54':
                    # Read the rest of the data
                    data = self.header + self.lidar_ser.read(46)
                    hex_data = [int(hex(byte), 16) for byte in data]
                    start_angle = self.parse_lidar_frame(hex_data)
                    if self.last_start_angle > start_angle:
                        break
                    self.last_start_angle = start_angle
                else:
                    self.lidar_ser.flushInput()

            self.last_start_angle = start_angle
            self.lidar_angles_show = self.lidar_angles.copy()
            self.lidar_distances_show = self.lidar_distances.copy()
            self.lidar_angles.clear()
            self.lidar_distances.clear()
        except Exception as e:
            print(f"[base_ctrl.lidar_data_recv] error: {e}")
            self.lidar_ser = serial.Serial(glob.glob('/dev/ttyACM*')[0], 230400, timeout=1)


class BaseController:

    def __init__(self, uart_dev_set, buad_set):
        try:
            self.ser = serial.Serial(uart_dev_set, buad_set, timeout=1)
            print(f"시리얼 포트 {uart_dev_set} 연결 성공")
        except Exception as e:
            print(f"시리얼 포트 연결 오류: {e}")
            # 오류 발생 시 가상 시리얼 모드로 전환 (테스트용)
            self.virtual_mode = True
            print("가상 모드로 실행합니다. 명령은 콘솔에 출력됩니다.")
            return
            
        self.virtual_mode = False
        self.rl = ReadLine(self.ser)
        self.command_queue = queue.Queue()
        self.command_thread = threading.Thread(target=self.process_commands, daemon=True)
        self.command_thread.start()

        self.base_light_status = 0
        self.head_light_status = 0

        self.data_buffer = None
        self.base_data = None
        self.imu_data = None

        # config.yaml에서 설정 로드
        self.use_lidar = f['base_config']['use_lidar']
        self.extra_sensor = f['base_config']['extra_sensor']
        self.robot_name = f['base_config']['robot_name']
        print(f"로봇 이름: {self.robot_name}, 라이다 사용: {self.use_lidar}, 추가 센서: {self.extra_sensor}")

    def feedback_data(self):
        if self.virtual_mode:
            # 가상 모드에서는 더미 IMU 데이터 반환 (필요시)
            # return {"T": 1002, "r": 0.0, "p": 0.0, "y": 0.0, "ax": 0.0, "ay": 0.0, "az": 9.8, "gx": 0.0, "gy": 0.0, "gz": 0.0, "temp": 25.0}
            return {"T": 1003, "virtual": True} # 기본 가상 응답 유지
            
        try:
            # 시리얼 버퍼에서 데이터 읽기
            while self.rl.s.in_waiting > 0:
                # 한 라인씩 읽고 디코딩 (오류 무시)
                line = self.rl.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    try:
                        # JSON 파싱 시도
                        data = json.loads(line)
                        
                        # 'T' 필드가 있는지 확인
                        if 'T' in data:
                            # T=1003 데이터는 self.base_data에 저장 (기존 로직 유지)
                            if data['T'] == 1003:
                                self.base_data = data
                                # print(f"Received 1003: {self.base_data}") # 디버깅 출력 (필요시)
                                
                            # T=1002 데이터는 IMU 데이터로 간주하고 self.imu_data에 저장
                            elif data['T'] == 1002:
                                self.imu_data = data # IMU 데이터 저장
                                # print(f"Received IMU (1002): {self.imu_data}") # 디버깅 출력 (필요시)
                            
                            # 다른 타입의 데이터도 필요에 따라 처리 가능
                            # elif data['T'] == 126:
                            #     print(f"Received command response (126): {data}") # 126 응답 자체는 데이터가 아닐 수 있음
                                
                    except json.JSONDecodeError:
                        # JSON 파싱 오류 발생 시 출력
                        print(f"[feedback_data] JSON Decode Error: {line}")
                    except Exception as e:
                        # 그 외 오류 발생 시 출력
                        print(f"[feedback_data] Processing Error: {e} for line: {line}")
                        
            # 최신 base_data (T=1003)를 반환하거나, 없으면 None 반환
            # IMU 데이터(T=1002)는 내부 변수에 저장되어 get_latest_imu_data()로 접근
            return self.base_data

        except Exception as e:
            # 전체 예외 발생 시 출력
            print(f"[base_ctrl.feedback_data] overall error: {e}")
            return {"error": str(e)}

    # IMU 데이터 가져오는 메서드
    def get_latest_imu_data(self):
        """가장 최근에 수신된 T=1002 타입의 IMU 데이터를 반환합니다."""
        return self.imu_data

    def on_data_received(self):
        if self.virtual_mode:
            return {"T": 1000, "virtual": True}
            
        self.ser.reset_input_buffer()
        data_read = json.loads(self.rl.readline().decode('utf-8'))
        return data_read

    def send_command(self, data):
        if self.virtual_mode:
            print(f"가상 명령 전송: {data}")
            return
            
        self.command_queue.put(data)

    def process_commands(self):
        while True:
            data = self.command_queue.get()
            if self.virtual_mode:
                print(f"가상 명령 처리: {data}")
            else:
                try:
                    self.ser.write((json.dumps(data) + '\n').encode("utf-8"))
                except serial.serialutil.SerialException as e:
                    print(f"[process_commands] Serial write failed: {e}")
                    # Depending on desired behavior, might try to reconnect or just log
                    # For now, just log and continue the loop

    def base_json_ctrl(self, input_json):
        self.send_command(input_json)

    def gimbal_emergency_stop(self):
        data = {"T":0}
        self.send_command(data)

    def base_speed_ctrl(self, input_left, input_right):
        data = {"T":f['cmd_config']['cmd_movition_ctrl'],"L":input_left,"R":input_right}
        self.send_command(data)

    # 새로운 속도 제어 메서드 (선형 속도 + 각속도)
    def base_velocity_ctrl(self, linear_x, angular_z):
        """
        Control the base using linear velocity (X) and angular velocity (Z)
        
        Args:
            linear_x (float): Linear velocity in m/s
            angular_z (float): Angular velocity in rad/s
        """
        data = {"T":13,"X":linear_x,"Z":angular_z}
        self.send_command(data)

    def gimbal_ctrl(self, input_x, input_y, input_speed, input_acceleration):
        data = {"T":f['cmd_config']['cmd_gimbal_ctrl'],"X":input_x,"Y":input_y,"SPD":input_speed,"ACC":input_acceleration}
        self.send_command(data)

    def gimbal_base_ctrl(self, input_x, input_y, input_speed):
        data = {"T":f['cmd_config']['cmd_gimbal_base_ctrl'],"X":input_x,"Y":input_y,"SPD":input_speed}
        self.send_command(data)

    def base_oled(self, input_line, input_text):
        data = {"T":3,"lineNum":input_line,"Text":input_text}
        self.send_command(data)

    def base_default_oled(self):
        data = {"T":-3}
        self.send_command(data)

    def bus_servo_id_set(self, old_id, new_id):
        data = {"T":f['cmd_config']['cmd_set_servo_id'],"raw":old_id,"new":new_id}
        self.send_command(data)

    def bus_servo_torque_lock(self, input_id, input_status):
        data = {"T":f['cmd_config']['cmd_servo_torque'],"id":input_id,"cmd":input_status}
        self.send_command(data)

    def bus_servo_mid_set(self, input_id):
        data = {"T":f['cmd_config']['cmd_set_servo_mid'],"id":input_id}
        self.send_command(data)

    def lights_ctrl(self, pwmA, pwmB):
        data = {"T":132,"IO4":pwmA,"IO5":pwmB}
        self.send_command(data)
        self.base_light_status = pwmA
        self.head_light_status = pwmB

    def base_lights_ctrl(self):
        if self.base_light_status != 0:
            self.base_light_status = 0
        else:
            self.base_light_status = 255
        self.lights_ctrl(self.base_light_status, self.head_light_status)

    def gimbal_dev_close(self):
        if not self.virtual_mode:
            self.ser.close()

    def change_breath_light_flag(self, input_cmd):
        self.breath_light_flag = input_cmd

    def breath_light(self, input_time):
        self.change_breath_light_flag(True)
        breath_start_time = time.monotonic()
        while time.monotonic() - breath_start_time < input_time:
            for i in range(0, 128, 10):
                if not self.breath_light_flag:
                    self.lights_ctrl(0, 0)
                    return
                self.lights_ctrl(i, 128-i)
                time.sleep(0.1)
            for i in range(0, 128, 10):
                if not self.breath_light_flag:
                    self.lights_ctrl(0, 0)
                    return
                self.lights_ctrl(128-i, i)
                time.sleep(0.1)
        self.lights_ctrl(0, 0)

# 테스트 코드
if __name__ == "__main__":
    print("BaseController 테스트")
    # 시리얼 포트 찾기 시도
    try:
        available_ports = glob.glob('/dev/ttyUSB*')
        if available_ports:
            port = available_ports[0]
            print(f"사용 가능한 시리얼 포트: {port}")
            ctrl = BaseController(port, 115200)
            # 테스트 명령 보내기
            ctrl.base_velocity_ctrl(0.1, 0)
            time.sleep(1)
            ctrl.base_velocity_ctrl(0, 0)
            print("테스트 완료")
            # 예시: 1~4번 서보 모두 토크 ON
            for servo_id in [1, 2, 3, 4]:
                ctrl.bus_servo_torque_lock(servo_id, 1)
        else:
            print("시리얼 포트를 찾을 수 없습니다. 가상 모드로 실행합니다.")
            ctrl = BaseController("VIRTUAL", 115200)
            ctrl.base_velocity_ctrl(0.1, 0)
    except Exception as e:
        print(f"테스트 오류: {e}")