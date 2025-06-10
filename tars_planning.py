# 02_tars_planning.py

import numpy as np
import time
from ultralytics import YOLO
from tars_config import (
    MAX_STEER, MAX_SPEED, MIN_SPEED, STRAIGHT_SPEED, TURN_THRESHOLD,
    WHEELBASE, LOOKAHEAD_DISTANCE, MIN_DETECTION_AREAS, DESTINATION_ARRIVAL_THRESHOLD, DESTINATION_CLASSES
)
import cv2
import json

# 클래스 정보 매핑
CLASS_INFO = {
    # 0: {"name": "straight sign", "color": (255, 221, 51)},
    # 1: {"name": "left sign", "color": (183, 209, 52)},
    # 2: {"name": "right sign", "color": (51, 255, 221)},
    3: {"name": "pedestrian sign", "color": (83, 179, 36)},
    4: {"name": "stop sign", "color": (245, 61, 184)},
    5: {"name": "car", "color": (250, 183, 50)},
    6: {"name": "bus", "color": (51, 204, 255)},
    7: {"name": "motorcycle", "color": (51, 153, 204)},
    8: {"name": "traffic light", "color": (245, 61, 61)},
    9: {"name": "green light", "color": (30, 230, 64)},
    10: {"name": "yellow light", "color": (55, 250, 250)},
    11: {"name": "red light", "color": (50, 50, 250)},
    12: {"name": "lane", "color": (77, 106, 255)},
    13: {"name": "office", "color": (255, 128, 0)},  # 주황색
    14: {"name": "school", "color": (128, 0, 255)},  # 보라색
    15: {"name": "home", "color": (0, 255, 128)},    # 연두색
    16: {"name": "airport", "color": (255, 0, 128)},  # 분홍색
    17: {"name": "passenger", "color": (255, 255, 255)},
}

class EnhancedLanePlanner:
    def __init__(self, model_path="obj.pt", destination_model_path="destination.pt"):
        self.MAX_STEER = MAX_STEER
        self.MAX_SPEED = MAX_SPEED
        self.MIN_SPEED = MIN_SPEED
        self.STRAIGHT_SPEED = STRAIGHT_SPEED
        self.TURN_THRESHOLD = 0.04  # 0.15에서 0.08로 감소
        
        # 차선 관련 상수 추가
        self.LANE_WIDTH_PX = 640  # 기본 차선 폭 (픽셀)
        
        # Pure Pursuit 매개변수 조정 - 더 민감한 조향을 위해
        self.WHEELBASE = 0.1  # 기존보다 작게 설정
        self.LOOKAHEAD_DISTANCE = 0.3  # 기존보다 작게 설정
        
        self.device = 0
        self.sign_model = YOLO(model_path).to(self.device)
        self.destination_model = YOLO(destination_model_path).to(self.device)
        self.vehicle_classes = [5, 6, 7]  # car, bus, motorcycle
        self.destination_classes = [13, 14, 15, 16]  # office, school, home, airport
        self.min_detection_areas = MIN_DETECTION_AREAS
        
        # 목적지 관련 변수 추가
        self.current_destination = None
        self.destination_arrived = False
        self.DESTINATION_ARRIVAL_THRESHOLD = DESTINATION_ARRIVAL_THRESHOLD
        
        self.last_seen_sign = None
        self.last_action_time = 0
        self.SIGN_COOLDOWN_FRAMES = 15  # 2초 (30fps * 2)
        self.PEDESTRIAN_COOLDOWN_FRAMES = 60
        self.current_frame_count = 0
        
        # 회전 관련 상태 변수 추가
        self.is_turning = False
        self.turn_start_frame = 0
        self.turn_direction = None
        self.LEFT_TURN_DURATION_FRAMES = 15  # 좌회전에 필요한 프레임 수
        self.RIGHT_TURN_DURATION_FRAMES = 10  # 우회전에 필요한 프레임 수
        self.STRAIGHT_AFTER_TURN_FRAMES = 20  # 회전 후 직진 시간
        self.turn_phase = "none"  # none, turning, straight, stop
        
        # 정지 표지판 관련 플래그
        self.stop_sign_detected = False
        self.stop_sign_start_frame = 0
        self.STOP_SIGN_DURATION_FRAMES = 60  # 2초 (30fps * 2)
        self.stop_sign_cooldown_frames = 150  # 5초 (30fps * 5)
        self.stop_sign_last_detected_frame = 0
        
        # 보행자 표지판 관련 플래그
        self.is_pedestrian_sign_active = False
        self.pedestrian_sign_start_time = None
        self.PEDESTRIAN_SIGN_DURATION = 2.0
        
        self.current_state = "lane_following"
        self.state_start_time = time.time()
        
        # 승객 관련 변수 추가
        self.passenger_detected = False
        self.PASSENGER_DETECTION_THRESHOLD = 10000  # 승객 감지 면적 임계값
        
        # Pure Pursuit 조향 게인 - 더 민감하게 조정
        self.STEERING_GAIN = 0.5  # 1.2에서 1.5로 증가
        # self.SPEED_REDUCTION_FACTOR = 0.7  # Factor to reduce speed during turns
        
        self.original_lane_center = None
        self.recovery_start_time = None
        self.RECOVERY_DURATION = 1.0  # 1초 동안 복귀
        self.is_recovering = False

        self.avoidance_active = False
        self.avoidance_direction = None
        self.avoidance_start_frame = 0
        self.AVOIDANCE_DURATION_FRAMES = 105
        self.AVOIDANCE_COOLDOWN_FRAMES = 90
        self.last_avoidance_frame = -1000

        self.passenger_objects = []
        self.destination_objects = []
        self.detected_objects = []
        
        # 주행 관련 변수 추가
        self.trip_start_time = None
        self.trip_status = "idle"  # idle, moving, arrived
        self.base_fare = 3000  # 기본 요금 (원)
        self.per_second_fare = 100  # 초당 요금 (원) - 100원/분을 초당으로 변환
        self.trip_info = {
            "duration": 0,
            "fare": 0,
            "distance": 0  # 미터 단위
        }
        
        print(f"✅ Enhanced Lane Planner initialized with device: {self.device}")
        print(f"📊 Pure Pursuit Parameters: WHEELBASE={self.WHEELBASE}, LOOKAHEAD={self.LOOKAHEAD_DISTANCE}")

    def detect_objects(self, frame):
        det_results = self.sign_model.predict(frame, verbose=False, imgsz=320)
        boxes = det_results[0].boxes
        detected_objects = []
        
        if boxes is not None:
            # 차량용 ROI 설정 - 이미지 하단 60%만 사용
            vehicle_roi_height = int(frame.shape[0] * 0.6)
            vehicle_roi_y = frame.shape[0] - vehicle_roi_height
            
            # 표지판용 ROI 설정 - 이미지 하단 80% 사용
            sign_roi_height = int(frame.shape[0] * 0.9)
            sign_roi_y = frame.shape[0] - sign_roi_height
            
            for i in range(len(boxes)):
                xyxy = boxes[i].xyxy[0].cpu().numpy()
                cls_id = int(boxes[i].cls[0].item())
                conf = float(boxes[i].conf[0].item())
                x1, y1, x2, y2 = map(int, xyxy)
                
                # 객체의 중심점 계산
                obj_center_y = (y1 + y2) / 2
                
                # 차량 클래스인 경우 차량용 ROI 체크
                if cls_id in self.vehicle_classes:
                    if obj_center_y < vehicle_roi_y:
                        continue  # 차량 ROI 밖의 객체는 무시
                
                # 표지판 클래스(0,1,2,3,4)인 경우 표지판용 ROI 체크
                elif cls_id in [0, 1, 2, 3, 4]:
                    continue  # 표지판 ROI 밖의 객체는 무시
                
                area = (x2 - x1) * (y2 - y1)
                
                # 차량 클래스인 경우 위치 정보 추가
                if cls_id in self.vehicle_classes:
                    vehicle_center_x = (x1 + x2) / 2
                    detected_objects.append({
                        'class': cls_id,
                        'confidence': conf,
                        'area': area,
                        'bbox': (x1, y1, x2, y2),
                        'position': 'left' if vehicle_center_x < frame.shape[1] / 2 else 'right'
                    })
                # 신호등 불(9,10,11)만 min_area 체크 제외
                elif cls_id in [9, 10, 11] or area >= self.min_detection_areas.get(cls_id, 3000):
                    detected_objects.append({
                        'class': cls_id,
                        'confidence': conf,
                        'area': area,
                        'bbox': (x1, y1, x2, y2)
                    })
            self.detected_objects = detected_objects
        return detected_objects
    
    def detect_destination_and_passenger(self, frame):
        """📍 목적지와 승객 객체 감지용 모델 실행"""
        det_results = self.destination_model.predict(frame, verbose=False, imgsz=320)
        boxes = det_results[0].boxes
        destination_objects = []
        passenger_objects = []

        if boxes is not None:
            for i in range(len(boxes)):
                xyxy = boxes[i].xyxy[0].cpu().numpy()
                raw_cls_id = int(boxes[i].cls[0].item())
                conf = float(boxes[i].conf[0].item())
                x1, y1, x2, y2 = map(int, xyxy)
                area = (x2 - x1) * (y2 - y1)

                # 목적지/승객 클래스 ID 매핑 (0->13, 1->14, 2->15, 3->16, 4->17)
                cls_id = raw_cls_id + 13
                
                # 디버깅 정보 출력 (30프레임마다)
                print(f"목적지/승객 감지: {CLASS_INFO[cls_id]['name']} (신뢰도: {conf:.2f}, 면적: {area})")

                obj_info = {
                    'class': cls_id,
                    'confidence': conf,
                    'area': area,
                    'bbox': (x1, y1, x2, y2),
                    'raw_class': raw_cls_id
                }

                # 승객인 경우 passenger_objects에 추가
                if cls_id == 17:  # passenger
                    passenger_objects.append(obj_info)
                else:  # 목적지인 경우
                    destination_objects.append(obj_info)

                self.passenger_objects = passenger_objects
                self.destination_objects = destination_objects

        return destination_objects, passenger_objects

    def process_traffic_light(self, detected_objects):
        class_ids = [obj["class"] for obj in detected_objects]
        
        if 8 in class_ids:  # 신호등이 감지된 경우
            has_red = has_yellow = has_green = has_none = False
            if 11 in class_ids:
                has_red = True
            elif 10 in class_ids:
                has_yellow = True
            elif 9 in class_ids:
                has_green = True
            else:
                has_none = True

            if has_red or has_yellow:
                print("🚨 빨간불 또는 노란불 감지 - 정지")
                return "stop", (0.0, 0.0)
            elif has_green:
                print("🟢 초록불 감지 - 통과")
                return "go", (self.STRAIGHT_SPEED, None)  # steering을 None으로 설정하여 차선 추종 사용
            else:
                print("🚦 신호등 감지 없음 - 직진")
                return "no_signal", (self.STRAIGHT_SPEED, None)  # steering을 None으로 설정하여 차선 추종 사용
        
        return None, None

    def process_traffic_signs(self, detected_objects, lane_center_x=None, image_center_x=None):
        class_ids = [obj["class"] for obj in detected_objects]
        
        # 프레임 카운터 증가
        self.current_frame_count += 1
        
        # 승객 감지 처리
        for obj in detected_objects:
            if obj['class'] == 17:  # passenger 클래스
                area = obj['area']
                if area >= self.PASSENGER_DETECTION_THRESHOLD:
                    if not self.passenger_detected:
                        print("👥 승객 감지 - 일시정지")
                        self.passenger_detected = True
                        return "passenger_detected", (0.0, 0.0)
        
        # 회전 동작 중인 경우
        if self.is_turning:
            elapsed = self.current_frame_count - self.turn_start_frame
            
            if self.turn_phase == "turning":
                # 좌/우회전에 따라 다른 회전 시간 적용
                turn_duration = self.LEFT_TURN_DURATION_FRAMES if self.turn_direction == "left" else self.RIGHT_TURN_DURATION_FRAMES
                if elapsed < turn_duration:
                    # 90도 회전 수행
                    steering = -self.MAX_STEER * 0.8 if self.turn_direction == "left" else self.MAX_STEER * 0.8
                    return "turning", (self.MIN_SPEED, steering)
                else:
                    # 회전 완료, 직진 단계로 전환
                    self.turn_phase = "straight"
                    self.turn_start_frame = self.current_frame_count
                    return "straight_after_turn", (self.STRAIGHT_SPEED, 0.0)
            
            elif self.turn_phase == "straight":
                if elapsed < self.STRAIGHT_AFTER_TURN_FRAMES:
                    # 직진 유지
                    return "straight_after_turn", (self.STRAIGHT_SPEED, 0.0)
                else:
                    # 정지 단계로 전환
                    self.turn_phase = "stop"
                    return "stop_after_turn", (0.0, 0.0)
            
            # elif self.turn_phase == "stop":
            #     # 정지 상태 유지
            #     return "stop_after_turn", (0.0, 0.0)
        
        # 정지 표지판 처리
        if 4 in class_ids and not self.stop_sign_detected:
            # 마지막 정지 표지판 감지 후 5초가 지났는지 확인
            if self.current_frame_count - self.stop_sign_last_detected_frame >= self.stop_sign_cooldown_frames:
                print("✅ 정지 표지판 감지 - 완전정지")
                self.stop_sign_detected = True
                self.stop_sign_start_frame = self.current_frame_count
                self.stop_sign_last_detected_frame = self.current_frame_count
                return "stop_sign", (0.0, 0.0)
        
        # 정지 표지판 감지 후 2초 동안 정지
        if self.stop_sign_detected:
            if self.current_frame_count - self.stop_sign_start_frame < self.STOP_SIGN_DURATION_FRAMES:
                return "stop_sign", (0.0, 0.0)
            else:
                self.stop_sign_detected = False
                self.stop_sign_start_frame = 0
        
        # 이전에 감지된 표지판이 있고 쿨다운 프레임이 지나지 않았다면 이전 동작 유지
        if (self.last_seen_sign == 3 and 
            self.current_frame_count - self.last_action_time < self.PEDESTRIAN_COOLDOWN_FRAMES):
            if lane_center_x is not None and image_center_x is not None:
                _, lane_steering, _ = self.calculate_lane_following(lane_center_x, image_center_x, [])
                if abs(lane_steering) > self.TURN_THRESHOLD:
                    return "pedestrian_sign", (self.MAX_SPEED * 0.99, lane_steering)
            return "pedestrian_sign", (self.MAX_SPEED * 0.4, None)
        elif (self.last_seen_sign is not None and self.current_frame_count - self.last_action_time < self.SIGN_COOLDOWN_FRAMES):
            if self.last_seen_sign == 0:
                return "straight_sign", None
            elif self.last_seen_sign == 1:
                return "left_turn_sign", (self.MIN_SPEED, -self.MAX_STEER * 0.8)
            elif self.last_seen_sign == 2:
                return "right_turn_sign", (self.MIN_SPEED, self.MAX_STEER * 0.8)
    
        # 새로운 표지판 감지
        if 0 in class_ids:
            print("✅ 직진 표지판 감지")
            self.last_seen_sign = 0
            self.last_action_time = self.current_frame_count
            return "straight_sign", None
            
        elif 1 in class_ids:
            print("✅ 좌회전 표지판 감지")
            self.last_seen_sign = 1
            self.last_action_time = self.current_frame_count
            # 회전 상태 초기화
            self.is_turning = True
            self.turn_start_frame = self.current_frame_count
            self.turn_direction = "left"
            self.turn_phase = "turning"
            return "left_turn_sign", (self.MIN_SPEED, -self.MAX_STEER * 0.8)
            
        elif 2 in class_ids:
            print("✅ 우회전 표지판 감지")
            self.last_seen_sign = 2
            self.last_action_time = self.current_frame_count
            # 회전 상태 초기화
            self.is_turning = True
            self.turn_start_frame = self.current_frame_count
            self.turn_direction = "right"
            self.turn_phase = "turning"
            return "right_turn_sign", (self.MIN_SPEED, self.MAX_STEER * 0.8)

        elif 3 in class_ids:
            print("✅ 보행자 표지판 감지 - 서행")
            self.last_seen_sign = 3
            self.last_action_time = self.current_frame_count
            if lane_center_x is not None and image_center_x is not None:
                # 차선 추종을 위한 조향각 계산
                _, lane_steering, _ = self.calculate_lane_following(lane_center_x, image_center_x, [])
                # 조향각이 클 때는 더 많은 파워 제공
                if abs(lane_steering) > self.TURN_THRESHOLD:
                    return "pedestrian_sign", (self.MAX_SPEED * 0.99, lane_steering)
            return "pedestrian_sign", (self.MAX_SPEED * 0.4, None)
        
        return None, None

    def calculate_pure_pursuit(self, target_x, lookahead_distance=None): 
        """
        Pure Pursuit 알고리즘을 사용하여 조향각을 계산하는 함수
        
        Args:
            target_x (float): 목표점의 x 좌표 (차량 기준 좌표계)
            lookahead_distance (float, optional): 전방 주시 거리. None인 경우 기본값 사용
            
        Returns:
            float: 계산된 조향각
        """
        if lookahead_distance is None:
            lookahead_distance = self.LOOKAHEAD_DISTANCE
            
        steering_angle = np.arctan2(2 * self.WHEELBASE * target_x, lookahead_distance**2)
        return np.clip(steering_angle * self.STEERING_GAIN, -self.MAX_STEER, self.MAX_STEER)

    def calculate_lane_following(self, lane_center_x, image_center_x, detected_objects=None):
        # 장애물 감지 시 Pure Pursuit 기반 회피
        if detected_objects:
            for obj in detected_objects:
                if obj['class'] in self.vehicle_classes:
                    area = obj['area']
                    if area > self.min_detection_areas.get(obj['class'], 3000):
                        # 장애물이 왼쪽에 있으면 오른쪽으로, 오른쪽에 있으면 왼쪽으로 회피
                        if 'position' in obj:
                            if obj['position'] == 'left':
                                # 오른쪽으로 회피하면서 직진 구간 추가
                                target_x = self.LOOKAHEAD_DISTANCE * 0.3  # 직진 구간
                                steering = self.calculate_pure_pursuit(target_x)
                                speed = self.MIN_SPEED
                                deviation = target_x / self.LOOKAHEAD_DISTANCE
                                print(f"🚧 Pure Pursuit 회피 동작: 장애물 {obj['position']} (직진 후 회피)")
                            else:
                                # 왼쪽으로 회피하면서 직진 구간 추가
                                target_x = -self.LOOKAHEAD_DISTANCE * 0.3  # 직진 구간
                                steering = self.calculate_pure_pursuit(target_x)
                                speed = self.MIN_SPEED
                                deviation = target_x / self.LOOKAHEAD_DISTANCE
                                print(f"🚧 Pure Pursuit 회피 동작: 장애물 {obj['position']} (직진 후 회피)")
                            return speed, steering, deviation

        # 정상 차선 추종 계산
        deviation = (lane_center_x - image_center_x) / image_center_x
        deviation = np.clip(deviation, -1.0, 1.0)

        target_x = self.LOOKAHEAD_DISTANCE * deviation
        steering = self.calculate_pure_pursuit(target_x)

        # 조향 각도에 따라 속도 조정
        if abs(steering) > self.TURN_THRESHOLD:
            speed = self.MIN_SPEED
        else:
            speed = self.STRAIGHT_SPEED

        print(f"🚗 정상 주행: 속도={speed}, 조향={steering}, 편차={deviation}")
        return speed, steering, deviation

    def process_llm_command(self, llm_output, lane_center_x, image_center_x):
        """로컬 JSON 파일의 명령을 처리하여 주행 명령을 생성"""
        try:
            # 이미 JSON 객체인 경우 바로 사용
            command = llm_output if isinstance(llm_output, dict) else json.loads(llm_output)
            
            task_type = command.get("task_type", "unknown")
            action = command.get("action", "")
            parameters = command.get("parameters", {})
            
            # 속도 제어 파라미터 처리
            speed_control = parameters.get("speed_control", "normal")
            base_speed = self.STRAIGHT_SPEED  # 기본 속도
            
            # 속도 제어에 따른 속도 조정
            if speed_control == "slow":
                base_speed = 0.2
            elif speed_control == "fast":
                base_speed = self.MAX_SPEED
            elif speed_control == "normal":
                base_speed = self.STRAIGHT_SPEED
            
            if task_type == "manual_command":
                if action == "stop":
                    return 0.0, 0.0, 0.0, "stop", []  # 속도 0, 조향 0
                elif action == "go_forward":
                    # 차선 추종을 사용한 직진
                    if lane_center_x is None:
                        return base_speed, 0.0, 0.0, "no_lane", []
                    speed, steering, deviation = self.calculate_lane_following(lane_center_x, image_center_x, [])
                    return speed, steering, deviation, action, []
                elif action == "go_backward":
                    return -base_speed, 0.0, 0.0, action, []  # 후진 속도, 조향 0
                elif action == "turn_left":
                    # Pure Pursuit을 사용한 좌회전
                    target_x = -self.LOOKAHEAD_DISTANCE * 0.5  # 왼쪽으로 회전
                    steering = self.calculate_pure_pursuit(target_x)
                    return base_speed * 0.5, steering, -0.5, action, []  # 회전 시 속도 감소
                elif action == "turn_right":
                    # Pure Pursuit을 사용한 우회전
                    target_x = self.LOOKAHEAD_DISTANCE * 0.5  # 오른쪽으로 회전
                    steering = self.calculate_pure_pursuit(target_x)
                    return base_speed * 0.5, steering, 0.5, action, []  # 회전 시 속도 감소
                elif action == "turn_around":
                    # Pure Pursuit을 사용한 180도 회전
                    target_x = self.LOOKAHEAD_DISTANCE  # 완전한 회전
                    steering = self.calculate_pure_pursuit(target_x)
                    return base_speed * 0.3, steering, 1.0, action, []  # 180도 회전 시 더 느리게
            
            elif task_type == "navigate":
                destination = parameters.get("destination")
                if destination:
                    print(f"🎯 새로운 목적지 설정: {destination}")
                    self.current_destination = destination
                    self.destination_arrived = False
                    self.trip_status = "moving"
                    self.trip_start_time = time.time()
                    # 목적지 설정만 하고, 주행은 차선 추종만 하다가 목적지가 보이면 정지
                    if lane_center_x is None:
                        return base_speed, 0.0, 0.0, "no_lane", []
                    speed, steering, deviation = self.calculate_lane_following(lane_center_x, image_center_x, [])
                    return speed, steering, deviation, "navigating", []
            
            # 알 수 없는 명령이나 task_type이 unknown인 경우
            # 기본 차선 추종으로 대체
            if lane_center_x is None:
                return base_speed, 0.0, 0.0, "no_lane", []
            speed, steering, deviation = self.calculate_lane_following(lane_center_x, image_center_x, [])
            return speed, steering, deviation, "lane_following", []
            
        except Exception as e:
            print(f"명령 처리 중 오류 발생: {e}")
            # 오류 발생 시 기본 차선 추종으로 대체
            if lane_center_x is None:
                return self.STRAIGHT_SPEED, 0.0, 0.0, "no_lane", []
            speed, steering, deviation = self.calculate_lane_following(lane_center_x, image_center_x, [])
            return speed, steering, deviation, "lane_following", []

    def check_destination_arrival(self, destination_objects):
        """목적지 도착 여부 확인"""
        if not self.current_destination or self.destination_arrived:
            return False

        # 목적지 클래스 매핑
        destination_mapping = {
            "office": 13,
            "school": 14,
            "home": 15,
            "airport": 16
        }

        target_class = destination_mapping.get(self.current_destination)
        if target_class is None:
            print(f"⚠️ 알 수 없는 목적지: {self.current_destination}")
            return False

        for obj in destination_objects:
            if obj['class'] == target_class:
                area = obj['area']
                conf = obj['confidence']
                
                print(f"목적지 감지 중: {self.current_destination} (면적: {area}, 신뢰도: {conf:.2f})")
                
                if area >= self.DESTINATION_ARRIVAL_THRESHOLD and conf > 0.5:
                    print(f"🎯 목적지 도착 감지: {self.current_destination}")
                    self.destination_arrived = True
                    
                    # 주행 정보 계산
                    if self.trip_start_time is not None:
                        trip_duration = time.time() - self.trip_start_time
                        fare = self.base_fare + (trip_duration * self.per_second_fare)  # 초당 요금 적용
                        
                        self.trip_info = {
                            "duration": round(trip_duration, 2),
                            "fare": round(fare, 2),
                            "distance": round(trip_duration * self.STRAIGHT_SPEED, 2)  # 대략적인 거리 계산
                        }
                        
                        print(f"📊 주행 정보:")
                        print(f"   - 주행 시간: {self.trip_info['duration']}초")
                        print(f"   - 요금: {self.trip_info['fare']}원")
                        print(f"   - 주행 거리: {self.trip_info['distance']}m")
                    
                    self.current_destination = None
                    self.trip_start_time = None
                    self.trip_status = "idle"
                    return True
        return False

    def plan_with_objects(self, frame, lane_center_x, image_center_x, current_command=None):
        """객체 인식을 포함한 전체 계획 수립 + LLM 주행 중에도 감지 상황 반영"""
        # 프레임 디버깅 정보 출력
        # print(f"\n📊 프레임 정보:")
        # print(f"   - 프레임 크기: {frame.shape}")
        # # print(f"   - 차선 중심점: {lane_center_x}")
        # # print(f"   - 이미지 중심점: {image_center_x}")
        # print(f"   - 현재 프레임 카운트: {self.current_frame_count}")
        
        
        detected_objects = self.detected_objects
        destination_objects, passenger_objects = self.destination_objects, self.passenger_objects

        # ✅ 목적지 도착 여부 확인
        if self.current_destination and not self.destination_arrived:
            if self.check_destination_arrival(destination_objects):
                print("🎯 목적지 도착! 정지합니다.")
                # 상태 초기화
                self.last_seen_sign = None
                self.last_action_time = 0
                self.is_turning = False
                self.turn_phase = "none"
                self.stop_sign_detected = False
                self.is_pedestrian_sign_active = False
                self.current_state = "destination_arrived"
                return 0.0, 0.0, 0.0, "destination_arrived", detected_objects

        # 승객 감지 처리
        for passenger in passenger_objects:
            if passenger['area'] >= self.PASSENGER_DETECTION_THRESHOLD:
                if not self.passenger_detected:
                    print("👥 승객 감지 - 일시정지")
                    self.passenger_detected = True
                    return 0.0, 0.0, 0.0, "passenger_detected", detected_objects

        # 기본 차선 추종 계산 - 한 번만 계산
        if lane_center_x is not None:
            base_speed, base_steering, base_deviation = self.calculate_lane_following(lane_center_x, image_center_x, detected_objects)
        else:
            base_speed, base_steering, base_deviation = self.STRAIGHT_SPEED, 0.0, 0.0

        # ✅ LLM 명령 우선 처리
        if current_command:
            # process_llm_command를 통해 목적지 설정 및 기본 주행 명령 처리
            speed, steering, deviation, action, _ = self.process_llm_command(current_command, lane_center_x, image_center_x)
            
            # 🚦 신호등 감지 우선
            traffic_result, traffic_control = self.process_traffic_light(detected_objects)
            if traffic_result == "stop":
                return 0.0, 0.0, 0.0, "red_light_stop", detected_objects
            elif traffic_result == "go":
                pass  # 초록불 → continue

            # 🚧 장애물 감지 및 회피
            for obj in detected_objects:
                if obj['class'] in self.vehicle_classes and obj['area'] > self.min_detection_areas.get(obj['class'], 3000):
                    print("⚠️ LLM 주행 중 장애물 감지 - 회피 수행")
                    speed, steering, deviation = self.calculate_lane_following(lane_center_x, image_center_x, detected_objects)
                    return speed, steering, deviation, "avoidance", detected_objects

            # 🛑 표지판 감지
            sign_result, sign_control = self.process_traffic_signs(detected_objects, lane_center_x, image_center_x)
            if sign_result and sign_control:
                s, st = sign_control
                return s, st if st is not None else steering, deviation, sign_result, detected_objects

            # 기본적으로 LLM 계획대로 실행
            return speed, steering, deviation, action, detected_objects

        # ✅ 일반 주행 모드
        # 1. 신호등 처리
        traffic_result, traffic_control = self.process_traffic_light(detected_objects)
        if traffic_result:
            if traffic_control:
                speed, steering = traffic_control
                if steering is None:
                    _, lane_steering, deviation = self.calculate_lane_following(lane_center_x, image_center_x, detected_objects)
                    steering = lane_steering
                return speed, steering, 0.0, traffic_result, detected_objects

        # 2. 표지판 처리
        sign_result, sign_control = self.process_traffic_signs(detected_objects, lane_center_x, image_center_x)
        if sign_result:
            if sign_control:
                speed, steering = sign_control
                if steering is None:
                    _, lane_steering, deviation = self.calculate_lane_following(lane_center_x, image_center_x, detected_objects)
                    steering = lane_steering
                else:
                    deviation = 0.0
                return speed, steering, deviation, sign_result, detected_objects

        # 3. 차선 추종 (기본)
        speed, steering, deviation = self.calculate_lane_following(lane_center_x, image_center_x, detected_objects)
        return speed, steering, deviation, "lane_following", detected_objects


    def get_detection_info(self):
        """현재 감지 상태 정보 반환"""
        return {
            "last_seen_sign": self.last_seen_sign,
            "last_action_time": self.last_action_time,
            "current_state": self.current_state
        }

    def get_trip_info(self):
        """목적지 도착 후의 주행 정보 반환"""
        if self.trip_status == "idle" and self.trip_info["duration"] > 0:
            return self.trip_info
        return None

    def plan(self, lane_center_x, image_center_x, frame=None, llm_output=None):
        """메인 계획 함수"""
        # LLM 명령이 있는 경우 우선 처리
        if llm_output:
            return self.process_llm_command(llm_output, lane_center_x, image_center_x)
            
        if frame is not None:
            speed, steering, deviation, state, objects = self.plan_with_objects(frame, lane_center_x, image_center_x)
            return speed, steering, deviation
        else:
            return self.calculate_lane_following(lane_center_x, image_center_x)

    def visualize_detections(self, frame, detected_objects):
        """감지된 객체를 시각화하여 반환"""
        vis_frame = frame.copy()
        
        for obj in detected_objects:
            x1, y1, x2, y2 = obj['bbox']
            cls_id = obj['class']
            
            # 클래스 정보 가져오기
            if cls_id in CLASS_INFO:
                class_name = CLASS_INFO[cls_id]['name']
                color = CLASS_INFO[cls_id]['color']
            else:
                class_name = f"Unknown_{cls_id}"
                color = (128, 128, 128)  # 회색
            
            # 바운딩 박스 그리기
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # 라벨 텍스트 준비
            conf = obj['confidence']
            label = f"{class_name}: {conf:.2f}"
            
            # 텍스트 크기 계산
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # 라벨 배경 그리기
            cv2.rectangle(vis_frame, 
                         (x1, y1 - text_height - baseline - 5), 
                         (x1 + text_width, y1), 
                         color, -1)
            
            # 라벨 텍스트 그리기
            cv2.putText(vis_frame, label, 
                       (x1, y1 - baseline - 2), 
                       font, font_scale, (255, 255, 255), thickness)
            
            # 목적지인 경우 특별한 표시 추가
            if cls_id in [13, 14, 15, 16]:  # 목적지 클래스
                # 중앙점 표시
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(vis_frame, (center_x, center_y), 5, color, -1)
                
                # 목적지 도착 임계값 표시
                if obj['area'] >= self.DESTINATION_ARRIVAL_THRESHOLD:
                    cv2.putText(vis_frame, "ARRIVAL", 
                               (x1, y2 + 20), 
                               font, 0.7, (0, 255, 0), 2)
        
        return vis_frame
