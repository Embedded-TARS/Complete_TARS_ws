import numpy as np
import time
import torch
from ultralytics import YOLO
from tars_config import (
    MAX_STEER, MAX_SPEED, MIN_SPEED, STRAIGHT_SPEED, TURN_THRESHOLD,
    WHEELBASE, LOOKAHEAD_DISTANCE
)

# 클래스 정보 매핑
CLASS_INFO = {
    0: {"name": "straight sign", "color": (255, 221, 51)},
    1: {"name": "left sign", "color": (183, 209, 52)},
    2: {"name": "right sign", "color": (51, 255, 221)},
    3: {"name": "pedestrian sign", "color": (83, 179, 36)},
    4: {"name": "stop sign", "color": (245, 61, 184)},
    5: {"name": "car", "color": (250, 183, 50)},
    6: {"name": "bus", "color": (51, 204, 255)},
    7: {"name": "motorcycle", "color": (51, 153, 204)},
    8: {"name": "traffic light", "color": (245, 61, 61)},
    9: {"name": "green light", "color": (30, 230, 64)},
    10: {"name": "yellow light", "color": (55, 250, 250)},
    11: {"name": "red light", "color": (50, 50, 250)},
    12: {"name": "lane", "color": (77, 106, 255)}
}

class EnhancedLanePlanner:
    def __init__(self, model_path="obj.pt"):
        self.MAX_STEER = MAX_STEER
        self.MAX_SPEED = MAX_SPEED
        self.MIN_SPEED = MIN_SPEED
        self.STRAIGHT_SPEED = STRAIGHT_SPEED
        self.TURN_THRESHOLD = TURN_THRESHOLD
        self.WHEELBASE = WHEELBASE
        self.LOOKAHEAD_DISTANCE = LOOKAHEAD_DISTANCE
        
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.sign_model = YOLO(model_path).to(self.device)
        self.vehicle_classes = [5, 6, 7]  # car, bus, motorcycle
        self.min_detection_area = 5000
        
        self.last_seen_sign = None
        self.last_action_time = 0
        self.SIGN_COOLDOWN_SEC = 3
        
        self.stop_sign_detected = False
        self.stop_start_time = None
        self.last_stop_sign_time = None
        self.STOP_DURATION = 2
        self.COOLDOWN_DURATION = 5
        
        self.pedestrian_sign_detected = False
        self.last_pedestrian_sign_time = None
        self.PEDESTRIAN_COOLDOWN_SEC = 3  # 보행자 표지판 쿨다운 시간
        
        self.current_state = "lane_following"
        self.state_start_time = time.time()
        
        print(f"✅ Enhanced Lane Planner initialized with device: {self.device}")

    def detect_objects(self, frame):
        det_results = self.sign_model.predict(frame, verbose=False)
        boxes = det_results[0].boxes
        detected_objects = []
        
        if boxes is not None:
            for i in range(len(boxes)):
                xyxy = boxes[i].xyxy[0].cpu().numpy()
                cls_id = int(boxes[i].cls[0].item())
                conf = float(boxes[i].conf[0].item())
                x1, y1, x2, y2 = map(int, xyxy)
                area = (x2 - x1) * (y2 - y1)
                
                if area >= self.min_detection_area:
                    detected_objects.append({
                        "bbox": [x1, y1, x2, y2],
                        "class": cls_id,
                        "confidence": conf,
                        "area": area
                    })
        
        return detected_objects, boxes

    def process_traffic_light(self, detected_objects, boxes):
        class_ids = [obj["class"] for obj in detected_objects]
        
        if 8 in class_ids:
            has_red = has_yellow = has_green = False
            
            if boxes is not None:
                for i in range(len(boxes)):
                    cls_id = int(boxes[i].cls[0].item())
                    if cls_id == 11:
                        has_red = True
                    elif cls_id == 10:
                        has_yellow = True
                    elif cls_id == 9:
                        has_green = True
            
            if has_red:
                print("🔴 빨간불 감지 - 정지")
                return "red_light", (0.0, 0.0)
            elif has_yellow:
                print("🟡 노란불 감지 - 감속")
                return "yellow_light", (self.MIN_SPEED, None)
            elif has_green:
                print("🟢 초록불 감지 - 통과")
                return "green_light", None
            else:
                print("⚠️ 신호등 감지, 불빛 미확인")
                return "traffic_light_unknown", (self.MIN_SPEED, None)
        
        return None, None

    def process_traffic_signs(self, detected_objects):
        class_ids = [obj["class"] for obj in detected_objects]
        current_time = time.time()
        
        # 보행자 표지판 쿨다운
        if 3 in class_ids and (self.last_pedestrian_sign_time is None or current_time - self.last_pedestrian_sign_time >= self.PEDESTRIAN_COOLDOWN_SEC):
            print("✅ 보행자 표지판 감지 - 서행")
            self.last_pedestrian_sign_time = current_time
            return "pedestrian_sign", (self.MIN_SPEED * 0.05, None)

        # 쿨다운 시간에 따른 신호 처리
        if (self.last_seen_sign is not None and current_time - self.last_action_time < self.SIGN_COOLDOWN_SEC):
            return None, None
        
        if 0 in class_ids:
            print("✅ 직진 표지판 감지")
            self.last_seen_sign = 0
            self.last_action_time = current_time
            return "straight_sign", None
            
        elif 1 in class_ids:
            print("✅ 좌회전 표지판 감지")
            self.last_seen_sign = 1
            self.last_action_time = current_time
            return "left_turn_sign", (self.MIN_SPEED, -self.MAX_STEER * 0.8)
            
        elif 2 in class_ids:
            print("✅ 우회전 표지판 감지")
            self.last_seen_sign = 2
            self.last_action_time = current_time
            return "right_turn_sign", (self.MIN_SPEED, self.MAX_STEER * 0.8)
            
        elif 4 in class_ids:
            if not self.stop_sign_detected:
                print("✅ 정지 표지판 감지 - 완전정지")
                self.stop_sign_detected = True
                self.stop_start_time = current_time
                self.current_state = "stopping"
                return "stop_sign", (0.0, 0.0)
            elif current_time - self.stop_start_time < self.STOP_DURATION:
                return "stop_sign", (0.0, 0.0)
            else:
                if self.last_stop_sign_time is None or current_time - self.last_stop_sign_time > self.COOLDOWN_DURATION:
                    print("✅ 정지 완료 - 출발")
                    self.stop_sign_detected = False
                    self.stop_start_time = None
                    self.current_state = "lane_following"
                    self.last_stop_sign_time = current_time
                    return None, None
                else:
                    return None, None
        
        return None, None

        def process_vehicles(self, detected_objects, image_center_x):
        """차량 회피 로직"""
        vehicle_objects = [obj for obj in detected_objects if obj["class"] in self.vehicle_classes]
        
        if not vehicle_objects:
            return None, None

        # 가장 큰 차량 선택
        largest_vehicle = max(vehicle_objects, key=lambda x: x["area"])
        x1, y1, x2, y2 = largest_vehicle["bbox"]
        vehicle_center_x = (x1 + x2) // 2
        deviation_threshold = image_center_x * 0.3

        # 차량이 센터포인트 기준으로 일정 범위 내에 있는 경우 회피 로직 시작
        if abs(vehicle_center_x - image_center_x) < deviation_threshold:
            if vehicle_center_x > image_center_x:
                print("↪️ 우측 차량 회피 - 좌측으로 크게 이동 후 직진 후 복귀")
                # 회피 로직 실행
                actions = [
                    (self.MIN_SPEED, self.MAX_STEER * 2, 2),  # 크게 왼쪽으로 꺾기 (2초)
                    (self.MIN_SPEED, 0.0, 2),  # 직진 (2초)
                    (self.MIN_SPEED, -self.MAX_STEER * 2, 2),  # 복귀 곡선 (2초)
                    (self.STRAIGHT_SPEED, 0.0, 2)  # 차선 복귀 (2초)
                ]
            else:
                print("↩️ 좌측 차량 회피 - 우측으로 크게 이동 후 직진 후 복귀")
                # 회피 로직 실행
                actions = [
                    (self.MIN_SPEED, -self.MAX_STEER * 2, 2),  # 크게 오른쪽으로 꺾기 (2초)
                    (self.MIN_SPEED, 0.0, 2),  # 직진 (2초)
                    (self.MIN_SPEED, self.MAX_STEER * 2, 2),  # 복귀 곡선 (2초)
                    (self.STRAIGHT_SPEED, 0.0, 2)  # 차선 복귀 (2초)
                ]

            # 쿨다운 타이머 갱신 (회피 로직이 끝난 후 적용)
            self.last_action_time = time.time() + sum(action[2] for action in actions)  # 모든 동작의 지속 시간 합산
            return "avoid_vehicle", actions

        return None, None

    def calculate_lane_following(self, lane_center_x, image_center_x):
        """기본 차선 추종 로직"""
        if lane_center_x is None:
            return 0.0, 0.0, 0.0
        
        deviation = (lane_center_x - image_center_x) / image_center_x
        deviation = np.clip(deviation, -1.0, 1.0)
        
        target_x = self.LOOKAHEAD_DISTANCE * deviation
        steering_angle = np.arctan2(2 * self.WHEELBASE * target_x, 
                                  self.LOOKAHEAD_DISTANCE**2)
        steering = np.clip(steering_angle, -self.MAX_STEER, self.MAX_STEER)
        
        if abs(steering) > 0.08:
            linear_speed = self.MIN_SPEED
        else:
            linear_speed = self.STRAIGHT_SPEED
            
        linear_speed = np.clip(linear_speed, -self.MAX_SPEED, self.MAX_SPEED)
        
        return linear_speed, steering, deviation

    def plan_with_objects(self, frame, lane_center_x, image_center_x):
        """객체 인식을 포함한 전체 계획 수립"""
        detected_objects, boxes = self.detect_objects(frame)
        
        traffic_result, traffic_control = self.process_traffic_light(detected_objects, boxes)
        if traffic_result:
            if traffic_control:
                speed, steering = traffic_control
                if steering is None:
                    _, lane_steering, deviation = self.calculate_lane_following(lane_center_x, image_center_x)
                    steering = lane_steering
                return speed, steering, 0.0, traffic_result, detected_objects
        
        sign_result, sign_control = self.process_traffic_signs(detected_objects)
        if sign_result:
            if sign_control:
                speed, steering = sign_control
                if steering is None:
                    _, lane_steering, deviation = self.calculate_lane_following(lane_center_x, image_center_x)
                    steering = lane_steering
                else:
                    deviation = 0.0
                return speed, steering, deviation, sign_result, detected_objects
        
        vehicle_result, vehicle_control = self.process_vehicles(detected_objects, image_center_x)
        if vehicle_result:
            if vehicle_control:
                # 동작 리스트를 반복적으로 처리
                for speed, steering, duration in vehicle_control:
                    print(f"🚗 차량 회피 동작: 속도={speed}, 조향={steering}, 지속 시간={duration}")
                # 마지막 동작을 반환
                speed, steering, _ = vehicle_control[-1]
                deviation = 0.0
                return speed, steering, deviation, vehicle_result, detected_objects
                
        speed, steering, deviation = self.calculate_lane_following(lane_center_x, image_center_x)
        return speed, steering, deviation, "lane_following", detected_objects

    def plan(self, lane_center_x, image_center_x, frame=None):
        if frame is not None:
            speed, steering, deviation, state, objects = self.plan_with_objects(frame, lane_center_x, image_center_x)
            return speed, steering, deviation
        else:
            return self.calculate_lane_following(lane_center_x, image_center_x)

    def get_detection_info(self):
        return {
            "last_seen_sign": self.last_seen_sign,
            "last_action_time": self.last_action_time,
            "current_state": self.current_state
        }
