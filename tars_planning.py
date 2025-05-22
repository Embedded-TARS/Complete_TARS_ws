# 02_tars_planning.py

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
        # 기존 파라미터
        self.MAX_STEER = MAX_STEER
        self.MAX_SPEED = MAX_SPEED
        self.MIN_SPEED = MIN_SPEED
        self.STRAIGHT_SPEED = STRAIGHT_SPEED
        self.TURN_THRESHOLD = TURN_THRESHOLD
        self.WHEELBASE = WHEELBASE
        self.LOOKAHEAD_DISTANCE = LOOKAHEAD_DISTANCE
        
        # 객체 검출 관련 파라미터
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.sign_model = YOLO(model_path).to(self.device)
        self.vehicle_classes = [5, 6, 7]  # car, bus, motorcycle
        self.min_detection_area = 5000
        
        # 상태 추적 변수
        self.last_seen_sign = None
        self.last_action_time = 0
        self.SIGN_COOLDOWN_SEC = 3
        
        # 정지 표지판 관련 변수
        self.stop_sign_detected = False
        self.stop_start_time = None
        self.last_stop_sign_time = None
        self.STOP_DURATION = 2  # 정지 시간 (초)
        self.COOLDOWN_DURATION = 5  # 정지 표지판 쿨다운 시간 (초)
        
        # 주행 상태
        self.current_state = "lane_following"  # lane_following, stopping, turning, avoiding
        self.state_start_time = time.time()
        
        print(f"✅ Enhanced Lane Planner initialized with device: {self.device}")

    def detect_objects(self, frame):
        """프레임에서 객체 검출 수행"""
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
        """신호등 처리 로직"""
        class_ids = [obj["class"] for obj in detected_objects]
        
        if 8 in class_ids:  # 신호등 감지
            has_red = has_yellow = has_green = False
            
            if boxes is not None:
                for i in range(len(boxes)):
                    cls_id = int(boxes[i].cls[0].item())
                    if cls_id == 11:    # 빨간불
                        has_red = True
                    elif cls_id == 10:  # 노란불
                        has_yellow = True
                    elif cls_id == 9:   # 초록불
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
        """교통 표지판 처리 로직"""
        class_ids = [obj["class"] for obj in detected_objects]
        current_time = time.time()
        
        # 쿨다운 체크
        if (self.last_seen_sign is not None and 
            current_time - self.last_action_time < self.SIGN_COOLDOWN_SEC):
            return None, None
        
        # 각 표지판별 처리
        if 0 in class_ids:  # 직진
            print("✅ 직진 표지판 감지")
            self.last_seen_sign = 0
            self.last_action_time = current_time
            return "straight_sign", None
            
        elif 1 in class_ids:  # 좌회전
            print("✅ 좌회전 표지판 감지")
            self.last_seen_sign = 1
            self.last_action_time = current_time
            return "left_turn_sign", (self.MIN_SPEED, self.MAX_STEER * 0.8)
            
        elif 2 in class_ids:  # 우회전
            print("✅ 우회전 표지판 감지")
            self.last_seen_sign = 2
            self.last_action_time = current_time
            return "right_turn_sign", (self.MIN_SPEED, -self.MAX_STEER * 0.8)
            
        elif 3 in class_ids:  # 보행자
            print("✅ 보행자 표지판 감지 - 서행")
            self.last_seen_sign = 3
            self.last_action_time = current_time
            return "pedestrian_sign", (self.MIN_SPEED * 0.5, None)
            
        elif 4 in class_ids:  # 정지 표지판
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
            
        largest_vehicle = max(vehicle_objects, key=lambda x: x["area"])
        x1, y1, x2, y2 = largest_vehicle["bbox"]
        vehicle_center_x = (x1 + x2) // 2
        deviation_threshold = image_center_x * 0.3
        
        if abs(vehicle_center_x - image_center_x) < deviation_threshold:
            if vehicle_center_x > image_center_x:
                print("↩️ 우측 차량 회피 - 좌측으로 이동")
                return "avoid_right_vehicle", (self.MIN_SPEED, self.MAX_STEER * 0.6)
            else:
                print("↪️ 좌측 차량 회피 - 우측으로 이동")
                return "avoid_left_vehicle", (self.MIN_SPEED, -self.MAX_STEER * 0.6)
        
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
                speed, steering = vehicle_control
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