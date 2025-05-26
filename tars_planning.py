# 02_tars_planning.py

import numpy as np
import time
import torch
from ultralytics import YOLO
from tars_config import (
    MAX_STEER, MAX_SPEED, MIN_SPEED, STRAIGHT_SPEED, TURN_THRESHOLD,
    WHEELBASE, LOOKAHEAD_DISTANCE, MIN_DETECTION_AREAS
)
import cv2

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
        
        # 차선 관련 상수 추가
        self.LANE_WIDTH_PX = 640  # 기본 차선 폭 (픽셀)
        
        # Pure Pursuit 매개변수 조정 - 더 민감한 조향을 위해
        self.WHEELBASE = 0.1  # 기존보다 작게 설정
        self.LOOKAHEAD_DISTANCE = 0.3  # 기존보다 작게 설정
        
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.sign_model = YOLO(model_path).to(self.device)
        self.vehicle_classes = [5, 6, 7]  # car, bus, motorcycle
        self.min_detection_areas = MIN_DETECTION_AREAS
        
        self.last_seen_sign = None
        self.last_action_time = 0
        self.SIGN_COOLDOWN_FRAMES = 15  # 2초 (30fps * 2)
        self.PEDESTRIAN_COOLDOWN_FRAMES = 60
        self.current_frame_count = 0
        
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
        
        # Pure Pursuit 조향 게인 - 더 민감하게 조정
        self.STEERING_GAIN = 1.2  # 1.0에서 1.2로 증가
        self.SPEED_REDUCTION_FACTOR = 0.7  # Factor to reduce speed during turns
        
        self.original_lane_center = None
        self.recovery_start_time = None
        self.RECOVERY_DURATION = 1.0  # 1초 동안 복귀
        self.is_recovering = False

        self.avoidance_active = False
        self.avoidance_direction = None
        self.avoidance_start_frame = 0
        self.AVOIDANCE_DURATION_FRAMES = 90
        self.AVOIDANCE_COOLDOWN_FRAMES = 90
        self.last_avoidance_frame = -1000
        
        print(f"✅ Enhanced Lane Planner initialized with device: {self.device}")
        print(f"📊 Pure Pursuit Parameters: WHEELBASE={self.WHEELBASE}, LOOKAHEAD={self.LOOKAHEAD_DISTANCE}")

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
        
        return detected_objects

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

    def process_traffic_signs(self, detected_objects):
        class_ids = [obj["class"] for obj in detected_objects]
        
        # 프레임 카운터 증가
        self.current_frame_count += 1
        
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
            return "left_turn_sign", (self.MIN_SPEED, self.MAX_STEER * 0.8)
            
        elif 2 in class_ids:
            print("✅ 우회전 표지판 감지")
            self.last_seen_sign = 2
            self.last_action_time = self.current_frame_count
            return "right_turn_sign", (self.MIN_SPEED, -self.MAX_STEER * 0.8)

        elif 3 in class_ids:
            print("✅ 보행자 표지판 감지 - 서행")
            self.last_seen_sign = 3
            self.last_action_time = self.current_frame_count
            return "pedestrian_sign", (self.MAX_SPEED * 0.4, None)
            # 기존 calculate_lane_following 메서드 활용

        
        return None, None

    def calculate_lane_following(self, lane_center_x, image_center_x, detected_objects=None):
        """
        Pure Pursuit 기반 차선 추종 로직 및 회피 주행 로직
        """
        self.current_frame_count += 1

        # 회피 동작 중이라면 회피 로직 수행
        if self.avoidance_active:
            elapsed = self.current_frame_count - self.avoidance_start_frame

            if elapsed < self.AVOIDANCE_DURATION_FRAMES // 5:
                # 첫 번째 단계: 장애물 반대 방향으로 크게 조향
                steering = -self.MAX_STEER * 0.8 if self.avoidance_direction == 'left' else self.MAX_STEER * 0.8
                speed = self.MIN_SPEED
            elif elapsed < self.AVOIDANCE_DURATION_FRAMES * 2 // 5:
                # 두 번째 단계: 적당히 원래 방향으로 복귀
                steering = self.MAX_STEER * 0.8 if self.avoidance_direction == 'left' else -self.MAX_STEER * 0.8
                speed = self.MIN_SPEED
            elif elapsed < self.AVOIDANCE_DURATION_FRAMES * 4 // 5:
                # 세 번째 단계: 직진
                steering = 0.0
                speed = 0.3
            elif elapsed < self.AVOIDANCE_DURATION_FRAMES:
                # 네 번째 단계: 장애물 방향으로 조향
                steering = self.MAX_STEER * 0.8 if self.avoidance_direction == 'left' else -self.MAX_STEER * 0.8
                speed = self.MIN_SPEED
            else:
                # 회피 동작 종료 및 쿨다운 시작
                self.avoidance_active = False
                self.last_avoidance_frame = self.current_frame_count
                print("✅ 회피 동작 완료 및 쿨다운 시작")
                return self.calculate_lane_following(lane_center_x, image_center_x, [])

            print(f"🚧 회피 동작 실행 중: {self.avoidance_direction} (frame {elapsed})")
            return speed, steering, 0.0

        # 차선 중심이 없으면 기본값 반환
        if lane_center_x is None:
            print("⚠️ 차선 중심이 감지되지 않았습니다.")
            return 0.0, 0.0, 0.0

        # 장애물 감지 시 회피 조건 체크
        if detected_objects:
            print(f"🔍 감지된 객체 수: {len(detected_objects)}")
            for obj in detected_objects:
# <<<<<<< HEAD
#                 if obj['class'] in self.vehicle_classes and 'position' in obj:
#                     x1, y1, x2, y2 = obj['bbox']
#                     object_width = min((y2 - y1) / 2, 30)
#                     if obj['position'] == 'right':
#                         # 오른쪽에 물체가 있으면 차선 중심점을 왼쪽으로 조정
#                         lane_center_x = lane_center_x - object_width
#                     elif obj['position'] == 'left':
#                         # 왼쪽에 물체가 있으면 차선 중심점을 오른쪽으로 조정
#                         lane_center_x = lane_center_x + object_width
#
#         # 편차 계산 (정규화)
# =======
                print(f"   - 객체 클래스: {obj['class']}, 면적: {obj['area']}, 위치: {obj.get('position', 'N/A')}")
                if obj['class'] in self.vehicle_classes:
                    area = obj['area']
                    if area > self.min_detection_areas.get(obj['class'], 2000):
                        print(f"   ✅ 객체 면적 조건 충족 (면적: {area})")
                    else:
                        print(f"   ❌ 객체 면적 조건 미충족 (면적: {area})")
                    if self.current_frame_count - self.last_avoidance_frame > self.AVOIDANCE_COOLDOWN_FRAMES:
                        print(f"   ✅ 쿨다운 조건 충족 (쿨다운: {self.current_frame_count - self.last_avoidance_frame} 프레임)")
                    else:
                        print(f"   ❌ 쿨다운 조건 미충족 (쿨다운: {self.current_frame_count - self.last_avoidance_frame} 프레임)")
                    if not self.avoidance_active:
                        print("   ✅ 회피 동작이 비활성화 상태입니다.")
                    else:
                        print("   ❌ 회피 동작이 이미 활성화 상태입니다.")

                    if (area > self.min_detection_areas.get(obj['class'], 3000) and
                        self.current_frame_count - self.last_avoidance_frame > self.AVOIDANCE_COOLDOWN_FRAMES and
                        not self.avoidance_active):
                        # 회피 동작 시작
                        self.avoidance_active = True
                        self.avoidance_start_frame = self.current_frame_count
                        self.avoidance_direction = 'left' if obj['position'] == 'right' else 'right'
                        print(f"⚠️ 장애물 감지 - 회피 시작 ({self.avoidance_direction})")
                        return self.MIN_SPEED, 0.0, 0.0

        # 정상 차선 추종 계산
# >>>>>>> jh
        deviation = (lane_center_x - image_center_x) / image_center_x
        deviation = np.clip(deviation, -1.0, 1.0)

        target_x = self.LOOKAHEAD_DISTANCE * deviation
        steering_angle = np.arctan2(2 * self.WHEELBASE * target_x, self.LOOKAHEAD_DISTANCE**2)
        steering = np.clip(steering_angle * self.STEERING_GAIN, -self.MAX_STEER, self.MAX_STEER)

        # 조향 각도에 따라 속도 조정
        if abs(steering) > self.TURN_THRESHOLD:
            speed = self.MIN_SPEED
        else:
            speed = self.STRAIGHT_SPEED

        print(f"🚗 정상 주행: 속도={speed}, 조향={steering}, 편차={deviation}")
        return speed, steering, deviation

    def plan_with_objects(self, frame, lane_center_x, image_center_x):
        """객체 인식을 포함한 전체 계획 수립"""
        detected_objects = self.detect_objects(frame)
        
        # 신호등 처리
        traffic_result, traffic_control = self.process_traffic_light(detected_objects)
        if traffic_result:
            if traffic_control:
                speed, steering = traffic_control
                if steering is None:
                    # 차선 추종을 위한 조향각 계산
                    _, lane_steering, deviation = self.calculate_lane_following(lane_center_x, image_center_x, detected_objects)
                    steering = lane_steering
                return speed, steering, 0.0, traffic_result, detected_objects
        
        # 교통표지판 처리
        sign_result, sign_control = self.process_traffic_signs(detected_objects)
        if sign_result:
            if sign_control:
                speed, steering = sign_control
                if steering is None:
                    _, lane_steering, deviation = self.calculate_lane_following(lane_center_x, image_center_x, detected_objects)
                    steering = lane_steering
                else:
                    deviation = 0.0
                return speed, steering, deviation, sign_result, detected_objects
        
        # 기본 차선 추종
        speed, steering, deviation = self.calculate_lane_following(lane_center_x, image_center_x, detected_objects)
        return speed, steering, deviation, "lane_following", detected_objects

    def plan(self, lane_center_x, image_center_x, frame=None):
        """메인 계획 함수"""
        if frame is not None:
            speed, steering, deviation, state, objects = self.plan_with_objects(frame, lane_center_x, image_center_x)
            return speed, steering, deviation
        else:
            return self.calculate_lane_following(lane_center_x, image_center_x)

    def get_detection_info(self):
        """현재 감지 상태 정보 반환"""
        return {
            "last_seen_sign": self.last_seen_sign,
            "last_action_time": self.last_action_time,
            "current_state": self.current_state
        }

    def visualize_detections(self, frame, detected_objects):
        # 감지된 객체를 시각화하여 반환
        vis_frame = frame.copy()
        
        for obj in detected_objects:
            x1, y1, x2, y2 = obj['bbox']
            cls_id = obj['class']
            color = CLASS_INFO[cls_id]['color']
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # 차량 클래스인 경우에만 위치 정보 표시
            if cls_id in self.vehicle_classes and 'position' in obj:
                label = f"{CLASS_INFO[cls_id]['name']} ({obj['position']})"
            else:
                label = CLASS_INFO[cls_id]['name']
                
            cv2.putText(vis_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_frame
