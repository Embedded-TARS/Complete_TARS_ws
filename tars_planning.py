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
import json

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
        
        # 목적지 관련 변수 추가
        self.current_destination = None
        self.destination_arrived = False
        self.DESTINATION_ARRIVAL_THRESHOLD = 50000  # 5만 픽셀
        
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
        self.AVOIDANCE_DURATION_FRAMES = 105
        self.AVOIDANCE_COOLDOWN_FRAMES = 90
        self.last_avoidance_frame = -1000
        
        print(f"✅ Enhanced Lane Planner initialized with device: {self.device}")
        print(f"📊 Pure Pursuit Parameters: WHEELBASE={self.WHEELBASE}, LOOKAHEAD={self.LOOKAHEAD_DISTANCE}")

    def detect_objects(self, frame):
        det_results = self.sign_model.predict(frame, verbose=False)
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
                    if obj_center_y < sign_roi_y:
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

    def process_traffic_signs(self, detected_objects, lane_center_x=None, image_center_x=None):
        class_ids = [obj["class"] for obj in detected_objects]
        
        # 프레임 카운터 증가
        self.current_frame_count += 1
        
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
            
            elif self.turn_phase == "stop":
                # 정지 상태 유지
                return "stop_after_turn", (0.0, 0.0)
        
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

    def calculate_lane_following(self, lane_center_x, image_center_x, detected_objects=None):
        """
        Pure Pursuit 기반 차선 추종 로직 및 회피 주행 로직
        """
        self.current_frame_count += 1
        # 회피 동작 단계 시간 조정
        AVOIDANCE_PHASE_1_DURATION = self.AVOIDANCE_DURATION_FRAMES * 0.2  # 20% 시간 - 직진
        AVOIDANCE_PHASE_2_DURATION = self.AVOIDANCE_DURATION_FRAMES * 0.15  # 15% 시간 - 회피 시작
        AVOIDANCE_PHASE_3_DURATION = self.AVOIDANCE_DURATION_FRAMES * 0.25  # 25% 시간 - 회피 유지
        AVOIDANCE_PHASE_4_DURATION = self.AVOIDANCE_DURATION_FRAMES * 0.2  # 20% 시간 - 복귀 시작
        AVOIDANCE_PHASE_5_DURATION = self.AVOIDANCE_DURATION_FRAMES * 0.2  # 20% 시간 - 복귀 완료

        # 회피 동작 중이라면 회피 로직 수행
        if self.avoidance_active:
            elapsed = self.current_frame_count - self.avoidance_start_frame

            if elapsed < AVOIDANCE_PHASE_1_DURATION:
                # 첫 번째 단계: 직진으로 거리 확보
                steering = 0.0
                speed = self.STRAIGHT_SPEED * 0.7  # 직진 속도의 70%
            elif elapsed < AVOIDANCE_PHASE_1_DURATION + AVOIDANCE_PHASE_2_DURATION:
                # 두 번째 단계: 부드럽게 회피 시작
                progress = (elapsed - AVOIDANCE_PHASE_1_DURATION) / AVOIDANCE_PHASE_2_DURATION
                steering = -self.MAX_STEER * 0.4 * progress if self.avoidance_direction == 'left' else self.MAX_STEER * 0.4 * progress
                speed = self.MIN_SPEED
            elif elapsed < AVOIDANCE_PHASE_1_DURATION + AVOIDANCE_PHASE_2_DURATION + AVOIDANCE_PHASE_3_DURATION:
                # 세 번째 단계: 회피 유지
                steering = -self.MAX_STEER * 0.4 if self.avoidance_direction == 'left' else self.MAX_STEER * 0.4
                speed = self.MIN_SPEED
            elif elapsed < AVOIDANCE_PHASE_1_DURATION + AVOIDANCE_PHASE_2_DURATION + AVOIDANCE_PHASE_3_DURATION + AVOIDANCE_PHASE_4_DURATION:
                # 네 번째 단계: 부드럽게 복귀 시작
                progress = (elapsed - (AVOIDANCE_PHASE_1_DURATION + AVOIDANCE_PHASE_2_DURATION + AVOIDANCE_PHASE_3_DURATION)) / AVOIDANCE_PHASE_4_DURATION
                steering = -self.MAX_STEER * 0.4 * (1 - progress) if self.avoidance_direction == 'left' else self.MAX_STEER * 0.4 * (1 - progress)
                speed = self.MIN_SPEED
            elif elapsed < AVOIDANCE_PHASE_1_DURATION + AVOIDANCE_PHASE_2_DURATION + AVOIDANCE_PHASE_3_DURATION + AVOIDANCE_PHASE_4_DURATION + AVOIDANCE_PHASE_5_DURATION:
                # 다섯 번째 단계: 복귀 완료 및 안정화
                steering = 0.0
                speed = self.STRAIGHT_SPEED * 0.5
            else:
                # 회피 동작 종료 및 쿨다운 시작
                self.avoidance_active = False
                self.last_avoidance_frame = self.current_frame_count
                print("✅ 회피 동작 완료 및 쿨다운 시작")
                return self.calculate_lane_following(lane_center_x, image_center_x, [])

            print(f"🚧 회피 동작 실행 중: {self.avoidance_direction} (frame {elapsed})")
            return speed, steering, 0.0

        # 장애물 감지 시 회피 조건 체크
        if detected_objects:
            print(f"🔍 감지된 객체 수: {len(detected_objects)}")
            for obj in detected_objects:
                if obj['class'] in self.vehicle_classes:
                    area = obj['area']
                    if (area > self.min_detection_areas.get(obj['class'], 3000) and
                        self.current_frame_count - self.last_avoidance_frame > self.AVOIDANCE_COOLDOWN_FRAMES and
                        not self.avoidance_active):
                        # 회피 동작 시작
                        self.avoidance_active = True
                        self.avoidance_start_frame = self.current_frame_count
                        self.avoidance_direction = 'left' if obj['position'] == 'right' else 'right'
                        print(f"⚠️ 장애물 감지 - 회피 시작 ({self.avoidance_direction})")
                        return self.STRAIGHT_SPEED * 0.7, 0.0, 0.0  # 직진으로 시작

        # 정상 차선 추종 계산
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

    def process_llm_command(self, llm_output):
        """LLM의 JSON 출력을 처리하여 주행 명령을 생성"""
        try:
            # JSON 문자열에서 실제 JSON 부분만 추출
            json_str = llm_output.split("```json")[1].split("```")[0].strip()
            command = json.loads(json_str)
            
            task_type = command.get("task_type", "unknown")
            action = command.get("action", "")
            parameters = command.get("parameters", {})
            
            if task_type == "manual_command":
                if action == "stop":
                    return 0.0, 0.0, 0.0  # 속도 0, 조향 0
                elif action == "go_forward":
                    return self.STRAIGHT_SPEED, 0.0, 0.0  # 직진 속도, 조향 0
                elif action == "go_backward":
                    return -self.STRAIGHT_SPEED, 0.0, 0.0  # 후진 속도, 조향 0
                elif action == "turn_left":
                    return self.MIN_SPEED, -self.MAX_STEER * 0.8, 0.0  # 좌회전
                elif action == "turn_right":
                    return self.MIN_SPEED, self.MAX_STEER * 0.8, 0.0  # 우회전
                elif action == "turn_around":
                    return self.MIN_SPEED, self.MAX_STEER, 0.0  # 180도 회전
            
            elif task_type == "navigate":
                destination = parameters.get("destination")
                if destination:
                    self.current_destination = destination
                    self.destination_arrived = False
                    speed_setting = parameters.get("speed", "normal")
                    if speed_setting == "fast":
                        speed = self.MAX_SPEED
                    elif speed_setting == "slow":
                        speed = self.MIN_SPEED
                    else:  # normal
                        speed = self.STRAIGHT_SPEED
                    
                    # 목적지에 따른 기본 조향 설정
                    return speed, 0.0, 0.0
            
            # 알 수 없는 명령이나 task_type이 unknown인 경우
            return self.MIN_SPEED, 0.0, 0.0
            
        except Exception as e:
            print(f"LLM 명령 처리 중 오류 발생: {e}")
            return self.MIN_SPEED, 0.0, 0.0

    def check_destination_arrival(self, detected_objects):
        """목적지 도착 여부 확인"""
        if not self.current_destination or self.destination_arrived:
            return False

        # 목적지에 해당하는 클래스 매핑
        destination_classes = {
            "home": 5,  # car
            "office": 6,  # bus
            "airport": 6,  # bus
            "school": 7   # motorcycle
        }

        target_class = destination_classes.get(self.current_destination)
        if target_class is None:
            return False

        for obj in detected_objects:
            if obj['class'] == target_class:
                area = obj['area']
                if area >= self.DESTINATION_ARRIVAL_THRESHOLD:
                    print(f"🎯 목적지 도착 감지: {self.current_destination} (면적: {area})")
                    self.destination_arrived = True
                    self.current_destination = None
                    return True
        return False

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
        
        # 목적지 도착 확인 (신호등과 교통표지판 처리 후)
        if self.check_destination_arrival(detected_objects):
            return 0.0, 0.0, 0.0, "destination_arrived", detected_objects
        
        # 기본 차선 추종
        speed, steering, deviation = self.calculate_lane_following(lane_center_x, image_center_x, detected_objects)
        return speed, steering, deviation, "lane_following", detected_objects

    def get_detection_info(self):
        """현재 감지 상태 정보 반환"""
        return {
            "last_seen_sign": self.last_seen_sign,
            "last_action_time": self.last_action_time,
            "current_state": self.current_state
        }

    def plan(self, lane_center_x, image_center_x, frame=None, llm_output=None):
        """메인 계획 함수"""
        # LLM 명령이 있는 경우 우선 처리
        if llm_output:
            return self.process_llm_command(llm_output)
            
        if frame is not None:
            speed, steering, deviation, state, objects = self.plan_with_objects(frame, lane_center_x, image_center_x)
            return speed, steering, deviation
        else:
            return self.calculate_lane_following(lane_center_x, image_center_x)

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
