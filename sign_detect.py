# 02_tars_planning.py

import numpy as np
import torch
from ultralytics import YOLO
import time

class LanePlanner:
    """차선 중심에 따라 속도와 조향 각을 계획하는 플래너 클래스"""
    
    def __init__(self):
        self.MAX_STEER = 0.5
        self.MAX_SPEED = 0.5
        self.MIN_SPEED = 0.5
        self.STRAIGHT_SPEED = 0.33
        self.TURN_THRESHOLD = 0.08  # 조향 임계값

    def plan(self, lane_center_x: float | None, image_center_x: int) -> tuple[float, float, float]:
        if lane_center_x is None:
            return 0.0, 0.0, 0.0

        deviation = (lane_center_x - image_center_x) / image_center_x
        deviation = np.clip(deviation, -1.0, 1.0)

        steering = self.MAX_STEER * deviation
        steering = np.clip(steering, -self.MAX_STEER, self.MAX_STEER)

        if abs(steering) > self.TURN_THRESHOLD:
            linear_speed = self.MIN_SPEED
        else:
            linear_speed = self.STRAIGHT_SPEED

        linear_speed = np.clip(linear_speed, -self.MAX_SPEED, self.MAX_SPEED)

        return linear_speed, steering, deviation


class SignDetectionModel:
    """표지판 및 신호 감지를 위한 YOLO 모델을 로드하고 관리합니다."""
    
    def __init__(self, model_path="detect.pt", target_classes=None):
        self.target_classes = target_classes or [
            "straight sign", "left sign", "rigth sign", "pedestrian sign",
            "stop sign", "green light", "yellow light", "red light"
        ]
        self.device = 0 if torch.cuda.is_available() else "cpu"
        print(f"YOLO 모델을 {self.device}에 로드하는 중...")
        self.model = YOLO(model_path).to(self.device)
        print(f"✅ YOLO 모델이 {self.device}에 로드되었습니다.")
        self.class_names = self.model.names

    def detect_signs(self, frame, conf_threshold=0.5):
        results = self.model.predict(frame, device=self.device, conf=conf_threshold)
        detections = results[0].boxes.data.cpu().numpy()
        
        signs = []
        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection
            class_id = int(class_id)
            class_name = self.class_names[class_id]
            
            if class_name in self.target_classes:
                signs.append({
                    "class": class_name,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(confidence)
                })
        
        return signs


def decide_action(
    signs: list,
    lane_center_x: float | None,
    image_center_x: int,
    prev_speed: float,
    prev_steering: float
) -> tuple[float, float]:
    
    planner = LanePlanner()

    if not signs:
        return prev_speed, prev_steering

    for sign in signs:
        cls = sign["class"]
        if cls == "stop sign" or cls == "red light":
            time.sleep(1.0)
            return 0.0, 0.0
        elif cls == "left sign":
            time.sleep(1.0)
            return 0.5, 0.5
        elif cls == "right sign":
            time.sleep(1.0)
            return 0.5, -0.5
        elif cls == "pedestrian sign" or cls == "yellow light":
            time.sleep(1.0)
            return 0.1, 0.0
        elif cls == "green light":
            return planner.plan(lane_center_x, image_center_x)[:2]

    return prev_speed, prev_steering
