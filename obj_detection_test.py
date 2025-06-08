import torch
from ultralytics import YOLO
import time
import cv2
import numpy as np
from jetcam.csi_camera import CSICamera
from tars_config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, MIN_DETECTION_AREAS

# 클래스 정보 매핑
CLASS_INFO = {
    0: {"name": "straight sign", "color": (255, 221, 51)},      # #33ddff -> BGR
    1: {"name": "left sign", "color": (183, 209, 52)},         # #34d1b7 -> BGR
    2: {"name": "right sign", "color": (51, 255, 221)},        # #ddff33 -> BGR
    3: {"name": "pedestrian sign", "color": (83, 179, 36)},    # #24b353 -> BGR
    4: {"name": "stop sign", "color": (245, 61, 184)},         # #b83df5 -> BGR
    5: {"name": "car", "color": (250, 183, 50)},               # #32b7fa -> BGR
    6: {"name": "bus", "color": (51, 204, 255)},               # #ffcc33 -> BGR
    7: {"name": "motorcycle", "color": (51, 153, 204)},        # #cc9933 -> BGR
    8: {"name": "traffic light", "color": (245, 61, 61)},      # #3d3df5 -> BGR
    9: {"name": "green light", "color": (30, 230, 64)},        # #40e61e -> BGR
    10: {"name": "yellow light", "color": (55, 250, 250)},     # #fafa37 -> BGR
    11: {"name": "red light", "color": (50, 50, 250)},         # #fa3253 -> BGR
    12: {"name": "lane", "color": (77, 106, 255)},              # #ff6a4d -> BGR
    13: {"name": "office", "color": (255, 128, 0)},  # 주황색
    14: {"name": "school", "color": (128, 0, 255)},  # 보라색
    15: {"name": "home", "color": (0, 255, 128)},    # 연두색
    16: {"name": "airport", "color": (255, 0, 128)}  # 분홍색
}

def hex_to_bgr(hex_color):
    """HEX 색상을 BGR로 변환"""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])  # RGB to BGR

def setup_object_detection():
    """객체 검출 모델 초기화"""
    device = 0 if torch.cuda.is_available() else "cpu"
    sign_model = YOLO("obj.pt").to(device)
    dest_model = YOLO("destination.pt").to(device)
    return sign_model, dest_model

def detect_objects(frame, sign_model, dest_model, min_area=1000):
    """프레임에서 객체 검출 수행"""
    det_results = sign_model.predict(frame, verbose=False)
    dest_results = dest_model.predict(frame, verbose=False)
    boxes = det_results[0].boxes
    dest_boxes = dest_results[0].boxes
    detected_objects = []
    
    if boxes is not None:
        for i in range(len(boxes)):
            xyxy = boxes[i].xyxy[0].cpu().numpy()
            cls_id = int(boxes[i].cls[0].item())
            conf = float(boxes[i].conf[0].item())
            x1, y1, x2, y2 = map(int, xyxy)
            area = (x2 - x1) * (y2 - y1)
            
            # 신호등 불(9,10,11)만 min_area 체크 제외
            if cls_id in [9, 10, 11] or area >= MIN_DETECTION_AREAS.get(cls_id, 3000):
                detected_objects.append({
                    "bbox": [x1, y1, x2, y2],
                    "class": cls_id,
                    "confidence": conf,
                    "area": area
                })
    
    # 목적지 객체 감지 결과 추가
    if dest_boxes is not None:
        for i in range(len(dest_boxes)):
            xyxy = dest_boxes[i].xyxy[0].cpu().numpy()
            raw_cls_id = int(dest_boxes[i].cls[0].item())
            # 목적지 클래스 ID 매핑 (0->13, 1->14, 2->15, 3->16)
            cls_id = raw_cls_id + 13
            conf = float(dest_boxes[i].conf[0].item())
            x1, y1, x2, y2 = map(int, xyxy)
            area = (x2 - x1) * (y2 - y1)
            
            if area >= MIN_DETECTION_AREAS.get(cls_id, 3000):
                detected_objects.append({
                    "bbox": [x1, y1, x2, y2],
                    "class": cls_id,
                    "confidence": conf,
                    "area": area,
                    "raw_class": raw_cls_id  # 원본 클래스 ID도 저장
                })
                print(f"목적지 감지: {CLASS_INFO[cls_id]['name']} (원본 ID: {raw_cls_id} -> 매핑 ID: {cls_id})")
    
    return detected_objects

def draw_detections(frame, detected_objects):
    """검출된 객체를 프레임에 그리기"""
    annotated_frame = frame.copy()
    
    for obj in detected_objects:
        cls_id = obj["class"]
        conf = obj["confidence"]
        x1, y1, x2, y2 = obj["bbox"]
        
        # 클래스 정보 가져오기
        if cls_id in CLASS_INFO:
            class_name = CLASS_INFO[cls_id]["name"]
            color = CLASS_INFO[cls_id]["color"]
        else:
            class_name = f"Unknown_{cls_id}"
            color = (128, 128, 128)  # 회색
        
        # 바운딩 박스 그리기
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # 신호등 관련 객체는 더 두껍게 표시
        if cls_id in [8, 9, 10, 11]:  # 신호등과 불
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
        
        # 픽셀 크기 계산
        width = x2 - x1
        height = y2 - y1
        
        # 라벨 텍스트 준비
        label = f"{class_name}: {conf:.2f} ({width}x{height})"
        
        # 텍스트 크기 계산
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # 라벨 배경 그리기
        cv2.rectangle(annotated_frame, 
                     (x1, y1 - text_height - baseline - 5), 
                     (x1 + text_width, y1), 
                     color, -1)
        
        # 라벨 텍스트 그리기
        cv2.putText(annotated_frame, label, 
                   (x1, y1 - baseline - 2), 
                   font, font_scale, (255, 255, 255), thickness)
        
        # 중앙점 표시
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(annotated_frame, (center_x, center_y), 3, color, -1)
    
    return annotated_frame

def process_traffic_light(detected_objects, boxes):
    """신호등 처리 로직"""
    class_ids = [obj["class"] for obj in detected_objects]
    
    if 8 in class_ids:  # 신호등 감지
        print("\n=== 신호등 감지 디버깅 ===")
        print(f"전체 감지된 객체 수: {len(boxes)}")
        
        has_red = has_yellow = has_green = False
        red_conf = yellow_conf = green_conf = 0.0
        
        # 모든 감지된 객체 정보 출력
        print("\n감지된 모든 객체:")
        for i in range(len(boxes)):
            cls_id = int(boxes[i].cls[0].item())
            conf = float(boxes[i].conf[0].item())
            xyxy = boxes[i].xyxy[0].cpu().numpy()
            print(f"객체 {i}: 클래스={cls_id}, 신뢰도={conf:.2f}, 위치={xyxy}")
            
            if cls_id == 11:    # 빨간불
                has_red = True
                red_conf = conf
                print(f"🔴 빨간불 감지 - 신뢰도: {conf:.2f}")
            elif cls_id == 10:  # 노란불
                has_yellow = True
                yellow_conf = conf
                print(f"🟡 노란불 감지 - 신뢰도: {conf:.2f}")
            elif cls_id == 9:   # 초록불
                has_green = True
                green_conf = conf
                print(f"🟢 초록불 감지 - 신뢰도: {conf:.2f}")
        
        # 신호등 상태 요약
        print("\n=== 신호등 상태 요약 ===")
        print(f"빨간불: {'감지됨' if has_red else '미감지'} (신뢰도: {red_conf:.2f})")
        print(f"노란불: {'감지됨' if has_yellow else '미감지'} (신뢰도: {yellow_conf:.2f})")
        print(f"초록불: {'감지됨' if has_green else '미감지'} (신뢰도: {green_conf:.2f})")
        print("========================\n")
        
        if has_red:
            return "red", (0.0, 0.0)  # 정지
        elif has_yellow:
            return "yellow", None  # 기본 주행 계속
        elif has_green:
            return "green", None   # 기본 주행 계속
        else:
            print("⚠️ 신호등은 있지만 불빛 없음")
            return "traffic_light_no_signal", None
    
    return None, None

def process_traffic_signs(detected_objects, last_seen_sign, last_action_time, SIGN_COOLDOWN_SEC=3):
    """교통 표지판 처리 로직"""
    class_ids = [obj["class"] for obj in detected_objects]
    current_time = time.time()
    
    # 각 표지판별 처리 (쿨다운 적용)
    if 0 in class_ids and (last_seen_sign != 0 or current_time - last_action_time > SIGN_COOLDOWN_SEC):
        print("✅ 직진 표지판 감지됨")
        return "straight", None, 0, current_time
    
    elif 1 in class_ids and (last_seen_sign != 1 or current_time - last_action_time > SIGN_COOLDOWN_SEC):
        print("✅ 좌회전 표지판 감지됨")
        return "left_turn", (0.5, 0.5), 1, current_time
    
    elif 2 in class_ids and (last_seen_sign != 2 or current_time - last_action_time > SIGN_COOLDOWN_SEC):
        print("✅ 우회전 표지판 감지됨")
        return "right_turn", (0.5, -0.5), 2, current_time
    
    elif 3 in class_ids and (last_seen_sign != 3 or current_time - last_action_time > SIGN_COOLDOWN_SEC):
        print("✅ 보행자 표지판 감지됨")
        return "pedestrian", (0.1, 0.0), 3, current_time
    
    elif 4 in class_ids and (last_seen_sign != 4 or current_time - last_action_time > SIGN_COOLDOWN_SEC):
        print("✅ 정지 표지판 감지됨 → 2초간 정지")
        return "stop", (0.0, 0.0), 4, current_time
    
    return None, None, last_seen_sign, last_action_time

def process_vehicles(detected_objects, image_center_x, vehicle_classes=[5, 6, 7]):
    """차량 회피 로직"""
    vehicle_objects = [obj for obj in detected_objects if obj["class"] in vehicle_classes]
    
    if vehicle_objects:
        vehicle = vehicle_objects[0]  # 첫 번째 차량 선택
        x1, y1, x2, y2 = vehicle["bbox"]
        center_x = (x1 + x2) // 2
        
        if center_x > image_center_x:
            print("↩️ 우측에 vehicle → 좌회전 곡선 회피")
            return "avoid_right_vehicle", (0.3, 0.5)
        else:
            print("↪️ 좌측에 vehicle → 우회전 곡선 회피")
            return "avoid_left_vehicle", (0.3, -0.5)
    
    return None, None

def object_detection_pipeline(frame, sign_model, dest_model, image_center_x, last_seen_sign, last_action_time):
    """전체 객체 검출 파이프라인"""
    # 1. 객체 검출
    detected_objects = detect_objects(frame, sign_model, dest_model, min_area=1000)
    boxes = sign_model.predict(frame, verbose=False)[0].boxes
    
    # 2. 신호등 처리
    traffic_light_result, traffic_light_control = process_traffic_light(detected_objects, boxes)
    if traffic_light_result:
        return traffic_light_result, traffic_light_control, last_seen_sign, last_action_time
    
    # 3. 교통 표지판 처리
    sign_result, sign_control, new_last_seen_sign, new_last_action_time = process_traffic_signs(
        detected_objects, last_seen_sign, last_action_time
    )
    if sign_result:
        return sign_result, sign_control, new_last_seen_sign, new_last_action_time
    
    # 4. 차량 회피 처리
    vehicle_result, vehicle_control = process_vehicles(detected_objects, image_center_x)
    if vehicle_result:
        return vehicle_result, vehicle_control, last_seen_sign, last_action_time
    
    # 5. 아무것도 검출되지 않음
    return None, None, last_seen_sign, last_action_time

def add_info_overlay(frame, detected_objects, fps):
    """프레임에 정보 오버레이 추가"""
    overlay = frame.copy()
    
    # 배경 영역 생성
    cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
    
    # 텍스트 정보
    info_text = [
        f"FPS: {fps:.1f}",
        f"Objects: {len(detected_objects)}",
        f"Resolution: {frame.shape[1]}x{frame.shape[0]}"
    ]
    
    # 검출된 객체별 카운트
    class_counts = {}
    for obj in detected_objects:
        cls_id = obj["class"]
        class_name = CLASS_INFO.get(cls_id, {}).get("name", f"Class_{cls_id}")
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    if class_counts:
        info_text.append("Detected:")
        for class_name, count in class_counts.items():
            info_text.append(f"  {class_name}: {count}")
    
    # 텍스트 그리기
    for i, text in enumerate(info_text):
        y_pos = 30 + i * 20
        cv2.putText(overlay, text, (15, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 투명도 적용
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    return frame

def main():
    # 초기화
    print("Initializing YOLO models...")
    sign_model, dest_model = setup_object_detection()
    print("✅ YOLO models initialized")
    
    # 카메라 초기화
    print("Initializing camera...")
    try:
        camera = CSICamera(width=CAMERA_WIDTH, height=CAMERA_HEIGHT, capture_fps=CAMERA_FPS)
        camera.running = True
        print(f"Camera initialized with settings: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}fps")
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return

    print("Waiting for camera to be ready...")
    wait_start = time.time()
    while camera.value is None:
        if time.time() - wait_start > 5.0:  # 5초 타임아웃
            print("Camera initialization timeout!")
            return
        print("Waiting for camera frame...")
        time.sleep(0.1)
    print("✅ Camera ready!")
    
    prev_t = time.time()
    frame_count = 0
    last_seen_sign = None
    last_action_time = time.time()
    image_center_x = CAMERA_WIDTH // 2
    
    try:
        while True:
            frame = camera.value
            
            if frame is None:
                print("No frame received from camera")
                time.sleep(0.01)
                continue
                
            frame_count += 1
            
            # 객체 검출 파이프라인 실행
            result, control, new_last_seen_sign, new_last_action_time = object_detection_pipeline(
                frame, sign_model, dest_model, image_center_x, last_seen_sign, last_action_time
            )
            
            # 상태 업데이트
            if new_last_seen_sign is not None:
                last_seen_sign = new_last_seen_sign
            if new_last_action_time is not None:
                last_action_time = new_last_action_time
            
            # 객체 검출 수행
            detected_objects = detect_objects(frame, sign_model, dest_model, min_area=1000)
            
            # FPS 계산
            current_time = time.time()
            fps = 1.0 / (current_time - prev_t) if current_time - prev_t > 0 else 0
            prev_t = current_time
            
            # 검출 결과를 프레임에 그리기
            annotated_frame = draw_detections(frame, detected_objects)
            
            # 정보 오버레이 추가
            final_frame = add_info_overlay(annotated_frame, detected_objects, fps)
            
            # 결과 출력 (30프레임마다)
            if frame_count % 30 == 0:
                print(f"\n=== Frame {frame_count} ===")
                print(f"FPS: {fps:.1f}")
                if detected_objects:
                    print("검출된 객체:")
                    for obj in detected_objects:
                        cls_id = obj["class"]
                        class_name = CLASS_INFO.get(cls_id, {}).get("name", f"Unknown_{cls_id}")
                        conf = obj["confidence"]
                        area = obj["area"]
                        x1, y1, x2, y2 = obj["bbox"]
                        width = x2 - x1
                        height = y2 - y1
                        print(f"  {class_name}: 신뢰도 {conf:.2f}, 영역 {area}, 크기 {width}x{height}")
                else:
                    print("검출된 객체 없음")
            
            # 화면에 표시
            cv2.imshow('Object Detection with Annotations', final_frame)
            
            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quit signal received")
                break
                
    finally:
        # 정리
        print("Releasing camera resources...")
        camera.running = False
        if hasattr(camera, 'cap') and hasattr(camera.cap, 'release'):
            print("Attempting to release camera.cap...")
            camera.cap.release()
            print("✅ camera.cap released.")
        del camera
        cv2.destroyAllWindows()
        print("Camera stopped.")

if __name__ == "__main__":
    main()
