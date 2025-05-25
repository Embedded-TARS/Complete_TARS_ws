# tars_config.py

"""
TARS 로봇 시스템의 전역 설정값을 관리하는 모듈
"""

# 카메라 설정
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# 차선 인식 설정
ROI_RATIO = 0.6  # ROI 시작 위치 (이미지 높이의 비율)
LANE_WIDTH_PX = 635  # Calibrated bottom lane width
MIDDLE_LANE_WIDTH_PX = 478  # Calibrated middle lane width
POLY_DEG = 3  # 차선 곡선 피팅에 사용할 다항식 차수
EMA_ALPHA = 0.9  # 차선 중앙 위치 스무딩 계수

# 자율주행 제어 설정
MAX_STEER = 0.3     # 최대 조향 각속도 (rad/s)
MAX_SPEED = 0.5
MIN_SPEED = 0.5
STRAIGHT_SPEED = 0.3
TURN_THRESHOLD = 0.12

# Pure Pursuit 설정
WHEELBASE = 0.1  # 차량 축간 거리 (100mm)
LOOKAHEAD_DISTANCE = 0.2  # Pure Pursuit 전방 주시 거리

# 수동 제어 설정
STEP_STEER = 0.03  # 조향 단계 값
STEP_SPEED = 0.02  # 속도 단계 값
UPDATE_INTERVAL = 0.05  # 업데이트 간격(초)

# 영상 저장 설정
CAPTURE_DIR = "./captures"
VIDEO_DIR = "./videos"

# ROI 슬라이스 계산 함수
def get_roi_slice(img_height):
    """
    이미지 높이에 기반하여 ROI 슬라이스를 계산합니다.
    
    Parameters
    ----------
    img_height : int
        원본 이미지의 높이
        
    Returns
    -------
    slice
        ROI 영역을 나타내는 슬라이스 (y 방향)
    """
    roi_start = int(img_height * ROI_RATIO)
    return slice(roi_start, img_height)
