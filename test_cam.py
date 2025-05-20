# camera_test.py
from tars_camera import CameraManager
import cv2

def main():
    # 싱글톤 인스턴스 가져오기
    cam_manager = CameraManager.get_instance()

    # 카메라 초기화 (/dev/video0 → device_index=0)
    cam = cam_manager.initialize_camera(device_index=0)

    print("카메라 테스트 시작 – 창에서 'q' 키를 누르면 종료됩니다.")
    try:
        while True:
            frame = cam_manager.get_frame()
            if frame is not None:
                #cv2.imshow("Camera Test (/dev/video0)", frame)
                print(frame.shape)
            # 'q' 키 입력 시 루프 탈출
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # 자원 정리
        cam_manager.release_camera()
        cv2.destroyAllWindows()
        print("카메라 테스트 종료")

if __name__ == "__main__":
    main()
