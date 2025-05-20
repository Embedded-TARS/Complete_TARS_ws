# tars_camera.py
from jetcam.csi_camera import CSICamera
import time
import cv2


class CameraManager:
    """
    카메라 리소스를 중앙에서 관리하는 싱글톤 클래스.
    여러 모듈에서 동일한 카메라 인스턴스에 접근할 수 있게 합니다.
    """
    _instance = None
    _camera = None
    _is_initialized = False
    _reference_count = 0

    @classmethod
    def get_instance(cls):
        """싱글톤 인스턴스를 반환합니다."""
        if cls._instance is None:
            cls._instance = CameraManager()
        return cls._instance

    def initialize_camera(
        self,
        width: int = 640,
        height: int = 480,
        capture_fps: int = 30,
        device_index: int = 0,           # ⬅ 추가: 기본 장치는 /dev/video0
    ):
        """
        카메라를 초기화합니다.
        이미 초기화된 경우 참조 카운트만 증가합니다.

        Args:
            width (int):  프레임 너비
            height (int): 프레임 높이
            capture_fps (int): 캡처 FPS
            device_index (int): 사용할 비디오 장치 번호 (/dev/video{n})
        """
        if not self._is_initialized:
            print("카메라 초기화 중...")
            # JetCam CSICamera에서 특정 장치를 지정하려면 capture_device 파라미터 사용
            self._camera = CSICamera(
                width=width,
                height=height,
                capture_fps=capture_fps,
                capture_device=device_index,   # 핵심 변경
            )
            self._camera.running = True

            # 카메라가 프레임을 읽을 준비가 될 때까지 대기
            print("카메라가 준비될 때까지 대기 중...")
            while self._camera.value is None:
                time.sleep(0.05)
            print("✅ 카메라가 준비되었습니다! (/dev/video{})".format(device_index))

            self._is_initialized = True

        # 참조 카운트 증가
        self._reference_count += 1
        print(f"카메라 사용 시작 (참조 카운트: {self._reference_count})")
        return self._camera

    def release_camera(self):
        """
        카메라 참조 카운트를 감소시키고,
        참조 카운트가 0이 되면 리소스를 해제합니다.
        """
        if not self._is_initialized:
            print("카메라가 초기화되지 않았습니다.")
            return

        self._reference_count -= 1
        print(f"카메라 사용 종료 (참조 카운트: {self._reference_count})")

        if self._reference_count <= 0:
            print("카메라 리소스를 해제합니다...")
            self._camera.running = False

            # 추가적인 cleanup이 필요한 경우
            if hasattr(self._camera, "cap") and hasattr(self._camera.cap, "release"):
                print("camera.cap 객체를 해제합니다...")
                self._camera.cap.release()
                print("✅ camera.cap 해제 완료!")

            self._camera = None
            self._is_initialized = False
            self._reference_count = 0
            print("✅ 카메라 리소스 해제 완료!")

    def get_camera(self):
        """초기화된 카메라 인스턴스를 반환합니다."""
        if not self._is_initialized:
            raise RuntimeError("카메라가 초기화되지 않았습니다. initialize_camera()를 먼저 호출하세요.")
        return self._camera

    def get_frame(self):
        """현재 카메라 프레임을 반환합니다."""
        if not self._is_initialized:
            raise RuntimeError("카메라가 초기화되지 않았습니다. initialize_camera()를 먼저 호출하세요.")
        return self._camera.value

    def is_initialized(self) -> bool:
        """카메라가 초기화되었는지 여부를 반환합니다."""
        return self._is_initialized

    def __del__(self):
        """소멸자: 인스턴스가 파괴될 때 리소스를 안전하게 해제합니다."""
        if self._is_initialized:
            self._camera.running = False
            if hasattr(self._camera, "cap") and hasattr(self._camera.cap, "release"):
                self._camera.cap.release()
            print("카메라 리소스 해제 (소멸자)")
