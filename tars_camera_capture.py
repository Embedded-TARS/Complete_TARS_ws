import cv2
import time
import pathlib
from jetcam.csi_camera import CSICamera

# Configuration
OUTDIR = "./captures"
WIDTH, HEIGHT = 640, 480 # Change camera resolution to 640x480
FPS = 30 # Not strictly needed for single capture but good practice

def main():
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    path = pathlib.Path(OUTDIR).resolve()
    path.mkdir(exist_ok=True, parents=True)
    filepath = str(path / f"{timestamp}.jpg") # Save as JPG

    print("Initializing camera...")
    # Use default resolution if not specified
    # camera = CSICamera(capture_fps=FPS)
    # Use specified resolution
    camera = CSICamera(width=WIDTH, height=HEIGHT, capture_fps=FPS)
    camera.running = True

    # Wait for camera to be ready
    print("Waiting for camera to be ready...")
    while camera.value is None:
        time.sleep(0.1)
    print("✅ Camera ready!")

    # Add a short delay to allow the camera to stabilize
    stabilization_delay = 0.5 # seconds
    print(f"Waiting for camera stabilization ({stabilization_delay}s)...")
    time.sleep(stabilization_delay)
    print("✅ Stabilization complete!")

    try:
        print(f"Capturing image to {filepath}...")
        frame = camera.value
        if frame is not None:
            # jetcam provides BGR format by default, which is suitable for cv2.imwrite
            cv2.imwrite(filepath, frame)
            print(f"✅ Image captured successfully: {filepath}")
        else:
            print("❌ Failed to capture frame.")

    except Exception as e:
        print(f"An error occurred during capture: {e}")

    finally:
        # Release camera resources
        print("Releasing camera resources...")
        camera.running = False
        # Attempt to explicitly release the underlying capture object
        if hasattr(camera, 'cap') and hasattr(camera.cap, 'release'):
            print("Attempting to release camera.cap...")
            camera.cap.release()
            print("✅ camera.cap released.")
        # Explicitly delete the camera object to ensure resource release
        del camera
        # No need for cv2.destroyAllWindows() for single capture without imshow
        print("Camera stopped.")

if __name__ == "__main__":
    main()
