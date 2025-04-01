import cv2
from interface.detector import run_detection
from utils.config import load_label_map

# Load label map
label_map = load_label_map()

def get_label_name(label_id):
    return label_map.get(str(label_id), "Unknown")


def run_inference(config):
    input_config = config.get("input", {})
    source = input_config.get("source", 0)
    fallback_video = input_config.get("fallback_video", "data/raw/demo_video.mp4")

    print(f"🎥 Trying to open video source: {source}")
    cap = cv2.VideoCapture(source)

    # If webcam fails, try fallback video
    if not cap.isOpened():
        print("❌ Webcam not detected. Falling back to demo video...")
        cap = cv2.VideoCapture(fallback_video)

        if not cap.isOpened():
            print(f"🚫 Failed to open fallback video: {fallback_video}")
            return

    # Set video properties (only applies to webcam, mostly)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, input_config.get("frame_width", 640))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, input_config.get("frame_height", 480))
    cap.set(cv2.CAP_PROP_FPS, input_config.get("fps", 30))

    run_detection(cap, config)
