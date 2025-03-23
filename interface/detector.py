import cv2
from utils.video_utils import preprocess_clip
from models.action_recognition.model import MockActionRecognitionModel
from interface.visualizer import draw_prediction

def run_detection(cap, config):
    """
    Handles real-time frame capture, preprocessing, fake inference, and visualization.

    Args:
        cap (cv2.VideoCapture): OpenCV video capture object.
        config (dict): Loaded configuration settings.
    """
    # Load config values
    sequence_length = config["models"]["action_recognition"]["sequence_length"]
    resize_shape = tuple(config["processing"]["resize"])
    frame_stride = config["processing"].get("frame_stride", 1)
    show_video = config["output"].get("show_video", True)

    # Initialize fake model
    model = MockActionRecognitionModel()

    clip_buffer = []
    frame_count = 0

    print("🚀 Detection started. Press 'q' to quit.")

    while True:
        # Inside run_detection(), after reading the frame
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame down to a smaller resolution (e.g., 480x270)
        frame = cv2.resize(frame, (480, 270))

        # Add every Nth frame to buffer
        if frame_count % frame_stride == 0:
            clip_buffer.append(frame)

        frame_count += 1

        # Trim buffer to correct size
        if len(clip_buffer) > sequence_length:
            clip_buffer.pop(0)

        # If buffer full, preprocess and get prediction
        if len(clip_buffer) == sequence_length:
            clip = preprocess_clip(clip_buffer, size=resize_shape)

            # Run mock model prediction
            result = model.predict(clip)
            label = result["label"]
            confidence = result["confidence"]

            print(f"🧠 Prediction: {label} ({confidence * 100:.1f}%)")

            # Draw result on current frame
            frame = draw_prediction(frame, label=label, confidence=confidence)

            # Slide the window
            clip_buffer.pop(0)

        # Show the video
        if show_video:
            cv2.imshow("Theft Detection", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Exiting detection loop.")
            break

    cap.release()
    cv2.destroyAllWindows()
