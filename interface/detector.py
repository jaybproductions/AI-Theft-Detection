import cv2
import numpy as np
import tensorflow as tf
from utils.video_utils import preprocess_clip
from models.action_recognition.model import load_action_recognition_model
from interface.visualizer import draw_prediction
from utils.config import load_label_map

# Load label map
label_map = load_label_map()

def get_label_name(label_id):
    return label_map.get(str(label_id), "Unknown")


def run_detection(cap, config):
    """
    Real-time detection pipeline: capture frames, build clips, run model, display results.

    Args:
        cap (cv2.VideoCapture): OpenCV capture object.
        config (dict): Project configuration loaded from config.yaml
    """
    # === Load config values ===
    model_path = config["models"]["action_recognition"]["model_path"]
    input_size = tuple(config["processing"]["resize"])  # (W, H)
    sequence_length = config["models"]["action_recognition"]["sequence_length"]
    frame_stride = config["processing"].get("frame_stride", 1)
    show_video = config["output"].get("show_video", True)

    # === Load real model ===
    model = load_action_recognition_model(model_path)
    labels = ["Normal", "Suspicious"]  # Can later be loaded from file/config

    clip_buffer = []
    frame_count = 0

    print("🚀 Detection started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame capture failed.")
            break

                    #make the video out smaller
        frame = cv2.resize(frame, (640, 480))

        # Add every Nth frame to buffer
        if frame_count % frame_stride == 0:
            clip_buffer.append(frame)

        frame_count += 1

        # Trim buffer to correct size
        if len(clip_buffer) > sequence_length:
            clip_buffer.pop(0)

        # If full clip ready, preprocess and predict
        if len(clip_buffer) == sequence_length:
            clip = preprocess_clip(clip_buffer, size=input_size)      # (T, H, W, C)
            clip = np.expand_dims(clip, axis=0)                        # (1, T, H, W, C)

            pred = model.predict(clip, verbose=0)[0]                   # (num_classes,)
            pred_index = int(tf.argmax(pred))
            label = labels[pred_index]
            confidence = float(pred[pred_index])

            print(f"🧠 Prediction: {label} ({confidence * 100:.1f}%)")

            # Draw prediction on the current frame
            frame = draw_prediction(frame, label=label, confidence=confidence)

            # Slide the window
            clip_buffer.pop(0)

        # Display video
        if show_video:
            cv2.imshow("Theft Detection", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Exiting detection loop.")
            break

    cap.release()
    cv2.destroyAllWindows()
