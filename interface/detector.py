import cv2
import numpy as np
import tensorflow as tf
import time
from utils.video_utils import preprocess_clip
from models.action_recognition.model import load_action_recognition_model
from interface.visualizer import draw_prediction
from utils.config import load_label_map

label_map = load_label_map()

def get_label_name(label_id):
    return label_map.get(str(label_id), "Unknown")

def run_detection(cap, config):
    model_path = config["models"]["action_recognition"]["model_path"]
    input_size = tuple(config["processing"]["resize"])
    sequence_length = config["models"]["action_recognition"]["sequence_length"]
    frame_stride = config["processing"].get("frame_stride", 1)
    show_video = config["output"].get("show_video", True)

    DISPLAY_EVERY_N_FRAMES = 2  # Show only every 2nd frame
    PREDICTION_INTERVAL = 10     # Run prediction every N frames

    model = load_action_recognition_model(model_path)
    labels = ["Normal", "Suspicious"]

    clip_buffer = []
    frame_count = 0
    last_label = "Processing..."
    last_confidence = 0.0

    # FPS counter
    fps_start = time.time()
    fps_counter = 0

    print("🚀 Detection started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame capture failed.")
            break

        frame = cv2.resize(frame, (640, 480))

        if frame_count % frame_stride == 0:
            clip_buffer.append(frame)

        frame_count += 1

        if len(clip_buffer) > sequence_length:
            clip_buffer.pop(0)

        # Predict every few frames only
        if len(clip_buffer) == sequence_length and frame_count % PREDICTION_INTERVAL == 0:
            clip = preprocess_clip(clip_buffer, size=input_size)
            clip = np.expand_dims(clip, axis=0)

            pred = model.predict(clip, verbose=0)[0]
            pred_index = int(tf.argmax(pred))
            last_label = labels[pred_index]
            last_confidence = float(pred[pred_index])

            print(f"🧠 Prediction: {last_label} ({last_confidence * 100:.1f}%)")

            clip_buffer.pop(0)

        # Always draw the latest prediction
        frame = draw_prediction(frame, label=last_label, confidence=last_confidence)

        # Show only every N frames for smoothness
        if show_video and frame_count % DISPLAY_EVERY_N_FRAMES == 0:
            cv2.imshow("Theft Detection", frame)

        # FPS debug print
        fps_counter += 1
        if fps_counter >= 30:
            fps_now = 30 / (time.time() - fps_start)
            print(f"📸 Estimated FPS: {fps_now:.2f}")
            fps_start = time.time()
            fps_counter = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Exiting detection loop.")
            break

    cap.release()
    cv2.destroyAllWindows()
