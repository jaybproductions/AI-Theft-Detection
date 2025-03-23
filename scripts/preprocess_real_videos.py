import os
import cv2
import numpy as np
import csv
from tqdm import tqdm

SEQUENCE_LENGTH = 16
RESIZE_SHAPE = (224, 224)
FRAME_STRIDE = 5
MAX_CLIPS_PER_VIDEO = 100

CLIP_OUTPUT_DIR = "data/processed/clips"
LABELS_CSV_PATH = "data/processed/labels.csv"

def extract_clips_from_video(video_path, label, start_index):
    cap = cv2.VideoCapture(video_path)
    clips = []
    buffer = []
    clip_index = 0
    frame_index = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_path)}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        pbar.update(1)

        # Skip frames based on stride
        if frame_index % FRAME_STRIDE != 0:
            continue

        frame = cv2.resize(frame, RESIZE_SHAPE)
        buffer.append(frame)

        if len(buffer) == SEQUENCE_LENGTH:
            clip = np.array(buffer).astype(np.float32) / 255.0
            clips.append((clip, label))
            buffer = []  # non-overlapping clips
            clip_index += 1

            if clip_index >= MAX_CLIPS_PER_VIDEO:
                break

    cap.release()
    pbar.close()
    return clips

def preprocess_videos(video_files, label_map):
    os.makedirs(CLIP_OUTPUT_DIR, exist_ok=True)

    index = 1
    with open(LABELS_CSV_PATH, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["clip", "label"])

        for video_name in video_files:
            label = label_map[video_name]
            video_path = os.path.join("data/raw", video_name)

            clips = extract_clips_from_video(video_path, label, index)

            for clip, label in clips:
                clip_name = f"real_clip_{index:03}.npy"
                clip_path = os.path.join(CLIP_OUTPUT_DIR, clip_name)
                np.save(clip_path, clip)
                writer.writerow([clip_name, label])
                index += 1

    print(f"✅ Done! Saved {index - 1} clips to {CLIP_OUTPUT_DIR}")
    print(f"📝 Labels saved to {LABELS_CSV_PATH}")

if __name__ == "__main__":
    video_files = [
        "shopping1.mp4",
        "shopping2.mp4",
        "stealing1.mp4",
        "stealing2.mp4",
    ]

    label_map = {
        "shopping1.mp4": 0,
        "shopping2.mp4": 0,
        "stealing1.mp4": 1,
        "stealing2.mp4": 1,
    }

    preprocess_videos(video_files, label_map)
