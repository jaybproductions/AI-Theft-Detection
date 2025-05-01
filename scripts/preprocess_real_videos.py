import os
import cv2
import numpy as np
import csv
import glob
from tqdm import tqdm

SEQUENCE_LENGTH = 16
RESIZE_SHAPE = (224, 224)
FRAME_STRIDE = 4
MAX_CLIPS_PER_VIDEO = 50

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

        if frame_index % FRAME_STRIDE != 0:
            continue

        frame = cv2.resize(frame, RESIZE_SHAPE)
        buffer.append(frame)

        if len(buffer) == SEQUENCE_LENGTH:
            clip = np.array(buffer).astype(np.float32) / 255.0
            clips.append((clip, label))
            buffer.pop(0)  # sliding window instead of resetting buffer
            clip_index += 1

            if clip_index >= MAX_CLIPS_PER_VIDEO:
                break

    cap.release()
    pbar.close()
    return clips

def preprocess_videos(video_files_with_paths, label_map):
    os.makedirs(CLIP_OUTPUT_DIR, exist_ok=True)

    index = 1
    with open(LABELS_CSV_PATH, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["clip", "label"])

        for video_path in video_files_with_paths:
            filename = os.path.basename(video_path)
            label = label_map[filename]

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
    normal_dir = "data/raw/normal"
    suspicious_dir = "data/raw/suspicious"

    normal_files = glob.glob(os.path.join(normal_dir, "*.mp4"))
    suspicious_files = glob.glob(os.path.join(suspicious_dir, "*.mp4"))

    all_files = normal_files + suspicious_files
    video_files_with_paths = all_files

    label_map = {}
    for f in normal_files:
        label_map[os.path.basename(f)] = 0
    for f in suspicious_files:
        label_map[os.path.basename(f)] = 1

    preprocess_videos(video_files_with_paths, label_map)
    print("🎥 Preprocessing complete!")