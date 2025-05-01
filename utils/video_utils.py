import cv2
import numpy as np
import os
from keras.api.keras.utils import Sequence  # ✅ Correct import

# === Frame and Clip Preprocessing ===

def preprocess_frame(frame, size=(224, 224)):
    """
    Resize and normalize a single video frame.

    Args:
        frame (np.array): Original frame (H x W x C).
        size (tuple): Desired output size (width, height).

    Returns:
        np.array: Preprocessed frame.
    """
    frame_resized = cv2.resize(frame, size)
    frame_normalized = frame_resized / 255.0
    return frame_normalized.astype(np.float32)

def preprocess_clip(frames, size=(224, 224)):
    """
    Preprocess a list of frames into a clip for model input.

    Args:
        frames (list of np.array): Raw frames.
        size (tuple): Resize shape (W, H).

    Returns:
        np.array: Shape (sequence_length, H, W, C)
    """
    processed = [preprocess_frame(f, size) for f in frames]
    return np.stack(processed, axis=0)


# === Data Generator ===

class ClipDataGenerator(Sequence):
    def __init__(self, clip_dir, labels_csv, batch_size=16, input_shape=(16, 224, 224, 3), shuffle=True):
        self.clip_dir = clip_dir
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.clips = []
        self.labels = []

        with open(labels_csv, "r") as f:
            next(f)  # skip header
            for line in f:
                name, label = line.strip().split(",")
                self.clips.append(name)
                self.labels.append(int(label))

        self.indices = np.arange(len(self.clips))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.clips) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_x = []
        batch_y = []

        for i in batch_indices:
            clip_path = os.path.join(self.clip_dir, self.clips[i])
            try:
                clip = np.load(clip_path)

                # Validate shape (avoid memory crash)
                if clip.shape != self.input_shape:
                    print(f"⚠️ Skipping {clip_path}: shape {clip.shape} != {self.input_shape}")
                    continue

                batch_x.append(clip)
                batch_y.append(self.labels[i])

            except Exception as e:
                print(f"❌ Failed to load {clip_path}: {e}")
                continue

        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
