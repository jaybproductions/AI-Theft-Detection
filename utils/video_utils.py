import cv2
import numpy as np

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
    frame_normalized = frame_resized / 255.0  # Normalize to [0, 1]
    return frame_normalized.astype(np.float32)

def preprocess_clip(frames, size=(224, 224)):
    """
    Preprocess a list of frames into a clip for model input.

    Args:
        frames (list of np.array): List of raw frames.
        size (tuple): Target resize shape (W, H).

    Returns:
        np.array: Array of shape (sequence_length, H, W, C)
    """
    processed = [preprocess_frame(f, size) for f in frames]
    return np.stack(processed, axis=0)
