import os
import numpy as np

# Configurable parameters
output_dir = "data/processed/clips"
num_clips = 10
sequence_length = 16
frame_size = (224, 224, 3)

def generate_mock_clip():
    """
    Generates a fake video clip of shape (T, H, W, C)
    with random noise to simulate a clip.
    """
    return np.random.rand(sequence_length, *frame_size).astype(np.float32)

def save_mock_clips():
    os.makedirs(output_dir, exist_ok=True)
    for i in range(1, num_clips + 1):
        clip = generate_mock_clip()
        filename = f"clip_{i:03}.npy"
        path = os.path.join(output_dir, filename)
        np.save(path, clip)
        print(f"✅ Saved {path}")

if __name__ == "__main__":
    save_mock_clips()
