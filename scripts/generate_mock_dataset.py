import os
import csv
import random
import numpy as np

# === Configuration ===
output_dir = "data/processed/clips"
labels_csv = "data/processed/labels.csv"
num_clips = 10
sequence_length = 16
frame_size = (224, 224, 3)
label_choices = [0, 1]  # 0 = Normal, 1 = Suspicious

def generate_mock_clip():
    """
    Generates a fake video clip of shape (T, H, W, C)
    with random noise to simulate input data.
    """
    return np.random.rand(sequence_length, *frame_size).astype(np.float32)

def generate_mock_dataset():
    os.makedirs(output_dir, exist_ok=True)

    # Open CSV for writing labels
    with open(labels_csv, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["clip", "label"])

        # Generate clips + labels
        for i in range(1, num_clips + 1):
            clip_filename = f"clip_{i:03}.npy"
            clip_path = os.path.join(output_dir, clip_filename)

            clip = generate_mock_clip()
            np.save(clip_path, clip)
            print(f"✅ Saved {clip_path}")

            label = random.choice(label_choices)
            writer.writerow([clip_filename, label])

    print(f"📄 labels.csv written to: {labels_csv}")

if __name__ == "__main__":
    generate_mock_dataset()
