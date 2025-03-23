import os
import csv
import random

CLIPS_DIR = "data/processed/clips"
OUTPUT_CSV = "data/processed/labels.csv"
LABELS = [0, 1]  # 0 = Normal, 1 = Suspicious

def generate_labels_csv():
    clips = [f for f in os.listdir(CLIPS_DIR) if f.endswith(".npy")]
    with open(OUTPUT_CSV, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["clip", "label"])

        for clip in sorted(clips):
            label = random.choice(LABELS)
            writer.writerow([clip, label])
    print(f"✅ labels.csv written to {OUTPUT_CSV}")

if __name__ == "__main__":
    generate_labels_csv()
