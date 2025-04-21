import os
import numpy as np
import pandas as pd
import tensorflow as tf
from models.action_recognition.model import build_cnn_rnn_model

def load_dataset(clips_dir, labels_csv, input_shape, sequence_length):
    """
    Loads training data from .npy files and labels from a CSV.

    Returns:
        X (np.array): shape (N, T, H, W, C)
        y (np.array): shape (N,)
    """
    labels_df = pd.read_csv(labels_csv)
    X = []
    y = []

    for _, row in labels_df.iterrows():
        clip_path = os.path.join(clips_dir, row["clip"])
        if not os.path.exists(clip_path):
            print(f"⚠️ Missing: {clip_path}")
            continue

        clip = np.load(clip_path)  # shape: (T, H, W, C)
        if clip.shape[0] != sequence_length:
            print(f"⚠️ Invalid length: {clip_path}")
            continue

        X.append(clip)
        y.append(int(row["label"]))

    return np.array(X), np.array(y)

def train(config):
    # === Load config ===
    model_config = config["models"]["action_recognition"]
    processing = config["processing"]
    input_size = tuple(processing["resize"]) + (3,)  # (H, W, C)
    sequence_length = model_config["sequence_length"]
    num_classes = model_config.get("num_classes", 2)

    clips_dir = config["training"]["clips_dir"]
    labels_csv = config["training"]["labels_csv"]
    save_path = model_config["model_path"]

    print("📦 Loading dataset...")
    X, y = load_dataset(clips_dir, labels_csv, input_size, sequence_length)
    print(f"✅ Loaded {len(X)} samples. Shape: {X.shape}")

    # === Build and train model ===
    model = build_cnn_rnn_model(input_shape=(sequence_length, *input_size), num_classes=num_classes)
    model.summary()

    history = model.fit(
        X, y,
        batch_size=config["training"].get("batch_size", 8),
        epochs=config["training"].get("epochs", 10),
        validation_split=0.2,
        shuffle=True
    )

    print(f"💾 Saving model to {save_path}")
    model.save(save_path)


    return model, history
