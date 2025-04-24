import os
import tensorflow as tf
from models.action_recognition.model import build_cnn_rnn_model
from utils.video_utils import ClipDataGenerator

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

    print("📦 Loading data generator...")
    train_generator = ClipDataGenerator(
        clip_dir=clips_dir,
        labels_csv=labels_csv,
        batch_size=16,
        input_shape=(sequence_length, *input_size),
        shuffle=True,
    )

    print("🧠 Building model...")
    model = build_cnn_rnn_model(
        input_shape=(sequence_length, *input_size),
        num_classes=num_classes
    )
    model.summary()

    print("🚀 Training model...")
    model.fit(
        train_generator,
        epochs=config["training"]["epochs"],
        verbose=1
    )

    print(f"💾 Saving model to {save_path}")
    model.save(save_path)

    return model
