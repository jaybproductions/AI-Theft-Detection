import os
import tensorflow as tf
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
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
    batch_size = config["training"].get("batch_size", 16)
    save_path = model_config["model_path"]
    initial_epoch = config["training"].get("initial_epoch", 0)
    total_epochs = config["training"]["epochs"]

    # === Load Data ===
    print("📦 Loading data generator...")
    train_generator = ClipDataGenerator(
        clip_dir=clips_dir,
        labels_csv=labels_csv,
        batch_size=batch_size,
        input_shape=(sequence_length, *input_size),
        shuffle=True,
    )

    # === Load model if exists, else build new ===
    if os.path.exists(save_path):
        print(f"🔁 Resuming from: {save_path}")
        model = load_model(save_path)
    else:
        print("🧠 Building new model...")
        model = build_cnn_rnn_model(
            input_shape=(sequence_length, *input_size),
            num_classes=num_classes
        )
    model.summary()

    # === Callbacks ===
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint(
            filepath="models/action_recognition/best_model.keras",
            monitor="loss",
            save_best_only=True,
            verbose=1
        )
    ]

    # === Train ===
    print(f"🚀 Training from epoch {initial_epoch} to {total_epochs}")
    model.fit(
        train_generator,
        epochs=total_epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=1
    )

    # === Save final model ===
    print(f"💾 Saving final model to: {save_path}")
    model.save(save_path)

    return model
