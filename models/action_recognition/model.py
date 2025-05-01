import random
import tensorflow as tf
import os

def build_cnn_rnn_model(input_shape=(16, 224, 224, 3), num_classes=2):
    inputs = tf.keras.Input(shape=input_shape)

    # Improved TimeDistributed CNN
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'))(inputs)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2)))(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2)))(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x)  # Better than Flatten

    x = tf.keras.layers.GRU(128, return_sequences=False)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def load_action_recognition_model(model_path):
    """
    Loads a trained TensorFlow model from disk.

    Args:
        model_path (str): Path to the .h5 model file.

    Returns:
        tf.keras.Model: Loaded model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    print(f"✅ Loaded model from {model_path}")
    return model

class MockActionRecognitionModel:
    def __init__(self, labels=None):
        self.labels = labels or ["Normal", "Suspicious"]

    def predict(self, clip):
        """
        Simulates prediction on a video clip.

        Args:
            clip (np.array): Preprocessed clip of shape (sequence_length, H, W, C)

        Returns:
            dict: { "label": str, "confidence": float }
        """
        label = random.choice(self.labels)
        confidence = round(random.uniform(0.6, 0.99), 2)  # Simulated confidence
        return {"label": label, "confidence": confidence}
