import random

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
