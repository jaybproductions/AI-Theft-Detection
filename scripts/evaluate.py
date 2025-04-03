
from utils.config import load_label_map
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# Load label map
label_map = load_label_map()

def get_label_name(label_id):
    return label_map.get(str(label_id), "Unknown")

def evaluate_model(y_true, y_pred):
    """Compute standard evaluation metrics for binary classification."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

    print("Evaluation Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # False Alarm Rate = FP / (FP + TN)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    print(f"False Alarm Rate: {false_alarm_rate:.4f}")

# Example usage (mock data)
if __name__ == "__main__":
    # Ground truth and predictions as label IDs
    y_true = [0, 1, 1, 0, 1, 0, 0, 1]
    y_pred = [0, 1, 0, 0, 1, 0, 1, 1]
    evaluate_model(y_true, y_pred)
