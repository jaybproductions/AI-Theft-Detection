# evaluate.py
# TODO: Evaluate trained models using metrics like accuracy, precision, recall, F1, and false alarm rate.
from utils.config import load_label_map

# Load label map
label_map = load_label_map()

def get_label_name(label_id):
    return label_map.get(str(label_id), "Unknown")
