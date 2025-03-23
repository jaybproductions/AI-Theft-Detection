# 🛡️ AI-Based Theft Detection System

This project aims to develop a real-time AI-based theft detection system for grocery stores using **computer vision**, **action recognition**, and **anomaly detection**. It leverages **TensorFlow** for deep learning and OpenCV for video processing.

---

## 🚀 Features

- Real-time video analysis using live camera feeds or stored footage
- Deep learning models for recognizing theft-related actions
- Anomaly detection to spot unusual shopping behavior
- Visual feedback with overlays for flagged events
- Easily configurable with YAML-based settings
- Modular and extensible architecture

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **TensorFlow 2.15.0**
- **OpenCV**
- **Scikit-learn**
- **Streamlit** (optional, for dashboards/UI)

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ai-theft-detection.git
cd ai-theft-detection
```

### 2. Create and Activate a Virtual Environment (Windows)

```bash
python -m venv venv
venv\Scripts\activate
```

Mac/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```


---

## 🧠 Training the Model

### Step 1: Generate Mock Dataset

This will create synthetic `.npy` video clips and a `labels.csv` file in `data/processed/`:

```bash
python scripts/generate_mock_dataset.py
```

You can adjust the number of clips and sequence length in the script.

---

### Step 2: Train the Model

This will load the generated clips and labels, train the CNN+RNN model, and save it to `models/action_recognition/model.h5`.

```bash
python scripts/train_model.py
```

Model configuration (input size, sequence length, training parameters) is defined in `config.yaml`.

---

### Example Config (in `config.yaml`)

```yaml
models:
  action_recognition:
    model_path: models/action_recognition/model.h5
    input_size: [224, 224, 3]
    sequence_length: 16
    num_classes: 2

training:
  clips_dir: data/processed/clips
  labels_csv: data/processed/labels.csv
  batch_size: 8
  epochs: 10
```

---

## ⚙️ Configuration

All project settings (model paths, thresholds, video source, etc.) are stored in:

```
config.yaml
```

Customize it for:
- Model parameters
- Input/output video paths
- Alert thresholds
- Logging options

---

## ▶️ How to Run

### Run the full real-time detection pipeline:

```bash
python run.py
```

### Run the video preprocessor:

```bash
python scripts/prepare_data.py --input ./data/raw/video1.mp4
```

---

## 🧪 Folder Structure

```
ai_theft_detection/
│
├── data/                # Raw and processed data
├── models/              # Action recognition and anomaly detection
├── inference/           # Real-time video processing and overlay
├── scripts/             # Preprocessing, training, evaluation
├── services/            # Alerting, logging, notifications
├── utils/               # Helper functions and configuration loader
├── run.py               # Main runner for inference
├── config.yaml          # All project settings
├── requirements.txt
└── README.md
```

---

## 📊 Evaluation

We evaluate the model using:
- Accuracy (normal vs suspicious behavior)
- False positive/negative rate
- Response time
- Robustness in real-world environments

---

## 📚 References

- [SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982)
- [I3D: Inflated 3D ConvNets](https://arxiv.org/abs/1705.07750)
- TensorFlow, OpenCV, Scikit-learn Docs

---

## 🧑‍💻 Author

Chris - Braxton - Troy — Auburn University | Master's in Software Engineering  
For questions or contributions, feel free to reach out or submit a PR.
