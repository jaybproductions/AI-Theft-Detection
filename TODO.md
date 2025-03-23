# 📝 Project TODO List: AI-Based Theft Detection

This document tracks the remaining implementation tasks for the theft detection system using computer vision and deep learning.

---

## ✅ Already Implemented

- [x] Project structure and setup
- [x] `run.py` and `config.yaml` loader
- [x] Webcam/video input handling with fallback to demo video
- [x] Frame buffering and preprocessing (`preprocess_clip()`)
- [x] Mock action recognition model for testing
- [x] Real-time detection and visual overlays using `draw_prediction()`

---

## 🔧 TODO: Model Integration

### 🔹 Action Recognition

- [ ] Replace `MockActionRecognitionModel` with a real TensorFlow model
- [ ] Add model loading using `tf.keras.models.load_model()`
- [ ] Normalize input shape, dtype, and batch dimension
- [ ] Create and use a label map (e.g., `label_map.json`)

### 🔹 Anomaly Detection (Optional)

- [ ] Implement anomaly detection model (e.g., autoencoder, IsolationForest)
- [ ] Integrate anomaly scoring
- [ ] Combine action and anomaly predictions into one unified output

---

## 🧹 TODO: Data & Training Scripts

### 📁 `scripts/`

- [ ] `prepare_data.py`: Extract and save preprocessed clips
- [ ] `evaluate.py`: Compute evaluation metrics (accuracy, F1, etc.)
- [ ] `generate_synthetic.py`: Generate synthetic clips for data augmentation

---

## 📦 TODO: Models

### 📁 `models/`

- [ ] `action_recognition/model.py`: Load or define real model architecture
- [ ] `action_recognition/train.py`: Training logic for action recognition
- [ ] `anomaly_detection/autoencoder.py`: Define a basic anomaly model
- [ ] `anomaly_detection/train.py`: Train anomaly detection model

---

## 🔔 TODO: Alerts & Logging

### 📁 `services/`

- [ ] `alerting.py`: Alert system (log, email, webhook)
- [ ] `logger.py`: Save detection events to structured logs

---

## 🧪 TODO: Testing

### 📁 `tests/`

- [ ] Unit tests for `config.py`, `video_utils.py`
- [ ] Model inference tests (mocked or real)
- [ ] Visualizer overlay rendering test

---

## 📊 Optional Features

- [ ] Save annotated video to file (`output_path`)
- [ ] Streamlit dashboard for monitoring
- [ ] Configurable confidence threshold
- [ ] Mode to show only suspicious clips

---
