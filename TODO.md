# 📝 Project TODO List: AI-Based Theft Detection

This document tracks the remaining implementation tasks for the theft detection system using computer vision and deep learning.

---

## ✅ Already Implemented

- [x] Project structure and setup
- [x] `run.py` and `config.yaml` loader
- [x] Webcam/video input handling with fallback to demo video
- [x] Frame buffering and preprocessing (`preprocess_clip()`)
- [x] Real-time detection and visual overlays using `draw_prediction()`
- [x] Mock model replaced with real TensorFlow model
- [x] Model training pipeline using `.npy` clips and `labels.csv`
- [x] Script to generate mock `.npy` clips
- [x] Script to generate `labels.csv`
- [x] Combined script to generate mock dataset
- [x] Model saved as `model.h5` and loaded in detection pipeline

---

## 🔧 TODO: Model Integration

### 🔹 Action Recognition

- [x] Replace `MockActionRecognitionModel` with a real TensorFlow model
- [x] Add model loading using `tf.keras.models.load_model()`
- [x] Normalize input shape, dtype, and batch dimension
- [ ] Create and use a label map (e.g., `label_map.json`)

### 🔹 Anomaly Detection (Optional)

- [ ] Implement anomaly detection model (e.g., autoencoder, IsolationForest)
- [ ] Integrate anomaly scoring
- [ ] Combine action and anomaly predictions into one unified output

---

## 🧹 TODO: Data & Training Scripts

### 📁 `scripts/`

- [x] `generate_mock_dataset.py`: Create clips + labels
- [x] `train_model.py`: Train and save model
- [ ] `evaluate.py`: Compute evaluation metrics (accuracy, F1, etc.)

---

## 📦 TODO: Models

### 📁 `models/`

- [x] `action_recognition/model.py`: Load and build model architecture
- [x] `action_recognition/train.py`: Training logic for action recognition
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
