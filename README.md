# 🚧 AI-Based Road Damage Detection System

## 📌 Project Overview

An AI-based road damage detection application developed using YOLO11n, Python, and Streamlit. It identifies different types of road damage from uploaded images and displays the results with bounding boxes and confidence scores.

## ✨ Features

- Upload road images in JPG, JPEG, and PNG formats.
- Detect multiple road damage instances.
- Display damage classes and confidence scores.
- Visualize detected damage using bounding boxes.
- Adjust the confidence threshold.
- Download annotated detection images.

## 🛠️ Technologies Used

- Python
- YOLO11n
- Ultralytics
- PyTorch
- OpenCV
- Streamlit

## 📊 Dataset

- Dataset: RDD2022 India Road Damage Dataset
- Number of classes: 10
- Training, validation, and test image sets.
- Dataset license: CC BY 4.0

## 📈 Model Evaluation

The trained model was evaluated on a held-out test dataset.

| Metric | Result |
|---|---:|
| Precision | 32.9% |
| Recall | 45.8% |
| mAP@50 | 34.6% |
| mAP@50-95 | 16.3% |

*Note: These are object-detection evaluation metrics, not accuracy percentages.*

## 🚀 Run Locally

1. Clone this repository:

   `git clone https://github.com/varsaleshravani-oss/AI-Based-Road-Damage-Detection.git`

2. Open the project folder.

3. Install the required dependencies:

   `pip install -r requirements.txt`

4. Run the Streamlit application:

   `streamlit run app.py`

## 🌐 Live Demo

[Open AI-Based Road Damage Detection App](https://road-damage-detection-shravani.streamlit.app/)

## 📁 Model Weights

The application uses the trained YOLO11n model stored in `best_v2.pt`.

## 👩‍💻 Author

Shravani Varsale

B.Tech – Artificial Intelligence
