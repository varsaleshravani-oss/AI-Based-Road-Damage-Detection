# AI-Based Road Damage Detection System

## Project Overview
This project uses a YOLO11 object detection model to identify road damage from images.

Users can upload road images and view detected damage with bounding boxes, confidence scores, and downloadable results.

## Technologies Used
- Python
- YOLO11
- Ultralytics
- OpenCV
- Streamlit
- PyTorch

## Dataset
RDD2022 India Road Damage Dataset.

## Features
- Upload road images.
- Detect multiple road damage instances.
- Display damage classes and confidence scores.
- Adjust the detection confidence threshold.
- Download annotated detection images.

## Model Evaluation
The improved model achieved the following results on the held-out test dataset:

- Precision: 32.9%
- Recall: 45.8%
- mAP@50: 34.6%
- mAP@50-95: 16.3%

## How to Run

1. Clone or download this repository.
2. Install the dependencies:

   pip install -r requirements.txt

3. Run the application:

   streamlit run app.py

## Model Weights
The application uses the `best_v2.pt` trained model.

## Author
Shravani Varsale