# Underwater Vision and Sonar Object Detection System

A desktop-based underwater object detection and monitoring system built using Python, OpenCV, PySide6, Arduino serial communication, and machine learning.

## Features

- Real-time object detection
- Underwater image enhancement
- Marine object classification
- Arduino integration
- Sensor fusion dashboard
- Manual simulation mode
- Event logging

## Technologies Used

- Python
- OpenCV
- PySide6
- NumPy
- TensorFlow
- PySerial
- Streamlit

## Installation

```bash
pip install -r requirements-desktop.txt
python standalone_app.py
```

## Project Structure

- app.py – Streamlit dashboard
- standalone_app.py – Desktop application
- train_marine_classifier.py – Model training
- capture_marine_dataset.py – Dataset collection
- prepare_marine_dataset.py – Dataset preparation
- download_marine_images.py – Dataset downloader

## Requirements

- Webcam (optional)
- Arduino Uno (optional)

The application can run in manual simulation mode without hardware.
