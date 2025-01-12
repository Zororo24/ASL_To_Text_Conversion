# ASL to Text Conversion

Real-time American Sign Language (ASL) detection and text conversion using computer vision and machine learning.

## Overview

This project implements a machine learning model that can detect and interpret American Sign Language gestures in real-time using your computer's webcam. The system utilizes OpenCV for image processing and MediaPipe for hand landmark detection, converting ASL signs into text with high accuracy.

## Features

- Real-time ASL gesture detection through webcam feed
- Hand landmark detection and tracking using MediaPipe
- Custom dataset training capability
- Random Forest Classifier for gesture classification
- High-confidence predictions with threshold filtering
- Support for basic ASL alphabet

## Technologies Used

- Python 3.8+
- OpenCV
- MediaPipe
- scikit-learn (Random Forest Classifier)
- Pickle
- NumPy

## Data Pipeline

1. **Image Capture**: Raw images are captured using OpenCV
2. **Hand Detection**: MediaPipe identifies hand presence in the frame
3. **Landmark Extraction**: 42 hand landmarks are extracted per frame
4. **Feature Engineering**: Landmarks are processed into a numerical dataset
5. **Classification**: Random Forest model predicts the corresponding text

## Future Improvements

- [ ] Support for continuous sign language sentences
- [ ] Web interface for easy access
- [ ] Support for more complex ASL gestures

## Contact

Abhishek Shinde - abhivshinde24@gmail.com
Project Link: https://github.com/Zororo24/ASL_To_Text_Conversion
