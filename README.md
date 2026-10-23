# SignLingo 🤟

**An AI-powered sign language learning assistant.**

## General Idea

SignLingo is a hackathon project exploring how computer vision and machine learning can make sign language learning more interactive. It uses a webcam to detect hand gestures and classify them into their corresponding signs.

The project explores two approaches to gesture classification: pixel-based CNNs and fully connected neural networks trained on hand landmarks.

## Project Structure

```text
.
├── data/
│   ├── loaders.py
│   ├── pixel-data/
│   ├── landmark-data/
│   ├── guide-images/
│   └── test_loaders.py
├── models/
│   ├── landmark_model.py
│   ├── pixel_models.py
│   └── test_models.py
├── detector/
│   └── hand_detector.py
├── video/
│   └── camera.py
├── train_landmark_model.py
├── train_pixel_model.py
├── test_landmark_model.py
├── test_pixel_model.py
├── pixel_pretrained.pt
├── landmark_pretrained.pt
└── app.py
```

## Data

The project uses hand-sign images for pixel-based classification and hand landmark coordinates for landmark-based classification. The two representations allow us to experiment with different approaches to gesture recognition.


## Tech Stack

-   **Python** — Main programming language.
    
-   **PyTorch** — Building, training, and running the classification models.
    
-   **MediaPipe** — Real-time hand detection and landmark extraction.
    
-   **OpenCV** — Webcam access, frame processing, and visualization.
    
-   **NumPy** — Numerical operations and landmark data processing.
    
-   **Pillow (PIL)** — Loading and processing image datasets.
    
-   **scikit-learn** — Dataset splitting for training, validation, and testing.
    

## Data

SignLingo uses two representations of hand gestures:

-   **Pixel data:** Hand images resized to 400 × 400 pixels and used as input for the CNN.
    
-   **Landmark data:** The 21 hand landmarks extracted by MediaPipe, each represented by its normalized x, y, and z coordinates, producing a 63-value input vector.
    

This setup allows us to experiment with image-based recognition and a more compact, geometry-based approach.

## Loaders

`data/loaders.py` contains reusable dataset-loading utilities built with PyTorch, Pillow, and NumPy. It handles loading and preparing the data, assigning class labels, and creating training, validation, and test splits.

The loaders provide a consistent interface for both pixel and landmark datasets, keeping data preparation separate from model implementation.

## Detector Model

`detector/hand_detector.py` uses MediaPipe Hands to locate hands in webcam frames and extract their landmarks.

It supports two processing paths: extracting hand landmarks for the landmark classifier and obtaining a cropped hand image for the pixel classifier. This keeps hand detection separate from the classification models.

## Classification Models

SignLingo implements two approaches using PyTorch:

-   **Pixel model** (`models/pixel_models.py`): A convolutional neural network (CNN) that learns visual features directly from hand images.
    
-   **Landmark model** (`models/landmark_model.py`): A fully connected neural network that classifies gestures from the 63 extracted landmark coordinates.
    

Each model has its own training and testing scripts, allowing the approaches to be trained and evaluated independently.

## Video Logic

`video/camera.py` handles webcam input using OpenCV and connects the live video stream to the hand detector and classifier.

It processes frames in real time, prepares the detected hand in the appropriate format, and supports displaying recognition results. This layer bridges the computer vision pipeline and the trained models.

## App Logic

`app.py` is the main entry point of SignLingo. It brings together the video pipeline, hand detector, and selected classification model to provide a real-time sign recognition experience.

The goal is to keep the main application logic separate from data preparation, model training, and hand detection, making the project easier to test and extend.
