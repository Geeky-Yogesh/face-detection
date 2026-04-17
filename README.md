# Real-Time Stable Face Detection & Tracking

A modular Python project for real-time, stable face detection and tracking using webcam with OpenCV Haar cascades.

## Features

- **Face Detection**: Real-time face detection using OpenCV Haar cascade
- **Stable Tracking**: Alpha-blend smoothing for flicker-free bounding boxes
- **Center Tracking**: Track face center coordinates and movement deltas
- **Missing Face Handling**: Hold last position for 5-10 frames when face is lost
- **Clean UI**: Bounding boxes, center dots, coordinates display, and status text
- **Real-time Performance**: Optimized for 15-30 FPS

## Installation

```bash
pip install -r requirements.txt
```

## Usage

**With Webcam:**
```bash
python main.py
```

**Demo Mode (no webcam required):**
```bash
python main.py --demo
```

Press 'q' to quit the application.

**Demo Mode Features:**
- Simulated face movement with smooth patterns
- Demonstrates all tracking features without camera
- Shows stable bounding box smoothing
- Displays movement tracking and trajectory

## Project Structure

- `face_detection.py` - Face detection module using OpenCV Haar cascades
- `smoothing.py` - Alpha-blend smoothing for stable tracking
- `tracking.py` - Center tracking and movement delta calculation
- `main.py` - Main application with real-time processing and UI

## Requirements

- Python 3.7+
- OpenCV 4.8+
- NumPy 1.24+
- Webcam
