# Real-Time Stable Face Detection & Tracking

A modular Python project for real-time, stable face detection and tracking using webcam with OpenCV Haar cascades.

## Key Features

- **Face Detection**: Real-time face detection using OpenCV Haar cascade
- **Distance Estimation**: Camera-to-face distance measurement in meters
- **Color-coded Bounding Boxes**: Visual distance indicators (green=close, red=far)
- **Calibration System**: Improve accuracy with known distance measurements
- **Stable Tracking**: Alpha-blend smoothing eliminates flicker during fast movements
- **Size Stabilization**: Prevents bounding box shrinking when turning sideways  
- **Movement Analysis**: Real-time center tracking with directional movement deltas
- **Demo Mode**: Test without webcam using simulated face movement
- **Modular Architecture**: Clean separation of detection, smoothing, and tracking logic
- **Optimized Performance**: 15-30 FPS with efficient OpenCV implementation

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

**Distance Estimation Controls:**
- Press '1' to calibrate for 1 meter distance
- Press '2' to calibrate for 2 meter distance

**Demo Mode Features:**
- Simulated face movement with smooth patterns
- Demonstrates all tracking features without camera
- Shows stable bounding box smoothing
- Displays movement tracking and trajectory

**Distance Estimation Features:**
- Real-time camera-to-face distance measurement
- Color-coded bounding boxes (green=close, red=far)
- Calibration system for improved accuracy
- Distance display in meters with 2-decimal precision

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
