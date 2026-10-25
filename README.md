# Somno-Vision

**Real-Time Eye Monitoring and Drowsiness Alert System**

Somno-Vision is a Python project that uses computer vision to monitor eye states through a webcam. It calculates the Eye Aspect Ratio (EAR) to detect prolonged eye closure and display a possible drowsiness warning.

## Features

- Real-time webcam monitoring.
- Face detection using MediaPipe.
- Facial landmark detection around the eyes.
- Eye Aspect Ratio (EAR) calculation for both eyes.
- Eye-state classification: open or closed.
- Visual warning after approximately 5 seconds of continuous eye closure.
- On-screen display of EAR values and eye status.
- Automatic alert reset when the face is no longer detected or the eyes remain open for at least 0.3 seconds.

## Technologies

- **Python** — application logic.
- **OpenCV** — webcam capture and visual interface.
- **MediaPipe** — face detection and facial landmark tracking.
- **Math** — distance calculations.
- **Time** — eye-closure duration tracking.

## Project Structure

```text
Somno-Vision/
├── models/
│   ├── blaze_face_short_range.tflite
│   └── face_landmarker.task
├── main.py
└── README.md
```

## Requirements

- Python 3.9 or a compatible version supported by your installed MediaPipe release.
- A working webcam.
- The required MediaPipe model files in the `models/` directory.

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/tuidlara/somno-vision
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

**Linux/macOS**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install opencv-python mediapipe
```

### 4. Run the application

Make sure the model files are present in the `models/` directory, then execute:

```bash
python main.py
```

The application will open a window displaying the webcam feed and real-time eye monitoring information.

Press **Q** to close the application.

## How It Works

Somno-Vision processes webcam frames using MediaPipe's face landmark detection.

The application calculates the Eye Aspect Ratio (EAR), a value based on the relationship between the vertical and horizontal dimensions of the eyes.

The current detection logic uses the following parameters:

| Parameter | Value |
|---|---|
| Eye closure threshold | EAR ≤ 0.19 for both eyes |
| Warning trigger | Approximately 5 seconds of continuous eye closure |
| Alert reset | Eyes remain open for at least 0.3 seconds |
| Face not detected | Eye timers and alert are reset |

When both eyes remain below the EAR threshold for the configured duration, the application displays a red warning on the screen.

## Limitations

This project is intended for educational and demonstration purposes. Detection accuracy may vary depending on lighting, camera quality, head position, and facial landmark tracking.

The warning indicates a possible sign of drowsiness. This project is not a medical diagnostic tool or a validated driver-safety system.

## Autor

Arthur de Lara Zilli - [LinkedIn](https://www.linkedin.com/in/arthur-de-lara-zilli-957512420/)
