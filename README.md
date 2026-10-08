# Webcam Cheating Detection

Python prototypes that estimate a live cheating score from a webcam feed.

<sub>ARCHIVED EXPERIMENT · 2025 · PYTHON · OPENCV · TENSORFLOW · DLIB · DEEPFACE · MEDIAPIPE</sub>

## Overview

The scripts try to flag possible cheating in front of a webcam by combining several visual cues. An SSD MobileNet V2 detector from TensorFlow Hub counts people and looks for phones or books, an OpenCV Haar cascade checks where the face sits in the frame, and dlib facial landmarks compare lip distance against a closed-lips calibration. Later versions only score frames after DeepFace verifies the person against a reference image, and a separate script estimates head pose with MediaPipe Face Mesh. The files are standalone scripts numbered in the order they were written, ending with `cheat_detection.py`.

## Contents

| Path | Description |
| --- | --- |
| `01_first_attempt.py` | First version: object detection, face position, and lip distance combined into a cheating percentage drawn on the webcam feed. |
| `02_second_attempt.py` | Same pipeline as the first attempt, with TensorFlow log messages suppressed. |
| `03_face_verification.py` | Checks webcam frames against a reference image with DeepFace every 30 frames and shows the result in a Jupyter notebook. |
| `04_face_verification_with_scoring.py` | Combines the two: the cheating score is computed only while the face matches the reference image. |
| `05_head_pose_estimation.py` | Separate approach: head angles from MediaPipe Face Mesh and `solvePnP`, a smoothed cheating probability, and a live Matplotlib plot. |
| `cheat_detection.py` | Final version: face verification plus scoring, where a detected phone or book, or a face below or beside the frame center, sets the score to 100%. Output is shown in an OpenCV window. |

## Usage

`cheat_detection.py` needs a webcam, the dlib `shape_predictor_68_face_landmarks.dat` model, and a reference face image. Both file paths are hardcoded in the script and must be edited before running.

```bash
pip install opencv-python numpy tensorflow tensorflow-hub scipy dlib deepface pillow
python cheat_detection.py
```

The script first collects 50 lip measurements with the lips closed, then runs until `q` is pressed. `05_head_pose_estimation.py` instead uses `opencv-python`, `numpy`, `mediapipe`, and `matplotlib`, and stops on Esc.

## Notes

Exploratory prototype kept for reference. The scripts repeat most of their code, rely on hand-set weights and thresholds, and point to local model and image files that are not included. No evaluation data or results are part of the repository.
