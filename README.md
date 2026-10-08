# Webcam Cheating Detection

This repository holds Python prototypes that estimate how likely it is that the person in front of a webcam is cheating.

<sub>ARCHIVED EXPERIMENT · 2025 · PYTHON · OPENCV · TENSORFLOW</sub>

## Overview

The scripts try to flag possible cheating from a live webcam feed by combining several visual cues. An SSD MobileNet V2 detector from TensorFlow Hub counts people and looks for phones or books, an OpenCV Haar cascade checks where the face sits in the frame, and dlib facial landmarks compare lip distance against a closed-lips calibration. Later versions score frames only while DeepFace matches the person to a reference image, and a separate script estimates head pose with MediaPipe Face Mesh instead. Earlier scripts are kept in `experiments/` and the latest version is in `src/`.

## Contents

| Path | Description |
| --- | --- |
| `experiments/` | Earlier standalone scripts, numbered in the order they were added. `01_first_attempt.py` and `02_second_attempt.py` combine object detection, face position and lip distance into a cheating percentage; the second only adds quieter TensorFlow logging. `03_face_verification.py` checks webcam frames against a reference image with DeepFace every 30 frames, and `04_face_verification_with_scoring.py` computes the score only while the face matches. `05_head_pose_estimation.py` is a separate approach that turns head angles from MediaPipe Face Mesh and `solvePnP` into a smoothed cheating probability with a live Matplotlib plot. |
| `src/` | `cheat_detection.py`, the latest version: face verification plus scoring, where a detected phone or book, or a face below or beside the frame center, raises the score to its maximum. Output is shown in an OpenCV window. |
| `requirements.txt` | Python packages imported across all scripts, without pinned versions. |
| `README.md` | This file. |

## Usage

`src/cheat_detection.py` needs a webcam, the dlib `shape_predictor_68_face_landmarks.dat` model and a reference face image. Neither file is included, and both paths are hardcoded in the script (and in the experiments that use them), so edit them before running. The SSD MobileNet V2 model is loaded from a TensorFlow Hub URL, so network access is needed at least on the first run.

```bash
pip install -r requirements.txt
python src/cheat_detection.py                  # latest version
python experiments/05_head_pose_estimation.py  # head pose experiment
```

`src/cheat_detection.py` first collects lip measurements while the lips are kept closed, then runs until `q` is pressed. `experiments/05_head_pose_estimation.py` closes its camera window on Esc, but its plot thread keeps running until the process is stopped. `03_face_verification.py` and `04_face_verification_with_scoring.py` display frames through IPython, so they are meant for a Jupyter notebook, and they open the camera through the Windows DirectShow backend (`cv2.CAP_DSHOW`).

## Notes

Exploratory prototype kept for reference. The scripts repeat most of their code, rely on hand-set weights and thresholds, and point to local model and image files that are not included. No evaluation data or results are part of the repository.

<sub>This repository follows the [Repository Standard](https://github.com/Atadbz/Atadbz/blob/main/REPOSITORY_STANDARD.md).</sub>
