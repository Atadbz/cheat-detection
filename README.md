# Webcam Cheating Detection

Python prototypes for flagging possible cheating from a live webcam feed.

<sub>ARCHIVED EXPERIMENT · 2024 · PYTHON · OPENCV · TENSORFLOW</sub>

## Overview

Most scripts try to flag possible cheating in a live webcam feed by combining several visual cues. An SSD MobileNet V2 detector from TensorFlow Hub counts people and finds phones or books, and an OpenCV Haar cascade checks face position. The dlib facial landmarks compare lip distance with a closed-lips calibration, and later versions score frames only while DeepFace matches the person to a reference image. Two scripts differ: `03_face_verification.py` only checks identity, and `05_head_pose_estimation.py` uses head angles from MediaPipe Face Mesh alone.

## Contents

| Path | Description |
| --- | --- |
| `src/` | `cheat_detection.py`, the final version. It scores only while the face matches. A detected phone or book raises the score to its maximum. So does a face below or beside the frame center. Output is shown in an OpenCV window. |
| `experiments/` | Five earlier scripts, numbered in commit order. `01_first_attempt.py` and `02_second_attempt.py` combine object detection, face position, and lip distance into a cheating percentage. The second only adds quieter TensorFlow logging. `03_face_verification.py` checks webcam frames against a reference image with DeepFace every 30 frames and computes no score. `04_face_verification_with_scoring.py` computes the score only while the face matches. `05_head_pose_estimation.py` is a separate approach. It turns head angles from MediaPipe Face Mesh and `solvePnP` into an accumulating cheating probability with a live Matplotlib plot. |
| `requirements.txt` | Third-party packages imported by the scripts, unpinned. |

## Usage

`src/cheat_detection.py` needs a webcam, the dlib `shape_predictor_68_face_landmarks.dat` model, and a reference face image. Neither the model nor the image is included. Both paths are hardcoded in the script and in the experiments that use them, so edit them before running. The SSD MobileNet V2 model is loaded from a TensorFlow Hub URL, so network access is needed at least on the first run.

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the final version or the head pose experiment:

```bash
python src/cheat_detection.py
python experiments/05_head_pose_estimation.py
```

`src/cheat_detection.py` first collects lip measurements while the lips are kept closed, then runs until `q` is pressed. `experiments/05_head_pose_estimation.py` closes its camera window on Esc, but its plot thread keeps running until the process is stopped. `03_face_verification.py` and `04_face_verification_with_scoring.py` display frames through IPython, so they are meant for a Jupyter notebook. They also open the camera through the Windows DirectShow backend (`cv2.CAP_DSHOW`).

## Notes

Archived experiment kept for reference. No evaluation data or results are included, and every script except 05 points to local model or image files that are not in the repository. Scripts 01, 02, and 04 share most of their code with `src/cheat_detection.py`, and every script except 03 uses hand-set weights and thresholds.

<sub>This repository follows the [Repository Standard](https://github.com/Atadbz/Atadbz/blob/main/REPOSITORY_STANDARD.md).</sub>
