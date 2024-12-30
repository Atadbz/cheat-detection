import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from collections import deque
import dlib
from scipy.spatial import distance as dist

# Load the SSD MobileNet V2 model from TensorFlow Hub
model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

# Load a pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize dlib's pre-trained face detector and shape predictor
dlib_detector = dlib.get_frontal_face_detector()
dlib_predictor = dlib.shape_predictor("C:/Users/V I S I O N/shape_predictor_68_face_landmarks.dat")

# Constants for lip landmarks
LIP_TOP = list(range(50, 53)) + list(range(61, 64))
LIP_BOTTOM = list(range(56, 59)) + list(range(65, 68))

# Function to calculate lip distance
def lip_distance(shape):
    top_lip = shape[LIP_TOP]
    bottom_lip = shape[LIP_BOTTOM]
    top_mean = np.mean(top_lip, axis=0)
    bottom_mean = np.mean(bottom_lip, axis=0)
    distance = dist.euclidean(top_mean, bottom_mean)
    return distance

# Function to perform object detection
def detect_objects(frame):
    resized_frame = cv2.resize(frame, (300, 300))
    input_tensor = tf.convert_to_tensor(resized_frame, dtype=tf.uint8)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = model(input_tensor)
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_scores = detections['detection_scores'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy()

    return detection_boxes, detection_scores, detection_classes

# Function to check for cheating
def is_cheating(detection_boxes, detection_scores, detection_classes, face_detected, face_position, lips_open):
    cheating_score = 0
    num_detections = detection_boxes.shape[0]
    person_count = 0

    for i in range(num_detections):
        if detection_scores[i] > 0.5:
            if detection_classes[i] == 1:
                person_count += 1
            elif detection_classes[i] in [77, 84]:
                cheating_score += 0.8

    if person_count == 0:
        cheating_score += 0.8

    if person_count > 1:
        cheating_score += 0.8

    if not face_detected or face_position == 'out_of_frame':
        cheating_score += 0.5
    elif face_position == 'downward':
        cheating_score += 0.1
    elif face_position == 'sideward':
        cheating_score += 0.5

    if lips_open:
        cheating_score = max(cheating_score, 0.8)

    return min(cheating_score, 1.0)

# Function to detect face and determine its position
def detect_face_position(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return False, 'out_of_frame'

    for (x, y, w, h) in faces:
        face_center_x = x + w // 2
        face_center_y = y + h // 2
        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2

        if face_center_y > frame_center_y + h // 2:
            return True, 'downward'
        elif abs(face_center_x - frame_center_x) > w // 2:
            return True, 'sideward'

    return True, 'forward'

# Function to detect if lips are open
def detect_lips_open(frame, calibration_distance):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = dlib_detector(gray, 0)

    for rect in rects:
        shape = dlib_predictor(gray, rect)
        shape = np.array([(p.x, p.y) for p in shape.parts()])
        lip_dist = lip_distance(shape)
        threshold = calibration_distance + 3.0

        if lip_dist > threshold:
            return True, lip_dist
        else:
            return False, lip_dist

    return False, 0

# Initialize webcam
cap = cv2.VideoCapture(0)

# Calibration for lip distance
calibration_distances = []
calibration_samples = 50

print("Please keep your lips closed for calibration...")

while len(calibration_distances) < calibration_samples:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = dlib_detector(gray, 0)

    for rect in rects:
        shape = dlib_predictor(gray, rect)
        shape = np.array([(p.x, p.y) for p in shape.parts()])
        lip_dist = lip_distance(shape)
        calibration_distances.append(lip_dist)

    cv2.imshow("Calibration", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

calibration_distance = np.mean(calibration_distances)
print(f"Calibration distance set to: {calibration_distance}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    boxes, scores, classes = detect_objects(frame)
    face_detected, face_position = detect_face_position(frame)
    lips_open, lip_dist = detect_lips_open(frame, calibration_distance)

    cheating_score = is_cheating(boxes, scores, classes, face_detected, face_position, lips_open)
    cheating_percentage = cheating_score * 100

    print(f'Cheating Percentage: {cheating_percentage:.2f}%')
    cv2.putText(frame, f'Cheating: {cheating_percentage:.2f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Webcam Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
