import cv2
import mediapipe as mp
import numpy as np
import threading as th
import time
import matplotlib.pyplot as plt

# Global variables for head pose
x, y = 0, 0
X_AXIS_CHEAT, Y_AXIS_CHEAT = 0, 0

# Global variables for cheating detection
PLOT_LENGTH = 200
GLOBAL_CHEAT = 0
PERCENTAGE_CHEAT = 0
CHEAT_THRESH = 0.6
XDATA = list(range(PLOT_LENGTH))
YDATA = [0] * PLOT_LENGTH

# Head pose detection function
def pose():
    global x, y, X_AXIS_CHEAT, Y_AXIS_CHEAT, PERCENTAGE_CHEAT

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d, face_2d = [], []
        face_ids = [33, 263, 1, 61, 291, 199]

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in face_ids:
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])
                dist_matrix = np.zeros((4, 1), dtype=np.float64)
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                rmat, _ = cv2.Rodrigues(rot_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                x, y = angles[0] * 360, angles[1] * 360

                if y < -10 or y > 10:
                    X_AXIS_CHEAT = 1
                else:
                    X_AXIS_CHEAT = 0

                if x < -5:
                    Y_AXIS_CHEAT = 1
                else:
                    Y_AXIS_CHEAT = 0

                cv2.putText(image, f"x: {int(x)} y: {int(y)}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        process()  # Update PERCENTAGE_CHEAT here
        cheat_text = f"Cheating Probability: {PERCENTAGE_CHEAT * 100:.2f}%"
        cv2.putText(image, cheat_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Head Pose Estimation', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# Cheating detection logic
def avg(current, previous):
    if previous > 1:
        return 0.65
    if current == 0:
        if previous < 0.01:
            return 0.01
        return previous / 1.01
    if previous == 0:
        return current
    return 1 * previous + 0.1 * current

def process():
    global GLOBAL_CHEAT, PERCENTAGE_CHEAT, CHEAT_THRESH

    if GLOBAL_CHEAT == 0:
        if X_AXIS_CHEAT == 0:
            if Y_AXIS_CHEAT == 0:
                PERCENTAGE_CHEAT = avg(0, PERCENTAGE_CHEAT)
            else:
                PERCENTAGE_CHEAT = avg(0.2, PERCENTAGE_CHEAT)
        else:
            if Y_AXIS_CHEAT == 0:
                PERCENTAGE_CHEAT = avg(0.1, PERCENTAGE_CHEAT)
            else:
                PERCENTAGE_CHEAT = avg(0.4, PERCENTAGE_CHEAT)
    else:
        if X_AXIS_CHEAT == 0:
            if Y_AXIS_CHEAT == 0:
                PERCENTAGE_CHEAT = avg(0.6, PERCENTAGE_CHEAT)
            else:
                PERCENTAGE_CHEAT = avg(0.85, PERCENTAGE_CHEAT)

    if PERCENTAGE_CHEAT > CHEAT_THRESH:
        GLOBAL_CHEAT = 1
        print("CHEATING!")
    else:
        GLOBAL_CHEAT = 0
    print("Cheat percent:", PERCENTAGE_CHEAT)

# Real-time plotting and monitoring
def run_detection():
    global XDATA, YDATA

    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlim(0, PLOT_LENGTH)
    ax.set_ylim(0, 1)
    line, = ax.plot(XDATA, YDATA, 'r-')

    while True:
        YDATA.pop(0)
        YDATA.append(PERCENTAGE_CHEAT)
        line.set_ydata(YDATA)
        plt.pause(0.05)


def additional_plot():
    xdata = []
    ydata = []

    plt.figure()
    axes = plt.gca()
    axes.set_xlim(0, 100)
    axes.set_ylim(0, 1)
    line, = axes.plot(xdata, ydata, 'r-')

    for i in range(100):
        xdata.append(i)
        ydata.append(i / 100)
        line.set_xdata(xdata)
        line.set_ydata(ydata)
        plt.draw()
        plt.pause(1e-17)

    plt.show()

if __name__ == "__main__":
    t1 = th.Thread(target=pose)
    t2 = th.Thread(target=run_detection)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    additional_plot()
