import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
from gpiozero import Button
from time import sleep, time
import scipy.optimize
import requests
from stereo_image_utils import (
    get_detections, get_cost, get_horiz_dist_corner_tl, get_horiz_dist_corner_br,
    get_dist_to_centre_tl, get_dist_to_centre_br
)
import os
import smbus

# ✅ Initialize text-to-speech engine
engine = pyttsx3.init()

def speak_message(message):
    print(message)
    engine.say(message)
    engine.runAndWait()

# ✅ Initialize MPU6050
bus = smbus.SMBus(1)
MPU6050_ADDR = 0x68
bus.write_byte_data(MPU6050_ADDR, 0x6B, 0)  # Wake up MPU6050

def read_mpu6050():
    """Read accelerometer data from MPU6050."""
    accel_x = bus.read_byte_data(MPU6050_ADDR, 0x3B) << 8 | bus.read_byte_data(MPU6050_ADDR, 0x3C)
    accel_y = bus.read_byte_data(MPU6050_ADDR, 0x3D) << 8 | bus.read_byte_data(MPU6050_ADDR, 0x3E)
    accel_z = bus.read_byte_data(MPU6050_ADDR, 0x3F) << 8 | bus.read_byte_data(MPU6050_ADDR, 0x40)
    
    # Convert to signed values
    if accel_x > 32768:
        accel_x -= 65536
    if accel_y > 32768:
        accel_y -= 65536
    if accel_z > 32768:
        accel_z -= 65536
        
    return accel_x, accel_y, accel_z

def is_moving(threshold=5000):
    """Determine if the device is in motion based on MPU6050 readings."""
    accel_x, accel_y, accel_z = read_mpu6050()
    magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    return abs(magnitude - 16384) > threshold  # 16384 = 1g in raw value

# ✅ Camera URLs
URL_left = "http://192.168.87.220:81/stream"
URL_right = "http://192.168.87.142:81/stream"

# ✅ Save directory
SAVE_DIR = "stereo_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# ✅ Stereo processing
def stereo():
    cap_left = cv2.VideoCapture(URL_left)
    cap_right = cv2.VideoCapture(URL_right)

    if not cap_left.isOpened() or not cap_right.isOpened():
        speak_message("Stereo cameras not available.")
        return

    ret_l, frame_l = cap_left.read()
    ret_r, frame_r = cap_right.read()

    if not ret_l or not ret_r:
        speak_message("Stereo cameras not providing frames.")
        cap_left.release()
        cap_right.release()
        return

    try:
        imgs = [frame_l, frame_r]
        det, lbls, scores = get_detections(model, imgs)
        sz1 = frame_r.shape[1]
        centre = sz1 / 2

        if det:
            cost = get_cost(det, lbls=lbls, sz1=centre)
            tracks = scipy.optimize.linear_sum_assignment(cost)

            dists_tl = get_horiz_dist_corner_tl(det)
            dists_br = get_horiz_dist_corner_br(det)

            final_dists = []
            dctl = get_dist_to_centre_tl(det[0], cntr=centre)
            dcbr = get_dist_to_centre_br(det[0], cntr=centre)

            for i, j in zip(*tracks):
                if dctl[i] < dcbr[i]:
                    final_dists.append(dists_tl[i][j])
                else:
                    final_dists.append(dists_br[i][j])

            dists_away = (9.2 / 2) * sz1 * (1 / 0.4105566946974735) / np.array(final_dists) - 3.374924435067257

            for i, j in zip(*tracks):
                if i < len(lbls) and i < len(dists_away):
                    object_number = int(lbls[i])
                    object_name = model.names.get(object_number, f"Unknown({object_number})")
                    distance = 1.1 * dists_away[i]
                    message = f'{object_name} is {abs(distance)/100:.1f} meters away'
                    speak_message(message)

            for i, imgi in enumerate(imgs):
                results = model(imgi)
                img = results[0].plot()
                filename = f"{SAVE_DIR}/{'left' if i == 0 else 'right'}_eye_{int(time())}.jpg"
                cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                print(f"Saved: {filename}")

    except Exception as e:
        print(f"Stereo error: {e}")
        speak_message("Stereo can't work right now.")

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

# ✅ Moving mode
def moving(results, frame):
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = int(box.cls[0])
            object_name = model.names[label]
            center_x = (x1 + x2) // 2
            direction = "left" if center_x <= 215 else "right" if center_x >= 430 else "middle"
            message = f"{object_name} is incoming {direction}"
            speak_message(message)

# ✅ Stationary mode
def stationary(results, frame, prev_positions):
    current_positions = {}
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = int(box.cls[0])
            object_name = model.names[label]
            center_x = (x1 + x2) // 2
            direction = "left" if center_x <= 215 else "right" if center_x >= 430 else "middle"
            current_positions[label] = (center_x, (y1 + y2) // 2)
            message = f"{object_name} is on {direction}"
            speak_message(message)
    prev_positions.update(current_positions)

# ✅ Initialize model
model = YOLO("epoch21.pt")
prev_positions = {}

cap = cv2.VideoCapture(0)
mode = "stationary"
last_frame_time = time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, -1)
    moving_state = is_moving()
    new_mode = "moving" if moving_state else "stationary"

    if new_mode != mode:
        mode = new_mode
        cv2.destroyAllWindows()
        speak_message(f"Switching to {mode} mode")

    current_time = time()
    if mode == "stationary" and current_time - last_frame_time >= 1.5:
        results = model(frame, stream=True)
        stationary(results, frame, prev_positions)
        last_frame_time = current_time

    elif mode == "moving" and current_time - last_frame_time >= 3:
        results = model(frame, stream=True)
        moving(results, frame)
        last_frame_time = current_time

    cv2.imshow(mode,frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('x'):
        break
    elif key == ord('s'):
        stereo()

cap.release()
cv2.destroyAllWindows()
