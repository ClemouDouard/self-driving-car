import cv2
import numpy as np
from camera import Camera
from picarx import Picarx
from pygame import time
from display import Display
from ultralytics import YOLO
from object_detection import ObjectDetection

WIDTH = 640
HEIGHT = 480

Y = HEIGHT - 10
REGION_HEIGHT = 150
CENTER = WIDTH // 2

FPS = 30

MAX_STEERING = 35
STEERING_MULTIPLIER = 40
MAX_ERROR = WIDTH // 2


class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, error, dt):
        self.integral += error * dt
        # Optional: Clamp the integral to avoid wind-up
        self.integral = max(min(self.integral, 100), -100)

        derivative = (error - self.prev_error) / dt if dt > 0 else 0

        output = (
                self.kp * error +
                self.ki * self.integral +
                self.kd * derivative
        )

        self.prev_error = error
        return output


def process_frame(frame):
    if frame is None:
        print("Warning")
        return None, frame
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    region = gray[Y:Y + REGION_HEIGHT, :]
    blurred = cv2.GaussianBlur(region, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edge_indices = np.where(edges > 0)[1]

    if len(edge_indices) >= 2:
        line_center = (np.min(edge_indices) + np.max(edge_indices)) // 2
        print(f"line_center: {line_center}")

        error = CENTER - line_center
        print(f"error: {error}")

        return error, frame
    else:
        return None, frame


pid = PID(kp=0.9, ki=0, kd=0)

running = True

px = Picarx()
clock = time.Clock()

camera = Camera(
    size=(640, 480),  # Resolution (width, height)
    vflip=True,  # Vertical flip
    hflip=True  # Horizontal flip
)
camera.start()
camera.show_fps(False)



object_detection = ObjectDetection(
    camera=camera,
    model_filename='my_yolo.pt',
    model_image_size=(224, 224),
    is_image_thread=True
)

display = Display(object_detection)
display.show(
    local=False,  # Show in local window
    web=True,  # Enable web streaming
    port=9000  # Port for web streaming
)

object_detection.start()

timer = 0
frame = None
dt = 0
stopped = False
parked = False
while running:

    if timer > 10:



        if 0 in object_detection.detected_classes:
            px.forward(0)
            if not parked:
                print('PARKING')
                parked = True
        elif 1 in object_detection.detected_classes:
            px.forward(0)
            if not stopped:
                print('STOPPING')
                stopped = True
        else:
            stopped = False
            parked = False

            frame = camera.get_image()
            error, processed_frame = process_frame(frame)

            if error is not None:
                px.forward(0.1)
                raw_steering = pid.compute(error, dt)
                print(f"raw_steering: {raw_steering}")
                scaled_steering = np.clip((raw_steering / MAX_ERROR) * STEERING_MULTIPLIER, -MAX_STEERING, MAX_STEERING)
                px.set_dir_servo_angle(-scaled_steering)
                print(f"scaled_steering: {scaled_steering}")
            else:
                print("Line not detected")
                px.set_dir_servo_angle(0)
                px.forward(0)





    timer += 1

    dt = clock.tick(FPS) / 1000
