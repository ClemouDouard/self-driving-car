import cv2
import numpy as np
from camera import Camera
from picarx import Picarx
from pygame import time
from display import Display

WIDTH = 640
HEIGHT = 480

Y = 0
REGION_HEIGHT = 100
CENTER = WIDTH//2

FPS = 30

MAX_STEERING = 30
MAX_ERROR = WIDTH // 2

class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, error):
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    region = gray[Y:Y+REGION_HEIGHT, :]
    blurred = cv2.GaussianBlur(region, (5,5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edge_indices = np.where(edges > 0 )[1]

    if len(edge_indices) >= 2:
        line_center = (np.min(edge_indices) + np.max(edge_indices)) // 2
        print(f"line_center: {line_center}")

        error = CENTER - line_center
        print(f"error: {error}")

        return error, frame
    else:
        return None, frame

pid = PID(kp=0.4, ki=0.01, kd=0.1)

running = True

px = Picarx()
clock = time.Clock()
camera = Camera(
    size=(640, 480),  # Resolution (width, height)
    vflip=True,  # Vertical flip
    hflip=True  # Horizontal flip
)
camera.start()
camera.show_fps(True)

display = Display(camera)
display.show(
    local=True,  # Show in local window
    web=True,  # Enable web streaming
    port=9000  # Port for web streaming
)

px.forward(1)
timer = 0

while running:

    if timer > 10:
        frame = camera.get_image()

        error, processed_frame = process_frame(frame)
        dt = 1/FPS
        if error is not None:
            raw_steering = pid.compute(error)
            print(f"raw_steering: {raw_steering}")
            scaled_steering = np.clip((raw_steering / MAX_ERROR) * MAX_STEERING, -30, 30)
            px.set_dir_servo_angle(scaled_steering)
            print(f"scaled_steering: {scaled_steering}")
        else:
            print("Line not detected")
    else:
        timer += 1

    clock.tick(FPS)
