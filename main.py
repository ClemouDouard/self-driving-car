from picarx import Picarx
from camera import Camera
from display import Display
from pygame import time
from pygame import mixer
from robot_hat import PWM, Music, Buzzer, set_volume, enable_speaker, disable_speaker
import os
import numpy as np
import cv2
from ultralytics import YOLO

WIDTH = 640
HEIGHT = 480

Y = HEIGHT - 100
REGION_HEIGHT = 50
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

pid = PID(kp=0.9, ki=0, kd=0)

def main():
    px = Picarx()
    clock = time.Clock()
    FPS = 30

    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO('yolov8n.pt')  # Your trained model file
    print("Model loaded successfully!")

    timer = 0

    try:
        # Initialize camera
        camera = Camera(
            size=(640, 480),
            vflip=True,
            hflip=True
        )

        print("Starting camera...")
        camera.start()
        camera.show_fps(True)
        camera.enable_detection_overlay(confidence=True)

        # Initialize display
        display = Display(camera)
        display.show(
            local=True,
            web=True,
            port=9000
        )

        px.set_cam_tilt_angle(0)
        px.set_cam_pan_angle(0)

        while True:
            if timer > 30:
                frame = camera.get_image()

                if frame is not None:
                    error, processed_frame = process_frame(frame)

                    dt = 1/FPS
                    if error is not None:
                        raw_steering = pid.compute(error, dt)
                        print(f"raw_steering: {raw_steering}")
                        scaled_steering = np.clip((raw_steering / MAX_ERROR) * MAX_STEERING, -30, 30)
                        px.set_dir_servo_angle(-scaled_steering)
                        print(f"scaled_steering: {scaled_steering}")
                    else:
                        print("Line not detected")

    except KeyboardInterrupt:
        px.forward(0)
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        px.forward(0)
        camera.stop()

if __name__ == "__main__":
    main()
