# scripts/track_face.py

from picarx import Picarx
from time import sleep
from pygame import time
from roboeye.main import RoboEye

def clamp(value, min_val, max_val):
    return max(min(value, max_val), min_val)

def main():
    px = Picarx()
    robo = RoboEye(use_face_detection=True)
    clock = time.Clock()
    x_angle = y_angle = 0

    try:
        robo.start()

        while True:
            robo.update()
            _, face = robo.detect_face()

            if face["n"] > 0:
                x, y = face["x"], face["y"]

                x_angle += (x * 10 / 640) - 5
                x_angle = clamp(x_angle, -35, 35)
                px.set_cam_pan_angle(x_angle)

                y_angle -= (y * 10 / 480) - 5
                y_angle = clamp(y_angle, -35, 35)
                px.set_cam_tilt_angle(y_angle)

            clock.tick(30)

    finally:
        robo.stop()
        px.stop()
        print("Stopped and cleaned up.")
        sleep(0.1)

if __name__ == "__main__":
    main()
