from PIL import Image
import os

dir = "self_driving_car/stop_sign/parking_sign_raw"
num_im = 1

for file in os.listdir(dir):
    filename = os.fsdecode(file)
    im = Image.open(rf"{dir}/{filename}")

    im = im.resize((640,480))
    im.save(f"self_driving_car/stop_sign/parking_sign{num_im}.jpg")

    num_im += 1
