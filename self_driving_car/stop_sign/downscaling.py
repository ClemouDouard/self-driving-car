from PIL import Image
import os

dir = "self_driving_car/stop_sign/drive-download-20250623T132809Z-1-001"
num_im = 1

for file in os.listdir(dir):
    filename = os.fsdecode(file)
    im = Image.open(rf"self_driving_car/stop_sign/drive-download-20250623T132809Z-1-001/{filename}")

    im = im.resize((640,480))
    im.save(f"self_driving_car/stop_sign/stop_sign{num_im}.jpg")

    num_im += 1
