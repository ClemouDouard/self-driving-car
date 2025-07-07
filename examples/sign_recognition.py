import torch
from torchvision import transforms
from PIL import Image
from picarx import Picarx
from pygame import time

from self_driving_car.models.sign import SignClassifier
from camera import Camera

sign_model = SignClassifier()
sign_model.load_state_dict(torch.load("../sign_classifier.pth"))
sign_model.eval()

running = True

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

px = Picarx()
clock = time.Clock()
camera = Camera(
    size=(640, 480),  # Resolution (width, height)
    vflip=False,  # Vertical flip
    hflip=False  # Horizontal flip
)
camera.start()

while running:

    img = camera.get_image()
    img = Image.fromarray(img).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = sign_model(img_tensor)
        _, predicted = torch.max(output, 1)

    if predicted.item() == 0:
        print("parking")
    else:
        print("stop")

    clock.tick(1)

camera.stop()
