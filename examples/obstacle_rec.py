import torch
from torchvision import transforms
from PIL import Image
from picarx import Picarx
from pygame import time

from self_driving_car.models.forward import ForwardClassifier
from camera import Camera

model = ForwardClassifier()
model.load_state_dict(torch.load("../forward_classifier.pth"))
model.eval()

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
    
    camera.take_photo("live_photo")
    img = Image.open("live_photo.jpg").convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    
    if predicted.item() == 0:
        print("car can go")
        px.forward(1)
    else:
        print("car cannot go")
        ### IMPL other img classifier
    
    clock.tick(20)