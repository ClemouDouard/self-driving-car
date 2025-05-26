import torch
from torchvision import transforms
from PIL import Image

from self_driving_car.models.forward import ForwardClassifier

model = ForwardClassifier()
model.load_state_dict(torch.load("../forward_classifier.pth"))
model.eval()

running = True

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

while running:
    ### IMG GETTING ###
    ### img = 
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    
    if predicted.item() == 0:
        print("car can go")
        ### Forward car ###
    else:
        print("car cannot go")
        ### IMPL other img classifier