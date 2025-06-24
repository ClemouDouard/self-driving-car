import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

class SignClassifier(nn.Module):
    def __init__(self):
        super(SignClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
    ])

    dataset = ImageFolder(root="self_driving_car/sign", transform=transform)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    device = torch.device("cpu")
    model = SignClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 10
    for epoch in range(n_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"epoch {epoch+1}/{n_epochs}, perte {running_loss:.4f}")

        torch.save(model.state_dict(), "sign_classifier.pth")
        print("model saved")
