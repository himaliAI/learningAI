# CIFAR-10 is a color image, so need to consider three channels - RGB

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 1. Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2. Transformations (normalize CIFAR-10 to [-1,1])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 3 channels
])

# 3. Load datasets
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)

# 4. DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 5. Define CNN model
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)   # input: 3 channels
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)   # after 3 poolings, 32x32 → 4x4
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # Input: [batch, 3, 32, 32]
        x = self.pool(F.relu(self.conv1(x)))   # [batch, 32, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))   # [batch, 64, 8, 8]
        x = self.pool(F.relu(self.conv3(x)))   # [batch, 128, 4, 4]
        x = x.view(-1, 128 * 4 * 4)            # flatten → [batch, 2048]
        x = F.relu(self.fc1(x))                # [batch, 256]
        x = self.fc2(x)                        # [batch, 10]
        return x
    
# 6. Initialize model, loss, optimizer
model = CIFAR10CNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. Training loop
n_epochs = 10
for epoch in range(n_epochs):
    # Training
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(f"Epoch {epoch+1}: Val Loss={val_loss/len(test_loader):.4f}, "
          f"Val Acc={100*correct/len(test_loader.dataset):.2f}%")
    
# 8. Shape transformation table (for reference)
print("\nLayer-by-layer shape transformations:")
print("Input:        [batch, 3, 32, 32]")
print("Conv1+Pool →  [batch, 32, 16, 16]")
print("Conv2+Pool →  [batch, 64, 8, 8]")
print("Conv3+Pool →  [batch, 128, 4, 4]")
print("Flatten →     [batch, 2048]")
print("FC1 →         [batch, 256]")
print("FC2 →         [batch, 10]")