# Transfer Learning with ResNet18 on CIFAR-10

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models

# 2. Transformations (standard normalization for CIFAR-10)
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#2.5 device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 3. Load datasets
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=train_transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=test_transform
)

# 4. DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 5. Load pretrained ResNet18
resnet18 = models.resnet18(pretrained=True)

# Freeze all layers except the final classifier
for param in resnet18.parameters():
    param.requires_grad = False

# Replace the final fully connected layer (ImageNet has 1000 classes, CIFAR-10 has 10)
num_features = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_features, 10)

model = resnet18.to(device)

# 6. Loss and optimizer (only train the new classifier head)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 7. Training loop
n_epochs = 5
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

# pretrained ResNet18 - already trained on Imagenet (1.2M images, 1000 classes)
# Freezing layers - keeps the learned filters (edges, textures, shapes)
# Replacing classifier head: adapts the model to CIFAR-10's 10 classes
# Training only the head: faster, less risk of overfitting on small data
