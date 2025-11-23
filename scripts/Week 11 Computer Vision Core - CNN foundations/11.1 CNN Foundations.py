# CNN (Convolutional neural network) core

# Define CNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 1. device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Transformations (normalize MNIST to [-1, 1])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 3. Load datasets
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# 4. DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 5. Define CNN model
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        
        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3, 3, 1)

        # fully connected layers
        self.fc1 = nn.Linear(64*7*7, 128) # after two pooling image size reduces to 7x7
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # Convolution + ReLU + Pooling
        x = self.pool(F.relu(self.conv1(x))) # [batch, 32, 14, 14]
        x = self.pool(F.relu(self.conv2(x))) # [batch, 64, 7, 7]

        # flatten
        x = x.view(-1, 64*7*7)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x 


# 6. Training setup (model, loss, optimizer)
model = CNNClassifier().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. training loop
n_epochs = 5
for epoch in range(n_epochs):
    # training
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

    # validation
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    print(f"Epoch {epoch+1}: Val Loss = {val_loss/len(test_loader):.4f}")
    print(f"Val Acc = {100*vorrect/len(test_loader.dataset):.2f}%")
    