import torch
import torch.nn as nn
import torch.optim as optim
import torchvision # we will use torchvision.dataset.MNIST which handles downloading and preprocessing
import torchvision.transforms as transforms

# device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transform: convert to tensor + normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
        # MNIST images are grayscale, and after toTensor() each pixel value is between 0 and 1
        # .Normalize((0.5), (0.5)) normalize them to mean and SD of 0.5 each
        # Original pixel range: [0, 1]; after normalization: [-1, 1]
])

# load train and test sets
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # train_dataset has len(train_dataset) no. of items. each item is a tuple of (image_tensor, label)
    # any sample in train_dataset can be assessed by {image, label = train_dataset[index]}
        # image is torch.Size([1, 28, 28]) i.e 1 grayscale channel, 28x28 heightxwidth
        # label is any number between 0 to 9

# Data Loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    # data_loaders (train_loader) feeds batches of data from dataset (eg train_dataset)
    # it may randomly shuffle data while feeding
    # also provide easy loop without manually indexing
    # and, can load data parallelly (multiple threads)

# Define models
    # input: 28 * 28 = 784 features
    # hidden: 128 units
    # output: 10 logits (digits 0-9)
class MNISTClassifier(nn.Module): # inherit the class rom nn.Module
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.fc1 = nn.Linear(28*28, 128) # first fully connected linear layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5) # Dropout layer
            # during training, dropout randomly zeros out 50% of hidden units each batch
            # dropout is automatically disabled during evaluation i.e model.eval()
            # prevents overfitting
        self.fc2 = nn.Linear(128, 10) # second fully connected linear layer

    def forward(self, x):
        x = x.view(-1, 28*28) # flatten; PyTorch .view is similar to Numpy's .reshape()
        x = self.fc1(x) # apply first layer
        x = self.relu(x) # apply activation
        self.dropout = nn.Dropout(p=0.5) # Dropout layer
        x = self.fc2(x) # apply second layer
        return x 
    
model = MNISTClassifier().to(device)

# Loss and Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
    # if you want to use weight decay, add it to the optimizer eg weight_decay=1e-4
    # 1e-4 is typical, 
    # larger values, stronger penalty (risk of underfitting)
    # smaller values, weaker penalty (risk of overfitting) 

# Training loop
import copy

def train(model, loader, optimizer, loss_fn, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        total_loss +- loss.item()
    print(f"Epoch {epoch}: Avg train loss = {total_loss/len(loader):.4f}")

def train_with_early_stopping(model, train_loader, test_loader, optimizer, loss_fn, n_epochs=20, patience=3):
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(1, n_epochs + 1):
        # -------Training----------
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, input)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # ------ Validation -------
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += loss_fn(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            avg_val_loss = val_loss / len(test_loader)
            val_acc = 100. * correct / len(test_loader.dataset)
        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val loss = {avg_val_loss:.4f}, Val Acc = {val_acc:.2f}%")

        # ------ Early Stopping Check -----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

def test(model, loader, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loader)
    accuracy = 100. * correct / len(loader.dataset)
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Run training
n_epochs = 5
for epoch in range(1, n_epochs+1):
    train(model, train_loader, optimizer, loss_fn, epoch)
    test(model, test_loader, loss_fn)

