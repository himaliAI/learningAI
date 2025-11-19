import torch

# training data
x = torch.tensor([[1.0],
                  [2.0],
                  [3.0],
                  [4.0]])
y_true = torch.tensor([[3.0],
                       [5.0],
                       [7.0],
                       [9.0]])

# model
import torch.nn as nn
model = nn.Linear(1, 1) # 1 input feature, 1 output target
    # nn.Linear(n_features, n_classes) for classification

# loss function
loss_fn = nn.MSELoss()
    # nn.CrossEntropyLoss() for classification

# Optimizer - We are using Adam to update weights
import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=0.1)

# training loop
    # forward pass -> computes predictions
    # calculation of loss
    # backward pass -> compute gradients
    # optimizer setup -> update weights
for epoc in range(100):
    y_pred = model(x) # forward pass
    loss = loss_fn(y_pred, y_true) # calculation of loss
    optimizer.zero_grad() # PyTorch accumulates gradients, without this step gradients from previous batches would add up
    loss.backward() # backward pass to compute gradients, add new gradients to whatever is already stored
    optimizer.step() # update weights
    print(f"Epoch {epoc}: Loss = {loss.item():.4f}")

# Evaluation:
print(f"Final prediction: {model(x).detach()}")