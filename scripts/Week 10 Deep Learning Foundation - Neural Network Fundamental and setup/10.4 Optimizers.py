import torch
import torch.nn as nn
import torch.optim as optim

# Simple dataset: y = 2x + 1
x = torch.tensor([[1.0],
                  [2.0],
                  [3.0]])
y_true = torch.tensor([[3.0],
                       [5.0],
                       [7.0]])

# model
model_sgd = nn.Linear(1, 1)
model_adam = nn.Linear(1, 1)

# loss function
loss_fn = nn.MSELoss()

# Optimizers
optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.01)
optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.01)

# Training loop
for step in range(10):
    # SGD
    optimizer_sgd.zero_grad()
    loss_sgd = loss_fn(model_sgd(x), y_true)
    loss_sgd.backward()
    optimizer_sgd.step()

    # Adam
    optimizer_adam.zero_grad()
    loss_adam = loss_fn(model_adam(x), y_true)
    loss_adam.backward()
    optimizer_adam.step()

    print(f"Step {step}: SGD Loss = {loss_sgd.item():.4f}, Adam Loss = {loss_adam.item():.4f}")
