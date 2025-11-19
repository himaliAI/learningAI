import torch

# Example dataset: 6 patients, 5 features each
# [age, bp, cholesterol, gender(0/1), smoker(0/1)]
X = torch.tensor([
    [25, 120, 200, 0, 1],
    [45, 140, 230, 1, 0],
    [35, 130, 210, 0, 0],
    [50, 150, 250, 1, 1],
    [28, 110, 190, 0, 0],
    [60, 160, 260, 1, 1]
], dtype=torch.float32)

y_true = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)

# model
import torch.nn as nn
model = nn.Sequential(
    nn.Linear(5, 8), # 5 input features to 8 hidden units
    nn.ReLU(),
    nn.Linear(8, 2) # 8 hidden units to 2 outputs (logits for binary classes)
)
    # for linear regression: model = nn.Linear(5, 1) # no hidden layers, no activation

# loss function: CrossEntropyLoss for classification
loss_fn = nn.CrossEntropyLoss()
    # for linear regression: loss_fn = nn.MSELoss()

# optimizer: we will use Adam
import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=0.01)
    
# training loop
for epoch in range(100):
    # forward pass
    logits = model(X)
    loss = loss_fn(logits, y_true)

    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# prediction: after training, convert logits to probabilities to predicted class
with torch.no_grad():
    logits = model(X)
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)

print(f"Probalilities:\n{probs}")
print(f"Predicted classes:\n{preds}")