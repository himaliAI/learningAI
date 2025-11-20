import torch

# Features: [age, bp, cholesterol, gender(0/1), smoker(0/1)]
X = torch.tensor([
    [25, 120, 200, 0, 1],
    [45, 140, 230, 1, 0],
    [35, 130, 210, 0, 0],
    [50, 150, 250, 1, 1],
    [28, 110, 190, 0, 0],
    [60, 160, 260, 1, 1]
], dtype=torch.float32)

# Labels: 0 = low risk, 1 = high risk
y_true = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)

# model
import torch.nn as nn
model = nn.Sequential(
    nn.Linear(5,10),
    nn.ReLU(),
    nn.Linear(10, 2)
)

# loss function
loss_fn = nn.CrossEntropyLoss()

# optimizer: Adam for adaptive learning
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

# evaluation
    # logits -> probabilities -> predicted class
with torch.no_grad():
    logits = model(X)
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)

print(f"Probabilites: {probs}")
print(f"Predicted classes: {preds}")

# Metrics
accuracy = (preds == y_true).float().mean()
print(f"Accuracy: {accuracy.item()}")

