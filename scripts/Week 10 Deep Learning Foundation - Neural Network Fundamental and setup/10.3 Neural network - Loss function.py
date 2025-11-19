# compare MSE (regression) vs. Cross-Entropy (classification) in PyTorch
import torch
import torch.nn as nn

'''
# example: Predict y = 2x + 1
x = torch.tensor([[1.0],
                  [2.0],
                  [3.0]])
y_true = torch.tensor([[3.0],
                       [5.0],
                       [7.0]])
# Simple linear model
model = nn.Linear(1, 1) 
    # (1, 1) stands for in_features and out_features respectively.
    # in our case, we have 1 input feature and 1 output target value, so both dimensions are 1

# Loss function: MSE
loss_fn = nn.MSELoss()

# forward pass
y_pred = model(x)
loss = loss_fn(y_pred, y_true)

print(f"Predictions: {y_pred}")
print(f"MSE Loss: {loss.item()}")
'''

# Classification with cross-Entropy
# Example: 3 classes, true label = class 2
logits = torch.tensor([[2.0, 1.0, 0.1]]) # raw scores
y_true = torch.tensor([0]) # class index from logits(0 = first class)

# loss function: CorssEntropy
loss_fn = nn.CrossEntropyLoss()

loss = loss_fn(logits, y_true)

print(f"Logits: {logits}") # logits are raw outputs
print(f"Cross-Entropy Loss: {loss.item()}") # CrossEntropyLoss internally applies softmax + log likelihood
