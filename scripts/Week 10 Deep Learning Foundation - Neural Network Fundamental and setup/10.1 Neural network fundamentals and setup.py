# Tensors in PyTorch
import torch

# scaler
scaler = torch.tensor(5)
print(scaler, scaler.shape) # tensor(5), torch.Size([])

# Vector
vector = torch.tensor([1, 2, 3])
print(vector, vector.shape) # tensor([1, 2, 3]), torch.Size([3])

# matrix
matrix = torch.tensor([[1, 2],
                       [3, 4]])
print(matrix, matrix.shape) # tensor([[1, 2], [3, 4]]), torch.Size([2, 2])

# 3D tensor (like a stack of matrices)
tensor3D = torch.rand(2, 3, 4) # shape -> 2 blocks, 3 rows, 4 cols
print(tensor3D.shape) # torch.Size([2, 3, 4])

# GPU usage with .to(device)
# check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create tensor on CPU
x = torch.rand(3, 3)
print(f"CPU tensor: {x.device}")

# move tensor to GPU if available
x_gpu = x.to(device)
print(f"Tensor on: {x_gpu.device}")