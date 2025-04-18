# Resource Notebook: https://www.learnpytorch.io/00_pytorch_fundamentals/#importing-pytorch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

print(torch.__version__)

# PyTorch tensors are created using `torch.tensor()`
scalar = torch.tensor(10)
print(scalar.ndim)
print(scalar.item())
print(scalar.shape)


vector = torch.tensor([50, 29, 50, 10])
print(vector.ndim)
print(vector.shape)


matrix = torch.tensor(
    [
        [50, 20, 60, 90],
        [50, 20, 60, 90]
    ]
)
print(matrix.ndim)
print(matrix.shape)


# Tensors are anything greater than 2D Array such as this 3D array or 4D array etc.
TENSOR3D = torch.tensor(
    [
        [
            [50, 20, 60, 90]
        ],
        [
            [50, 20, 60, 90]
        ]
    ]
)
print(TENSOR3D.ndim)
print(TENSOR3D.shape)


# Random tensors are important because the way Neural Networks learn is that they start with tensors full of random numbers and then adjust those random numbers to better represent the data.

random_image_size_tensor = torch.rand(224, 224, 3) # height, width, color channel (RGB)
print(random_image_size_tensor.shape)
print(random_image_size_tensor.ndim)


# Create a tensor of all zeros
zeros = torch.zeros(5, 5)
print(zeros)


# Create a tensor of all ones
ones = torch.ones(5, 5)
ones[:, 1] = 900
print(ones)


# Creating tensors with the same shape, and dim as another tensor (`like` another tensor)
like = torch.zeros_like(ones)
print(like)
print(like.dtype)
print(like.device)


# Creating ranges
range_1 = torch.arange(1, 21)
range_2 = torch.arange(start=0, end=1000, step=50)
print(range_1)
print(range_2)
print(range_2.dtype)


multiplication = torch.tensor([50, 30, 10, 60])

print(multiplication * 2)
# Element-wise matrix multiplication
print(torch.matmul(multiplication, multiplication)) #torch.mm is the the same as torch.matmul
print(torch.matmul(matrix, multiplication))
# Transposing a matrix
print(matrix.T)


# Tensor Aggregations (Finding the min, max, mean, sum, etc)
x = torch.arange(0, 100, 10)
print(x)
print(torch.min(x))
print(torch.max(x))
print(torch.mean(x.type(torch.float32)))
print(x.sum())


# Find the position in tensor that has the min value with argmin() -> returns index position of target tensor where the min value occurs.
print(x.argmin())
print(x[x.argmin()])

# Find the position in tensor that has the max value with argmax()
print(x.argmax())
print(x[x.argmax()])


# Add an extra dimension
print(matrix.shape)
# Valid reshape → new shape must contain the exact same number of elements as the original.
# When you multiply all the dimensions of a tensor’s shape—no matter if it’s 1D, 2D, 3D, or even higher-dimensional—you get the total number of elements in the tensor. Example, 1 * 8 = 8 total elements in matrix 2D array.
x_reshaped = matrix.reshape(1, 8)
x_reshaped[:, 2] = 200
print(x_reshaped)


# Stacking Tensors on top of each other
x_stacked = torch.stack([multiplication, multiplication, multiplication], dim=1)
print(x_stacked)


# torch.squeeze() - removes all single dimensions (1's) from a target tensors shape
x_squeezed = x_reshaped.squeeze()
print(x_squeezed)


# toch.unsqueeze() - adds a single dimension to a target tensor at a specific dimension dim=1 means add it on index 1 of the shape of x_unsqueezed
x_unsqueezed = x_squeezed.unsqueeze(dim=1)
print(x_unsqueezed)


# NumPy array to tensor
array = np.arange(1, 10)
tensor = torch.from_numpy(array)
print(array, tensor)

# tensor to numPy array
testTensor = torch.tensor([50, 40, 65])
print(testTensor)
changed_tensor = torch.Tensor.numpy(testTensor)
print(testTensor, changed_tensor)





# Set the random seed (reproducibility trying to take the random out of random)
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
random_tensor_C = torch.rand(3, 4)

torch.manual_seed(RANDOM_SEED)
random_tensor_D = torch.rand(3, 4)
print(random_tensor_C)
print(random_tensor_D)
print(random_tensor_C == random_tensor_D)


# Running tensors and pytorch objects on GPUs (and making faster computations) GPUs = faster computation on numbers, thanks to CUDA + NVIDIA hardware + PyTorch working behind the scenes to make everything good. Checking for GPU access with PyTorch

print(torch.cuda.is_available()) # This is the basic check to see if my machine has access to a CUDA-capable GPU (like my RTX 3050), and if PyTorch is correctly set up to use it.

print(torch.cuda.device_count())


# Setup device agnostic code
device = 'cuda' if torch.cuda.device_count() else 'cpu'
print(device)