import matplotlib.pyplot as plt
import torch
import math


# Activation Function
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def forwardProp(input, bias=5):
    torch.manual_seed(42)
    X = torch.rand(3, 3, 3)
    # Create an empty tensor to store multiplied values
    Z = torch.zeros_like(X)
    for i in range(len(input)):
        for j in range(len(input[i][0])):
            for c in range(len(input[i][j])):
                Z[i][j][c] = X[i][j][c] * input[i][j][c]
    Z += bias
    return sigmoid(Z)




# Manually calculating L1 Loss without using any methods
def L1_loss(predictions, targets):
    total_diff = 0
    num_elements = 0
    
    # Iterate over all elements and calculate absolute difference
    for i in range(len(predictions)):
        for j in range(len(predictions[i])):
            for c in range(len(predictions[i][j])):
                total_diff += abs(predictions[i][j][c] - targets[i][j][c])  # Calculate absolute difference
                num_elements += 1  # Count the number of elements
    
    # Return the mean absolute difference
    return total_diff / num_elements



# Manually implementing a Gradient Descent Optimizer
def gradient_descent_optimizer(X, loss, learning_rate=0.01):
    # Compute the gradients (derivative of L1 loss with respect to X)
    gradients = torch.sign(X)  # The derivative of L1 loss is the sign of the input (1 or -1)
    
    # Update X using the gradient and learning rate
    X = X - learning_rate * gradients
    
    return X



image_tensor = torch.tensor([
    [ [255, 0, 0], [0, 255, 0], [0, 0, 255]],  # Row 1: Red, Green, Blue
    [[255, 255, 0], [0, 255, 255], [255, 0, 255]],  # Row 2: Yellow, Cyan, Magenta
    [[128, 128, 128], [0, 0, 0], [255, 255, 255]]
])  # Row 3: Grey, Black, White

output = forwardProp(image_tensor)

print(output)


# Calculate L1 Loss between output and image_tensor
loss = L1_loss(output, image_tensor.float())  # Convert image_tensor to float for loss calculation
print(f"L1 Loss: {loss:.4f}")