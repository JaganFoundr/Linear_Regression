# ----------------------------------------------
# IMPORTING LIBRARIES
# ----------------------------------------------
import torch
import numpy as np

# ----------------------------------------------
# DATA PREPARATION
# ----------------------------------------------

# Inputs (features: temperature, rainfall, humidity)
inputs = np.array([[73, 67, 45],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70]], dtype='float32')

# Targets (outputs: apples, oranges)
targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119]], dtype='float32')

# Convert inputs and targets to PyTorch tensors
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

# ----------------------------------------------
# INITIALIZING MODEL PARAMETERS
# ----------------------------------------------

# Randomly initialize weights and biases
weights = torch.randn(2, 3, requires_grad=True)
bias = torch.randn(2, requires_grad=True)

# ----------------------------------------------
# MODEL AND LOSS FUNCTION
# ----------------------------------------------

# Define the linear regression model
def model(x):
    return x @ weights.t() + bias

# Define the Mean Squared Error (MSE) loss function
def mse(x, y):
    diff = x - y
    return torch.sum(diff * diff) / diff.numel()

# ----------------------------------------------
# TRAINING THE MODEL
# ----------------------------------------------

# Training loop: 1000 epochs
for epoch in range(1000):
    # Forward pass: predict using the model
    prediction = model(inputs)

    # Calculate the loss
    loss = mse(prediction, targets)

    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/1000], Loss: {loss.item():.4f}")

    # Backward pass: compute gradients
    loss.backward()

    # Update weights and biases
    with torch.no_grad():
        lr = 0.00001  # Learning rate

        # Update weights
        weights -= weights.grad * lr

        # Update bias
        bias -= bias.grad * lr

        # Zero the gradients after updating
        weights.grad.zero_()
        bias.grad.zero_()

# ----------------------------------------------
# FINAL LOSS
# ----------------------------------------------
print(f"Final Loss: {loss.item():.4f}")
