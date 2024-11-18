# ----------------------------------------------
# IMPORTING LIBRARIES
# ----------------------------------------------
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np

# ----------------------------------------------
# DATA PREPARATION
# ----------------------------------------------

# Inputs (features: temperature, rainfall, humidity)
inputs = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58],
                   [102, 43, 37], [69, 96, 70], [73, 67, 43],
                   [91, 88, 64], [87, 134, 58], [102, 43, 37],
                   [69, 96, 70], [73, 67, 43], [91, 88, 64],
                   [87, 134, 58], [102, 43, 37], [69, 96, 70]], dtype='float32')

# Targets (outputs: apples, oranges)
targets = np.array([[56, 70], [81, 101], [119, 133],
                    [22, 37], [103, 119], [56, 70],
                    [81, 101], [119, 133], [22, 37],
                    [103, 119], [56, 70], [81, 101],
                    [119, 133], [22, 73], [103, 119]], dtype='float32')

# Convert inputs and targets to PyTorch tensors
inputs = torch.tensor(inputs)
targets = torch.tensor(targets)

# Combine inputs and targets into a dataset
training_data = TensorDataset(inputs, targets)

# ----------------------------------------------
# DATA LOADER
# ----------------------------------------------

# Set batch size
batch_size = 5

# DataLoader for shuffling and batching the data
training_data = DataLoader(training_data, batch_size=batch_size, shuffle=True)

# ----------------------------------------------
# MODEL SETUP
# ----------------------------------------------

# Linear regression model: 3 input features → 2 output features
model = nn.Linear(3, 2)

# Loss function: Mean Squared Error
loss_fn = F.mse_loss

# Optimizer: Stochastic Gradient Descent (SGD)
optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)

# ----------------------------------------------
# TRAINING FUNCTION
# ----------------------------------------------

def train_function(nepochs, model, loss_fn, optimizer):
    for epoch in range(nepochs):
        for x, y in training_data:  # Iterate over batches
            # Forward pass: predict using the model
            prediction = model(x)

            # Calculate the loss
            loss = loss_fn(prediction, y)

            # Backward pass: compute gradients
            loss.backward()

            # Update parameters
            optimizer.step()

            # Zero the gradients after updating
            optimizer.zero_grad()

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{nepochs}], Loss: {loss.item():.4f}")

# Train the model for 1000 epochs
train_function(1000, model, loss_fn, optimizer)

# ----------------------------------------------
# MODEL PREDICTIONS
# ----------------------------------------------
prediction = model(inputs)

# Print predictions and targets for comparison
print("\nModel Predictions:\n", prediction)
print("\nTarget Values:\n", targets)
