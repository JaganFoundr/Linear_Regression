import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define weight, bias, and create data
weight = 0.5
bias = 0.2
x = torch.arange(0, 10, 0.0002).unsqueeze(dim=1).to(device)
y = (weight * x + bias).to(device)

# Split data into training and testing
train_split = int(0.8 * len(x))
x_train, y_train = x[:train_split], y[:train_split]
x_test, y_test = x[train_split:], y[train_split:]

# Plotting function
def plot_data_with_predictions(x_train, y_train, x_test, y_test, y_pred=None):
    plt.figure(figsize=(20, 10))

    # Plot training data
    plt.scatter(x_train.cpu(), y_train.cpu(), color='red', label='Training Data', alpha=0.7)

    # Plot test data
    plt.scatter(x_test.cpu(), y_test.cpu(), color='green', label='Test Data', alpha=0.7)

    # Plot predictions if provided
    if y_pred is not None:
        plt.scatter(x_test.cpu(), y_pred.cpu(), color='blue', label='Predictions', alpha=0.7, marker='x', s=100)

    # Add labels, legend, and title
    plt.title("Training vs Test Data (and Predictions)", fontsize=14)
    plt.xlabel("X Values (Features)", fontsize=12)
    plt.ylabel("Y Values (Targets)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # Show plot
    plt.show()

# Visualize data
plot_data_with_predictions(x_train, y_train, x_test, y_test)

# Define Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear1(x)

# Set random seed and initialize model
torch.manual_seed(42)
model = LinearRegression().to(device)
print(model.state_dict())  # Check model parameters

# Make initial predictions with the model
with torch.inference_mode():
    test_prediction = model(x_test)

# Visualize data with initial predictions
plot_data_with_predictions(x_train, y_train, x_test, y_test, test_prediction)

# Define Loss Function and Optimizer
loss_function = nn.L1Loss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Print initial predictions and true values
print("Initial Predictions:", test_prediction[:10])
print("True Test Values:", y_test[:10])

# Training the model
epochs = 100
for epoch in range(epochs):
    # Training phase
    model.train()
    train_pred = model(x_train)
    train_loss = loss_function(train_pred, y_train)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # Evaluation phase
    model.eval()
    with torch.inference_mode():
        test_pred = model(x_test)
        test_loss = loss_function(test_pred, y_test)

    # Print progress every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}: Training Loss: {train_loss.item():.4f}, Testing Loss: {test_loss.item():.4f}")

# Visualize data with final predictions
plot_data_with_predictions(x_train, y_train, x_test, y_test, test_pred)

# Check predictions vs true values
print("Final Predictions:", test_pred[:10])
print("True Train Values:", y_train[:10])
print("True Test Values:", y_test[:10])

# Save the model
torch.save(model.state_dict(), 'LinearRegression.pth')
print("Model saved successfully!")

# Load the model for testing
saved_model = LinearRegression().to(device)
saved_model.load_state_dict(torch.load('LinearRegression.pth'))
print("Loaded Model Parameters:", saved_model.state_dict())
