#importing all the important libraries
import torch
import numpy as np

#inputs (temperature, rainfall, humidity)
inputs=np.array([[73,67,45],
                 [91,88,64],
                 [87,134,58],
                 [102,43,37],
                 [69,96,70]],dtype='float32')

#targets (apples, oranges)
targets=np.array([[56,70],
                 [81,101],
                 [119,133],
                 [22,37],
                 [103,119]],dtype='float32')

#converting inputs and targets from numpy format to tensor format
inputs=torch.from_numpy(inputs)
targets=torch.from_numpy(targets)

# defining random weights and biases
weights=torch.randn(2,3,requires_grad=True)
bias=torch.randn(2,requires_grad=True)

#function for the prediction equation for the model
def model(x):
  return x @ weights.t()+bias

#function for the loss
def mse(x,y):
  diff=x-y
  return torch.sum(diff*diff)/diff.numel()

#training loop with 1000 epochs
for i in range(1000):

  #predicting using the inputs
  prediction=model(inputs)

  # checking the loss of the prediction
  loss=mse(prediction, targets)
  print(loss)

  # backpropogating to compute gradients and update the weights
  loss.backward()

  #updating the weights without tracking the gradient
  with torch.no_grad():

    #learning rate value
    lr=0.00001

    #updating weights
    weights-=weights.grad*lr

    #updating bias
    bias-=bias.grad*lr

    #emptying the gradients of weights and bias
    weights.grad.zero_()
    bias.grad.zero_()

print(loss)
