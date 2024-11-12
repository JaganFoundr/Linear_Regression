#MORE COMPLEX LINEAR REGRESSION

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
import torch.nn.functional as F
import numpy as np


inputs=np.array([[73,67,43],[91,88,64],[87,134,58],
                 [102,43,37],[69,96,70],[73,67,43],
                 [91,88,64],[87,134,58],[102,43,37],
                 [69,96,70],[73,67,43],[91,88,64],
                 [87,134,58],[102,43,37],[69,96,70]],dtype='float32')

targets=np.array([[56,70],[81,101],[119,133],
                 [22,37],[103,119],[56,70],
                 [81,101],[119,133],[22,37],
                 [103,119],[56,70],[81,101],
                 [119,133],[22,73],[103,119]],dtype='float32')

inputs=torch.tensor(inputs)
targets=torch.tensor(targets)

training_data=TensorDataset(inputs,targets)

batch_size = 5

training_data=DataLoader(training_data, batch_size, shuffle=True )

model=nn.Linear(3,2)

list(model.parameters())

loss_fn = F.mse_loss

optimizer=torch.optim.SGD(model.parameters(),lr=0.00001)

def train_function(nepochs, model, loss_fn, optimizer):
  for epochs in range(nepochs):
    for x,y in training_data:

      prediction = model(x)

      loss = loss_fn(prediction, y)

      loss.backward()

      optimizer.step()

      optimizer.zero_grad()

    if (epochs+1)%10==0:
      print(f"epochs: {epochs+1}/{nepochs} , loss: {loss.item()}")

train_function(1000, model, loss_fn, optimizer)

prediction=model(inputs)

print(prediction)

print(targets)
