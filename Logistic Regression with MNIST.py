#LOGISTIC REGRESSION USING MNIST

#importing libraries  
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import SubsetRandomSampler, DataLoader

#main dataset and testdata
dataset=MNIST(download=True, train=True, root="./data", transform=transforms.ToTensor())
testset=MNIST(root="./data", train=False, transform=transforms.ToTensor())

#plotting the dataset
image,labels=dataset[1000]
plt.imshow(image[0,10:25,10:25], cmap="gray")
plt.show()
print("label: ", labels)

#splitting the whole dataset into validation data and training data
def split_data(dataset, validation_percent):
  validation_data=int(dataset*validation_percent)
  shuffled=np.random.permutation(dataset)
  return shuffled[validation_data:], shuffled[:validation_data]

training_data,validation_data = split_data(len(dataset), 0.3)
print("length of training data: ", len(training_data))
print("length of validation data: ", len(validation_data))

print("portion of validation data: ",validation_data[:20])

#putting all the splitted data to the sampler and then into the dataloader
train_data_sampler=SubsetRandomSampler(training_data)
valid_data_sampler=SubsetRandomSampler(validation_data)

batch_size=100

training_loader=DataLoader(dataset, batch_size, sampler=train_data_sampler)
validation_loader=DataLoader(dataset, batch_size, sampler=valid_data_sampler)

#defining the model
input_size=28*28
num_classes=10

class MNISTmodel(nn.Module):
  def __init__(self):
    super().__init__()
    self.Linear=nn.Linear(input_size,num_classes)

  def forward(self, size):
    size=size.reshape(-1,784)
    output=self.Linear(size)
    return output

model=MNISTmodel()

#putting the training loader in the for loop as inputs and outputs for prediction
for images, labels in training_loader:
  prediction=model(images)

#display predictions and sum of the predictions of each array
print(prediction[:2])
sum=torch.sum(prediction[2])
print(sum)

#changing the sum of the probablities of the predictions close to 1 and then checking the sum again
prob=F.softmax(prediction)
print(prob[:2])
sum=torch.sum(prob[2])
print(sum)

#displaying the exact predicted labels by the model
max_prob, pred = torch.max(prob, dim=1)
print(pred)
print(max_prob)

#displaying the actual target labels
print(labels)

#defining the loss
loss_fn=F.cross_entropy

#defining the optimizer
opt=torch.optim.SGD(model.parameters(), lr=0.0001)

# accuracy metrics
def accuracy(outputs, labels):
  _,pred=torch.max(outputs, dim=1)
  return (torch.sum(pred==labels).item()/len(pred))*100

# loss batch function for loss computation, gradient computation, updating weights, resetting gradients, accuracy computation
def loss_batch(model, loss_fn, images, labels, opt, metrics=accuracy):
  prediction=model(images)
  loss=loss_fn(prediction, labels)

  if opt is not None:    
    loss.backward()
    opt.step()
    opt.zero_grad()

  metric_result=None
  if metrics is not None:
    metric_result=metrics(prediction, labels)

  return loss.item(), len(images), metric_result

# function for evaluating the average loss and average accuracy of the validation set
def evaluate(model, loss_fn, validation_loader, metrics=accuracy):
  with torch.no_grad():
    validation_prediction=[loss_batch(model, loss_fn, images, labels, opt=None, metrics=accuracy) for images, labels in validation_loader]

    losses, nums, metric=zip(*validation_prediction)

    total=np.sum(nums)

    average_loss = np.sum(np.multiply(losses, nums))/total

    average_metrics=None
    if metrics is not None:
      average_metrics = np.sum(np.multiply(metric, nums))/total

  return average_loss.item(), total, average_metrics

#function for explicit training
def fit(nepochs, model, images, labels, training_loader, validation_loader, opt, metrics=accuracy):
  for epoch in range(nepochs):
    for images, labels in training_loader:
      train_loss,_, train_accuracy=loss_batch(model, loss_fn, images, labels, opt, metrics=accuracy)

    valid_loss, _, valid_accuracy= evaluate(model, loss_fn, validation_loader, metrics=accuracy)

    print(f"Epoch: {epoch+1}/{nepochs}")
    print(f"Training loss: {train_loss:.4f} and Validation loss: {valid_loss:.4f}.")
    print(f"Training accuracy: {train_accuracy:.2f}% and Validation accuracy: {valid_accuracy:.2f}%.")
    print("--------------------------------------------------------------------------------------------")

  return train_loss, _, train_accuracy, valid_loss, _, valid_accuracy

train_loss,_, train_accuracy, valid_loss, _, valid_accuracy = fit(6, model, images, labels, training_loader, validation_loader, opt, metrics=accuracy)

print("--")
print(f"The train accuracy is {train_accuracy:.2f} % and loss is {train_loss:.4f}.")
print("--------------------------------------------")
print(f"The validation accuracy is {valid_accuracy:.2f} % and loss is {valid_loss:.4f}")

# testing the model with the testing dataset
#function for predicting the test images
def predict_image(image, model):
  input=image.unsqueeze(0)
  output=model(input)
  _,preds=torch.max(output, dim=1)

  return preds[0].item()

#predicting and displaying different labels
image,labels=testset[10]
plt.imshow(image[0], cmap="gray")
plt.show()
print("label: ", labels)
print("predicted: ", predict_image(image, model))

image,labels=testset[100]
plt.imshow(image[0], cmap="gray")
plt.show()
print("label: ", labels)
print("predicted: ", predict_image(image, model))

image,labels=testset[1000]
plt.imshow(image[0], cmap="gray")
plt.show()
print("label: ", labels)
print("predicted: ", predict_image(image, model))

image,labels=testset[905]
plt.imshow(image[0], cmap="gray")
plt.show()
print("label: ", labels)
print("predicted: ", predict_image(image, model))

#checking the loss and accuracy on the test set
test_loader=DataLoader(testset, batch_size=200)
test_loss, total, test_accuracy=evaluate(model, loss_fn, test_loader, metrics=accuracy)
print(f"The test set loss is {test_loss:.4f} and the accuracy is {test_accuracy:.2f}%.")

#saving and loading the model
torch.save(model.state_dict(),'MNISTlogistic.pth')
model.state_dict()

savedmodel=MNISTmodel()
savedmodel.load_state_dict(torch.load('MNISTlogistic.pth'))
savedmodel.state_dict()
