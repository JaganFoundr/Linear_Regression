{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOckzp1Q1y4DSSsvC0WR8dJ",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/JaganFoundr/PyTorchNN/blob/main/Deep_Learning.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#LINEAR REGRESSION\n",
        "\n",
        "#importing all the important libraries\n",
        "import torch\n",
        "import numpy as np\n",
        "\n",
        "#inputs (temperature, rainfall, humidity)\n",
        "inputs=np.array([[73,67,45],\n",
        "                 [91,88,64],\n",
        "                 [87,134,58],\n",
        "                 [102,43,37],\n",
        "                 [69,96,70]],dtype='float32')\n",
        "\n",
        "#targets (apples, oranges)\n",
        "targets=np.array([[56,70],\n",
        "                 [81,101],\n",
        "                 [119,133],\n",
        "                 [22,37],\n",
        "                 [103,119]],dtype='float32')\n",
        "\n",
        "#converting inputs and targets from numpy format to tensor format\n",
        "inputs=torch.from_numpy(inputs)\n",
        "targets=torch.from_numpy(targets)\n",
        "\n",
        "# defining random weights and biases\n",
        "weights=torch.randn(2,3,requires_grad=True)\n",
        "bias=torch.randn(2,requires_grad=True)\n",
        "\n",
        "#function for the prediction equation for the model\n",
        "def model(x):\n",
        "  return x @ weights.t()+bias\n",
        "\n",
        "#function for the loss\n",
        "def mse(x,y):\n",
        "  diff=x-y\n",
        "  return torch.sum(diff*diff)/diff.numel()\n",
        "\n",
        "#training loop with 1000 epochs\n",
        "for i in range(1000):\n",
        "\n",
        "  #predicting using the inputs\n",
        "  prediction=model(inputs)\n",
        "\n",
        "  # checking the loss of the prediction\n",
        "  loss=mse(prediction, targets)\n",
        "  print(loss)\n",
        "\n",
        "  # backpropogating to compute gradients and update the weights\n",
        "  loss.backward()\n",
        "\n",
        "  #updating the weights without tracking the gradient\n",
        "  with torch.no_grad():\n",
        "\n",
        "    #learning rate value\n",
        "    lr=0.00001\n",
        "\n",
        "    #updating weights\n",
        "    weights-=weights.grad*lr\n",
        "\n",
        "    #updating bias\n",
        "    bias-=bias.grad*lr\n",
        "\n",
        "    #emptying the gradients of weights and bias\n",
        "    weights.grad.zero_()\n",
        "    bias.grad.zero_()\n",
        "\n",
        "print(loss)"
      ],
      "metadata": {
        "id": "CdEho-pTaSMz"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#MORE COMPLEX LINEAR REGRESSION\n",
        "\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "from torch.utils.data import TensorDataset,DataLoader\n",
        "import torch.nn.functional as F\n",
        "import numpy as np\n",
        "\n",
        "\n",
        "inputs=np.array([[73,67,43],[91,88,64],[87,134,58],\n",
        "                 [102,43,37],[69,96,70],[73,67,43],\n",
        "                 [91,88,64],[87,134,58],[102,43,37],\n",
        "                 [69,96,70],[73,67,43],[91,88,64],\n",
        "                 [87,134,58],[102,43,37],[69,96,70]],dtype='float32')\n",
        "\n",
        "targets=np.array([[56,70],[81,101],[119,133],\n",
        "                 [22,37],[103,119],[56,70],\n",
        "                 [81,101],[119,133],[22,37],\n",
        "                 [103,119],[56,70],[81,101],\n",
        "                 [119,133],[22,73],[103,119]],dtype='float32')\n",
        "\n",
        "inputs=torch.tensor(inputs)\n",
        "targets=torch.tensor(targets)\n",
        "\n",
        "training_data=TensorDataset(inputs,targets)\n",
        "\n",
        "batch_size = 5\n",
        "\n",
        "training_data=DataLoader(training_data, batch_size, shuffle=True )\n",
        "\n",
        "model=nn.Linear(3,2)\n",
        "\n",
        "list(model.parameters())\n",
        "\n",
        "loss_fn = F.mse_loss\n",
        "\n",
        "optimizer=torch.optim.SGD(model.parameters(),lr=0.00001)\n",
        "\n",
        "def train_function(nepochs, model, loss_fn, optimizer):\n",
        "  for epochs in range(nepochs):\n",
        "    for x,y in training_data:\n",
        "\n",
        "      prediction = model(x)\n",
        "\n",
        "      loss = loss_fn(prediction, y)\n",
        "\n",
        "      loss.backward()\n",
        "\n",
        "      optimizer.step()\n",
        "\n",
        "      optimizer.zero_grad()\n",
        "\n",
        "    if (epochs+1)%10==0:\n",
        "      print(f\"epochs: {epochs+1}/{nepochs} , loss: {loss.item()}\")\n",
        "\n",
        "train_function(1000, model, loss_fn, optimizer)\n",
        "\n",
        "prediction=model(inputs)\n",
        "\n",
        "print(prediction)\n",
        "\n",
        "print(targets)"
      ],
      "metadata": {
        "id": "ZJN2ACY-uplj",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "collapsed": true,
        "outputId": "daa1ec29-3977-456f-d862-4b62d1f97842"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "epochs: 10/1000 , loss: 238.8297882080078\n",
            "epochs: 20/1000 , loss: 425.7377014160156\n",
            "epochs: 30/1000 , loss: 217.41061401367188\n",
            "epochs: 40/1000 , loss: 120.44380950927734\n",
            "epochs: 50/1000 , loss: 37.2009162902832\n",
            "epochs: 60/1000 , loss: 107.33747863769531\n",
            "epochs: 70/1000 , loss: 23.355182647705078\n",
            "epochs: 80/1000 , loss: 109.0358657836914\n",
            "epochs: 90/1000 , loss: 81.80258178710938\n",
            "epochs: 100/1000 , loss: 72.1570053100586\n",
            "epochs: 110/1000 , loss: 56.62866973876953\n",
            "epochs: 120/1000 , loss: 73.27456665039062\n",
            "epochs: 130/1000 , loss: 26.875635147094727\n",
            "epochs: 140/1000 , loss: 43.62718200683594\n",
            "epochs: 150/1000 , loss: 78.890869140625\n",
            "epochs: 160/1000 , loss: 86.25858306884766\n",
            "epochs: 170/1000 , loss: 34.824241638183594\n",
            "epochs: 180/1000 , loss: 23.114429473876953\n",
            "epochs: 190/1000 , loss: 20.999698638916016\n",
            "epochs: 200/1000 , loss: 15.804906845092773\n",
            "epochs: 210/1000 , loss: 79.90194702148438\n",
            "epochs: 220/1000 , loss: 20.205957412719727\n",
            "epochs: 230/1000 , loss: 18.537242889404297\n",
            "epochs: 240/1000 , loss: 33.73158264160156\n",
            "epochs: 250/1000 , loss: 77.61127471923828\n",
            "epochs: 260/1000 , loss: 8.863539695739746\n",
            "epochs: 270/1000 , loss: 30.53778076171875\n",
            "epochs: 280/1000 , loss: 18.733600616455078\n",
            "epochs: 290/1000 , loss: 15.911836624145508\n",
            "epochs: 300/1000 , loss: 21.290199279785156\n",
            "epochs: 310/1000 , loss: 15.45386791229248\n",
            "epochs: 320/1000 , loss: 71.94413757324219\n",
            "epochs: 330/1000 , loss: 16.404926300048828\n",
            "epochs: 340/1000 , loss: 3.423675060272217\n",
            "epochs: 350/1000 , loss: 16.53108787536621\n",
            "epochs: 360/1000 , loss: 4.358015060424805\n",
            "epochs: 370/1000 , loss: 71.71493530273438\n",
            "epochs: 380/1000 , loss: 15.771695137023926\n",
            "epochs: 390/1000 , loss: 3.8115763664245605\n",
            "epochs: 400/1000 , loss: 16.557565689086914\n",
            "epochs: 410/1000 , loss: 15.749751091003418\n",
            "epochs: 420/1000 , loss: 26.686126708984375\n",
            "epochs: 430/1000 , loss: 17.538860321044922\n",
            "epochs: 440/1000 , loss: 29.510366439819336\n",
            "epochs: 450/1000 , loss: 14.381510734558105\n",
            "epochs: 460/1000 , loss: 69.44481658935547\n",
            "epochs: 470/1000 , loss: 3.8050777912139893\n",
            "epochs: 480/1000 , loss: 81.62145233154297\n",
            "epochs: 490/1000 , loss: 15.876832962036133\n",
            "epochs: 500/1000 , loss: 80.14299011230469\n",
            "epochs: 510/1000 , loss: 70.10478973388672\n",
            "epochs: 520/1000 , loss: 69.99870300292969\n",
            "epochs: 530/1000 , loss: 3.621614933013916\n",
            "epochs: 540/1000 , loss: 3.630280017852783\n",
            "epochs: 550/1000 , loss: 14.341386795043945\n",
            "epochs: 560/1000 , loss: 72.78015899658203\n",
            "epochs: 570/1000 , loss: 5.099566459655762\n",
            "epochs: 580/1000 , loss: 5.011298179626465\n",
            "epochs: 590/1000 , loss: 13.81164264678955\n",
            "epochs: 600/1000 , loss: 2.7010388374328613\n",
            "epochs: 610/1000 , loss: 13.769948959350586\n",
            "epochs: 620/1000 , loss: 80.50200653076172\n",
            "epochs: 630/1000 , loss: 74.64453887939453\n",
            "epochs: 640/1000 , loss: 12.618977546691895\n",
            "epochs: 650/1000 , loss: 27.5805606842041\n",
            "epochs: 660/1000 , loss: 12.994125366210938\n",
            "epochs: 670/1000 , loss: 82.65612030029297\n",
            "epochs: 680/1000 , loss: 14.81744384765625\n",
            "epochs: 690/1000 , loss: 82.20123291015625\n",
            "epochs: 700/1000 , loss: 14.026227951049805\n",
            "epochs: 710/1000 , loss: 72.42861938476562\n",
            "epochs: 720/1000 , loss: 13.11204719543457\n",
            "epochs: 730/1000 , loss: 13.88226318359375\n",
            "epochs: 740/1000 , loss: 70.36982727050781\n",
            "epochs: 750/1000 , loss: 13.901385307312012\n",
            "epochs: 760/1000 , loss: 26.490036010742188\n",
            "epochs: 770/1000 , loss: 14.575761795043945\n",
            "epochs: 780/1000 , loss: 3.4752533435821533\n",
            "epochs: 790/1000 , loss: 3.710793972015381\n",
            "epochs: 800/1000 , loss: 79.93730163574219\n",
            "epochs: 810/1000 , loss: 13.650426864624023\n",
            "epochs: 820/1000 , loss: 73.53709411621094\n",
            "epochs: 830/1000 , loss: 80.42716979980469\n",
            "epochs: 840/1000 , loss: 5.643105506896973\n",
            "epochs: 850/1000 , loss: 80.0959701538086\n",
            "epochs: 860/1000 , loss: 12.013420104980469\n",
            "epochs: 870/1000 , loss: 74.50337219238281\n",
            "epochs: 880/1000 , loss: 80.88932800292969\n",
            "epochs: 890/1000 , loss: 3.273789644241333\n",
            "epochs: 900/1000 , loss: 13.789217948913574\n",
            "epochs: 910/1000 , loss: 15.523408889770508\n",
            "epochs: 920/1000 , loss: 24.141185760498047\n",
            "epochs: 930/1000 , loss: 3.3357186317443848\n",
            "epochs: 940/1000 , loss: 25.26515769958496\n",
            "epochs: 950/1000 , loss: 12.124983787536621\n",
            "epochs: 960/1000 , loss: 13.68928050994873\n",
            "epochs: 970/1000 , loss: 80.7054443359375\n",
            "epochs: 980/1000 , loss: 4.940096855163574\n",
            "epochs: 990/1000 , loss: 2.44824481010437\n",
            "epochs: 1000/1000 , loss: 24.850187301635742\n",
            "tensor([[ 57.0929,  73.0788],\n",
            "        [ 82.1150, 102.6467],\n",
            "        [118.8509, 132.2457],\n",
            "        [ 21.1156,  46.8291],\n",
            "        [101.6864, 116.2038],\n",
            "        [ 57.0929,  73.0788],\n",
            "        [ 82.1150, 102.6467],\n",
            "        [118.8509, 132.2457],\n",
            "        [ 21.1156,  46.8291],\n",
            "        [101.6864, 116.2038],\n",
            "        [ 57.0929,  73.0788],\n",
            "        [ 82.1150, 102.6467],\n",
            "        [118.8509, 132.2457],\n",
            "        [ 21.1156,  46.8291],\n",
            "        [101.6864, 116.2038]], grad_fn=<AddmmBackward0>)\n",
            "tensor([[ 56.,  70.],\n",
            "        [ 81., 101.],\n",
            "        [119., 133.],\n",
            "        [ 22.,  37.],\n",
            "        [103., 119.],\n",
            "        [ 56.,  70.],\n",
            "        [ 81., 101.],\n",
            "        [119., 133.],\n",
            "        [ 22.,  37.],\n",
            "        [103., 119.],\n",
            "        [ 56.,  70.],\n",
            "        [ 81., 101.],\n",
            "        [119., 133.],\n",
            "        [ 22.,  73.],\n",
            "        [103., 119.]])\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#LOGISTIC REGRESSION USING MNIST\n",
        "\n",
        "#importing libraries\n",
        "import torch\n",
        "from torchvision.datasets import MNIST\n",
        "import torchvision.transforms as transforms\n",
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "import torch.nn.functional as F\n",
        "import torch.nn as nn\n",
        "from torch.utils.data import SubsetRandomSampler, DataLoader\n",
        "\n",
        "#main dataset and testdata\n",
        "dataset=MNIST(download=True, train=True, root=\"./data\", transform=transforms.ToTensor())\n",
        "testset=MNIST(root=\"./data\", train=False, transform=transforms.ToTensor())\n",
        "\n",
        "#plotting the dataset\n",
        "image,labels=dataset[1000]\n",
        "plt.imshow(image[0,10:25,10:25], cmap=\"gray\")\n",
        "plt.show()\n",
        "print(\"label: \", labels)"
      ],
      "metadata": {
        "id": "ZQV72q6kWpJf"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#splitting the whole dataset into validation data and training data\n",
        "def split_data(dataset, validation_percent):\n",
        "  validation_data=int(dataset*validation_percent)\n",
        "  shuffled=np.random.permutation(dataset)\n",
        "  return shuffled[validation_data:], shuffled[:validation_data]\n",
        "\n",
        "training_data,validation_data = split_data(len(dataset), 0.3)\n",
        "print(\"length of training data: \", len(training_data))\n",
        "print(\"length of validation data: \", len(validation_data))\n",
        "\n",
        "print(\"portion of validation data: \",validation_data[:20])"
      ],
      "metadata": {
        "id": "dlorU70oXLps"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#putting all the splitted data to the sampler and then into the dataloader\n",
        "train_data_sampler=SubsetRandomSampler(training_data)\n",
        "valid_data_sampler=SubsetRandomSampler(validation_data)\n",
        "\n",
        "batch_size=100\n",
        "\n",
        "training_loader=DataLoader(dataset, batch_size, sampler=train_data_sampler)\n",
        "validation_loader=DataLoader(dataset, batch_size, sampler=valid_data_sampler)"
      ],
      "metadata": {
        "id": "JIlbxkGddWAM"
      },
      "execution_count": 37,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#defining the model\n",
        "input_size=28*28\n",
        "num_classes=10\n",
        "\n",
        "class MNISTmodel(nn.Module):\n",
        "  def __init__(self):\n",
        "    super().__init__()\n",
        "    self.Linear=nn.Linear(input_size,num_classes)\n",
        "\n",
        "  def forward(self, size):\n",
        "    size=size.reshape(-1,784)\n",
        "    output=self.Linear(size)\n",
        "    return output\n",
        "\n",
        "model=MNISTmodel()"
      ],
      "metadata": {
        "id": "3Gej-4W87N2t"
      },
      "execution_count": 38,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#putting the training loader in the for loop as inputs and outputs for prediction\n",
        "for images, labels in training_loader:\n",
        "  prediction=model(images)"
      ],
      "metadata": {
        "collapsed": true,
        "id": "zZXbi9a5Pv87"
      },
      "execution_count": 39,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#display predictions and sum of the predictions of each array\n",
        "print(prediction[:2])\n",
        "sum=torch.sum(prediction[2])\n",
        "print(sum)\n",
        "\n",
        "#changing the sum of the probablities of the predictions close to 1 and then checking the sum again\n",
        "prob=F.softmax(prediction)\n",
        "print(prob[:2])\n",
        "sum=torch.sum(prob[2])\n",
        "print(sum)\n",
        "\n",
        "#displaying the exact predicted labels by the model\n",
        "max_prob, pred = torch.max(prob, dim=1)\n",
        "print(pred)\n",
        "print(max_prob)\n",
        "\n",
        "#displaying the actual target labels\n",
        "print(labels)"
      ],
      "metadata": {
        "id": "XGqPcJUW9P4s"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#defining the loss\n",
        "loss_fn=F.cross_entropy"
      ],
      "metadata": {
        "id": "ivMZt7kCgQQn"
      },
      "execution_count": 41,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#defining the optimizer\n",
        "opt=torch.optim.SGD(model.parameters(), lr=0.0001)"
      ],
      "metadata": {
        "id": "iDN3RG9YjUEG"
      },
      "execution_count": 51,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# accuracy metrics\n",
        "def accuracy(outputs, labels):\n",
        "  _,pred=torch.max(outputs, dim=1)\n",
        "  return (torch.sum(pred==labels).item()/len(pred))*100"
      ],
      "metadata": {
        "id": "bSGzZGi4S59k"
      },
      "execution_count": 52,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# loss batch function for loss computation, gradient computation, updating weights, resetting gradients, accuracy computation\n",
        "def loss_batch(model, loss_fn, images, labels, opt, metrics=accuracy):\n",
        "  prediction=model(images)\n",
        "  loss=loss_fn(prediction, labels)\n",
        "\n",
        "  if opt is not None:\n",
        "    loss.backward()\n",
        "    opt.step()\n",
        "    opt.zero_grad()\n",
        "\n",
        "  metric_result=None\n",
        "  if metrics is not None:\n",
        "    metric_result=metrics(prediction, labels)\n",
        "\n",
        "  return loss.item(), len(images), metric_result"
      ],
      "metadata": {
        "id": "oObTH91hPsnK"
      },
      "execution_count": 53,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# function for evaluating the average loss and average accuracy of the validation set\n",
        "def evaluate(model, loss_fn, validation_loader, metrics=accuracy):\n",
        "  with torch.no_grad():\n",
        "    validation_prediction=[loss_batch(model, loss_fn, images, labels, opt=None, metrics=accuracy) for images, labels in validation_loader]\n",
        "\n",
        "    losses, nums, metric=zip(*validation_prediction)\n",
        "\n",
        "    total=np.sum(nums)\n",
        "\n",
        "    average_loss = np.sum(np.multiply(losses, nums))/total\n",
        "\n",
        "    average_metrics=None\n",
        "    if metrics is not None:\n",
        "      average_metrics = np.sum(np.multiply(metric, nums))/total\n",
        "\n",
        "  return average_loss.item(), total, average_metrics"
      ],
      "metadata": {
        "id": "OkoWI6TfbQyx"
      },
      "execution_count": 54,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#function for explicit training\n",
        "def fit(nepochs, model, images, labels, training_loader, validation_loader, opt, metrics=accuracy):\n",
        "  for epoch in range(nepochs):\n",
        "    for images, labels in training_loader:\n",
        "      train_loss,_, train_accuracy=loss_batch(model, loss_fn, images, labels, opt, metrics=accuracy)\n",
        "\n",
        "    valid_loss, _, valid_accuracy= evaluate(model, loss_fn, validation_loader, metrics=accuracy)\n",
        "\n",
        "    print(f\"Epoch: {epoch+1}/{nepochs}\")\n",
        "    print(f\"Training loss: {train_loss:.4f} and Validation loss: {valid_loss:.4f}.\")\n",
        "    print(f\"Training accuracy: {train_accuracy:.2f}% and Validation accuracy: {valid_accuracy:.2f}%.\")\n",
        "    print(\"--------------------------------------------------------------------------------------------\")\n",
        "\n",
        "  return train_loss, _, train_accuracy, valid_loss, _, valid_accuracy\n",
        "\n",
        "train_loss,_, train_accuracy, valid_loss, _, valid_accuracy = fit(6, model, images, labels, training_loader, validation_loader, opt, metrics=accuracy)\n",
        "\n",
        "print(\"--\")\n",
        "print(f\"The train accuracy is {train_accuracy:.2f} % and loss is {train_loss:.4f}.\")\n",
        "print(\"--------------------------------------------\")\n",
        "print(f\"The validation accuracy is {valid_accuracy:.2f} % and loss is {valid_loss:.4f}\")"
      ],
      "metadata": {
        "id": "TCttIo85b4ch"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# testing the model with the testing dataset\n",
        "#function for predicting the test images\n",
        "def predict_image(image, model):\n",
        "  input=image.unsqueeze(0)\n",
        "  output=model(input)\n",
        "  _,preds=torch.max(output, dim=1)\n",
        "\n",
        "  return preds[0].item()"
      ],
      "metadata": {
        "id": "v8W1V8lqiu3b"
      },
      "execution_count": 94,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#predicting and displaying different labels\n",
        "image,labels=testset[10]\n",
        "plt.imshow(image[0], cmap=\"gray\")\n",
        "plt.show()\n",
        "print(\"label: \", labels)\n",
        "print(\"predicted: \", predict_image(image, model))\n",
        "\n",
        "image,labels=testset[100]\n",
        "plt.imshow(image[0], cmap=\"gray\")\n",
        "plt.show()\n",
        "print(\"label: \", labels)\n",
        "print(\"predicted: \", predict_image(image, model))\n",
        "\n",
        "image,labels=testset[1000]\n",
        "plt.imshow(image[0], cmap=\"gray\")\n",
        "plt.show()\n",
        "print(\"label: \", labels)\n",
        "print(\"predicted: \", predict_image(image, model))\n",
        "\n",
        "image,labels=testset[905]\n",
        "plt.imshow(image[0], cmap=\"gray\")\n",
        "plt.show()\n",
        "print(\"label: \", labels)\n",
        "print(\"predicted: \", predict_image(image, model))"
      ],
      "metadata": {
        "id": "rmP4znRwkh7s"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#checking the loss and accuracy on the test set\n",
        "test_loader=DataLoader(testset, batch_size=200)\n",
        "test_loss, total, test_accuracy=evaluate(model, loss_fn, test_loader, metrics=accuracy)\n",
        "print(f\"The test set loss is {test_loss:.4f} and the accuracy is {test_accuracy:.2f}%.\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xCKU1FQKn-YG",
        "outputId": "fb9a20f5-080c-4f98-88b6-bdae4d1da73f"
      },
      "execution_count": 105,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "The test set loss is 0.7575 and the accuracy is 85.03%.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#saving and loading the model\n",
        "torch.save(model.state_dict(),'MNISTlogistic.pth')\n",
        "model.state_dict()"
      ],
      "metadata": {
        "id": "ILIq6GWqqLFV"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "savedmodel=MNISTmodel()\n",
        "savedmodel.load_state_dict(torch.load('MNISTlogistic.pth'))\n",
        "savedmodel.state_dict()"
      ],
      "metadata": {
        "id": "9s_BDeUdr7ZX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "wu3GnbgisZBY"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}