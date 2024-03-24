import copy
import time

import torch
from torchvision.datasets import MNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import LeNet
import torch.nn as nn
import pandas as pd


def train_val_data_process():
    train_data = MNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=False)

    train_data, val_data = Data.random_split(train_data, [round(0.8*len(train_data)), round(0.2*len(train_data))])
    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=64,
                                       shuffle=True,
                                       num_workers=2)

    val_dataloader = Data.DataLoader(dataset=val_data,
                                       batch_size=64,
                                       shuffle=True,
                                       num_workers=2)

    return train_dataloader, val_dataloader


def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    # Set the device used for training, there is a GPU then use the GPU and not a GPU use the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Using the Adam optimizer with a learning rate of 0.001, it is a gradient descent algorithm.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # The loss function is a cross entropy function, which calculates
    # the difference between the predicted value and the true value
    criterion = nn.CrossEntropyLoss()
    # Place the model into the training facility
    model = model.to(device)
    # Copy the parameters of the current model and use static storage to store the model that works best.
    best_model_wts = copy.deepcopy(model.state_dict())

    # Initialization parameter, used to store data
    # Define the highest accuracy parameters
    best_acc = 0.0
    # List of training set loss values
    train_loss_all = []
    # List of validation set loss values
    val_loss_all = []
    # Training set accuracy list
    train_acc_all = []
    # Validation set accuracy list
    val_acc_all = []
    # Get current time
    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print("-"*10)

        # initialization parameter
        # Loss of training set
        train_loss = 0.0
        # Accuracy of the training set
        train_corrects = 0
        # Verify the loss of set loss
        val_loss = 0.0
        # Validation set accuracy
        val_corrects = 0
        # Number of training set samples
        train_num = 0
        # The number of samples of the verification set
        val_num = 0

        for step, (b_x, b_y) in enumerate(train_dataloader):
            # Place the feature atlas into the training device
            b_x = b_x.to(device)
            # Place the tag set into the training device
            b_y = b_y.to(device)
            # Set the model to training mode
            model.train()

            # Forward propagation
            output = model(b_x)
            # Finds the row label corresponding to the maximum value in each row
            pre_lab = torch.argmax(output, dim=1)
            # Calculate the loss function for each batch
            loss = criterion(output, b_y)

            # The gradient is initialized to 0 in order to update the function later and
            # prevent the previous data from being used on a new batch of data
            optimizer.zero_grad()
            # back propagation
            loss.backward()
            # The gradient parameters of the network are updated
            # according to the gradient information of the network backpropagation
            optimizer.step()
            # Add up the loss function
            train_loss += loss.item() * b_x.size(0)
            # If the prediction is correct, the number of accurate results is added by 1
            train_corrects += torch.sum(pre_lab == b_y.data)
            # The number of samples currently used for training
            train_num += b_x.size(0)
        for step, (b_x, b_y) in enumerate(val_dataloader):
            # Place the validation atlas into the device
            b_x = b_x.to(device)
            # Place the label into the verification device
            b_y = b_y.to(device)
            # Set the model to evaluation mode
            model.eval()
            # Forward propagation process.
            # The input is a batch, and the output is the corresponding prediction in a batch
            output = model(b_x)
            # Finds the row label corresponding to the maximum value in each row
            pre_lab = torch.argmax(output, dim=1)
            # Calculate the loss of each batch
            loss = criterion(output, b_y)

            # Turn it off during verification
            # Add up the loss function
            val_loss += loss.item() * b_x.size(0)
            # If the prediction is correct, the correct result is added by 1
            val_corrects += torch.sum(pre_lab == b_y.data)
            # The number of samples currently used for validation
            val_num += b_x.size(0)
        # ----------------------------------------------------------------

        # Calculate and save the loss value and accuracy for each iteration

        # Calculate and save the loss value of the training set,
        # in which train_loss is the sum of each loss in the training set
        train_loss_all.append(train_loss / train_num)
        # Calculate and save the accuracy of the training set
        train_acc_all.append(train_corrects.double().item() / train_num)

        # Calculate and save the loss value for the validation set
        val_loss_all.append(val_loss / val_num)
        # Calculate and save the accuracy of the validation set
        val_acc_all.append(val_corrects.double().item() / val_num)

        print("{} train loss:{:.4f} train acc: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("{} val loss:{:.4f} val acc: {:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        # Compare the accuracy of the last round with the highest accuracy
        if val_acc_all[-1] > best_acc:
            # Save the current highest accuracy
            best_acc = val_acc_all[-1]
            # Save the current model parameters with the highest accuracy
            best_model_wts = copy.deepcopy(model.state_dict())

        # Calculate how long it takes to train and validate
        time_use = time.time() - since
        print("Time spent on training and validation{:.0f}m{:.0f}s".format(time_use//60, time_use%60))

    # ---------------------------------------------------------

    # Select the optimal parameters and save the model of the optimal parameters
    torch.save(best_model_wts, "best_model.pth")
    # torch.save(model, 'whole_model.pth')

    # draw pictures
    train_process = pd.DataFrame(data={"epoch":range(num_epochs),
                                       "train_loss_all":train_loss_all,
                                       "val_loss_all":val_loss_all,
                                       "train_acc_all":train_acc_all,
                                       "val_acc_all":val_acc_all, })

    return train_process


def matplot_acc_loss(train_process):
    # Displays the loss function and accuracy of the training set and validation set after each iteration
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss")
    plt.plot(train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Load the required model
    LeNet = LeNet()
    # Load data set
    train_data, val_data = train_val_data_process()
    # Use the existing model to train the model
    train_process = train_model_process(LeNet, train_data, val_data, num_epochs=20)
    matplot_acc_loss(train_process)