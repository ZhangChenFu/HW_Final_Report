import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import MNIST
from model import LeNet
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def test_data_process():
    test_data = MNIST(root='./data',
                              train=False,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=False)

    # Set the size of batch to 1.
    # At this point, the for loop in the following code does not have the step variable
    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=0)
    return test_dataloader


def test_model_process(model, test_dataloader):
    # Select test equipment
    device = ("cuda" if torch.cuda.is_available() else 'cpu')

    # Put the model into the training facility
    model = model.to(device)

    # initialization parameter
    test_corrects = 0.0
    test_num = 0

    # Only forward propagation is performed, and gradients are not calculated,
    # thus saving memory and speeding up operation
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            # Put the feature into the test device
            test_data_x = test_data_x.to(device)
            # Place the label into the test device
            test_data_y = test_data_y.to(device)
            # Set the model to evaluation mode
            model.eval()
            # Finds the row label corresponding to the maximum value in each row
            output= model(test_data_x)
            # Finds the row label corresponding to the maximum value in each row
            pre_lab = torch.argmax(output, dim=1)
            # If the prediction is correct, the accuracy is increased by 1
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            # All test samples are added together
            test_num += test_data_x.size(0)

    # Computational test accuracy
    test_acc = test_corrects.double().item() / test_num
    print("test accuracy:b ", test_acc)


if __name__=="__main__":
    # loading model
    model = LeNet()
    model.load_state_dict(torch.load('best_model.pth'))

    # Load test data
    test_dataloader = test_data_process()
    # Load the model test function
    test_model_process(model, test_dataloader)
