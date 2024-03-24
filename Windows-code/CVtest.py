import torch
import torch.utils.data as Data
from torchvision import transforms
import torch.nn.functional as F
from model import LeNet

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

# same process as CVmethod.py
def cv_process(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reverse = 255 - gray
    #ret1, binary_otsu = cv2.threshold(reverse, 180, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret1, binary_otsu = cv2.threshold(reverse, 180, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    image_erosion = cv2.erode(binary_otsu, kernel, iterations=1)
    image_dilate = cv2.dilate(image_erosion, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(image_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)

    cat = binary_otsu[y-20:y+h+20, x-20:x+w+20]
    final = Image.fromarray(cat)
    return final

# make the image can be tested
def torch_process(image):

    # We have to convert the image to a tensor in order to evaluate it
    test_transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
    image = test_transform(image)
    image = image.unsqueeze(0)
    return image


if __name__=="__main__":

    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model = LeNet()
    model.load_state_dict(torch.load('best_model.pth'))
    model.to(device)
    model.eval()
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    image1 = cv_process('picture/7.jpg')
    image1 = torch_process(image1)

    t = time.time()
    with torch.no_grad():
        image1 = image1.to(device)
        output = model(image1)
        pre_lab = torch.argmax(output, dim=1)
        result = pre_lab.item()

    print("predicted value：", classes[result])
    time_use1 = time.time() - t
    print("time used_1 on predicting {:.0f}m{:.0f}s".format(time_use1 // 60, time_use1 % 60))

