from torchvision.datasets import MNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt

train_data = MNIST(root='./data',
                        train=True,
                        transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                        download=False)


train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=64,
                               shuffle=True,
                               num_workers=0)

# Get a Batch of data
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break
# Remove the four-dimensional tensor from the
# first dimension and convert it to a Numpy array
batch_x = b_x.squeeze().numpy()

# When there is only one element in the dimension of a tensor, you
# can use .squeeze() to remove the dimension. If a dimension is not equal to 1, it remains the same.

batch_y = b_y.numpy()  # Convert tensors to Numpy arrays
class_label = train_data.classes  # The label of the training set
# print(class_label)
print("The size of batch in train data:", batch_x.shape)

# Visualize the image of a batch
plt.figure(figsize=(12, 5))
for ii in np.arange(len(batch_y)):
    plt.subplot(4, 16, ii + 1)
    plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)
    plt.title(class_label[batch_y[ii]], size=10)
    plt.axis("off")
    plt.subplots_adjust(wspace=0.05)
plt.show()
