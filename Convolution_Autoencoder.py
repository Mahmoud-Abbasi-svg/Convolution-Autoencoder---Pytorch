# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 01:12:19 2022

@author: HP_PC2
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os    
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# convert Image to torch.FloatTensor
transform = transforms.ToTensor()

# load the training and test data
train_data = datasets.CIFAR10(root='/data', train=True,
                              transform=transform, download=True)

test_data = datasets.CIFAR10(root='/data', train=False,
                             transform=transform, download=True)


# Create training and test dataloaders
batch_size = 20

train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=batch_size)

# Visualize the data
# a helper function to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5 # unnormalize the data (i.e., Images)
    plt.imshow(np.transpose(img, (1, 2, 0))) # convert from Tensor image
    
# specify the image classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

'''
# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
# display 20 images
for idx in np.arange(20): #note that batch size is 20 (20 images )
    ax = fig.add_subplot(2, 20//2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])
'''

# Convolutional Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # encoder layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=4,
                               kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # decoder layers
        self.t_conv1 = nn.ConvTranspose2d(in_channels=4, out_channels=16,
                                          kernel_size=2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(in_channels=16, out_channels=3,
                                          kernel_size=2, stride=2)
        
    def forward(self, x):
        # encode forward pass
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation
        
        # decode forward pass
        x = F.relu(self.t_conv1(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = F.sigmoid(self.t_conv2(x))
                
        return x

# initialize the model
model = ConvAutoencoder()
print(model)        
        
# Training the model
criterion = nn.BCELoss()
n_epochs = 1
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, n_epochs+1):
    train_loss = 0.0
    for data in train_loader:
        # no need to flatten images
        images,_ = data
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*images.size(0)
        
    train_loss = train_loss/len(train_loader)
    print(f'Epoch: {epoch} \ Training Loss: {train_loss:.4f}')


# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()

# get sample outputs
output = model(images)

# prepare images for display
images = images.numpy()

# output is resized into a batch of images
output = output.view(batch_size, 3, 32, 32)

# use detach when it's an output that requires_grad
output = output.detach().numpy()

# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(24,4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(output[idx])
    ax.set_title(classes[labels[idx]])
    
# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(24,4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])























