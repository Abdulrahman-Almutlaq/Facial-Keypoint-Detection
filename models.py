## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        self.conv_1 = nn.Conv2d(1, 32, 5)
        self.bnorm_1 = nn.BatchNorm2d(32)
        
        self.conv_2 = nn.Conv2d(32, 64, 5)
        self.bnorm_2 = nn.BatchNorm2d(64)
        
        self.conv_3 = nn.Conv2d(64, 128, 5,stride=2,padding=1)
        self.bnorm_3 = nn.BatchNorm2d(128)
        
        self.conv_4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bnorm_4 = nn.BatchNorm2d(256)
        
        self.max_pool = nn.MaxPool2d(2)
        
        self.drop = nn.Dropout(0.2)
        
        self.fc_1 = nn.Linear(256*13*13,256)
        self.bnorm_6 = nn.BatchNorm1d(256)
        
        
        self.fc_2 = nn.Linear(256, 136)        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch                     normalization) to avoid overfitting        

        
    def forward(self, x):
        
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.max_pool(F.relu(self.conv_1(x)))
        x = self.bnorm_1(x)
        
        # not sure if I have to normalize the batch before doing max pooling,
        # but my intution is this, I will think of the next layer as a new NN, thus the last step before feeding the data to
        # it should be normlizaion. Maybe it does not matter which I perform first, I'm not sure.
        
        x = self.max_pool(F.relu(self.conv_2(x)))
        x = self.bnorm_2(x)
        
        x = self.max_pool(F.relu(self.conv_3(x)))
        x = self.bnorm_3(x)
        
        x = F.relu(self.conv_4(x))
        x = self.bnorm_4(x)
        
        x = x.view(-1,256*13*13)
        
        x = F.relu(self.fc_1(x))
        x = self.bnorm_6(x)
        
        x = self.fc_2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        
        return x
