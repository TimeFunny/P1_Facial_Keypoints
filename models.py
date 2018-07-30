## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
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
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.pool1 = nn.MaxPool2d(2)
        #self.drop1 = nn.Dropout2d(p=0.5)
        
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        #self.batch2 = nn.BatchNorm2d(64) 
        self.pool2 = nn.MaxPool2d(2)       
        #self.drop2 = nn.Dropout2d(p=0.6)
        
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.batch3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)      
        #self.drop3 = nn.Dropout2d(p=0.5)
        
        self.conv4 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.batch4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2)                
        #self.drop4 = nn.Dropout2d(p=0.5)
        
        self.conv5 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.batch5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2)                
        #self.drop5 = nn.Dropout2d(p=0.2)        
       
        self.linear1 = nn.Linear(4608 ,1024)
        self.drop6 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(1024,1024)
        self.drop7 = nn.Dropout(p=0.4)
        self.linear3 = nn.Linear(1024 ,136)       

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.batch3(self.conv3(x))))
        x = self.pool4(F.relu(self.batch4(self.conv4(x))))
        x = self.pool5(F.relu(self.batch5(self.conv5(x))))
      
        x = x.view(x.size(0), -1)
        x = self.drop6(F.relu(self.linear1(x)))
        x = self.drop7(F.relu(self.linear2(x)))     
        x = self.linear3(x)
        #x.reshape(-1, 2)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
