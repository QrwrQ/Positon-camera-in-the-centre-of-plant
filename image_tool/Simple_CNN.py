import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
import os
#size= 74

class Layers_2(nn.Module):
    def __init__(self):
        super(Layers_2,self).__init__()

        self.conv1=nn.Conv2d(3,16,7)
        self.pool1=nn.MaxPool2d(2,2)
        # return self.pool2

        self.conv3=nn.Conv2d(16,32,3)
        self.pool3=nn.MaxPool2d(2,2)


        self.fc1 = nn.Linear(16*16*32,600)
        self.fc1_5=nn.Linear(600,9)
        # self.fc2 = nn.Linear(120,84)
        # self.fc3 = nn.Linear(120,9)


    def forward(self,x):

        x=F.relu(self.conv1(x))
        x=self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x=x.view(-1,16*16*32)
        x=F.relu(self.fc1(x))
        # x = F.relu(self.fc1_5(x))
        # x=F.relu(self.fc2(x))
        x=self.fc1_5(x)
        return x