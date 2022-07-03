import numpy as np
import torch.nn as nn
from enum import Enum
import torch.nn.functional as F

class MobileBirdNet(nn.Module):
    def __init__(self):
        super(MobileBirdNet, self).__init__()

        self.layers = []
        #Pre-processing
        self.layers.append(nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), padding=1, stride=(1,2)))
        self.layers.append(nn.BatchNorm2d(num_features=8))
        self.layers.append(nn.ReLU(True))
        #self.layers.append(nn.MaxPool2d(2))

        #Conv1
        self.layers.append(nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), padding=1))
        self.layers.append(nn.BatchNorm2d(num_features=16))
        self.layers.append(nn.ReLU(True))
        self.layers.append(nn.MaxPool2d(2))
        self.layers.append(nn.Dropout())

        #Conv2
        self.layers.append(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), padding=1))
        self.layers.append(nn.BatchNorm2d(num_features=32))
        self.layers.append(nn.ReLU(True))
        self.layers.append(nn.MaxPool2d(2))
        self.layers.append(nn.Dropout())

        #Conv3
        self.layers.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=1))
        self.layers.append(nn.BatchNorm2d(num_features=64))
        self.layers.append(nn.ReLU(True))
        self.layers.append(nn.MaxPool2d(2))
        self.layers.append(nn.Dropout())

        #Conv4
        self.layers.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1))
        self.layers.append(nn.BatchNorm2d(num_features=128))
        self.layers.append(nn.ReLU(True))
        self.layers.append(nn.MaxPool2d(2))
        self.layers.append(nn.Dropout())

        #Classification
        self.layers.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4,12), padding=0))
        self.layers.append(nn.BatchNorm2d(num_features=128))
        self.layers.append(nn.ReLU(True))
        #self.layers.append(nn.MaxPool2d((2, 12)))

        self.layers.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,1), padding=0))
        self.layers.append(nn.BatchNorm2d(num_features=256))
        self.layers.append(nn.ReLU(True))

        self.layers.append(nn.Conv2d(in_channels=256, out_channels=83, kernel_size=(1,1), padding=0))
        self.layers.append(nn.BatchNorm2d(num_features=83))
        self.layers.append(nn.AdaptiveAvgPool2d(output_size=(1,1)))
        self.layers.append(nn.Sigmoid())

        self.classifier = nn.Sequential(*self.layers)


    
    def forward(self, x):
        return self.classifier(x)