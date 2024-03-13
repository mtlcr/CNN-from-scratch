

import torch
import torch.nn as nn
import numpy as np

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # self.bn = nn.BatchNorm2d(3)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(28800, 10)

    def forward(self, x):
        outs = None

        # outs = self.bn(x)
        outs = self.conv1(x)
        outs = self.relu(outs)

        outs = self.conv2(outs)
        outs = self.relu(outs)

        outs = self.conv3(outs)
        outs = self.relu(outs)

        outs = self.maxpool(outs)
        outs = outs.view(outs.size(0), -1)
        out_flatten = outs.reshape(outs.size(0), -1)
        outs = self.fc(out_flatten)
        return outs
