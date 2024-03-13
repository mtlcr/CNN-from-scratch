

import torch
import torch.nn as nn
import numpy as np

class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(5408, 10)

    def forward(self, x):
        outs = None
        n = np.shape(x)[0]
        x_flatten = x.reshape(n, -1)
        outs = self.conv1(x)
        outs = self.relu(outs)
        outs = self.maxpool(outs)
        outs = outs.view(outs.size(0), -1)
        out_flatten = outs.reshape(outs.size(0), -1)
        outs = self.fc(out_flatten)

        return outs
