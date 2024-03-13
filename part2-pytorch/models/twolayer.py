
import numpy as np
import torch
import torch.nn as nn


class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        """
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        """
        super(TwoLayerNet, self).__init__()
        # self.model = nn.Sequential(
        #     nn.Linear(input_dim, hidden_size),
        #     nn.Sigmoid(),
        #     nn.Linear(hidden_size, num_classes)
        # )

        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        out = None
        # out = self.model(x)
        n = np.shape(x)[0]
        x_flatten = x.reshape(n, -1)
        out = self.fc1(x_flatten)
        out = self.sigmoid(out)
        out = self.fc2(out)
        return out
