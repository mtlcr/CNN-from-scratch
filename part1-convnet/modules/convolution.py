

import numpy as np
import math

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        """
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output(aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel(both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        """
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        """
        out = None
        N, C, H, W = x.shape
        kernel_size = self.kernel_size
        stride = self.stride
        pad = self.padding
        x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)
        H_out = math.floor((H + 2 * pad - kernel_size) // stride + 1)
        W_out = math.floor((W + 2 * pad - kernel_size) // stride + 1)
        out = np.zeros((N, self.out_channels, H_out, W_out))
        for n in range(N):
            for c_out in range(self.out_channels):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        convol_area = x_pad[n, :, h_out * stride:h_out * stride + kernel_size,
                                   w_out * stride: w_out * stride + kernel_size]
                        out[n, c_out, h_out, w_out] = np.sum(convol_area * self.weight[c_out]) + self.bias[c_out]
        self.cache = x
        return out

    def backward(self, dout):
        """
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        """
        x = self.cache
        N, C, H, W = x.shape
        kernel_size = self.kernel_size
        stride = self.stride
        pad = self.padding
        dx, dw, db = np.zeros_like(x), np.zeros_like(self.weight), np.zeros_like(self.bias)
        x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)
        dx_pad = np.pad(dx, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)

        for n in range(N):
            for c_out in range(self.out_channels):
                for h_out in range(dout.shape[2]):
                    for w_out in range(dout.shape[3]):
                        x_region = x_pad[n, :, h_out * stride: h_out * stride + kernel_size,
                                   w_out * stride: w_out * stride + kernel_size]
                        dx_pad[n, :, h_out * stride: h_out * stride + kernel_size,
                        w_out * stride: w_out * stride + kernel_size] += self.weight[c_out] * dout[n, c_out, h_out, w_out]
                        dw[c_out] += x_region * dout[n, c_out, h_out, w_out]
                        db[c_out] += dout[n, c_out, h_out, w_out]

        self.dx = dx_pad[:, :, pad:-pad, pad:-pad]
        self.dw = dw
        self.db = db
        self.weight -= dw
        self.bias -= db
