

import numpy as np
import math

class MaxPooling:
    """
    Max Pooling of input
    """

    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None
        self.max_index = None

    def forward(self, x):
        """
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        """
        out = None

        N, C, H, W = x.shape
        kernel_size, stride = self.kernel_size, self.stride
        H_out = math.floor((H - kernel_size) // stride + 1)
        W_out = math.floor((W - kernel_size) // stride + 1)
        out = np.zeros((N, C, H_out, W_out))
        for i in range(H_out):
            for j in range(W_out):
                out[:, :, i, j] = np.max(x[:, :, i * stride:i * stride + kernel_size,
                                         j * stride:j * stride + kernel_size], axis=(2, 3))

        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        """
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return: nothing, but self.dx should be updated
        """
        x, H_out, W_out = self.cache

        N, C, _, _ = dout.shape
        self.dx = np.zeros_like(x)
        kernel_size, stride = self.kernel_size, self.stride

        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        pool = x[n, c, h*stride:h*stride+kernel_size, w*stride:w*stride+kernel_size]
                        mask = (pool == np.max(pool))
                        self.dx[n, c, h*stride:h*stride+kernel_size, w*stride:w*stride+kernel_size] = \
                            np.multiply(mask,dout[n,c,h,w])

