

from .softmax_ce import SoftmaxCrossEntropy
from .relu import ReLU
from .max_pool import MaxPooling
from .convolution import Conv2D
from .linear import Linear


class ConvNet:
    """
    Max Pooling of input
    """
    def __init__(self, modules, criterion):
        self.modules = []
        for m in modules:
            if m['type'] == 'Conv2D':
                self.modules.append(
                    Conv2D(m['in_channels'],
                           m['out_channels'],
                           m['kernel_size'],
                           m['stride'],
                           m['padding'])
                )
            elif m['type'] == 'ReLU':
                self.modules.append(
                    ReLU()
                )
            elif m['type'] == 'MaxPooling':
                self.modules.append(
                    MaxPooling(m['kernel_size'],
                               m['stride'])
                )
            elif m['type'] == 'Linear':
                self.modules.append(
                    Linear(m['in_dim'],
                           m['out_dim'])
                )
        if criterion['type'] == 'SoftmaxCrossEntropy':
            self.criterion = SoftmaxCrossEntropy()
        else:
            raise ValueError("Wrong Criterion Passed")

    def forward(self, x, y):
        """
        The forward pass of the model
        :param x: input data: (N, C, H, W)
        :param y: input label: (N, )
        :return:
          probs: the probabilities of all classes: (N, num_classes)
          loss: the cross entropy loss
        """
        probs = None
        loss = None
        out = x
        for module in self.modules:
            out = module.forward(out)

        probs = out
        loss = self.criterion.forward(probs, y)

        return probs, loss

    def backward(self):
        """
        The backward pass of the model
        :return: nothing but dx, dw, and db of all modules are updated
        """
        self.criterion.backward()
        dx = self.criterion.dx
        for module in reversed(self.modules):
            module.backward(dx)
            dx = module.dx
            # dw = module.dw
            # db = module.db
            print('module', module)
            print('module.dx.sum()',module.dx.sum())
            # print('module.dw.sum()',module.dw.sum())
            # print('module.db.sum()',module.db.sum())