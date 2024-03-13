
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def reweight(cls_num_list, beta=0.9999):
    """
    Implement reweighting by effective numbers
    :param cls_num_list: a list containing # of samples of each class
    :param beta: hyper-parameter for reweighting, see paper for more details
    :return:
    """
    per_cls_weights = None
    per_cls_weights = (1.0 - beta)/np.array(1.0 - np.power(beta, cls_num_list))
    per_cls_weights = per_cls_weights /np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.Tensor(per_cls_weights)
    return per_cls_weights


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        Implement forward of focal loss
        :param input: input predictions
        :param target: labels
        :return: tensor of focal loss in scalar
        """
        loss = None
        log_pt = torch.nn.functional.log_softmax(input, dim=-1)
        pt = torch.exp(log_pt)
        loss = torch.nn.functional.nll_loss(((1 - pt) ** self.gamma) * log_pt, target, self.weight)
        return loss
