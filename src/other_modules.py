import torch
import warnings
import torch.nn as nn
import numpy as np

class BPRLoss(nn.Module):
    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    # 调整为是否算mean的版本
    def forward(self, pos_score, neg_score,mean=True):
        if mean:
            loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        else:
            loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score))
        return loss


