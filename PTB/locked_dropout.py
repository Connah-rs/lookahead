# Copied from https://github.com/salesforce/awd-lstm-lm
# For model in paper: "Regularizing and Optimizing LSTM Language Models" (Merity et al.) https://arxiv.org/pdf/1708.02182.pdf

import torch
import torch.nn as nn
from torch.autograd import Variable

class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x