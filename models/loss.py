#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class EnetLoss(nn.Module):
    def __init__(self, weighted_values=None, size_average=None):
        super(EnetLoss, self).__init__()

        self.nll_loss = nn.NLLLoss(weighted_values, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)
