#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnetLoss(nn.Module):
    def __init__(self, weighted_values=None, size_average=None):
        super(EnetLoss, self).__init__()

        if weighted_values is not None:
            weighted_values = torch.FloatTensor(weighted_values)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            weighted_values = weighted_values.to(device)

        self.nll_loss = nn.NLLLoss(weighted_values, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)
