#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnetLoss(nn.Module):
    def __init__(self, weighted_values=None, size_average=None, encoder_only=False):
        super(EnetLoss, self).__init__()

        self._encoder_only = encoder_only

        if weighted_values is not None:
            weighted_values = torch.FloatTensor(weighted_values)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            weighted_values = weighted_values.to(device)

        self.loss_func = nn.CrossEntropyLoss(weighted_values, size_average)

    @property
    def encoder_only(self):
        return self._encoder_only

    @encoder_only.setter
    def encoder_only(self, flag):
        self._encoder_only = flag

    def forward(self, inputs, targets):

        if self._encoder_only:
            batch_size, h, w = targets.shape
            targets = torch.unsqueeze(targets, 0)
            targets = targets.type(torch.cuda.FloatTensor)
            targets = F.interpolate(targets, (h // 8, w // 8), None, mode="bilinear", align_corners=True)
            targets = targets.reshape((batch_size, h // 8, w // 8))
            targets = targets.type(torch.cuda.LongTensor)

        if not self._encoder_only:
            targets = targets.type(torch.cuda.LongTensor)

        return self.loss_func(inputs, targets)
