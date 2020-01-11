#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .data_loader_base import BaseDataset


class CamvidDataset(BaseDataset):
    def __init__(self, data_path, classes, colors):
        super(BaseDataset, self).__init__(data_path, classes, colors)
