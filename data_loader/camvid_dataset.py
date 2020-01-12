#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import cv2
import numpy as np
from .data_loader_base import BaseDataset

_CAMVID_CLASSES = [
    "Bicyclist",
    "Building",
    "Car",
    "Fence",
    'Pavement',
    "Pedestrian",
    "Pole",
    "Road",
    'SignSymbol',
    "Sky",
    "Tree",
    "Void",
]

_CAMVID_COLORS = [
    (0, 128, 192),
    (128, 0, 0),
    (64, 0, 128),
    (64, 64, 128),
    (60, 40, 222),
    (64, 64, 0),
    (192, 192, 128),
    (128, 64, 128),
    (192, 128, 128),
    (128, 128, 128),
    (128, 128, 0),
    (0, 0, 0)
]


class CamvidDataset(BaseDataset):
    def __init__(self, data_path, phase="test", transform=None):
        super(CamvidDataset, self).__init__(data_path, phase=phase,
                                            classes=_CAMVID_CLASSES, colors=_CAMVID_COLORS, transform=transform)

        _camvid_data_path = os.path.join(self._data_path, "CamVid")
        _image_data_paths = os.path.join(_camvid_data_path, phase)
        _gt_data_paths = os.path.join(_camvid_data_path, "{}annot".format(phase))

        self._image_paths = glob.glob(os.path.join(_image_data_paths, "*.png"))
        self._gt_paths = glob.glob(os.path.join(_gt_data_paths, "*.png"))
        self._image_paths.sort(key=BaseDataset.human_sort)
        self._gt_paths.sort(key=BaseDataset.human_sort)

        self._color_idx_dict = BaseDataset.color_to_color_idx_dict(self._colors)
