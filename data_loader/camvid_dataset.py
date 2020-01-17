#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import cv2
import numpy as np
from .data_loader_base import BaseDataset


class CamvidDatasetConfig:
    CAMVID_CLASSES = [
        "Sky",
        "Building",
        "Pole",
        "Road",
        "Pavement",
        "Tree",
        "SignSymbol",
        "Fence",
        "Car",
        "Pedestrian",
        "Bicyclist",
        "Void",
    ]

    CAMVID_COLORS = [
        (128, 128, 128),
        (0, 0, 128),
        (128, 192, 192),
        (128, 64, 128),
        (222, 40, 60),
        (0, 128, 128),
        (128, 128, 192),
        (128, 64, 64),
        (128, 0, 64),
        (0, 64, 64),
        (192, 128, 0),
        (0, 0, 0),
    ]

    @property
    def num_classes(self):
        return len(self.CAMVID_CLASSES)


class CamvidDataset(BaseDataset):
    def __init__(self, data_path, phase="test", transform=None):
        super(CamvidDataset, self).__init__(
            data_path,
            phase=phase,
            classes=CamvidDatasetConfig.CAMVID_CLASSES,
            colors=CamvidDatasetConfig.CAMVID_COLORS,
            transform=transform,
        )

        _camvid_data_path = os.path.join(self._data_path, "CamVid")
        _image_data_paths = os.path.join(_camvid_data_path, phase)
        _gt_data_paths = os.path.join(_camvid_data_path, "{}annot".format(phase))

        self._image_paths = glob.glob(os.path.join(_image_data_paths, "*.png"))
        self._gt_paths = glob.glob(os.path.join(_gt_data_paths, "*.png"))
        self._image_paths.sort(key=BaseDataset.human_sort)
        self._gt_paths.sort(key=BaseDataset.human_sort)

        self._color_idx_dict = BaseDataset.color_to_color_idx_dict(self._colors)

    def weighted_class(self):
        assert "Void" in self._classes
        class_idx_dict = BaseDataset.class_to_class_idx_dict(self._classes)
        weighted = super(CamvidDataset, self).weighted_class()
        weighted[class_idx_dict["Void"]] = 0

        return weighted
