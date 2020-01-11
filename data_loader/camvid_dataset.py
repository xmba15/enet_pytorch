#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import cv2
import numpy as np
from .data_loader_base import BaseDataset

_CAMVID_CLASSES = [
    "Animal",
    "Archway",
    "Bicyclist",
    "Bridge",
    "Building",
    "Car",
    "CartLuggagePram",
    "Child",
    "Column_Pole",
    "Fence",
    "LaneMkgsDriv",
    "LaneMkgsNonDriv",
    "Misc_Text",
    "MotorcycleScooter",
    "OtherMoving",
    "ParkingBlock",
    "Pedestrian",
    "Road",
    "RoadShoulder",
    "Sidewalk",
    "SignSymbol",
    "Sky",
    "SUVPickupTruck",
    "TrafficCone",
    "TrafficLight",
    "Train",
    "Tree",
    "Truck_Bus",
    "Tunnel",
    "VegetationMisc",
    "Void",
    "Wall",
]

_CAMVID_COLORS = [
    (64, 128, 64),
    (192, 0, 128),
    (0, 128, 192),
    (0, 128, 64),
    (128, 0, 0),
    (64, 0, 128),
    (64, 0, 192),
    (192, 128, 64),
    (192, 192, 128),
    (64, 64, 128),
    (128, 0, 192),
    (192, 0, 64),
    (128, 128, 64),
    (192, 0, 192),
    (128, 64, 64),
    (64, 192, 128),
    (64, 64, 0),
    (128, 64, 128),
    (128, 128, 192),
    (0, 0, 192),
    (192, 128, 128),
    (128, 128, 128),
    (64, 128, 192),
    (0, 0, 64),
    (0, 64, 64),
    (192, 64, 128),
    (128, 128, 0),
    (192, 128, 192),
    (64, 0, 64),
    (192, 192, 0),
    (0, 0, 0),
    (64, 192, 0),
]


class CamvidDataset(BaseDataset):
    def __init__(self, data_path):
        super(CamvidDataset, self).__init__(data_path, _CAMVID_CLASSES, _CAMVID_COLORS)
        _camvid_data_path = os.path.join(self._data_path, "camvid")

        _all_images = glob.glob(os.path.join(_camvid_data_path, "*.png"))
        _all_images.sort(key=BaseDataset.human_sort)

        self._gt_paths = [img_path for img_path in _all_images if img_path.endswith("_L.png")]
        self._image_paths = [img_path for img_path in _all_images if img_path not in self._gt_paths]

        self._color_idx_dict = BaseDataset.color_to_color_idx_dict(self._colors)

    def __getitem__(self, idx):
        image = cv2.imread(self._image_paths[idx])
        rgb_gt = cv2.imread(self._gt_paths[idx])
        rgb_gt = rgb_gt[:, :, [2, 1, 0]]

        gt = np.zeros(rgb_gt.shape[:2], dtype=np.int)

        for color in self._colors:
            gt[(rgb_gt == color).all(axis=2)] = self._color_idx_dict[color]

        return image, gt
