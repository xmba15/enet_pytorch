#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import torch.utils.data as data


class BaseDataset(data.Dataset):
    def __init__(self, data_path, classes, colors):
        super(BaseDataset, self).__init__()

        assert(os.path.isdir(data_path))
        self._data_path = data_path

        self._image_path = []
        self._gt_path = []

        self._classes = classes
        self._colors = colors
        self._legend = BaseDataset.show_color_chart(self.classes, self.__colors)

    def __len__(self):
        return len(self._image_path)

    def __getitem__(self, idx):
        image = self._image_path[idx]
        gt = self._gt_path[idx]

        return image, gt

    @property
    def colors(self):
        return self._colors

    @property
    def legend(self):
        return self._legend

    @property
    def classes(self):
        return self._classes

    def get_overlay_image(self, idx):
        image, label = self.__getitem__(idx)
        mask = self._colors[label]
        overlay = ((0.3 * image) + (0.7 * mask)).astype("uint8")

    @staticmethod
    def show_color_chart(classes, colors):
        legend = np.zeros(((len(classes) * 25) + 25, 300, 3), dtype="uint8")
        for (i, (class_name, color)) in enumerate(zip(classes, colors)):
            color = [int(c) for c in color]
            cv2.putText(legend, class_name, (5, (i * 25) + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(legend, (100, (i * 25)), (300, (i * 25) + 25), tuple(color), -1)

        return
