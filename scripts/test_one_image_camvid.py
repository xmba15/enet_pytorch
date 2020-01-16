#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import cv2
import torch
import numpy as np


_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, ".."))
try:
    from models import Enet
    from data_loader import CamvidDataTransform
    from data_loader import CamvidDatasetConfig
except Exception as e:
    print(e)
    sys.exit(-1)


parser = argparse.ArgumentParser()
parser.add_argument("--image_path", required=True)
parser.add_argument("--model_path", required=True)
parser.add_argument("--overlay_ratio", type=float, default=0.7)
parsed_args = parser.parse_args()


def main(args):
    colors = CamvidDatasetConfig.CAMVID_COLORS
    img = cv2.imread(args.image_path)
    img2 = cv2.imread(args.image_path)
    assert img is not None
    model = Enet(num_classes=12, img_size=(360, 480))
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    data_transform = CamvidDataTransform(num_classes=12, input_size=(360, 480))
    img = data_transform(image=img)

    if torch.cuda.is_available():
        model = model.cuda()
        img = img.cuda()

    img = img.unsqueeze(0)
    output = model(img)
    mask = output.data.max(1)[1].cpu().numpy().reshape(360, 480)

    color_mask = np.array(colors)[mask]
    alpha = args.overlay_ratio
    overlay = (((1 - alpha) * img2) + (alpha * color_mask)).astype("uint8")
    cv2.imwrite("overlay.png", overlay)


if __name__ == "__main__":
    main(args=parsed_args)
