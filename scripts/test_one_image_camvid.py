#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import cv2
import torch
import numpy as np
import torch


_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, ".."))
try:
    from models import Enet
    from data_loader import CamvidDatasetConfig
    from utils import process_one_image
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
    assert img is not None
    model = Enet(num_classes=12, img_size=(360, 480))
    model.load_state_dict(torch.load(args.model_path)["state_dict"])
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    overlay = process_one_image(model, img, colors, alpha=args.overlay_ratio)
    cv2.imwrite("overlay.png", overlay)


if __name__ == "__main__":
    main(args=parsed_args)
