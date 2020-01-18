#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import cv2
import torch
import numpy as np
import time
import tqdm


_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, ".."))
try:
    from models import Enet
    from data_loader import CamvidDatasetConfig, CamvidDataset
    from config import Config
    from utils import process_one_image
except Exception as e:
    print(e)
    sys.exit(-1)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path", default=os.path.join(_CURRENT_DIR, "../saved_models/checkpoint_Enet_epoch_299.pth"),
)
parser.add_argument("--save_gif", action="store_true")

parser.add_argument("--overlay_ratio", type=float, default=0.7)
parsed_args = parser.parse_args()


def main(args):
    assert os.path.isfile(args.model_path)

    colors = CamvidDatasetConfig.CAMVID_COLORS
    dt_config = Config()
    num_classes = CamvidDatasetConfig().num_classes

    model = Enet(num_classes=12, img_size=(360, 480))
    model.load_state_dict(torch.load(args.model_path)["state_dict"])
    model.eval()

    train_dataset = CamvidDataset(data_path=dt_config.DATA_PATH, phase="test",)

    if torch.cuda.is_available():
        model = model.cuda()

    if args.save_gif:
        results = []
    for img, _ in tqdm.tqdm(train_dataset):
        overlay = process_one_image(model, img, colors, alpha=args.overlay_ratio)
        result = np.hstack((img, overlay))
        if args.save_gif:
            results.append(result[:, :, [2, 1, 0]])
        else:
            cv2.imshow("result", result)
            time.sleep(0.05)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    if args.save_gif:
        import imageio

        gif_path = os.path.abspath(os.path.join(_CURRENT_DIR, "../docs/images", "camvid_test_results.gif"))
        with imageio.get_writer(gif_path, mode="I") as writer:
            print("start creating gif file...")
            for img in tqdm.tqdm(results):
                writer.append_data(img)
        print("gif file saved into {}".format(gif_path))


if __name__ == "__main__":
    main(args=parsed_args)
