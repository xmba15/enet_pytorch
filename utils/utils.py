#!/usr/bin/env python
# -*- coding: utf-8 -*-


def inf_loop(data_loader):
    from itertools import repeat

    for loader in repeat(data_loader):
        yield from loader


def process_one_image(model, img, colors, alpha=0.7):
    import torch
    import numpy as np

    processed_img = img / 255.0
    # [np.newaxis,:] equals unsqueeze(0)
    processed_img = torch.tensor(
        processed_img.transpose(2, 0, 1)[np.newaxis, :]
    ).float()
    if torch.cuda.is_available():
        processed_img = processed_img.cuda()

    output = model(processed_img)
    mask = (
        output.data.max(1)[1].cpu().numpy().reshape(img.shape[0], img.shape[1])
    )

    color_mask = np.array(colors)[mask]
    overlay = (((1 - alpha) * img) + (alpha * color_mask)).astype("uint8")

    return overlay
