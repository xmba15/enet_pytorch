#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import os
import math
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch
from config import Config
from models import Enet, EnetLoss, mean_iu_acc
from trainer import Trainer
from data_loader import CamvidDataset, CamvidDataTransform, CamvidDatasetConfig
from trainer import Trainer
from torchsummary import summary
import pandas as pd


def main():
    dt_config = Config()
    input_size = (360, 480)
    num_classes = CamvidDatasetConfig().num_classes

    ############################################################################
    # fmt: off
    # data loader preparation
    data_transform = CamvidDataTransform(
        num_classes=num_classes,
        input_size=input_size
    )
    train_dataset = CamvidDataset(
        data_path=dt_config.DATA_PATH,
        phase="train",
        transform=data_transform
    )
    val_dataset = CamvidDataset(
        data_path=dt_config.DATA_PATH,
        phase="val",
        transform=data_transform
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=dt_config.BATCH_SIZE,
        shuffle=True
    )
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=dt_config.BATCH_SIZE
    )
    data_loaders_dict = {
        "train": train_data_loader,
        "val": val_data_loader
    }
    # fmt: off
    ############################################################################

    ############################################################################
    # encoder_only
    encoder_only = True

    model = Enet(
        num_classes=num_classes, img_size=input_size, encoder_only=encoder_only
    )

    weighted_values = train_dataset.weighted_class()
    criterion = EnetLoss(
        weighted_values=weighted_values, encoder_only=encoder_only
    )
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=2e-4)
    scheduler = lr_scheduler.StepLR(
        optimizer=optimizer, step_size=100, gamma=0.1
    )

    # fmt: off
    camvid_trainer = Trainer(
        model=model,
        criterion=criterion,
        metric_func=None,
        optimizer=optimizer,
        num_epochs=dt_config.NUM_EPOCHS,
        save_period=dt_config.SAVED_PERIOD,
        config=dt_config,
        data_loaders_dict=data_loaders_dict,
        scheduler=scheduler,
    )
    # fmt: on

    encoder_logs = camvid_trainer.train()
    ############################################################################

    ############################################################################
    # encoder + decoder

    model = camvid_trainer.model

    model.remove_encoder_classifier()
    criterion.encoder_only = False
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=2e-4)
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=0.1)

    # fmt: off
    camvid_trainer = Trainer(
        model=model,
        criterion=criterion,
        metric_func=None,
        optimizer=optimizer,
        num_epochs=dt_config.NUM_EPOCHS,
        save_period=dt_config.SAVED_PERIOD,
        config=dt_config,
        data_loaders_dict=data_loaders_dict,
        scheduler=scheduler,
    )
    # fmt: on

    encoder_decoder_logs = camvid_trainer.train()
    ############################################################################

    encoder_df = pd.DataFrame(encoder_logs)
    encoder_decoder_df = pd.DataFrame(encoder_decoder_logs)
    encoder_df.to_csv(os.path.join(dt_config.SAVED_MODEL_PATH, "encoder.csv"))
    encoder_decoder_df.to_csv(os.path.join(dt_config.SAVED_MODEL_PATH, "encoder_decoder.csv"))


if __name__ == "__main__":
    main()
