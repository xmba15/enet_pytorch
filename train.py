#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from config import Config
from models import Enet
from trainer import Trainer
from data_loader import CamvidDataset


def main():
    dt_config = Config()
    dataset = CamvidDataset(data_path=dt_config.DATA_PATH, phase="train")


if __name__ == "__main__":
    main()
