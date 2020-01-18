#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import torch
import tqdm
from .base_trainer import BaseTrainer


_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, ".."))
try:
    from utils import inf_loop
except:
    print("cannot load modules")
    sys.exit(-1)


class Trainer(BaseTrainer):
    def __init__(
        self,
        model,
        criterion,
        metric_func,
        optimizer,
        num_epochs,
        save_period,
        config,
        data_loaders_dict,
        scheduler=None,
        device=None,
        len_epoch=None,
    ):
        super(Trainer, self).__init__(model, criterion, metric_func, optimizer, num_epochs, save_period, config, device)

        self.train_data_loader = data_loaders_dict["train"]
        self.val_data_loader = data_loaders_dict["val"]
        if len_epoch is None:
            self._len_epoch = len(self.train_data_loader)
        else:
            self.train_data_loader = inf_loop(self.train_data_loader)
            self._len_epoch = len_epoch

        self._do_validation = self.val_data_loader is not None
        self._scheduler = scheduler

    def _train_epoch(self, epoch):
        self._model.train()

        for batch_idx, (data, target) in tqdm.tqdm(enumerate(self.train_data_loader)):
            data, target = data.to(self._device), target.to(self._device)
            self._optimizer.zero_grad()
            output = self._model(data)
            train_loss = self._criterion(output, target)
            train_loss.backward()
            self._optimizer.step()

            if batch_idx == self._len_epoch:
                break

        if self._do_validation:
            val_loss = self._valid_epoch(epoch)

        if self._scheduler is not None:
            self._scheduler.step()

        return train_loss, val_loss

    def _valid_epoch(self, epoch):
        self._model.eval()

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_data_loader):
                data, target = data.to(self._device), target.to(self._device)

                output = self._model(data)
                loss = self._criterion(output, target)

        return loss
