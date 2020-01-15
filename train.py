#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import math
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from config import Config
from models import Enet, EnetLoss, mean_iu_acc
from trainer import Trainer
from data_loader import CamvidDataset, CamvidDataTransform, CamvidDatasetConfig
from trainer import Trainer
from torchsummary import summary
import time


def train_model(net, data_loaders_dict, criterion, optimizer, num_epochs, scheduler=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.to(device)

    torch.backends.cudnn.benchmark = True

    num_train_imgs = len(data_loaders_dict["train"].dataset)
    num_val_imgs = len(data_loaders_dict["val"].dataset)
    batch_size = data_loaders_dict["train"].batch_size

    iteration = 1
    logs = []

    batch_multiplier = 3

    for epoch in range(num_epochs):
        t_epoch_start = time.time()
        t_iter_start = time.time()
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        print("-------------")
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-------------")

        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                print("（train）")

            else:
                if (epoch + 1) % 5 == 0:
                    net.eval()
                    print("-------------")
                    print("（val）")
                else:
                    continue

            count = 0
            for images, anno_class_imges in data_loaders_dict[phase]:
                if images.size()[0] == 1:
                    continue

                images = images.to(device)
                anno_class_imges = anno_class_imges.to(device)

                # multiple minibatchでのパラメータの更新
                if (phase == "train") and (count == 0):
                    optimizer.step()
                    optimizer.zero_grad()
                    count = batch_multiplier

                with torch.set_grad_enabled(phase == "train"):
                    outputs = net(images)
                    loss = criterion(outputs, anno_class_imges.long()) / batch_multiplier

                    if phase == "train":
                        loss.backward()
                        count -= 1

                        if iteration % 10 == 0:
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            print(
                                "iteration {} || Loss: {:.4f} || 10 iter: {:.4f} sec.".format(
                                    iteration, loss.item() / batch_size * batch_multiplier, duration
                                )
                            )
                            t_iter_start = time.time()

                        epoch_train_loss += loss.item() * batch_multiplier
                        iteration += 1
                    else:
                        epoch_val_loss += loss.item() * batch_multiplier

        t_epoch_finish = time.time()
        print("-------------")
        print(
            "epoch {} || Epoch_TRAIN_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f}".format(
                epoch + 1, epoch_train_loss / num_train_imgs, epoch_val_loss / num_val_imgs
            )
        )
        print("timer:  {:.4f} sec.".format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

        log_epoch = {
            "epoch": epoch + 1,
            "train_loss": epoch_train_loss / num_train_imgs,
            "val_loss": epoch_val_loss / num_val_imgs,
        }
        logs.append(log_epoch)

    torch.save(net.state_dict(), "saved_models/{}_{}.pth".format(type(net).__name__, epoch + 1))


def main():
    dt_config = Config()
    input_size = (360, 480)

    num_classes = CamvidDatasetConfig().num_classes
    data_transform = CamvidDataTransform(num_classes=num_classes, input_size=input_size)
    train_dataset = CamvidDataset(data_path=dt_config.DATA_PATH, phase="train", transform=data_transform)
    val_dataset = CamvidDataset(data_path=dt_config.DATA_PATH, phase="val", transform=data_transform)

    train_data_loader = DataLoader(train_dataset, batch_size=dt_config.BATCH_SIZE, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=dt_config.BATCH_SIZE)
    data_loaders_dict = {"train": train_data_loader, "val": val_data_loader}

    model = Enet(num_classes=num_classes, img_size=input_size)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    weighted_values = train_dataset.weighted_class()
    criterion = EnetLoss(weighted_values=weighted_values)

    train_model(model, data_loaders_dict, criterion, optimizer, num_epochs=dt_config.NUM_EPOCHS)


if __name__ == "__main__":
    main()
