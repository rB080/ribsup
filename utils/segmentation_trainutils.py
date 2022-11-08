from scope_tools import logger, losses, segmentation_metrics
import torch.nn as nn
import os.path as osp
import os
from tqdm import tqdm
import cv2
import numpy as np


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_one_epoch(epoch, model, loader, dataset_size, optimizer, device, log_path):
    model.train()
    segloss = losses.segmentation_loss()
    epoch_data = {}
    for k in logger.PRESET_SEGMENTATION_LOGS:
        epoch_data[k] = 0
    epoch_data["epochs"] = epoch
    epoch_data["lr"] = get_lr(optimizer)
    print("==========================================================================")
    print("Training models: Epoch:", epoch)
    if osp.isfile(osp.join(log_path, "unet_train_logs.json")):
        log = logger.get_log(
            osp.join(log_path, "unet_train_logs.json"))
    else:
        log = logger.create_log(
            osp.join(log_path, "unet_train_logs.json"), mode="segmentation")
    dataset_iterator = tqdm(enumerate(loader), total=len(loader))
    for batch_idx, pack in dataset_iterator:
        img, map = pack["img"].to(device), pack["map"].to(device)
        pred = model(img)
        optimizer.zero_grad()
        loss = segloss(map, pred)
        loss.backward()
        optimizer.step()

        epoch_data["loss"] += loss.item()/dataset_size
        mets = segmentation_metrics.segmentation_metrics(pred, map)
        for k in mets.keys():
            epoch_data[k] += mets[k]/dataset_size
    logger.log_epoch(log, epoch_data)
    logger.save_log(log, osp.join(log_path, "unet_train_logs.json"))
    print("train_iou:", epoch_data["iou"])
    print("==========================================================================")
    print("==========================================================================")
    return epoch_data


def save_data(model, loader, dataset_size, device, save_path):
    model.eval()
    print("==========================================================================")
    print("Generating Masks")
    dataset_iterator = tqdm(enumerate(loader), total=len(loader))
    if osp.isdir(osp.join(save_path, "mapX")) == False:
        os.makedirs(osp.join(save_path, "mapX"))
    if osp.isdir(osp.join(save_path, "mapY")) == False:
        os.makedirs(osp.join(save_path, "mapY"))
    for batch_idx, pack in dataset_iterator:
        imX, imY, name = pack["imX"].to(
            device), pack["imY"].to(device), pack["name"]
        predX, predY = model(imX)*255.0, model(imY)*255.0
        predX = np.array(predX[0].detach().cpu(),
                         dtype=np.uint8).transpose(1, 2, 0)
        predY = np.array(predY[0].detach().cpu(),
                         dtype=np.uint8).transpose(1, 2, 0)

        cv2.imwrite(osp.join(save_path, "mapX",
                    name[0].split("/")[-1]), predX)
        cv2.imwrite(osp.join(save_path, "mapY",
                    name[0].split("/")[-1]), predY)
    print("Done!")
    print("==========================================================================")
    print("==========================================================================")
