from scope_tools import logger, losses, translation_metrics
import torch.nn as nn
import os.path as osp
import os
from tqdm import tqdm
import numpy as np
import cv2


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_one_epoch(epoch, Gxy, Gyx, Dx, Dy, loader, dataset_size, optimizer_G, optimizer_Dx, optimizer_Dy, device, log_path):
    Gxy.train()
    Gyx.train()
    Dx.train()
    Dy.train()
    epoch_data_Gxy = {"epochs": epoch, "lr": get_lr(
        optimizer_G), "loss": 0, "ssim": 0, "psnr": 0}
    epoch_data_Gyx = {"epochs": epoch, "lr": get_lr(
        optimizer_G), "loss": 0, "ssim": 0, "psnr": 0}
    print("==========================================================================")
    print("Training models: Epoch:", epoch)
    if osp.isfile(osp.join(log_path, "Gxy_train_logs.json")) and osp.isfile(osp.join(log_path, "Gyx_train_logs.json")):
        log_Gxy = logger.get_log(
            osp.join(log_path, "Gxy_train_logs.json"))
        log_Gyx = logger.get_log(
            osp.join(log_path, "Gyx_train_logs.json"))
    else:
        log_Gxy = logger.create_log(
            osp.join(log_path, "Gxy_train_logs.json"), mode="translation")
        log_Gyx = logger.create_log(
            osp.join(log_path, "Gyx_train_logs.json"), mode="translation")

    dataset_iterator = tqdm(enumerate(loader), total=len(loader))
    for batch_idx, pack in dataset_iterator:
        imX, imY, mapX, mapY = pack["imX"].to(device), pack["imY"].to(
            device), pack["mapX"].to(device), pack["mapY"].to(device)

        # Discriminator train:
        pyx = Gyx.forward(imY, mapY, deep_supervision=False)
        pxy = Gxy.forward(imX, mapX, deep_supervision=False)

        #Train: Dx
        optimizer_Dx.zero_grad()
        outX = Dx.forward(imX)
        outY = Dy.forward(pyx)

        Ldx_real = losses.real_mse_loss(outX)
        Ldx_fake = losses.fake_mse_loss(outY)
        dx_loss = Ldx_real + Ldx_fake
        dx_loss.backward()
        optimizer_Dx.step()

        #Train: Dy
        optimizer_Dy.zero_grad()
        outY = Dy.forward(imY)
        outX = Dy.forward(pxy)

        Ldy_real = losses.real_mse_loss(outY)
        Ldy_fake = losses.fake_mse_loss(outX)
        dy_loss = Ldy_real + Ldy_fake
        dy_loss.backward()
        optimizer_Dy.step()

        pyx, pxy = 0, 0
        # Generator train:
        pyx = Gyx.forward(imY, mapY, deep_supervision=False)
        pxy = Gxy.forward(imX, mapX, deep_supervision=False)
        optimizer_G.zero_grad()
        maxpool = nn.MaxPool2d(2)
        imX1 = maxpool(imX)
        imX2 = maxpool(imX1)
        imY1 = maxpool(imY)
        imY2 = maxpool(imY1)

        # Forward cycle: Y-X-Y
        Ly1 = losses.real_mse_loss(Dx(pyx))
        pyxy, pyxy1, pyxy2 = Gxy.forward(pyx, mapX, deep_supervision=True)
        Ly2 = losses.cycle_consistency_loss(imY, pyxy, 10) + losses.cycle_consistency_loss(
            imY1, pyxy1, 10) + losses.cycle_consistency_loss(imY2, pyxy2, 10)

        # Forward cycle: X-Y-X
        Lx1 = losses.real_mse_loss(Dy(pxy))
        pxyx, pxyx1, pxyx2 = Gyx.forward(pxy, mapY, deep_supervision=True)
        Lx2 = losses.cycle_consistency_loss(imX, pxyx, 10) + losses.cycle_consistency_loss(
            imX1, pxyx1, 10) + losses.cycle_consistency_loss(imX2, pxyx2, 10)

        #Loss and Backprop
        gLoss = Lx1 + Lx2 + Ly1 + Ly2
        gLoss.backward()
        optimizer_G.step()

        # Metric calculation
        epoch_data_Gxy["loss"] += (Lx1.item() + Lx2.item())/dataset_size
        epoch_data_Gyx["loss"] += (Ly1.item() + Ly2.item())/dataset_size
        trmets_xy = translation_metrics.translation_metrics(imY, pxy)
        trmets_yx = translation_metrics.translation_metrics(imX, pyx)
        for k in trmets_xy.keys():
            epoch_data_Gxy[k] += trmets_xy[k].item()/dataset_size
            epoch_data_Gyx[k] += trmets_yx[k].item()/dataset_size

    # Metric Logging
    logger.log_epoch(log_Gxy, epoch_data_Gxy)
    logger.save_log(log_Gxy, osp.join(log_path, "Gxy_train_logs.json"))
    logger.log_epoch(log_Gyx, epoch_data_Gyx)
    logger.save_log(log_Gyx, osp.join(log_path, "Gyx_train_logs.json"))
    print("loss:", epoch_data_Gxy["loss"]+epoch_data_Gyx["loss"])
    print("ssim_xy:", epoch_data_Gxy["ssim"],
          "ssim_yx", epoch_data_Gyx["ssim"])
    print("psnr_xy:", epoch_data_Gxy["psnr"],
          "psnr_yx", epoch_data_Gyx["psnr"])
    print("==========================================================================")
    return epoch_data_Gxy, epoch_data_Gyx


def test_one_epoch(epoch, Gxy, Gyx, loader, dataset_size, device, log_path):
    Gxy.eval()
    Gyx.eval()
    epoch_data_Gxy = {"epochs": epoch, "lr": 0,
                      "loss": 0, "ssim": 0, "psnr": 0}
    epoch_data_Gyx = {"epochs": epoch, "lr": 0,
                      "loss": 0, "ssim": 0, "psnr": 0}
    print("==========================================================================")
    print("Testing")
    if osp.isfile(osp.join(log_path, "Gxy_test_logs.json")) and osp.isfile(osp.join(log_path, "Gyx_test_logs.json")):
        log_Gxy = logger.get_log(
            osp.join(log_path, "Gxy_test_logs.json"))
        log_Gyx = logger.get_log(
            osp.join(log_path, "Gyx_test_logs.json"))
    else:
        log_Gxy = logger.create_log(
            osp.join(log_path, "Gxy_test_logs.json"), mode="translation")
        log_Gyx = logger.create_log(
            osp.join(log_path, "Gyx_test_logs.json"), mode="translation")

    dataset_iterator = tqdm(enumerate(loader), total=len(loader))
    for batch_idx, pack in dataset_iterator:
        imX, imY, mapX, mapY = pack["imX"].to(device), pack["imY"].to(
            device), pack["mapX"].to(device), pack["mapY"].to(device)
        pyx = Gyx.forward(imY, mapY, deep_supervision=False)
        pxy = Gxy.forward(imX, mapX, deep_supervision=False)
        trmets_xy = translation_metrics.translation_metrics(imY, pxy)
        trmets_yx = translation_metrics.translation_metrics(imX, pyx)
        for k in trmets_xy.keys():
            epoch_data_Gxy[k] += trmets_xy[k].item()/dataset_size
            epoch_data_Gyx[k] += trmets_yx[k].item()/dataset_size
    # Metric Logging
    logger.log_epoch(log_Gxy, epoch_data_Gxy)
    logger.save_log(log_Gxy, osp.join(log_path, "Gxy_test_logs.json"))
    logger.log_epoch(log_Gyx, epoch_data_Gyx)
    logger.save_log(log_Gyx, osp.join(log_path, "Gyx_test_logs.json"))
    print("ssim_xy:", epoch_data_Gxy["ssim"],
          "ssim_yx", epoch_data_Gyx["ssim"])
    print("psnr_xy:", epoch_data_Gxy["psnr"],
          "psnr_yx", epoch_data_Gyx["psnr"])
    print("==========================================================================")
    print("==========================================================================")
    return epoch_data_Gxy, epoch_data_Gyx


def save_translations(Gxy, Gyx, loader, dataset_size, device, save_path):
    Gxy.eval()
    Gyx.eval()
    print("==========================================================================")
    print("Generating Translations")
    dataset_iterator = tqdm(enumerate(loader), total=len(loader))
    for batch_idx, pack in dataset_iterator:
        imX, imY, mapX, mapY, name = pack["imX"].to(device), pack["imY"].to(
            device), pack["mapX"].to(device), pack["mapY"].to(device), pack["name"]
        pyx = Gyx.forward(imY, mapY, deep_supervision=False) * 255.0
        pxy = Gxy.forward(imX, mapX, deep_supervision=False) * 255.0
        pyx = np.array(pyx[0].detach().cpu(),
                       dtype=np.uint8).transpose(1, 2, 0)
        pxy = np.array(pxy[0].detach().cpu(),
                       dtype=np.uint8).transpose(1, 2, 0)
        cv2.imwrite(osp.join(save_path, "Y_to_X",
                    name[0].split("/")[-1]), pyx)
        cv2.imwrite(osp.join(save_path, "X_to_Y",
                    name[0].split("/")[-1]), pxy)
    print("Done!")
    print("==========================================================================")
    print("==========================================================================")
