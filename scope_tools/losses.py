import torch
from scope_tools.translation_metrics import *
import torch.nn as nn


def real_mse_loss(D_out):
    # how close is the produced output from being "real"?
    return torch.mean((D_out - 1) ** 2)


def fake_mse_loss(D_out):
    # how close is the produced output from being "fake"?
    return torch.mean(D_out ** 2)


def cycle_consistency_loss(real_im, reconstructed_im, lambda_weight, ssim_weight=0):
    # calculate reconstruction loss
    # return weighted loss
    ssim = SSIM().forward(real_im, reconstructed_im)
    # print("real_im",real_im.shape)
    # print("rec_im",reconstructed_im.shape)
    loss = torch.mean(torch.abs(real_im - reconstructed_im)
                      ) + (1-ssim) * ssim_weight
    return loss * lambda_weight


class segmentation_loss(nn.Module):
    def __init__(self, batch=True):
        super(segmentation_loss, self).__init__()
        self.batch = batch
        self.mae_loss = torch.nn.L1Loss()
        self.bce_loss = torch.nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def iou_loss(self, inputs, targets):
        smooth = 0.0
        #inputs = inputs.view(-1)
        #targets = targets.view(-1)

        intersection = (inputs * targets).sum(1).sum(1).sum(1)
        total = (inputs + targets).sum(1).sum(1).sum(1)
        union = total - intersection

        IoU = (intersection + smooth)/(union + smooth)

        return (1 - IoU.mean())

    def forward(self, y_true, y_pred):
        b = self.soft_dice_loss(y_true, y_pred)
        c = self.bce_loss(y_pred, y_true)
        d = self.iou_loss(y_pred, y_true)
        loss = b + d + c
        return loss
