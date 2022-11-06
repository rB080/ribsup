import torch
from scope_tools.translation_metrics import *


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
