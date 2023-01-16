from net import translator
from data import load_jsrt, load_unpaired
import torch
from utils import translation_trainutils as TRANS
import os.path as osp
import math

def lr_controller(initial_lr, epoch, set_num_epoch=100, decay_factor=10):
    ratio = math.pow(1/decay_factor, 1/(set_num_epoch-1))
    LR_i = initial_lr * math.pow(ratio, epoch - 1)
    return LR_i

def run(args):
    device = torch.device(args.device)
    print("train_translators device:", device)
    Gxy, Gyx, Dx, Dy = translator.get_model_set(args)
    
    if args.dataset == "jsrt":
        _, train_dataset_size, train_loader = load_jsrt.get_loader(
            args, split="train")
        _, test_dataset_size, test_loader = load_jsrt.get_loader(
            args, split="test")
    else: 
        _, train_dataset_size, train_loader = load_unpaired.get_loader(
            args, split="train")
        _, test_dataset_size, test_loader = load_unpaired.get_loader(
            args, split="test")
    
    best_ssim_xy = 0
    best_ssim_yx = 0
    for epoch in range(1, args.trans_epochs+1):
        LR = lr_controller(args.trans_lr, epoch, set_num_epoch=100, decay_factor=1.00001)
        optimizer_G = torch.optim.Adam(
        params=list(Gxy.parameters()) + list(Gyx.parameters()), lr=LR, weight_decay=0.0001)
        optimizer_Dx = torch.optim.Adam(
            params=Dx.parameters(), lr=LR, weight_decay=0.0001)
        optimizer_Dy = torch.optim.Adam(
            params=Dy.parameters(), lr=LR, weight_decay=0.0001)
        print("Learning rate:", LR)
        D1, D2 = TRANS.train_one_epoch(epoch, Gxy, Gyx, Dx, Dy, train_loader, train_dataset_size,
                                       optimizer_G, optimizer_Dx, optimizer_Dy, device, osp.join(args.base, args.workspace, "logs"), args.ssim_weight,
                                       args.deep_supervision, args.use_attention)
        D1, D2 = TRANS.test_one_epoch(epoch, Gxy, Gyx, test_loader, test_dataset_size, device, osp.join(
            args.base, args.workspace, "logs"), args.use_attention)
        torch.save(Gxy.state_dict(), osp.join(
            args.base, args.workspace, "saved_models", "Gxy_lm.pth"))
        torch.save(Gyx.state_dict(), osp.join(
            args.base, args.workspace, "saved_models", "Gyx_lm.pth"))
        torch.save(Dx.state_dict(), osp.join(
            args.base, args.workspace, "saved_models", "Dx.pth"))
        torch.save(Dy.state_dict(), osp.join(
            args.base, args.workspace, "saved_models", "Dy.pth"))

        if D1["ssim"] >= best_ssim_xy:
            torch.save(Gxy.state_dict(), osp.join(
                args.base, args.workspace, "saved_models", "Gxy_bm.pth"))
            best_ssim_xy = D1["ssim"]

        if D2["ssim"] >= best_ssim_yx:
            torch.save(Gyx.state_dict(), osp.join(
                args.base, args.workspace, "saved_models", "Gyx_bm.pth"))
            best_ssim_yx = D1["ssim"]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
