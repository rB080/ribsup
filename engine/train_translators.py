from net import translator
from data import load_jsrt
import torch
from utils import translation_trainutils as TRANS
import os.path as osp


def run(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    #device = torch.device('cpu')
    print("train_translators device:", device)
    Gxy, Gyx, Dx, Dy = translator.get_model_set(device)
    optimizer_G = torch.optim.Adam(
        params=list(Gxy.parameters()) + list(Gyx.parameters()), lr=args.trans_lr, weight_decay=0.0001)
    optimizer_Dx = torch.optim.Adam(
        params=Dx.parameters(), lr=args.trans_lr, weight_decay=0.0001)
    optimizer_Dy = torch.optim.Adam(
        params=Dy.parameters(), lr=args.trans_lr, weight_decay=0.0001)
    _, train_dataset_size, train_loader = load_jsrt.get_loader(
        args.data_root, split="train")
    _, test_dataset_size, test_loader = load_jsrt.get_loader(
        args.data_root, split="test")
    best_ssim_xy = 0
    best_ssim_yx = 0
    for epoch in range(1, args.trans_epochs+1):
        D1, D2 = TRANS.train_one_epoch(epoch, Gxy, Gyx, Dx, Dy, train_loader, train_dataset_size,
                                       optimizer_G, optimizer_Dx, optimizer_Dy, device, osp.join(args.base, args.workspace, "logs"))
        D1, D2 = TRANS.test_one_epoch(epoch, Gxy, Gyx, test_loader, test_dataset_size, device, osp.join(
            args.base, args.workspace, "logs"))
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
