from net import unet
from data import load_segdata
import torch
from utils import segmentation_trainutils as SEG
import os.path as osp


def run(args):
    device = torch.device(args.device)
    print("train_segmentor device:", device)
    model = unet.get_model(args)

    _, trainset_size, train_loader = load_segdata.get_loader(args)

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=args.seg_lr, weight_decay=0.0001)

    for epoch in range(1, args.seg_epochs+1):
        D = SEG.train_one_epoch(
            epoch, model, train_loader, trainset_size, optimizer, device, osp.join(args.base, args.workspace, "logs"))
        torch.save(model.state_dict(), osp.join(
            args.base, args.workspace, "saved_models", "segmentor.pth"))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
