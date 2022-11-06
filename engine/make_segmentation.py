from net import unet
from data import load_jsrt
import torch
from utils import segmentation_trainutils as SEG
import os.path as osp


def run(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    #device = torch.device('cpu')
    print("make_segmentation device:", device)
    model = unet.get_model()
    model.to(device)
    model.load_state_dict(torch.load(osp.join(
        args.base, args.workspace, "saved_models", "segmentor.pth"), map_location=device))
    _, dataset_size, loader = load_jsrt.get_loader(
        args.data_root, split="all", get_maps=False)
    SEG.save_data(model, loader, dataset_size, device, args.data_root)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
