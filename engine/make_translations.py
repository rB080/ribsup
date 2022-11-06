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
    print("make_translations device:", device)
    Gxy, Gyx, _, _ = translator.get_model_set(device)

    Gxy.load_state_dict(torch.load(osp.join(
        args.base, args.workspace, "saved_models", "Gxy_bm.pth"), map_location=device))
    Gyx.load_state_dict(torch.load(osp.join(
        args.base, args.workspace, "saved_models", "Gym_bm.pth"), map_location=device))

    _, dataset_size, loader = load_jsrt.get_loader(
        args.data_root, split="all", get_maps=False)
    TRANS.save_translations(Gxy, Gyx, loader, dataset_size, device, osp.join(
        args.base, args.workspace, "outputs"))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
