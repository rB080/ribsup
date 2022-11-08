from net import translator
from data import load_jsrt
import torch
from utils import translation_trainutils as TRANS
import os.path as osp


def run(args):
    tb, dp = args.trans_batch, args.dataparallel
    args.trans_batch = 1
    #args.dataparallel = False

    device = torch.device(args.device)
    print("make_translations device:", device)
    Gxy, Gyx, _, _ = translator.get_model_set(args)

    Gxy.load_state_dict(torch.load(osp.join(
        args.base, args.workspace, "saved_models", "Gxy_bm.pth")))
    Gyx.load_state_dict(torch.load(osp.join(
        args.base, args.workspace, "saved_models", "Gyx_bm.pth")))

    

    _, dataset_size, loader = load_jsrt.get_loader(
        args, split="all", get_maps=True)
    
    TRANS.save_translations(Gxy, Gyx, loader, dataset_size, device, osp.join(
        args.base, args.workspace, "outputs"))
    args.trans_batch = tb
    args.dataparallel = dp
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
