from net import translator
from data import load_jsrt, load_unpaired
import torch
from utils import translation_trainutils as TRANS
import os.path as osp
import os


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

    

    if args.dataset == "jsrt": 
        _, dataset_size, loader = load_jsrt.get_loader(
        args, split="all", get_maps=True)
        TRANS.save_translations(Gxy, Gyx, loader, dataset_size, device, osp.join(
        args.base, args.workspace, "outputs"), args.use_attention)
    else:
        if osp.isdir(osp.join(
        args.base, args.workspace, "outputs", "Train")) == False:
            os.makedirs(osp.join(
        args.base, args.workspace, "outputs", "Train"))
            os.makedirs(osp.join(
        args.base, args.workspace, "outputs", "Train", "X_to_Y"))
            os.makedirs(osp.join(
        args.base, args.workspace, "outputs", "Train", "Y_to_X"))

        if osp.isdir(osp.join(
        args.base, args.workspace, "outputs", "Test")) == False:
            os.makedirs(osp.join(
        args.base, args.workspace, "outputs", "Test"))
            os.makedirs(osp.join(
        args.base, args.workspace, "outputs", "Test", "X_to_Y"))
            os.makedirs(osp.join(
        args.base, args.workspace, "outputs", "Test", "Y_to_X"))

        _, dataset_size, loader = load_unpaired.get_unpaired_loader(
            args, split="train")
        TRANS.save_unpaired_translations(Gxy, Gyx, loader, dataset_size, device, osp.join(
        args.base, args.workspace, "outputs", "Train"), args.use_attention)

        _, dataset_size, loader = load_unpaired.get_unpaired_loader(
            args, split="test")
        TRANS.save_unpaired_translations(Gxy, Gyx, loader, dataset_size, device, osp.join(
        args.base, args.workspace, "outputs", "Test"), args.use_attention)
    
    
    args.trans_batch = tb
    args.dataparallel = dp
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
