from net import unet
from data import load_jsrt, load_unpaired
import torch
from utils import segmentation_trainutils as SEG
import os.path as osp


def run(args):
    device = torch.device(args.device)
    print("make_segmentation device:", device)
    model = unet.get_model(args)

    model.load_state_dict(torch.load(osp.join(
        args.base, args.workspace, "saved_models", "segmentor.pth"), map_location=device))
    
    tb = args.trans_batch
    args.trans_batch = 1

    if args.dataset == "jsrt":
        _, dataset_size, loader = load_jsrt.get_loader(
            args, split="all", get_maps=False)
        
        SEG.save_data(model, loader, dataset_size, device, args.data_root)
    else:
        _, dataset_size, loader = load_unpaired.get_unpaired_loader(
            args, split="train", get_maps=False)
        SEG.save_unpaired_data(model, loader, dataset_size, device, args.data_root+"/Train")

        _, dataset_size, loader = load_unpaired.get_unpaired_loader(
            args, split="test", get_maps=False)
        SEG.save_unpaired_data(model, loader, dataset_size, device, args.data_root+"/Test")
    
    args.trans_batch = tb
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
