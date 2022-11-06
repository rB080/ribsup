import argparse
import os
import os.path as osp
from engine import train_segmentor, train_translators, make_segmentation, make_translations
import json
from scope_tools.logger import Logger


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


defaults = {
    "base": r"E:\\Projects\\Ongoing\\Rib_Suppression_IIT\\Train_outputs",
    "data_root": r"E:\\Projects\\Datasets\\Biomedical\\JSRT\\X-Ray Shadow Suppression Kaggle",
    "segdata_root": r"E:\\Projects\\Datasets\\Biomedical\\CT_IITD"
}


def get_args_parser():
    parser = argparse.ArgumentParser()

    #Workspace and Paths
    parser.add_argument('--workspace', default='untitled_training', type=str)
    parser.add_argument('--log_name', default='new_log', type=str)
    parser.add_argument('--base', default=defaults["base"], type=str)
    parser.add_argument('--data_root', default=defaults["data_root"], type=str)
    parser.add_argument(
        '--segdata_root', default=defaults["segdata_root"], type=str)

    # Segmentation args
    parser.add_argument('--seg_epochs', default=50, type=int)
    parser.add_argument('--seg_lr', default=1e-4, type=float)

    # Translation args
    parser.add_argument('--trans_epochs', default=100, type=int)
    parser.add_argument('--trans_lr', default=1e-5, type=float)

    # Jobs
    parser.add_argument('--train_segmentor', default=False, type=str2bool)
    parser.add_argument('--make_segmentation', default=False, type=str2bool)
    parser.add_argument('--train_translators', default=False, type=str2bool)
    parser.add_argument('--make_translations', default=False, type=str2bool)

    args = parser.parse_args()
    return args


args = get_args_parser()
configs = vars(args)

# Workspace creation:
workbase = osp.join(args.base, args.workspace)
log_base = osp.join(workbase, "logs")
mod_base = osp.join(workbase, "saved_models")
analysis_base = osp.join(workbase, "analyses")
out_base_xy = osp.join(workbase, "outputs", "X_to_Y")
out_base_yx = osp.join(workbase, "outputs", "Y_to_X")

if osp.isdir(workbase) == False:
    os.makedirs(workbase)
if osp.isdir(log_base) == False:
    os.makedirs(log_base)
if osp.isdir(mod_base) == False:
    os.makedirs(mod_base)
if osp.isdir(analysis_base) == False:
    os.makedirs(analysis_base)
if osp.isdir(out_base_xy) == False:
    os.makedirs(out_base_xy)
if osp.isdir(out_base_yx) == False:
    os.makedirs(out_base_yx)
print("Workspace Created Successfully!", flush=True)

Logger(osp.join(workbase, args.log_name) + ".log")

# Show console settings:
print("Console Settings:")
print(configs)
file = open(os.path.join(workbase, 'configs.json'), 'w')
file.write(json.dumps(configs))
file.close()
print("Console settings saved!")

# Job lists:
if args.train_segmentor:  # Segmentation model training
    print("Training Segmentation Model now!")
    train_segmentor.run(args)
    print("Segmentation Model Training Done!!")

if args.make_segmentation:  # Save segmentation maps from pretrained model
    print("Saving attention maps from pretrained segmentation model now!")
    make_segmentation.run(args)
    print("Attention maps saved successfully!!")

if args.train_translators:  # Train the CycleGANs for image translation
    print("Starting CycleGAN trianing now!")
    train_translators.run(args)
    print("Translation models trained!!")

if args.make_translations:  # Generate translated images for visualization
    print("Generating translation predictions now!")
    make_translations.run(args)
    print("Translation predictions saved!!")
