import os.path as osp
import json
import numpy as np
import matplotlib.pyplot as plt

def run(args):
    print("Analysing Now...")
    workbase = osp.join(args.base, args.workspace)
    log_base = osp.join(workbase, "logs")
    mod_base = osp.join(workbase, "saved_models")
    analysis_base = osp.join(workbase, "analyses")
    f = open(osp.join(workbase, "configs.json"))
    training_configs = json.load(f)
    print("Workspace Data Loaded!")

    print("Analysing translator training!")
    f = open(osp.join(log_base, "Gxy_train_logs.json"))
    Gxy_tr = json.load(f)
    f = open(osp.join(log_base, "Gyx_train_logs.json"))
    Gyx_tr = json.load(f)
    f = open(osp.join(log_base, "Gxy_test_logs.json"))
    Gxy_te = json.load(f)
    f = open(osp.join(log_base, "Gyx_test_logs.json"))
    Gyx_te = json.load(f)

    ssim_seq_Gxy_train = np.array(Gxy_tr["ssim"])
    ssim_seq_Gyx_train = np.array(Gyx_tr["ssim"])
    ssim_seq_Gxy_test  = np.array(Gxy_te["ssim"])
    ssim_seq_Gyx_test  = np.array(Gyx_te["ssim"])
    lr_seq = np.array(Gyx_tr["lr"])

    plt.title("Training Curve: Train SSIM")
    plt.plot(ssim_seq_Gxy_train, label="SSIM for Gxy (Best:"+str(ssim_seq_Gxy_train.max())+")")
    plt.plot(ssim_seq_Gyx_train, label="SSIM for Gyx (Best:"+str(ssim_seq_Gyx_train.max())+")")
    plt.xlabel("Epochs")
    plt.ylabel("SSIM score")
    plt.legend()
    plt.savefig(osp.join(analysis_base, "train_plots.png"))
    plt.clf()

    plt.title("Training Curve: Test SSIM")
    plt.plot(ssim_seq_Gxy_test, label="SSIM for Gxy (Best:"+str(ssim_seq_Gxy_test.max())+")")
    plt.plot(ssim_seq_Gyx_test, label="SSIM for Gyx (Best:"+str(ssim_seq_Gyx_test.max())+")")
    plt.xlabel("Epochs")
    plt.ylabel("SSIM score")
    plt.legend()
    plt.savefig(osp.join(analysis_base, "test_plots.png"))
    plt.clf()

    plt.title("Training Curve: Learning rate reduction")
    plt.plot(lr_seq)
    plt.xlabel("Epochs")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.savefig(osp.join(analysis_base, "LR_plot.png"))
    plt.clf()
    print("Done!")