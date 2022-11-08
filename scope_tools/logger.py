import json
import sys

PRESET_TRANSLATION_LOGS = ["epochs", "lr", "loss", "ssim", "psnr"]
PRESET_SEGMENTATION_LOGS = ["epochs", "lr", "loss",
                            "acc", "sen", "pre", "rec", "dsc", "iou"]
log_dicts = {"translation": PRESET_TRANSLATION_LOGS,
             "segmentation": PRESET_SEGMENTATION_LOGS}


def create_log(path, mode="translation"):
    assert mode in ["translation", "segmentation"], "Invalid Mode!"
    quants = log_dicts[mode]
    log = {}
    for k in quants:
        log[k] = []
    with open(path, "w") as file:
        file.write(json.dumps(log))
    return log


def log_consistency(log):
    sizes = []
    for k in log.keys():
        sizes.append(len(log[k]))
    if len(set(sizes)) == 1:
        return True
    else:
        return False


def get_log(path):
    with open(path, "rb") as file:
        log = json.load(file)
    return log


def log_epoch(log, epoch_data):
    for k in epoch_data.keys():
        log[k].append(epoch_data[k])
    assert log_consistency(log), "Inconsistent number of epoch data in logfile"
    return log


def save_log(log, path):
    with open(path, "w") as file:
        file.write(json.dumps(log))


class Logger(object):
    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log = open(outfile, "a")
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
