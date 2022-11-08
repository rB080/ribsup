import torch.nn as nn
import torch.nn.functional as F
from net.utils import *


class UNet(nn.Module):
    def __init__(self, bilinear=False):
        super(UNet, self).__init__()
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.act = nn.Sigmoid()

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        self.up4 = Up(1024, 512 // factor, bilinear)
        self.up3 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128, bilinear)
        self.up1 = Up(128, 64, bilinear)

        self.inc = DoubleConv(3, 64)
        self.outconv = OutConv(64, 1)

    def forward(self, x):

        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x4 = self.down3(x3)

        xb = self.down4(x4)

        y4 = self.up4(xb, x4)
        y3 = self.up3(y4, x3)
        y2 = self.up2(y3, x2)
        y1 = self.up1(y2, x1)

        out = self.outconv(y1)
        out = self.act(out)
        return out


def get_model(args):
    model = UNet()
    if args.device == 'cuda' and args.dataparallel == True:
        model = torch.nn.DataParallel(model).cuda()
    elif args.device == 'cuda' and args.dataparallel == False:
        model.to(device)
    return model