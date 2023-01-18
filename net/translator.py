import torch
import torch.nn as nn
import torch.nn.functional as F
from net.utils import *


class Discriminator(nn.Module):

    def __init__(self, conv_dim=64):
        super(Discriminator, self).__init__()

        # Define all convolutional layers
        # Should accept an RGB image as input and output a single value
        self.layer_1 = conv(3, conv_dim, 4, batch_norm=False)
        self.layer_2 = conv(conv_dim, conv_dim*2, 4)
        self.layer_3 = conv(conv_dim*2, conv_dim*4, 4)
        self.layer_4 = conv(conv_dim*4, conv_dim*8, 4)
        self.layer_5 = conv(conv_dim*8, 1, 4, 1, batch_norm=False)

    def forward(self, x):
        # define feedforward behavior
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = F.relu(self.layer_4(x))

        x = self.layer_5(x)
        return x


class Generator_v4(nn.Module):
    def __init__(self, bilinear=False):
        super(Generator_v4, self).__init__()
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        # self.down4 = Down(512, 1024 // factor)

        # self.up4 = Up(1024, 512 // factor, bilinear)
        self.up3 = Up(512, 255 // factor, bilinear)
        self.up2 = Up(256, 127, bilinear)
        self.up1 = Up(128, 63, bilinear)

        # self.sa4 = SpatialAttention(out_channels=512)
        self.sa3 = SpatialAttention(out_channels=256)
        self.sa2 = SpatialAttention(out_channels=128)
        self.sa1 = SpatialAttention(out_channels=64)

        layers = []
        for n in range(9):
            layers.append(ResidualBlock(512))
        self.res_blocks = nn.Sequential(*layers)
        self.inc = DoubleConv(3, 64)
        self.outconv = OutConv(64, 3)
        self.outconv1 = OutConv(128, 3)
        self.outconv2 = OutConv(256, 3)

        # UNet 2 (used as attention)
        self.mp = torch.nn.MaxPool2d(2)

        # self.outa = UNet()

    def forward(self, x, attention_map, deep_supervision=True, attention=True):
        # unet 2
        #print(x.shape, attention_map.shape)
        amap1 = self.mp(attention_map)
        amap2 = self.mp(amap1)
        

        # unet main

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        xb = self.down3(x3)

        # xb = self.down4(x4)
        xb = self.res_blocks(xb)

        x1 = self.sa1(x1)
        x2 = self.sa2(x2)
        x3 = self.sa3(x3)
        # x4 = self.sa4(x4)

        # y4 = self.up4(xb, x4)
        
        y3 = self.up3(xb, x3)
        print(torch.mean(y3, 3).shape, y3.shape, amap2.shape)
        if attention: y3 = torch.cat((amap2, y3), 1)
        else: y3 = torch.cat((torch.mean(y3, 0), y3), 1)
        
        y2 = self.up2(y3, x2)
        if attention: y2 = torch.cat((amap1, y2), 1)
        else: y2 = torch.cat((torch.mean(y2, 0), y2), 1)

        y1 = self.up1(y2, x1)
        if attention: y1 = torch.cat((attention_map, y1), 1)
        else: y1 = torch.cat((torch.mean(y1, 0), y1), 1)

        out = self.outconv(y1)
        out1 = self.outconv1(y2)
        out2 = self.outconv2(y3)
        out = F.tanh(out)
        out1 = F.tanh(out1)
        out2 = F.tanh(out2)
        if deep_supervision:
            return out, out1, out2
        else:
            return out


def get_model_set(args):
    Gxy, Gyx, Dx, Dy = Generator_v4(), Generator_v4(), Discriminator(), Discriminator()
    if args.device == 'cuda' and args.dataparallel == True:
        Gxy = torch.nn.DataParallel(Gxy).cuda()
        Gyx = torch.nn.DataParallel(Gyx).cuda()
        Dx = torch.nn.DataParallel(Dx).cuda()
        Dy = torch.nn.DataParallel(Dy).cuda()
    elif args.device == 'cuda' and args.dataparallel == False:
        Gxy.to(args.device)
        Gyx.to(args.device)
        Dx.to(args.device)
        Dy.to(args.device)
    return Gxy, Gyx, Dx, Dy
