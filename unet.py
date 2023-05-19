# Adapted from https://github.com/jvanvugt/pytorch-unet/blob/master/unet.py

import torch
from torch import nn


def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        depth=5,
        wf=6,
        padding=True,
        batch_norm=False,
        slope=0.2
    ):
        super(UNet, self).__init__()
        self.padding = padding
        self.depth = depth
        prev_channels = self.get_input_chs(in_channels)
        self.down_path = nn.ModuleList()
        for i in range(depth):
            need_ds = True if i < depth - 1 else False
            self.down_path.append(
                UNetConvBlock(prev_channels, (2**i) * wf, padding, batch_norm, slope, need_ds)
            )
            prev_channels = (2**i) * wf

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, (2**i) * wf, padding, batch_norm, slope)
            )
            prev_channels = (2**i) * wf

        self.last = nn.Conv2d(prev_channels, out_channels, kernel_size=3, stride=1, padding=int(padding), bias=True)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            if i != len(self.down_path) - 1:
                x, x_same = down(x)
                blocks.append(x_same)
            else:
                x = down(x)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)
    
    def get_input_chs(self, in_chs):
        return in_chs
    
    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, slope=0, need_ds=True):
        super(UNetConvBlock, self).__init__()
        self.ds = need_ds
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        if slope == 0:
            block.append(nn.ReLU())
        else:
            block.append(nn.LeakyReLU(slope, inplace=True))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        if slope == 0:
            block.append(nn.ReLU())
        else:
            block.append(nn.LeakyReLU(slope, inplace=True))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

        if need_ds:
            self.ds = conv_down(out_size, out_size, bias=False)

    def forward(self, x):
        out_same = self.block(x)
        if self.ds:
            out_ds = self.ds(out_same)
            return out_ds, out_same
        else:
            return out_same


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm, slope=slope, need_ds=False)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)

        return out
