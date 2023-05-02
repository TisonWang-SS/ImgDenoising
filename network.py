#!/usr/bin/env python
# -*- coding:utf-8 -*-

# Network Architecture

from unet import UNet, conv_down
import torch
from torch import nn


class UNetR(UNet):
    def __init__(self, in_channels=3, out_channels=3, depth=5, wf=6, padding=True, batch_norm=False, slope=0.2):
        super().__init__(in_channels, out_channels, depth, wf, padding, batch_norm, slope)


class UNetG(UNet):
    def __init__(self, in_channels=3, out_channels=3, depth=5, wf=6, padding=True, batch_norm=False, slope=0.2):
        super().__init__(in_channels, out_channels, depth, wf, padding, batch_norm, slope)
    
    def get_input_chs(self, in_chs):
        return in_chs + 1

def sample_generator(netG, x):
    z = torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]], device=x.device)
    x1 = torch.cat([x, z], dim=1)
    noise = netG(x1)

    return x + noise


class Discriminator(nn.Module):
    def __init__(self, in_channels=6, ndf=64, slope=0.2):
        super().__init__()
        self.ndf = ndf

        blocks = []
        # input is N x C x 128 x 128
        blocks.append(conv_down(in_channels, ndf, bias=False))
        blocks.append(nn.LeakyReLU(slope, inplace=True))
        # state size: N x ndf x 64 x 64
        blocks.append(conv_down(ndf, ndf*2, bias=False))
        blocks.append(nn.LeakyReLU(slope, inplace=True))
        # state size: N x (ndf*2) x 32 x 32
        blocks.append(conv_down(ndf*2, ndf*4, bias=False))
        blocks.append(nn.LeakyReLU(slope, inplace=True))
        # state size: N x (ndf*4) x 16 x 16
        blocks.append(conv_down(ndf*4, ndf*8, bias=False))
        blocks.append(nn.LeakyReLU(slope, inplace=True))
        # state size: N x (ndf*8) x 8 x 8
        blocks.append(conv_down(ndf*8, ndf*16, bias=False))
        blocks.append(nn.LeakyReLU(slope, inplace=True))
        # state size: N x (ndf*16) x 4 x 4
        blocks.append(nn.Conv2d(ndf*16, ndf*32, 4, stride=1, padding=0, bias=False))
        blocks.append(nn.LeakyReLU(slope, inplace=True))
        # state size: N x (ndf*32) x 1 x 1
        self.block = nn.Sequential(*blocks)
        self.output = nn.Linear(ndf*32, 1)

    def forward(self, x):
        feature_map = self.block(x).view(-1, self.ndf * 32)
        out = self.output(feature_map)
        return out.view(-1)