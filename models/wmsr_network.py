import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
from torch.nn.utils import weight_norm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.wmsr_block import WAM, MeanShift
import matplotlib.pylab as plt
import torchvision
import os
import cv2
import random
def create_model(args):
    return WMSR(args)

class WMSR(nn.Module):
    def __init__(self, args):
        super(WMSR, self).__init__()

        self.scale = args.scale
        self.colors = args.colors
        self.window_sizes = args.window_sizes
        self.m_wmsr  = args.m_wmsr
        self.c_wmsr  = args.c_wmsr
        self.n_share = args.n_share
        self.r_expand = args.r_expand
        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [nn.Conv2d(self.colors, self.c_wmsr, kernel_size=3, stride=1, padding=1)]

        # define body module
        m_body = []
        for i in range(self.m_wmsr // (1+self.n_share)):
            if (i+1) % 2 == 1: 
                m_body.append(
                    WAM(hidden_dim=self.c_wmsr)
                )
            else:              
                m_body.append(
                    WAM(hidden_dim=self.c_wmsr)
                )
        # define tail module
        m_tail = [
            nn.Conv2d(self.c_wmsr, self.colors*self.scale*self.scale, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(self.scale)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        
        x = self.sub_mean(x)
        x = self.head(x)
        res1 = self.body(x)
        res = res1 + x

        x = self.tail(res)
        x = self.add_mean(x)

        return x[:, :, 0:H*self.scale, 0:W*self.scale]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        wsize = 2
        for i in range(1, len(self.window_sizes)):
            wsize = wsize*self.window_sizes[i] // math.gcd(wsize, self.window_sizes[i])
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


if __name__ == '__main__':
    pass
