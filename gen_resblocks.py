import math

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import Higham_norm
# from links import CategoricalConditionalBatchNorm2d


def _upsample(x):
    h, w = x.size()[2:]
    return F.interpolate(x, size=(h * 2, w * 2), mode='bilinear')


class Block(nn.Module):

    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1,
                 activation=F.relu, upsample=False, num_classes=0):
        super(Block, self).__init__()

        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_ch != out_ch or upsample
        if h_ch is None:
            h_ch = out_ch
        self.num_classes = num_classes

        # Register layrs
        self.c1 = Higham_norm.spectral_norm(nn.Conv2d(in_ch, h_ch, ksize, 1, pad), use_adaptivePC=False, pclevel=1)
        self.c2 = Higham_norm.spectral_norm(nn.Conv2d(h_ch, out_ch, ksize, 1, pad), use_adaptivePC=False, pclevel=1)
        self.b1 = nn.BatchNorm2d(in_ch)
        self.b2 = nn.BatchNorm2d(h_ch)
        if self.learnable_sc:
            self.c_sc = Higham_norm.spectral_norm(nn.Conv2d(in_ch, out_ch, 1), use_adaptivePC=False, pclevel=1)
        self._initialize()

    def _initialize(self):
        init.xavier_normal_(self.c1.weight.data)
        init.xavier_normal_(self.c2.weight.data)
        if self.learnable_sc:
            init.xavier_normal_(self.c_sc.weight.data, gain=1)

    def forward(self, x, y=None, z=None, **kwargs):
        return self.shortcut(x) + self.residual(x, y, z)

    def shortcut(self, x, **kwargs):
        if self.learnable_sc:
            if self.upsample:
                h = _upsample(x)
            h = self.c_sc(h)
            return h
        else:
            return x

    def residual(self, x, y=None, z=None, **kwargs):
        if y is not None:
            h = self.b1(x, y, **kwargs)
        else:
            h = self.b1(x)
        h = self.activation(h)
        if self.upsample:
            h = _upsample(h)
        h = self.c1(h)
        if y is not None:
            h = self.b2(h, y, **kwargs)
        else:
            h = self.b2(h)
        return self.c2(self.activation(h))


class DeepBlock(nn.Module):

    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1,
                 activation=F.relu, upsample=False, num_classes=0):
        super(DeepBlock, self).__init__()

        self.activation = activation
        self.upsample = upsample
        self.learnable_sc = in_ch != out_ch or upsample
        if h_ch is None:
            h_ch = out_ch // 4
        self.num_classes = num_classes
        self.in_channels = in_ch
        self.out_channels = out_ch

        # Register layrs
        self.c1 = nn.Conv2d(in_ch, h_ch, kernel_size=1, padding=0)
        self.c2 = nn.Conv2d(h_ch, h_ch, ksize, padding=pad)
        self.c3 = nn.Conv2d(h_ch, h_ch, ksize, padding=pad)
        self.c4 = nn.Conv2d(h_ch, out_ch, kernel_size=1, padding=0)
        if self.num_classes > 0:
            self.b1 = CategoricalConditionalBatchNorm2d(num_classes, in_ch)
            self.b2 = CategoricalConditionalBatchNorm2d(num_classes, h_ch)
        else:
            self.b1 = nn.BatchNorm2d(in_ch)
            self.b2 = nn.BatchNorm2d(h_ch)
            self.b3 = nn.BatchNorm2d(h_ch)
            self.b4 = nn.BatchNorm2d(h_ch)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1)
        self._initialize()

    def _initialize(self):
        init.xavier_normal_(self.c1.weight.data)
        init.xavier_normal_(self.c2.weight.data)
        init.xavier_normal_(self.c3.weight.data)
        init.xavier_normal_(self.c4.weight.data)
        if self.learnable_sc:
            init.xavier_normal_(self.c_sc.weight.data, gain=1)

    def forward(self, x, y=None, z=None, **kwargs):
        # Project down to channel ratio
        h = self.c1(self.activation(self.b1(x)))
        # Apply next BN-ReLU
        h = self.activation(self.b2(h))
        # Drop channels in x if necessary
        if self.in_channels != self.out_channels:
            x = x[:, :self.out_channels]      
        # Upsample both h and x at this point  
        if self.upsample:
            h = _upsample(h)
            x = _upsample(x)
        # 3x3 convs
        h = self.c2(h)
        h = self.c3(self.activation(self.b3(h)))
        # Final 1x1 conv
        h = self.c4(self.activation(self.b4(h)))
        if self.learnable_sc:
            return h+self.c_sc(x)
        else:
            return h + x
