import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
# from torch.nn import utils
import Higham_norm
from delta_orthogonal import makeDeltaOrthogonal
import SVDconv


class Block(nn.Module):

    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1,
                 activation=F.relu, downsample=False, power_iter=1, use_adaptivePC=True, pclevel=0, diter=1):
        super(Block, self).__init__()

        self.activation = activation
        self.downsample = downsample

        self.learnable_sc = (in_ch != out_ch) or downsample
        if h_ch is None:
            h_ch = in_ch
        else:
            h_ch = out_ch

        self.c1 = Higham_norm.spectral_norm(nn.Conv2d(in_ch, h_ch, ksize, 1, pad), n_power_iterations=power_iter,
                                            use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.c2 = Higham_norm.spectral_norm(nn.Conv2d(h_ch, out_ch, ksize, 1, pad), n_power_iterations=power_iter,
                                            use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        if self.learnable_sc:
            self.c_sc = Higham_norm.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0), n_power_iterations=power_iter,
                                                  use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        h = self.c1(self.activation(x))
        h = self.c2(self.activation(h))
        if self.downsample:
            h = F.avg_pool2d(h, 2)
        return h


class OptimizedBlock(nn.Module):

    def __init__(self, in_ch, out_ch, ksize=3, pad=1, activation=F.relu, power_iter=1, use_adaptivePC=True, pclevel=0, diter=1):
        super(OptimizedBlock, self).__init__()
        self.activation = activation

        self.c1 = Higham_norm.spectral_norm(nn.Conv2d(in_ch, out_ch, ksize, 1, pad), n_power_iterations=power_iter,
                                            use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.c2 = Higham_norm.spectral_norm(nn.Conv2d(out_ch, out_ch, ksize, 1, pad), n_power_iterations=power_iter,
                                            use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.c_sc = Higham_norm.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0), n_power_iterations=power_iter,
                                              use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        return self.c_sc(F.avg_pool2d(x, 2))

    def residual(self, x):
        h = self.activation(self.c1(x))
        return F.avg_pool2d(self.c2(h), 2)



class SVDBlock(nn.Module):

    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1,
                 activation=F.relu, downsample=False):
        super(SVDBlock, self).__init__()

        self.activation = activation
        self.downsample = downsample

        self.learnable_sc = (in_ch != out_ch) or downsample
        if h_ch is None:
            h_ch = in_ch
        else:
            h_ch = out_ch

        self.c1 = SVDconv.SVDConv2d(in_ch, h_ch, ksize, 1, pad)
        self.c2 = SVDconv.SVDConv2d(h_ch, out_ch, ksize, 1, pad)
        if self.learnable_sc:
            self.c_sc = SVDconv.SVDConv2d(in_ch, out_ch, 1, 1, 0)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        h = self.c1(self.activation(x))
        h = self.c2(self.activation(h))
        if self.downsample:
            h = F.avg_pool2d(h, 2)
        return h

    def loss_orth(self):
        loss =  self.c1.loss_orth() + self.c2.loss_orth() + self.c_sc.loss_orth()
        return loss


class SVDOptimizedBlock(nn.Module):

    def __init__(self, in_ch, out_ch, ksize=3, pad=1, activation=F.relu):
        super(SVDOptimizedBlock, self).__init__()
        self.activation = activation

        self.c1 = SVDconv.SVDConv2d(in_ch, out_ch, ksize, 1, pad)
        self.c2 = SVDconv.SVDConv2d(out_ch, out_ch, ksize, 1, pad)
        self.c_sc = SVDconv.SVDConv2d(in_ch, out_ch, 1, 1, 0)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        return self.c_sc(F.avg_pool2d(x, 2))

    def residual(self, x):
        h = self.activation(self.c1(x))
        return F.avg_pool2d(self.c2(h), 2)

    def loss_orth(self):
        loss =  self.c1.loss_orth() + self.c2.loss_orth() + self.c_sc.loss_orth()
        return loss


class DeepBlock(nn.Module):

    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1,
                 activation=F.relu, downsample=False, power_iter=1, use_adaptivePC=True, pclevel=0, diter=1):
        super(DeepBlock, self).__init__()

        self.activation = activation
        self.downsample = downsample

        self.learnable_sc = (in_ch != out_ch) or downsample
        if h_ch is None:
            h_ch = in_ch//4
        else:
            h_ch = out_ch//4

        self.c1 = Higham_norm.spectral_norm(nn.Conv2d(in_ch, h_ch, kernel_size=1, padding=0), n_power_iterations=power_iter,
                                            use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.c2 = Higham_norm.spectral_norm(nn.Conv2d(h_ch, h_ch, ksize, padding=pad), n_power_iterations=power_iter,
                                            use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.c3 = Higham_norm.spectral_norm(nn.Conv2d(h_ch, h_ch, ksize, padding=pad), n_power_iterations=power_iter,
                                            use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.c4 = Higham_norm.spectral_norm(nn.Conv2d(h_ch, out_ch, kernel_size=1, padding=0), n_power_iterations=power_iter,
                                            use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        
        if self.learnable_sc:
            self.c_sc = Higham_norm.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0), n_power_iterations=power_iter,
                                                  use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)

    def forward(self, x):
        # 1x1 bottleneck conv
        h = self.c1(F.relu(x))
        # 3x3 convs
        h = self.c2(self.activation(h))
        h = self.c3(self.activation(h))
        # relu before downsample
        h = self.activation(h)
        # downsample
        if self.downsample:
            h = F.avg_pool2d(h, 2)   
        # final 1x1 conv
        h = self.c4(h)
        return h + self.shortcut(x)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

class DeepBlock2(nn.Module):

    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1,
                 activation=F.relu, downsample=False, power_iter=1, use_adaptivePC=True, pclevel=0, diter=1):
        super(DeepBlock2, self).__init__()

        self.activation = activation
        self.downsample = downsample

        self.learnable_sc = (in_ch != out_ch) or downsample
        if h_ch is None:
            h_ch = in_ch//4
        else:
            h_ch = out_ch//4

        self.c1 = Higham_norm.spectral_norm(nn.Conv2d(in_ch, h_ch, kernel_size=1, padding=0), n_power_iterations=power_iter,
                                            use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.c2 = Higham_norm.spectral_norm(nn.Conv2d(h_ch, h_ch, ksize, padding=pad), n_power_iterations=power_iter,
                                            use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.c3 = Higham_norm.spectral_norm(nn.Conv2d(h_ch, h_ch, ksize, padding=pad), n_power_iterations=power_iter,
                                            use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.c4 = Higham_norm.spectral_norm(nn.Conv2d(h_ch, h_ch, ksize, padding=pad), n_power_iterations=power_iter,
                                            use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.c5 = Higham_norm.spectral_norm(nn.Conv2d(h_ch, h_ch, ksize, padding=pad), n_power_iterations=power_iter,
                                            use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.c6 = Higham_norm.spectral_norm(nn.Conv2d(h_ch, h_ch, ksize, padding=pad), n_power_iterations=power_iter,
                                            use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.c7 = Higham_norm.spectral_norm(nn.Conv2d(h_ch, h_ch, ksize, padding=pad), n_power_iterations=power_iter,
                                            use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.c8 = Higham_norm.spectral_norm(nn.Conv2d(h_ch, out_ch, kernel_size=1, padding=0), n_power_iterations=power_iter,
                                            use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        
        if self.learnable_sc:
            self.c_sc = Higham_norm.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0), n_power_iterations=power_iter,
                                                  use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)

    def forward(self, x):
        # 1x1 bottleneck conv
        h = self.c1(F.relu(x))
        # 3x3 convs
        h = self.c2(self.activation(h))
        h = self.c3(self.activation(h))
        h = self.c4(self.activation(h))
        h = self.c5(self.activation(h))
        h = self.c6(self.activation(h))
        h = self.c7(self.activation(h))
        # relu before downsample
        h = self.activation(h)
        # downsample
        if self.downsample:
            h = F.avg_pool2d(h, 2)   
        # final 1x1 conv
        h = self.c8(h)
        return h + self.shortcut(x)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x