# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import conv, Linear
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter


def _l2normalize(v, eps=1e-12):
    return v / (torch.norm(v) + eps)

def max_singular_value(W, u=None, Ip=1):
    """
    power iteration for weight parameter
    """
    #xp = W.data
    if not Ip >= 1:
        raise ValueError("Power iteration should be a positive integer")
    if u is None:
        u = torch.FloatTensor(1, W.size(0)).normal_(0, 1).cuda()
    _u = u
    for _ in range(Ip):
        _v = _l2normalize(torch.matmul(_u, W.data), eps=1e-12)
        _u = _l2normalize(torch.matmul(_v, torch.transpose(W.data, 0, 1)), eps=1e-12)
    sigma = torch.sum(F.linear(_u, torch.transpose(W.data, 0, 1)) * _v)
    return sigma, _u


class SNConv2d(conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(SNConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        self.register_buffer('u', torch.Tensor(1, out_channels).normal_())
        self.polar_iter = 1
        self.a = Parameter(torch.tensor(.5))
        self.b = Parameter(torch.tensor(1.5))

    @property
    def W_(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = max_singular_value(w_mat, self.u)
        self.u.copy_(_u)
        weight = self.weight / sigma

        if len(weight.shape) <= 2:
            weight = weight.t()
            for _ in range(self.polar_iter):
                # weight = 1.5 * weight - 0.5 * weight.mm(weight.t()).mm(weight)
                weight = self.b * weight - self.a * weight.mm(weight.t()).mm(weight)
            weight = weight.t()

        else:
            origin_shape = weight.shape
            weight = weight.view(weight.shape[0], -1)
            for _ in range(self.polar_iter):
                # weight = 1.5 * weight - 0.5 * weight.mm(weight.t()).mm(weight)
                weight = self.b * weight - self.a * weight.mm(weight.t()).mm(weight)
            weight = weight.view(origin_shape)
        return weight

    def forward(self, input):
        return F.conv2d(input, self.W_, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class SNLinear(Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SNLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('u', torch.Tensor(1, out_features).normal_())
        self.polar_iter = 1
        self.a = Parameter(torch.tensor(.5))
        self.b = Parameter(torch.tensor(1.5))

    @property
    def W_(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = max_singular_value(w_mat, self.u)
        self.u.copy_(_u)
        weight = self.weight / sigma
        if len(weight.shape) <= 2:
            weight = weight.t()
            for _ in range(self.polar_iter):
                # weight = 1.5 * weight - 0.5 * weight.mm(weight.t()).mm(weight)
                weight = self.b * weight - self.a * weight.mm(weight.t()).mm(weight)
            weight = weight.t()

        else:
            origin_shape = weight.shape
            weight = weight.view(weight.shape[0], -1)
            for _ in range(self.polar_iter):
                # weight = 1.5 * weight - 0.5 * weight.mm(weight.t()).mm(weight)
                weight = self.b * weight - self.a * weight.mm(weight.t()).mm(weight)
            weight = weight.view(origin_shape)
        return weight

    def forward(self, input):
        return F.linear(input, self.W_, self.bias)