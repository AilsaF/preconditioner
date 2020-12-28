import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import conv, Linear
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
from torch.nn import init

class SVDConv2d(conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(SVDConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        kh, kw = kernel_size
        self.W_shape = (out_channels, in_channels, kh, kw)
        self.total_in_dim = in_channels * kh * kw
        if self.out_channels <= self.total_in_dim:
            self.U = Parameter(torch.Tensor(out_channels, out_channels))
            self.D = Parameter(torch.Tensor(out_channels))
            self.V = Parameter(torch.Tensor(out_channels, self.total_in_dim))
        else:
            self.U = Parameter(torch.Tensor(self.total_in_dim, out_channels))
            self.D = Parameter(torch.Tensor(self.total_in_dim))
            self.V = Parameter(torch.Tensor(self.total_in_dim, self.total_in_dim))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters2()

    def update_sigma(self):
        self.D.data = self.D.data / torch.abs(self.D).data.max()

    @property
    def W_bar(self):
        self.update_sigma()
        _W = torch.matmul(self.U.t() * self.D, self.V)
        return _W.reshape(self.W_shape)


    def forward(self, input):
        return F.conv2d(input, self.W_bar, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def reset_parameters2(self):
        # init.kaiming_uniform_(self.U, a=math.sqrt(5))
        self.D.data.fill_(1.)
        init.orthogonal_(self.U)
        init.orthogonal_(self.V)
        # init.kaiming_uniform_(self.V, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def log_d_max(self):
        return torch.log(torch.max(torch.abs(self.D)))

    def loss_orth(self):
        penalty = 0

        W = self.U
        Wt = W.t()
        WWt = torch.matmul(W, Wt)
        I = torch.eye(WWt.shape[0]).cuda()
        penalty = penalty + torch.sum((WWt - I) ** 2)

        W = self.V
        Wt = W.T
        WWt = torch.matmul(W, Wt)
        I = torch.eye(WWt.shape[0]).cuda()
        penalty = penalty + torch.sum((WWt - I) ** 2)

        spectral_penalty = 0
        # if self.mode in (4, 5):
        #     if (self.D.size > 1):
        #         sd2 = 0.1 ** 2
        #         _d = self.D[cupy.argsort(self.D.data)]
        #         spectral_penalty += F.mean((1 - _d[:-1]) ** 2 / sd2 - F.log((_d[1:] - _d[:-1]) + 1e-7)) * 0.05
        # elif self.mode == 6:
        #     spectral_penalty += F.mean(self.D * F.log(self.D))
        # elif self.mode == 7:
        #     spectral_penalty += F.mean(F.exp(self.D))
        # elif self.mode == 8:
        spectral_penalty += -torch.mean(torch.log(self.D))

        return penalty + spectral_penalty * 0.1


class SVDLinear(Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SVDLinear, self).__init__(in_features, out_features, bias)
        if out_features <= in_features:
            self.U = Parameter(torch.Tensor(out_features, out_features))
            self.D = Parameter(torch.Tensor(out_features))
            self.V = Parameter(torch.Tensor(out_features, in_features))
        else:
            self.U = Parameter(torch.Tensor(in_features, out_features))
            self.D = Parameter(torch.Tensor(in_features))
            self.V = Parameter(torch.Tensor(in_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters2()

    def update_sigma(self):
        self.D.data = self.D.data / torch.abs(self.D).data.max()

    @property
    def W_bar(self):
        self.update_sigma()
        _W = torch.matmul(self.U * self.D, self.V)
        return _W

    def forward(self, input):
        return F.linear(input, self.W_bar, self.bias)

    def reset_parameters2(self):
        # init.kaiming_uniform_(self.U, a=math.sqrt(5))
        self.D.data.fill_(1.)
        init.orthogonal_(self.U)
        init.orthogonal_(self.V)
        # init.kaiming_uniform_(self.V, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def log_d_max(self):
        return torch.log(torch.max(torch.abs(self.D)))

    def loss_orth(self):
        penalty = 0

        W = self.U.t()
        Wt = W.t()
        WWt = torch.matmul(W, Wt)
        I = torch.eye(WWt.shape[0]).cuda()
        penalty = penalty + torch.sum((WWt - I) ** 2)

        W = self.V
        Wt = W.t()
        WWt = torch.matmul(W, Wt)
        I = torch.eye(WWt.shape[0]).cuda()
        penalty = penalty + torch.sum((WWt - I) ** 2)

        spectral_penalty = 0
        # if self.mode in (4, 5):
        #     if (self.D.size > 1):
        #         sd2 = 0.1 ** 2
        #         _d = self.D[cupy.argsort(self.D.data)]
        #         spectral_penalty += F.mean((1 - _d[:-1]) ** 2 / sd2 - F.log((_d[1:] - _d[:-1]) + 1e-7)) * 0.05
        # elif self.mode == 6:
        #     spectral_penalty += F.mean(self.D * F.log(self.D))
        # elif self.mode == 7:
        #     spectral_penalty += F.mean(F.exp(self.D))
        # elif self.mode == 8:
        spectral_penalty += -torch.mean(torch.log(self.D))

        return penalty + spectral_penalty * 0.1

