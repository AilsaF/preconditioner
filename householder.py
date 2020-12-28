import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import conv, Linear
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
from torch.nn import init
import math


def np_Hprod(H, u, k):
    # H.shape = (batch, n_h)
    # u.shape = (n_h,)
    alpha = 2* np.dot(H[:, -k:], u[-k:]) / np.dot(u[-k:],u[-k:]) # alpha.shape = (batch,)
    H_out = H.copy()
    H_out[:, -k:] -= np.outer(alpha, u[-k:])
    return H_out

def np_svdProd(H,U):
    #U_shape = U.get_shape().as_list()
    U_shape = U.shape
    n_r = U_shape[0]; n_h = U_shape[1]
    assert( H.shape[1] == n_h)
    H_copy = H.copy()
    for i in range(0, n_r):
        H_copy = np_Hprod(H_copy, U[i], n_h-i)
    return H_copy

def np_svdProd_inv(H,U):
    #U_shape = U.get_shape().as_list()
    U_shape = U.shape
    n_r = U_shape[0]; n_h = U_shape[1]
    assert( H.shape[1] == n_h)
    H_copy = H.copy()
    for i in range(n_r-1,-1,-1):
        H_copy = np_Hprod(H_copy, U[i], n_h-i)
    return H_copy



def Hprod(H, u, k):
    # H.shape = (batch, n_h)
    # u.shape = (n_h,)
    alpha = 2*(H[:, -k:] * u[-k:]).sum(dim=1) / torch.dot(u[-k:], u[-k:]) # alpha.shape = (batch,)
    H[:, -k:] -= torch.ger(alpha, u[-k:])
    return H


def svdProd(H, U):
    U_shape = U.shape
    n_r = U_shape[0]
    n_h = U_shape[1]
    for i in range(0, n_r):
        H = Hprod(H, U[i], n_h - i)
    return H

def svdProd_inv(H,U):
    U_shape = U.shape
    n_r = U_shape[0]
    n_h = U_shape[1]
    for i in range(n_r - 1, -1, -1):
        H = Hprod(H, U[i], n_h - i)
    return H


# output_size = 5
#
# x = torch.randn(1, 3, output_size ,output_size)
# V = torch.randn(output_size, output_size)
# V = torch.triu(V)
#
# x2 = x.clone().numpy()
# V2 = V.clone().numpy()
#
# print svdProd_inv(x, V)
# print np_svdProd_inv(x2, V2)




def Hprod2(u, k):
    I = torch.eye(u.shape[0])
    I[-k:, -k:] -= 2*torch.mm(u[-k:].view(-1,1), u[-k:].view(1,-1)) / torch.dot(u[-k:], u[-k:])
    return I


def svdProd2(U):
    U_shape = U.shape
    n_r = U_shape[0]
    n_h = U_shape[1]
    H = torch.eye(n_h)
    for i in range(n_r - 1, -1, -1):
        H = torch.mm(H, Hprod2(U[i], n_h - i))
    return H

def svdProd_inv2(U):
    U_shape = U.shape
    n_r = U_shape[0]
    n_h = U_shape[1]
    H = torch.eye(n_h)
    for i in range(0, n_r):
        H = torch.mm(H, Hprod2(U[i], n_h - i))
    return H


# output_size = 5

# x = torch.randn(3, output_size)
# V = torch.randn(2, output_size)
# V = torch.triu(V)
#
# x2 = x.clone()
# V2 = V.clone()
#
# print svdProd_inv(svdProd(x, V), V)
# for i in range(3):
#     print torch.mm(svdProd2(V2), (torch.mm(svdProd_inv2(V2), x2[i].view(-1,1))))


class SVDConv2d(conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', k1=None, k2=None):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(SVDConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        # self.register_buffer('u', torch.Tensor(1, out_channels).normal_())

        kh, kw = kernel_size
        self.W_shape = (out_channels, in_channels, kh, kw)
        self.total_in_dim = in_channels * kh * kw
        self.out_channels = out_channels

        self.k1 = k1 if k1 else out_channels/4
        self.k2 = k2 if k2 else self.total_in_dim/4


        self.U = Parameter(torch.Tensor(out_channels, out_channels))
        self.smallsize = min(out_channels, self.total_in_dim)
        self.D = Parameter(torch.Tensor(self.smallsize))
        self.V = Parameter(torch.Tensor(self.total_in_dim, self.total_in_dim))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters2()

    def Hprod(self, u, k):
        I = torch.eye(u.shape[0]).cuda()
        I[-k:, -k:] -= 2*torch.mm(u[-k:].view(-1,1), u[-k:].view(1,-1)) / torch.dot(u[-k:], u[-k:])
        return I

    def svdProd(self, U):
        U_shape = U.shape
        n_r = U_shape[0]
        n_h = U_shape[1]
        H = torch.eye(n_h).cuda()
        for i in range(n_r - 1, -1, -1):
            H = torch.mm(H, self.Hprod(U[i], n_h - i))
        return H

    def svdProd_inv(self, V):
        V_shape = V.shape
        n_r = V_shape[0]
        n_h = V_shape[1]
        H = torch.eye(n_h).cuda()
        for i in range(0, n_r):
            H = torch.mm(H, self.Hprod(V[i], n_h - i))
        return H

    @property
    def W_(self):
        V = torch.triu(self.V)
        U = torch.triu(self.U)
        D = torch.zeros(self.out_channels, self.total_in_dim).cuda()
        D[:self.smallsize, :self.smallsize] = torch.diag(self.D)
        W = torch.mm(self.svdProd(U), D).mm(self.svdProd_inv(V))
        W = W.reshape(self.W_shape)
        return W

    def forward(self, input):
        return F.conv2d(input, self.W_, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def reset_parameters2(self):
        self.D.data.fill_(1.)
        init.orthogonal_(self.U)
        init.orthogonal_(self.V)
        # init.kaiming_uniform_(self.V, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)


class SVDLinear(Linear):
    def __init__(self, in_features, out_features, bias=True, k1=None, k2=None):
        super(SVDLinear, self).__init__(in_features, out_features, bias)
        self.k1 = k1 if k1 else out_features / 4
        self.k2 = k2 if k2 else in_features / 4
        self.out_features = out_features
        self.in_features = in_features

        self.U = Parameter(torch.Tensor(out_features, out_features))
        self.smallsize = min(out_features, in_features)
        self.D = Parameter(torch.Tensor(self.smallsize))
        self.V = Parameter(torch.Tensor(in_features, in_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters2()

    def Hprod(self, u, k):
        I = torch.eye(u.shape[0]).cuda()
        I[-k:, -k:] -= 2*torch.mm(u[-k:].view(-1,1), u[-k:].view(1,-1)) / torch.dot(u[-k:], u[-k:])
        return I

    def svdProd(self, U):
        U_shape = U.shape
        n_r = U_shape[0]
        n_h = U_shape[1]
        H = torch.eye(n_h).cuda()
        for i in range(n_r - 1, -1, -1):
            H = torch.mm(H, self.Hprod(U[i], n_h - i))
        return H

    def svdProd_inv(self, V):
        V_shape = V.shape
        n_r = V_shape[0]
        n_h = V_shape[1]
        H = torch.eye(n_h).cuda()
        for i in range(0, n_r):
            H = torch.mm(H, self.Hprod(V[i], n_h - i))
        return H

    @property
    def W_(self):
        V = torch.triu(self.V)
        U = torch.triu(self.U)
        D = torch.zeros(self.out_features, self.in_features).cuda()
        D[:self.smallsize, :self.smallsize] = torch.diag(self.D)
        W = torch.mm(self.svdProd(U), D).mm(self.svdProd_inv(V))
        return W

    def forward(self, input):
        return F.linear(input, self.W_, self.bias)

    def reset_parameters2(self):
        self.D.data.fill_(1.)
        init.orthogonal_(self.U)
        init.orthogonal_(self.V)
        # init.kaiming_uniform_(self.V, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)


