import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import Parameter
import math
# import extension as my
# from extension.normalization.NormedConv import ONI_Conv2d as Conv2d_ONI

__all__ = ['resnetDebug20', 'resnetDebug32', 'resnetDebug44', 'resnetDebug110']


class IdentityModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(IdentityModule, self).__init__()

    def forward(self, input: torch.Tensor):
        return input

def _IdentityModule(x, *args, **kwargs):
    """return first input"""
    return IdentityModule()

class ONINorm(torch.nn.Module):
    def __init__(self, T=2, norm_groups=1, *args, **kwargs):
        super(ONINorm, self).__init__()
        self.T = T
        self.norm_groups = norm_groups
        self.eps = 1e-5

    # def matrix_power3(self, Input):
    #     B=torch.bmm(Input, Input)
    #     return torch.bmm(B, Input)

    # def forward(self, weight: torch.Tensor):
    #     assert weight.shape[0] % self.norm_groups == 0
    #     Z = weight.view(self.norm_groups, weight.shape[0] // self.norm_groups, -1)  # type: torch.Tensor
    #     # Z_ = weight.view(weight.shape[0] , -1)
    #     # print("before", torch.svd(Z_)[1][0], torch.svd(Z_)[1][0]/torch.svd(Z_)[1][-1])
    #     Zc = Z - Z.mean(dim=-1, keepdim=True)
    #     S = torch.matmul(Zc, Zc.transpose(1, 2))
    #     eye = torch.eye(S.shape[-1]).to(S).expand(S.shape)
    #     S = S + self.eps*eye
    #     norm_S = S.norm(p='fro', dim=(1, 2), keepdim=True)
    #     S = S.div(norm_S)
    #     B = [torch.Tensor([]) for _ in range(self.T + 1)]
    #     B[0] = torch.eye(S.shape[-1]).to(S).expand(S.shape)
    #     for t in range(self.T):
    #         #B[t + 1] = torch.baddbmm(1.5, B[t], -0.5, torch.matrix_power(B[t], 3), S)
    #         B[t + 1] = torch.baddbmm(1.5, B[t], -0.5, self.matrix_power3(B[t]), S)
    #     W = B[self.T].matmul(Zc).div_(norm_S.sqrt())
    #     #print(W.matmul(W.transpose(1,2)))
    #     # W = oni_py.apply(weight, self.T, ctx.groups)
    #     # Z_ = W.view_as(Z_)
    #     # print("after", torch.svd(Z_)[1][0], torch.svd(Z_)[1][0]/torch.svd(Z_)[1][-1])
    #     # print("===============")
    #     W = W.view_as(weight)
    #     return W

    def matrix_power32(self, Input):
        B=torch.mm(Input, Input)
        return torch.mm(B, Input)

    def forward(self, weight: torch.Tensor):
        Z = weight.view(weight.shape[0], -1)
        V = Z / Z.norm(p='fro')
        S = V.mm(V.t())
        eye = torch.eye(S.shape[-1]).to(S).expand(S.shape)
        S = S + self.eps*eye
        B = [torch.Tensor([]) for _ in range(self.T + 1)]
        B[0] = torch.eye(S.shape[-1]).to(S).expand(S.shape)
        for t in range(self.T):
            B[t + 1] = 1.5*B[t]-0.5*torch.mm(self.matrix_power32(B[t]), S)
        W = torch.mm(B[self.T], V)
        return W.view_as(weight)

    def extra_repr(self):
        fmt_str = ['T={}'.format(self.T)]
        if self.norm_groups > 1:
            fmt_str.append('groups={}'.format(self.norm_groups))
        return ', '.join(fmt_str)


class ONINorm_colum(torch.nn.Module):
    def __init__(self, T=2, norm_groups=1, *args, **kwargs):
        super(ONINorm_colum, self).__init__()
        self.T = T
        self.norm_groups = norm_groups
        self.eps = 1e-5
        #print(self.eps)

    def matrix_power3(self, Input):
        B=torch.bmm(Input, Input)
        return torch.bmm(B, Input)

    def forward(self, weight: torch.Tensor):
        assert weight.shape[0] % self.norm_groups == 0
        Z = weight.view(self.norm_groups, weight.shape[0] // self.norm_groups, -1)  # type: torch.Tensor
        Zc = Z - Z.mean(dim=-1, keepdim=True)
        S = torch.matmul(Zc.transpose(1, 2), Zc)
        eye = torch.eye(S.shape[-1]).to(S).expand(S.shape)
        S = S + self.eps*eye
        norm_S = S.norm(p='fro', dim=(1, 2), keepdim=True)
        #print(S.size())
        #S = S.div(norm_S)
        B = [torch.Tensor([]) for _ in range(self.T + 1)]
        B[0] = torch.eye(S.shape[-1]).to(S).expand(S.shape)
        for t in range(self.T):
            #B[t + 1] = torch.baddbmm(1.5, B[t], -0.5, torch.matrix_power(B[t], 3), S)
            B[t + 1] = torch.baddbmm(1.5, B[t], -0.5, self.matrix_power3(B[t]), S)
        W = Zc.matmul(B[self.T]).div_(norm_S.sqrt())
        #print(W.matmul(W.transpose(1,2)))
        # W = oni_py.apply(weight, self.T, ctx.groups)
        return W.view_as(weight)

    def extra_repr(self):
        fmt_str = ['T={}'.format(self.T)]
        if self.norm_groups > 1:
            fmt_str.append('groups={}'.format(self.norm_groups))
        return ', '.join(fmt_str)

class ONI_Conv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 T=2, norm_groups=1, norm_channels=0, NScale=1.414, adjustScale=False, ONIRow_Fix=False, *args, **kwargs):
        super(ONI_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        print('ONI channels:--OD:',out_channels, '--ID:', in_channels, '--KS',kernel_size)
        if out_channels <= (in_channels*kernel_size*kernel_size):
            if norm_channels > 0:
                norm_groups = out_channels // norm_channels
            #print('ONI_Conv_Row:--T=', T, '----norm_groups:', norm_groups, '---NScale:', NScale, '---adjust:', adjustScale)
            self.weight_normalization = ONINorm(T=T, norm_groups=norm_groups)
        else:
            if ONIRow_Fix:
              #  print('ONI_Conv_Row:--T=', T, '----norm_groups:', norm_groups, '---NScale:', NScale, '---adjust:', adjustScale)
                self.weight_normalization = ONINorm(T=T, norm_groups=norm_groups)
            else: 
               # print('ONI_Conv_Colum:--T=', T, '----norm_groups:', norm_groups, '---NScale:', NScale, '---adjust:', adjustScale)
                self.weight_normalization = ONINorm_colum(T=T, norm_groups=norm_groups)
        self.scale_ = torch.ones(out_channels, 1, 1, 1).fill_(NScale)
        if adjustScale:
            self.WNScale = Parameter(self.scale_)
        else:
           # self.scale = Variable(self.scale, requires_grad=False)
            self.register_buffer('WNScale', self.scale_)

    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        weight_q = self.weight_normalization(self.weight)
        weight_q = weight_q * self.WNScale
        out = F.conv2d(input_f, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out


Norm = _IdentityModule
NormConv = ONI_Conv2d

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    # return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    return NormConv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ONIBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ONIBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = Norm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = Norm(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            identity = torch.cat((identity, torch.zeros_like(identity)), 1)

        out += identity
        out = self.relu(out)
        return out


class ONIResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(ONIResNet, self).__init__()
        self.num_layers = sum(layers)
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16)
        self.bn1 = Norm(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
        # for m in self.modules():
        #     if isinstance(m, ONIBasicBlock):
        #         nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(2. / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
        #         # nn.init.normal_(m.conv2.weight, mean=0, std=np.sqrt(2. / (m.conv2.weight.shape[0] * np.prod(m.conv2.weight.shape[2:]))) * self.num_layers ** (-0.5))
        #         nn.init.constant_(m.conv2.weight, 0)
        #         # w = m.conv1.weight.data
        #         # w = w.view(w.shape[0], -1)
        #         # singular = torch.svd(w)[1][0]
        #         # nn.init.orthogonal_(m.conv1.weight, gain=singular)

        #         # w = m.conv2.weight.data
        #         # w = w.view(w.shape[0], -1)
        #         # singular = torch.svd(w)[1][0]
        #         # nn.init.orthogonal_(m.conv2.weight, gain=singular)

        #         # nn.init.orthogonal_(m.conv1.weight, gain=1)
        #         # nn.init.orthogonal_(m.conv2.weight, gain=1)
                
        #     elif isinstance(m, nn.Linear):
        #         nn.init.constant_(m.weight, 0)
        #         # nn.init.orthogonal_(m.weight, gain=self.num_layers ** (-0.5))
                
        #         # nn.init.normal_(m.weight, mean=0, std=np.sqrt(2. / m.weight.shape[0]) * self.num_layers ** (-0.5))
        #         # w = m.weight.data
        #         # singular = torch.svd(w)[1][0]
        #         # nn.init.orthogonal_(m.weight, gain=singular)
        #         nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.AvgPool2d(1, stride=stride)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnetDebug20(**kwargs):
    """Constructs a ONI-ResNet-20 model.

    """
    model = ONIResNet(ONIBasicBlock, [3, 3, 3], **kwargs)
    return model


def resnetDebug32(**kwargs):
    """Constructs a ONI-ResNet-32 model.

    """
    model = ONIResNet(ONIBasicBlock, [5, 5, 5], **kwargs)
    return model


def resnetDebug44(**kwargs):
    """Constructs a ONI-ResNet-44 model.

    """
    model = ONIResNet(ONIBasicBlock, [7, 7, 7], **kwargs)
    return model


def resnetDebug56(**kwargs):
    """Constructs a ONI-ResNet-56 model.

    """
    model = ONIResNet(ONIBasicBlock, [9, 9, 9], **kwargs)
    return model


def resnetDebug110(**kwargs):
    """Constructs a ONI-ResNet-110 model.

    """
    model = ONIResNet(ONIBasicBlock, [18, 18, 18], **kwargs)
    return model


def resnetDebug1202(**kwargs):
    """Constructs a ONI-ResNet-1202 model.

    """
    model = ONIResNet(ONIBasicBlock, [200, 200, 200], **kwargs)
    return model
