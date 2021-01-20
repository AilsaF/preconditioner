import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import Higham_norm
import math

__all__ = ['FixupResNet', 'fixup_resnet20', 'fixup_resnet32', 'fixup_resnet44', 'fixup_resnet56', 'fixup_resnet110', 'fixup_resnet1202']

class PCLayer(torch.nn.Module):
    def __init__(self, use_adaptivePC=False, pclevel=0, *args, **kwargs):
        super(PCLayer, self).__init__()
        self.use_adaptivePC = use_adaptivePC
        self.pclevel = pclevel
        self.cns = torch.tensor([]).cuda()
        self.called_time = 0
        self.In = None
        self.Im = None

    def preconditionertall(self, weight):
        if self.pclevel == 0:
            return weight

        if self.Im is None:
            n, m = weight.shape
            self.Im = torch.eye(m, device=torch.device('cuda'))
        wtw = weight.t().mm(weight)
        if self.pclevel == 1:
            weight = weight.mm(1.507 * self.Im - 0.507 * wtw)
        elif self.pclevel == 2:
            weight = weight.mm(2.083 * self.Im + wtw.mm(-1.643 * self.Im + 0.560 * wtw))
        # elif self.pclevel == 2.5:
        #     weight = weight.mm(2.615 * self.Im + wtw.mm(-3.548 * self.Im + wtw.mm(2.727 * self.Im - 0.795 * wtw)))
        elif self.pclevel == 3:
            weight = weight.mm(2.909 * self.Im + wtw.mm(-4.649 * self.Im + wtw.mm(4.023 * self.Im - 1.283 * wtw)))
        # elif self.pclevel == 3.5:
        #     weight = weight.mm(3.418 * self.Im + wtw.mm(-8.029 * self.Im + wtw.mm(11.552 * self.Im + wtw.mm(-8.152 * self.Im + 2.211 * wtw))))
        elif self.pclevel == 4:
            weight = weight.mm(3.625 * self.Im + wtw.mm(-9.261 * self.Im + wtw.mm(14.097 * self.Im + wtw.mm(-10.351 * self.Im + 2.890 * wtw))))
        elif self.pclevel == 5:
            weight = weight.mm(4.230 * self.Im + wtw.mm(-13.367 * self.Im + wtw.mm(23.356 * self.Im + wtw.mm(-18.866 * self.Im + 5.646* wtw))))
        else:
            raise ValueError("No pre-conditioner provided")
        return weight

    def preconditionerwide(self, weight):
        if self.pclevel == 0:
            return weight

        n, m = weight.shape
        if self.In is None:
            self.In = torch.eye(n, device=torch.device('cuda'))
        wwt = weight.mm(weight.t())
        if self.pclevel == 1:
            weight = (1.507 * self.In - 0.507 * wwt).mm(weight)
        elif self.pclevel == 2:
            weight = (2.083 * self.In + wwt.mm(-1.643 * self.In + 0.560 * wwt)).mm(weight)
        # elif self.pclevel == 2.5:
        #     weight = (2.615 * self.In + wwt.mm(-3.548 * self.In + wwt.mm(2.727 * self.In - 0.795 * wwt))).mm(weight)
        elif self.pclevel == 3:
            weight = (2.909 * self.In + wwt.mm(-4.649 * self.In + wwt.mm(4.023 * self.In - 1.283 * wwt))).mm(weight)
        # elif self.pclevel == 3.5:
        #     weight = (3.418 * self.In + wwt.mm(-8.029 * self.In + wwt.mm(11.552 * self.In + wwt.mm(-8.152 * self.In + 2.211 * wwt)))).mm(
        #         weight)
        elif self.pclevel == 4:
            weight = (3.625 * self.In + wwt.mm(-9.261 * self.In + wwt.mm(14.097 * self.In + wwt.mm(-10.351 * self.In + 2.890 * wwt)))).mm(
                weight)
        elif self.pclevel == 5:
            weight = (4.230 * self.In + wwt.mm(-13.367 * self.In + wwt.mm(23.356 * self.In + wwt.mm(-18.866 * self.In + 5.646 * wwt)))).mm(
                weight)
        else:
            raise ValueError("No pre-conditioner provided")
        return weight

    def adaptive(self, weight):
        if self.use_adaptivePC and self.called_time % (470*20) == 0:
            weight = weight.detach()
            S = torch.svd(weight)[1]
            sin_num = max(1, int(S.shape[0] * 0.1))
            condition_number = 1. / (S[-sin_num:]).mean()
            # print(self.cns.device)
            self.cns.data = torch.cat((self.cns.view(-1), condition_number.view(1)))[-5:]
            recent_cn_avg = self.cns.mean()
            if recent_cn_avg <= 5:
                self.pclevel = 2
            elif 5 < recent_cn_avg <= 10:
                self.pclevel = 3
            elif 10 < recent_cn_avg <= 20:
                self.pclevel = 4
            else:
                self.pclevel = 5
            print("CNs are {}, change preconditioner iteration to {}".format(self.cns, self.pclevel))
        self.called_time += 1

    def forward(self, weight):
        weight_shape = weight.shape
        weight = weight.view(weight.shape[0], -1)
        sigma = torch.norm(weight)
        # sigma = (torch.linalg.norm(weight, float('inf')) * torch.linalg.norm(weight, 1)) ** 0.5
        if sigma > 0.:
            weight = weight / sigma
            self.adaptive(weight)
            if self.pclevel:
                n, m = weight.shape
                if n >= m:
                    weight = self.preconditionertall(weight)
                else:
                    weight = self.preconditionerwide(weight)
                weight = weight.view(weight_shape)
            return weight #* sigma
        else:
            return weight.view(weight_shape) #* torch.tensor(1.)


class PC_Conv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 use_adaptivePC=False, pclevel=0, *args, **kwargs):
        super(PC_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.PCLayer = PCLayer(use_adaptivePC=use_adaptivePC, pclevel=pclevel)
        print(self._get_name(), use_adaptivePC, pclevel)

    def forward(self, input):
        pcweight = self.PCLayer(self.weight)
        out = F.conv2d(input, pcweight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out


class PC_Linear(torch.nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True, use_adaptivePC=False, pclevel=0, *args, **kwargs):
        super(PC_Linear, self).__init__(in_channels, out_channels, bias)
        self.PCLayer = PCLayer(use_adaptivePC=use_adaptivePC, pclevel=pclevel)
        print(self._get_name(), use_adaptivePC, pclevel)
    
    def forward(self, input):
        pcweight = self.PCLayer(self.weight)
        out = F.linear(input, pcweight, self.bias)
        return out


def conv3x3(in_planes, out_planes, stride=1, PC=0):
    """3x3 convolution with padding"""
    if PC == 0:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
    else:
        return PC_Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, use_adaptivePC=True, pclevel=PC)


class FixupBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, PC=0):
        super(FixupBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = conv3x3(inplanes, planes, stride, PC=PC)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = conv3x3(planes, planes, PC=PC)
        self.scale0 = nn.Parameter(torch.ones(1))
        self.scale = nn.Parameter(torch.ones(1))
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x + self.bias1a)
        out = self.relu(out * self.scale0 + self.bias1b)

        out = self.conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b

        if self.downsample is not None:
            identity = self.downsample(x + self.bias1a)
            identity = torch.cat((identity, torch.zeros_like(identity)), 1)

        out += identity
        out = self.relu(out)

        return out


class FixupResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, PC=0, init='fixup'):
        super(FixupResNet, self).__init__()
        self.num_layers = sum(layers)
        self.inplanes = 16
        self.conv1 = conv3x3(3, 16, PC=PC)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0], PC=PC)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2, PC=PC)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2, PC=PC)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bias2 = nn.Parameter(torch.zeros(1))
        if PC == 0:
            self.fc = nn.Linear(64, num_classes)
        else:
            self.fc = PC_Linear(64, num_classes, use_adaptivePC=True, pclevel=PC)
        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                if init == 'fixup':
                    nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(2. / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
                    # nn.init.normal_(m.conv2.weight, mean=0, std=np.sqrt(2. / (m.conv2.weight.shape[0] * np.prod(m.conv2.weight.shape[2:]))) * self.num_layers ** (-0.25))
                    nn.init.constant_(m.conv2.weight, 0)
                elif init == 'kaiming':
                    nn.init.kaiming_normal_(m.conv1.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.kaiming_normal_(m.conv2.weight, mode='fan_out', nonlinearity='relu')
                else:
                    nn.init.orthogonal_(m.conv1.weight, gain=self.num_layers ** (-0.5))
                    nn.init.orthogonal_(m.conv2.weight, gain=self.num_layers ** (-0.5))
                
            elif isinstance(m, nn.Linear):
                if init == 'fixup':
                    nn.init.constant_(m.weight, 0)
                    # nn.init.normal_(m.weight, mean=0, std=np.sqrt(2. / m.weight.shape[0]) * self.num_layers ** (-0.5))
                # kaiming
                elif init == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                else:
                    nn.init.orthogonal_(m.weight, gain=self.num_layers ** (-0.5))
                nn.init.constant_(m.bias, 0)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))

    def _make_layer(self, block, planes, blocks, stride=1, PC=0):
        downsample = None
        if stride != 1:
            downsample = nn.AvgPool2d(1, stride=stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, PC=PC))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes, PC=PC))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x * self.scale + self.bias1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x + self.bias2)

        return x


def fixup_resnet20(**kwargs):
    """Constructs a Fixup-ResNet-20 model.

    """
    model = FixupResNet(FixupBasicBlock, [3, 3, 3], **kwargs)
    return model


def fixup_resnet32(**kwargs):
    """Constructs a Fixup-ResNet-32 model.

    """
    model = FixupResNet(FixupBasicBlock, [5, 5, 5], **kwargs)
    return model


def fixup_resnet44(**kwargs):
    """Constructs a Fixup-ResNet-44 model.

    """
    model = FixupResNet(FixupBasicBlock, [7, 7, 7], **kwargs)
    return model


def fixup_resnet56(**kwargs):
    """Constructs a Fixup-ResNet-56 model.

    """
    model = FixupResNet(FixupBasicBlock, [9, 9, 9], **kwargs)
    return model


def fixup_resnet110(**kwargs):
    """Constructs a Fixup-ResNet-110 model.

    """
    model = FixupResNet(FixupBasicBlock, [18, 18, 18], **kwargs)
    return model


def fixup_resnet1202(**kwargs):
    """Constructs a Fixup-ResNet-1202 model.

    """
    model = FixupResNet(FixupBasicBlock, [200, 200, 200], **kwargs)
    return model