import torch
import torch.nn as nn
import numpy as np
# import Higham_norm
import torch.nn.functional as F
import math


__all__ = ['FixupResNet', 'fixup_resnet18', 'fixup_resnet34', 'fixup_resnet50', 'fixup_resnet101', 'fixup_resnet152']

class PCLayer(torch.nn.Module):
    def __init__(self, use_adaptivePC=False, pclevel=0, *args, **kwargs):
        super(PCLayer, self).__init__()
        self.use_adaptivePC = use_adaptivePC
        self.pclevel = pclevel
        self.called_time = 0
        self.In = None
        self.Im = None

    def preconditionertall(self, weight, pclevel):
        if pclevel == 0:
            return weight
        if self.Im is None:
            n, m = weight.shape
            self.Im = torch.eye(m, device=torch.device('cuda'))
        wtw = weight.t().mm(weight)
        if pclevel == 1:
            weight = weight.mm(1.507 * self.Im - 0.507 * wtw)
        elif pclevel == 2:
            weight = weight.mm(2.083 * self.Im + wtw.mm(-1.643 * self.Im + 0.560 * wtw))
        elif pclevel == 3:
            weight = weight.mm(2.909 * self.Im + wtw.mm(-4.649 * self.Im + wtw.mm(4.023 * self.Im - 1.283 * wtw)))
        elif pclevel == 3.5:
            weight = weight.mm(3.418 * self.Im + wtw.mm(-8.029 * self.Im + wtw.mm(11.552 * self.Im + wtw.mm(-8.152 * self.Im + 2.211 * wtw))))
        elif pclevel == 4:
            weight = weight.mm(3.625 * self.Im + wtw.mm(-9.261 * self.Im + wtw.mm(14.097 * self.Im + wtw.mm(-10.351 * self.Im + 2.890 * wtw))))
        else:
            raise ValueError("No pre-conditioner provided")
        return weight

    def preconditionerwide(self, weight, pclevel):
        if pclevel == 0:
            return weight
        n, m = weight.shape
        if self.In is None:
            self.In = torch.eye(n, device=torch.device('cuda'))
        wwt = weight.mm(weight.t())
        if pclevel == 1:
            weight = (1.507 * self.In - 0.507 * wwt).mm(weight)
        elif pclevel == 2:
            weight = (2.083 * self.In + wwt.mm(-1.643 * self.In + 0.560 * wwt)).mm(weight)
        elif pclevel == 3:
            weight = (2.909 * self.In + wwt.mm(-4.649 * self.In + wwt.mm(4.023 * self.In - 1.283 * wwt))).mm(weight)
        elif pclevel == 3.5:
            weight = (3.418 * self.In + wwt.mm(-8.029 * self.In + wwt.mm(11.552 * self.In + wwt.mm(-8.152 * self.In + 2.211 * wwt)))).mm(
                weight)
        elif pclevel == 4:
            weight = (3.625 * self.In + wwt.mm(-9.261 * self.In + wwt.mm(14.097 * self.In + wwt.mm(-10.351 * self.In + 2.890 * wwt)))).mm(
                weight)
        else:
            raise ValueError("No pre-conditioner provided")
        return weight

    def forward(self, weight):
        weight_shape = weight.shape
        weight = weight.view(weight.shape[0], -1)
        sigma = torch.norm(weight) / 3.
        # sigma = (torch.linalg.norm(weight, float('inf')) * torch.linalg.norm(weight, 1)) ** 0.5
        if sigma > 0.:
            weight = weight / sigma
            # if len(weight.shape) > 2:
            if self.pclevel:
                n, m = weight.shape
                if n >= m:
                    if self.Im is None:
                        self.Im = torch.eye(m, device=torch.device('cuda'))
                    wtw = weight.t().mm(weight)
                    weight = weight.mm(2.909 * self.Im + wtw.mm(-4.649 * self.Im + wtw.mm(4.023 * self.Im - 1.283 * wtw)))
                    # weight = self.preconditionertall(weight, self.pclevel)
                else:
                    if self.In is None:
                        self.In = torch.eye(n, device=torch.device('cuda'))
                    wwt = weight.mm(weight.t())
                    weight = (2.909 * self.In + wwt.mm(-4.649 * self.In + wwt.mm(4.023 * self.In - 1.283 * wwt))).mm(weight)
                    # weight = self.preconditionerwide(weight, self.pclevel)
                # for _ in range(self.pclevel):
                #     weight = 1.5 * weight - 0.5 * weight.mm(weight.t()).mm(weight)
                weight = weight.view(weight_shape)
            return weight 
        else:
            return weight.view(weight_shape) 


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
    if PC:
        return PC_Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, use_adaptivePC=False, pclevel=PC)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1, PC=0):
    """1x1 convolution"""
    # return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    if PC:
        return PC_Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, 
                    bias=False, use_adaptivePC=False, pclevel=PC)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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
        # self.scale0 = nn.Parameter(torch.ones(1))
        self.scale = nn.Parameter(torch.ones(1)*10.)
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x + self.bias1a)
        out = self.relu(out + self.bias1b)

        out = self.conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b

        if self.downsample is not None:
            identity = self.downsample(x + self.bias1a)

        out += identity
        out = self.relu(out)

        return out


class FixupBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, PC=0):
        super(FixupBottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = conv1x1(inplanes, planes, PC=PC)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = conv3x3(planes, planes, stride, PC=PC)
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.bias3a = nn.Parameter(torch.zeros(1))
        self.conv3 = conv1x1(planes, planes * self.expansion, PC=PC)
        # self.scale0 = nn.Parameter(torch.ones(1))
        # self.scale1 = nn.Parameter(torch.ones(1))
        self.scale = nn.Parameter(torch.ones(1)*10.)
        self.bias3b = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x + self.bias1a)
        out = self.relu(out + self.bias1b)

        out = self.conv2(out + self.bias2a)
        out = self.relu(out + self.bias2b)

        out = self.conv3(out + self.bias3a)
        out = out * self.scale + self.bias3b

        if self.downsample is not None:
            identity = self.downsample(x + self.bias1a)

        out += identity
        out = self.relu(out)

        return out


class FixupResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, PC=0, init='fixup'):
        super(FixupResNet, self).__init__()
        print(layers, PC, init)
        self.num_layers = sum(layers)
        self.inplanes = 64
        if PC:
            self.conv1 = PC_Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False, use_adaptivePC=False, pclevel=PC)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bias1 = nn.Parameter(torch.zeros(1))
        # self.scale = nn.Parameter(torch.ones(1))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], PC=PC)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, PC=PC)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, PC=PC)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, PC=PC)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bias2 = nn.Parameter(torch.zeros(1))
        if PC == 0:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        else:
            self.fc = PC_Linear(512 * block.expansion, num_classes, use_adaptivePC=False, pclevel=PC)
        if init == 'fixup':
            for m in self.modules():
                if isinstance(m, FixupBasicBlock):
                    nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(2. / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
                    nn.init.constant_(m.conv2.weight, 0)
                    if m.downsample is not None:
                        nn.init.normal_(m.downsample.weight, mean=0, std=np.sqrt(2. / (m.downsample.weight.shape[0] * np.prod(m.downsample.weight.shape[2:]))))
                elif isinstance(m, FixupBottleneck):
                    nn.init.normal_(m.conv1.weight, mean=0, std=np.sqrt(2. / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.25))
                    nn.init.normal_(m.conv2.weight, mean=0, std=np.sqrt(2. / (m.conv2.weight.shape[0] * np.prod(m.conv2.weight.shape[2:]))) * self.num_layers ** (-0.25))
                    nn.init.constant_(m.conv3.weight, 0)
                    if m.downsample is not None:
                        nn.init.normal_(m.downsample.weight, mean=0, std=np.sqrt(2. / (m.downsample.weight.shape[0] * np.prod(m.downsample.weight.shape[2:]))))
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)
        elif init == 'kaiming':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
            #     if isinstance(m, nn.Conv2d):
            #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #         m.weight.data.normal_(0, math.sqrt(2. / n))
            # for m in self.modules():
            #     if isinstance(m, FixupBasicBlock):
            #         nn.init.kaiming_normal_(m.conv1.weight, mode='fan_out', nonlinearity='relu')
            #         nn.init.kaiming_normal_(m.conv2.weight, mode='fan_out', nonlinearity='relu')
            #         if m.downsample is not None:
            #             nn.init.kaiming_normal_(m.downsample.weight, mode='fan_out', nonlinearity='relu')
            #     elif isinstance(m, FixupBottleneck):
            #         nn.init.kaiming_normal_(m.conv1.weight, mode='fan_out', nonlinearity='relu')
            #         nn.init.kaiming_normal_(m.conv2.weight, mode='fan_out', nonlinearity='relu')
            #         nn.init.kaiming_normal_(m.conv3.weight, mode='fan_out', nonlinearity='relu')
            #         if m.downsample is not None:
            #             nn.init.kaiming_normal_(m.downsample.weight, mode='fan_out', nonlinearity='relu')
            #     elif isinstance(m, nn.Linear):
            #         nn.init.kaiming_normal_(m.weight, mode='fan_out')
            #         nn.init.constant_(m.bias, 0)
            
        elif init == 'ortho':
            for m in self.modules():
                if isinstance(m, FixupBasicBlock):
                    nn.init.orthogonal_(m.conv1.weight, gain=self.num_layers ** (-0.5))
                    nn.init.orthogonal_(m.conv2.weight, gain=self.num_layers ** (-0.5))
                    if m.downsample is not None:
                        nn.init.orthogonal_(m.downsample.weight, gain=1)

                elif isinstance(m, FixupBottleneck):
                    nn.init.orthogonal_(m.conv1.weight, gain=self.num_layers ** (-0.25))
                    nn.init.orthogonal_(m.conv2.weight, gain=self.num_layers ** (-0.25))
                    nn.init.orthogonal_(m.conv3.weight, gain=self.num_layers ** (-0.25))
                    if m.downsample is not None:
                        nn.init.orthogonal_(m.downsample.weight, gain=1)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, PC=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = conv1x1(self.inplanes, planes * block.expansion, stride, PC=PC)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, PC=PC))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, PC=PC))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x + self.bias1)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x + self.bias2)

        return x


def fixup_resnet18(**kwargs):
    """Constructs a Fixup-ResNet-18 model.

    """
    model = FixupResNet(FixupBasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def fixup_resnet34(**kwargs):
    """Constructs a Fixup-ResNet-34 model.

    """
    model = FixupResNet(FixupBasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def fixup_resnet50(**kwargs):
    """Constructs a Fixup-ResNet-50 model.

    """
    model = FixupResNet(FixupBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def fixup_resnet101(**kwargs):
    """Constructs a Fixup-ResNet-101 model.

    """
    model = FixupResNet(FixupBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def fixup_resnet152(**kwargs):
    """Constructs a Fixup-ResNet-152 model.

    """
    model = FixupResNet(FixupBottleneck, [3, 8, 36, 3], **kwargs)
    return model