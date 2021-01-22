import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.nn import Parameter
# import extension as my
# from extension.normalization.NormedConv import ONI_Conv2d as Conv2d_ONI

__all__ = ['ResNetDebug', 'resnetDebug18', 'resnetDebug34', 'resnetDebug50', 'resnetDebug101', 'resnetDebug152']

# class IdentityModule(torch.nn.Module):
#     def __init__(self, *args, **kwargs):
#         super(IdentityModule, self).__init__()

#     def forward(self, input: torch.Tensor):
#         return input

# def _IdentityModule(x, *args, **kwargs):
#     """return first input"""
#     return IdentityModule()

class ONINorm(torch.nn.Module):
    def __init__(self, T=7, norm_groups=1, *args, **kwargs):
        super(ONINorm, self).__init__()
        self.T = T
    
    def forward(self, weight):
        weight_shape = weight.shape
        weight = weight.view(weight.shape[0], -1)
        sigma = torch.norm(weight)
        # sigma = (torch.linalg.norm(weight, float('inf')) * torch.linalg.norm(weight, 1)) ** 0.5
        if sigma > 0.:
            weight = weight / sigma
            # if len(weight.shape) > 2:
            if self.T:
                # n, m = weight.shape
                # if n >= m:
                #     weight = self.preconditionertall(weight, self.pclevel)
                # else:
                #     weight = self.preconditionerwide(weight, self.pclevel)
                for _ in range(self.T):
                    weight = 1.5 * weight - 0.5 * weight.mm(weight.t()).mm(weight)
                weight = weight.view(weight_shape)
            return weight #* sigma
        else:
            return weight.view(weight_shape) #* torch.tensor(1.)
        

class ONI_Conv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 T=7, norm_groups=1, norm_channels=0, NScale=1.414, adjustScale=False, ONIRow_Fix=True, *args, **kwargs):
        super(ONI_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        print('ONI channels:--OD:',out_channels, '--ID:', in_channels, '--KS',kernel_size)
        self.weight_normalization = ONINorm(T=T, norm_groups=1)
        # self.WNScale = 1.414 # Parameter([1.414])

    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        weight_q = self.weight_normalization(self.weight)
        # weight_q = weight_q * self.WNScale
        out = F.conv2d(input_f, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out

class ONI_Linear(torch.nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True,
                 T=7, norm_groups=1, norm_channels=0, NScale=1, adjustScale=False, *args, **kwargs):
        super(ONI_Linear, self).__init__(in_channels, out_channels, bias)
        self.weight_normalization = ONINorm(T=T, norm_groups=1)
        # self.WNScale = 1.414 

    def forward(self, input_f: torch.Tensor) -> torch.Tensor:
        weight_q = self.weight_normalization(self.weight)
        # weight_q = weight_q * self.WNScale
        out = F.linear(input_f, weight_q, self.bias)
        return out


# Norm = _IdentityModule
NormConv = ONI_Conv2d


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        #self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv1 = NormConv(inplanes, planes, 3, stride, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True) 
        self.conv2 = NormConv(planes, planes, 3, stride=1, padding=1, bias=False)
        self.downsample = downsample
        self.stride = stride

        self.bias1a = nn.Parameter(torch.zeros(1))
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.scale1 = nn.Parameter(torch.ones(1)*1.414)
        self.scale2 = nn.Parameter(torch.ones(1)*1.414)
        
    def forward(self, x):
        identity = x

        out = self.conv1(x + self.bias1a)
        out = self.relu(out * self.scale1 + self.bias1b)

        out = self.conv2(out + self.bias2a)
        out = out * self.scale2 + self.bias2b

        if self.downsample is not None:
            identity = self.downsample(x + self.bias1a)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = NormConv(inplanes, planes, kernel_size=1, bias=False)
        self.conv2 = NormConv(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = NormConv(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.bias1a = nn.Parameter(torch.zeros(1))
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.bias3a = nn.Parameter(torch.zeros(1))
        self.bias3b = nn.Parameter(torch.zeros(1))
        self.scale1 = nn.Parameter(torch.ones(1)*1.414)
        self.scale2 = nn.Parameter(torch.ones(1)*1.414)
        self.scale3 = nn.Parameter(torch.ones(1)*1.414)
         
    def forward(self, x):
        identity = x

        out = self.conv1(x + self.bias1a)
        out = self.relu(out * self.scale1 + self.bias1b)

        out = self.conv2(out + self.bias2a)
        out = self.relu(out * self.scale2 + self.bias2b)

        out = self.conv3(out + self.bias3a)
        out = out * self.scale3 + self.bias3b

        if self.downsample is not None:
            identity = self.downsample(x + self.bias1a)

        out += identity
        out = self.relu(out)

        return out



class ResNetDebug(nn.Module):

    def __init__(self, block, layers, num_classes=1000, **kwargs):
        self.inplanes = 64
        super(ResNetDebug, self).__init__()
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = NormConv(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = ONI_Linear(512 * block.expansion, num_classes, adjustScale=True)
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.scale1 = nn.Parameter(torch.ones(1)*1.414)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, nn.BatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                #nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                NormConv(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, ONIRow_Fix=True),
               )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x * self.scale1 + self.bias1)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x + self.bias2)

        return x


def resnetDebug18(pretrained=False, **kwargs):
    """Constructs a ResNetDebug-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetDebug(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnetDebug18']))
    return model


def resnetDebug34(pretrained=False, **kwargs):
    """Constructs a ResNetDebug-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetDebug(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnetDebug34']))
    return model


def resnetDebug50(pretrained=False, **kwargs):
    """Constructs a ResNetDebug-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetDebug(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnetDebug50']))
    return model


def resnetDebug101(pretrained=False, **kwargs):
    """Constructs a ResNetDebug-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetDebug(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnetDebug101']))
    return model


def resnetDebug152(pretrained=False, **kwargs):
    """Constructs a ResNetDebug-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetDebug(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnetDebug152']))
    return model