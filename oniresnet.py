import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.nn import Parameter
# import extension as my
# from extension.normalization.NormedConv import ONI_Conv2d as Conv2d_ONI

__all__ = ['ResNetDebug', 'resnetDebug18', 'resnetDebug34', 'resnetDebug50', 'resnetDebug101', 'resnetDebug152']

class IdentityModule(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(IdentityModule, self).__init__()

    def forward(self, input: torch.Tensor):
        return input

def _IdentityModule(x, *args, **kwargs):
    """return first input"""
    return IdentityModule()

class ONINorm(torch.nn.Module):
    def __init__(self, T=5, norm_groups=1, *args, **kwargs):
        super(ONINorm, self).__init__()
        self.T = T
        self.norm_groups = norm_groups
        self.eps = 1e-5

    def matrix_power3(self, Input):
        B=torch.bmm(Input, Input)
        return torch.bmm(B, Input)

    def forward(self, weight: torch.Tensor):
        assert weight.shape[0] % self.norm_groups == 0
        Z = weight.view(self.norm_groups, weight.shape[0] // self.norm_groups, -1)  # type: torch.Tensor
        Zc = Z - Z.mean(dim=-1, keepdim=True)
        S = torch.matmul(Zc, Zc.transpose(1, 2))
        eye = torch.eye(S.shape[-1]).to(S).expand(S.shape)
        S = S + self.eps*eye
        norm_S = S.norm(p='fro', dim=(1, 2), keepdim=True)
        S = S.div(norm_S)
        B = [torch.Tensor([]) for _ in range(self.T + 1)]
        B[0] = torch.eye(S.shape[-1]).to(S).expand(S.shape)
        for t in range(self.T):
            #B[t + 1] = torch.baddbmm(1.5, B[t], -0.5, torch.matrix_power(B[t], 3), S)
            B[t + 1] = torch.baddbmm(1.5, B[t], -0.5, self.matrix_power3(B[t]), S)
        W = B[self.T].matmul(Zc).div_(norm_S.sqrt())
        #print(W.matmul(W.transpose(1,2)))
        # W = oni_py.apply(weight, self.T, ctx.groups)
        return W.view_as(weight)

    def extra_repr(self):
        fmt_str = ['T={}'.format(self.T)]
        if self.norm_groups > 1:
            fmt_str.append('groups={}'.format(self.norm_groups))
        return ', '.join(fmt_str)


class ONINorm_colum(torch.nn.Module):
    def __init__(self, T=5, norm_groups=1, *args, **kwargs):
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
                 T=5, norm_groups=1, norm_channels=0, NScale=1.414, adjustScale=False, ONIRow_Fix=False, *args, **kwargs):
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
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        #self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv1 = NormConv(inplanes, planes, 3, stride, padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = Norm(planes)
        self.relu = nn.ReLU(inplace=True)
        #self.conv2 = conv3x3(planes, planes)
        #self.conv2 = Conv2d_ONI(planes, planes, 3, stride=1, padding=1, bias=False, T=7)
        self.conv2 = NormConv(planes, planes, 3, stride=1, padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = Norm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        #self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv1 = NormConv(inplanes, planes, kernel_size=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = Norm(planes)
        
        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = NormConv(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = Norm(planes)
        
        #self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.conv3 = NormConv(planes, planes * 4, kernel_size=1, bias=False)
        #self.bn3 = nn.BatchNorm2d(planes * 4)
        self.bn3 = Norm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetDebug(nn.Module):

    def __init__(self, block, layers, num_classes=1000, **kwargs):
        self.inplanes = 64
        super(ResNetDebug, self).__init__()
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = NormConv(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = Norm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                #nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                NormConv(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, ONIRow_Fix=True),
                #nn.BatchNorm2d(planes * block.expansion), )
                Norm(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

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