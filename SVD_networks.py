from torch.nn import init
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import SVDconv
from higham_disc_resblocks import SVDBlock as DiscBlock
from higham_disc_resblocks import SVDOptimizedBlock
from gen_resblocks import Block as GenBlock


# ===========  DCGAN for 32 mnist ================
# G(z)
class DCGenerator32(nn.Module):
    # initializers
    def __init__(self, dim_z=128, num_features=64, channel=3, first_kernel=4):
        super(DCGenerator32, self).__init__()
        self.dim_z = dim_z
        self.num_features = num_features
        self.first_kernel = first_kernel
        self.l1 = nn.Linear(dim_z, num_features * 8 * first_kernel * first_kernel)
        self.deconv1 = nn.ConvTranspose2d(num_features * 8, num_features * 4, 4, 2, 1)
        self.deconv1_bn = nn.BatchNorm2d(num_features * 4)
        self.deconv2 = nn.ConvTranspose2d(num_features * 4, num_features * 2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(num_features * 2)
        self.deconv3 = nn.ConvTranspose2d(num_features * 2, num_features, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(num_features)
        self.conv4 = nn.Conv2d(num_features, channel, 3, 1, 1)
        # self.initialize()

    # forward method
    def forward(self, input):
        x = self.l1(input)
        x = x.view(-1, self.num_features * 8, self.first_kernel, self.first_kernel)
        x = F.relu(self.deconv1_bn(self.deconv1(x)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.tanh(self.conv4(x))
        return x


class DCDiscriminator32(nn.Module):
    # initializers
    def __init__(self, num_features=64, channel=3, first_kernel=4, power_iter=1, usepolar=True, polar_iter=1):
        super(DCDiscriminator32, self).__init__()
        self.num_features = num_features
        # self.first_kernel = first_kernel
        self.conv1 = SVDconv.SVDConv2d(channel, num_features, 3, 1, 1)
        self.conv2 = SVDconv.SVDConv2d(num_features, num_features, 4, 2, 1)
        self.conv3 = SVDconv.SVDConv2d(num_features, num_features * 2, 3, 1, 1)
        self.conv4 = SVDconv.SVDConv2d(num_features * 2, num_features * 2, 4, 2, 1)
        self.conv5 = SVDconv.SVDConv2d(num_features * 2, num_features * 4, 3, 1, 1)
        self.conv6 = SVDconv.SVDConv2d(num_features * 4, num_features * 4, 4, 2, 1)
        self.conv7 = SVDconv.SVDConv2d(num_features * 4, num_features * 8, 3, 1, 1)
        self.proj = SVDconv.SVDLinear(num_features * 8 * first_kernel * first_kernel, 1, bias=False)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.1)  # (50, 128, 32, 32)
        x = F.leaky_relu(self.conv2(x), 0.1)  # (50, 256, 16, 16)
        x = F.leaky_relu(self.conv3(x), 0.1)  # (50, 512, 8, 8)
        x = F.leaky_relu(self.conv4(x), 0.1)  # (50, 512, 8, 8)
        x = F.leaky_relu(self.conv5(x), 0.1)
        x = F.leaky_relu(self.conv6(x), 0.1)
        x = F.leaky_relu(self.conv7(x), 0.1)
        x = x.view(x.size(0), -1)
        y = self.proj(x)
        return y

    def loss_orth(self):
        loss = self.conv1.loss_orth() + self.conv2.loss_orth() + \
               self.conv3.loss_orth() + self.conv4.loss_orth() + \
               self.conv5.loss_orth() + self.conv6.loss_orth() + \
               self.conv7.loss_orth() + self.proj.loss_orth()
        return loss


class DCGenerator256YT(nn.Module):
    def __init__(self, dim_z, numGfeature=1024):
        super(DCGenerator256YT, self).__init__()
        self.numGfeature = numGfeature
        self.fc = nn.Linear(dim_z, 4*4*numGfeature)
        self.fc_bn = nn.BatchNorm1d(4*4*numGfeature)
        self.deconv1 = nn.ConvTranspose2d(numGfeature, int(numGfeature/2), 4, 2, 1)
        self.deconv1_bn = nn.BatchNorm2d(int(numGfeature/2))
        self.deconv2 = nn.ConvTranspose2d(int(numGfeature/2), int(numGfeature/4), 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(int(numGfeature/4))
        self.deconv3 = nn.ConvTranspose2d(int(numGfeature/4), int(numGfeature/8), 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(int(numGfeature/8))
        self.deconv4 = nn.ConvTranspose2d(int(numGfeature/8), int(numGfeature/16), 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(int(numGfeature/16))
        self.deconv5 = nn.ConvTranspose2d(int(numGfeature/16), int(numGfeature/32), 4, 2, 1)
        self.deconv5_bn = nn.BatchNorm2d(int(numGfeature/32))
        self.deconv6 = nn.ConvTranspose2d(int(numGfeature/32), int(numGfeature/64), 4, 2, 1)
        self.deconv6_bn = nn.BatchNorm2d(int(numGfeature/64))
        self.conv7 = nn.Conv2d(int(numGfeature/64), int(numGfeature/64), 3, 1, 1)
        self.conv7_bn = nn.BatchNorm2d(int(numGfeature/64))
        self.conv8 = nn.Conv2d(int(numGfeature/64), 3, 3, 1, 1)

    def forward(self, input):
        x = self.fc_bn(self.fc(input))
        x = x.view(-1, self.numGfeature, 4, 4)
        x = F.relu(self.deconv1_bn(self.deconv1(x)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.relu(self.deconv5_bn(self.deconv5(x)))
        x = F.relu(self.deconv6_bn(self.deconv6(x)))
        x = F.relu(self.conv7_bn(self.conv7(x)))
        x = F.tanh(self.conv8(x))
        return x

class Discriminator256YT(nn.Module):
    def __init__(self, numDfeature=512):
        super(Discriminator256YT, self).__init__()
        self.conv1 = SVDconv.SVDConv2d(3, numDfeature/32, 4, 2, 1)
        self.conv2 = SVDconv.SVDConv2d(numDfeature/32, numDfeature/16, 4, 2, 1)
        self.conv3 = SVDconv.SVDConv2d(numDfeature/16, numDfeature/8, 4, 2, 1)
        self.conv4 = SVDconv.SVDConv2d(numDfeature/8, numDfeature/4, 4, 2, 1)
        self.conv5 = SVDconv.SVDConv2d(numDfeature/4, numDfeature/2, 4, 2, 1)
        self.conv6 = SVDconv.SVDConv2d(numDfeature/2, numDfeature, 4, 2, 1)
        self.fc = SVDconv.SVDLinear(4*4*numDfeature, 1, bias=False)

    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.1)  # (50, 128, 32, 32)
        x = F.leaky_relu(self.conv2(x), 0.1)  # (50, 256, 16, 16)
        x = F.leaky_relu(self.conv3(x), 0.1)  # (50, 512, 8, 8)
        x = F.leaky_relu(self.conv4(x), 0.1)  # (50, 512, 8, 8)
        x = F.leaky_relu(self.conv5(x), 0.1)
        x = F.leaky_relu(self.conv6(x), 0.1)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return y

    def loss_orth(self):
        loss = self.conv1.loss_orth() + self.conv2.loss_orth() + \
               self.conv3.loss_orth() + self.conv4.loss_orth() + \
               self.conv5.loss_orth() + self.conv6.loss_orth() + \
               self.fc.loss_orth()
        return loss

class DCGenerator512YT(nn.Module):
    def __init__(self, dim_z, numGfeature=1024):
        super(DCGenerator512YT, self).__init__()
        self.numGfeature = numGfeature
        self.fc = nn.Linear(dim_z, 4*4*numGfeature)
        self.fc_bn = nn.BatchNorm1d(4*4*numGfeature)
        self.deconv1 = nn.ConvTranspose2d(numGfeature, int(numGfeature/2), 4, 2, 1)
        self.deconv1_bn = nn.BatchNorm2d(int(numGfeature/2))
        self.deconv2 = nn.ConvTranspose2d(int(numGfeature/2), int(numGfeature/4), 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(int(numGfeature/4))
        self.deconv3 = nn.ConvTranspose2d(int(numGfeature/4), int(numGfeature/8), 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(int(numGfeature/8))
        self.deconv4 = nn.ConvTranspose2d(int(numGfeature/8), int(numGfeature/16), 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(int(numGfeature/16))
        self.deconv5 = nn.ConvTranspose2d(int(numGfeature/16), int(numGfeature/32), 4, 2, 1)
        self.deconv5_bn = nn.BatchNorm2d(int(numGfeature/32))
        self.deconv6 = nn.ConvTranspose2d(int(numGfeature/32), int(numGfeature/64), 4, 2, 1)
        self.deconv6_bn = nn.BatchNorm2d(int(numGfeature/64))
        self.deconv7 = nn.ConvTranspose2d(int(numGfeature/64), int(numGfeature/128), 4, 2, 1)
        self.deconv7_bn = nn.BatchNorm2d(int(numGfeature/128))
        self.conv8 = nn.Conv2d(int(numGfeature/128), int(numGfeature/128), 3, 1, 1)
        self.conv8_bn = nn.BatchNorm2d(int(numGfeature/128))
        self.conv9 = nn.Conv2d(int(numGfeature/128), 3, 3, 1, 1)

    def forward(self, input):
        x = self.fc_bn(self.fc(input))
        x = x.view(-1, self.numGfeature, 4, 4)
        for i in range(1, 8):
            x = getattr(self, 'deconv{}'.format(i))(x)
            x = getattr(self, 'deconv{}_bn'.format(i))(x)
            x = F.relu(x)
        x = F.relu(self.conv8_bn(self.conv8(x)))
        x = F.tanh(self.conv9(x))
        return x

class Discriminator512YT(nn.Module):
    def __init__(self, numDfeature=1024):
        super(Discriminator512YT, self).__init__()
        self.conv0 = SVDconv.SVDConv2d(3, numDfeature / 64, 4, 2, 1)
        self.conv1 = SVDconv.SVDConv2d(numDfeature / 64, numDfeature / 32, 4, 2, 1)
        self.conv2 = SVDconv.SVDConv2d(numDfeature / 32, numDfeature / 16, 4, 2, 1)
        self.conv3 = SVDconv.SVDConv2d(numDfeature / 16, numDfeature / 8, 4, 2, 1)
        self.conv4 = SVDconv.SVDConv2d(numDfeature / 8, numDfeature / 4, 4, 2, 1)
        self.conv5 = SVDconv.SVDConv2d(numDfeature / 4, numDfeature / 2, 4, 2, 1)
        self.conv6 = SVDconv.SVDConv2d(numDfeature / 2, numDfeature, 4, 2, 1)
        self.fc = SVDconv.SVDLinear(4 * 4 * numDfeature, 1, bias=False)

    def forward(self, input):
        x = input
        for i in range(0, 7):
            x = getattr(self, 'conv{}'.format(i))(x)
            x = F.leaky_relu(x, 0.1)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return y

    def loss_orth(self):
        loss = self.conv1.loss_orth() + self.conv2.loss_orth() + \
               self.conv3.loss_orth() + self.conv4.loss_orth() + \
               self.conv5.loss_orth() + self.conv6.loss_orth() + \
               self.fc.loss_orth() + self.conv0.loss_orth()
        return loss



# =======================================================
#                       RESNET
# =======================================================

# =======   MNIST and CIFAR ===========
class ResNetGenerator32(nn.Module):
    """Generator generates 32x32."""

    def __init__(self, num_features=256, dim_z=128, channel=3, bottom_width=4,
                 activation=F.relu, num_classes=0, distribution='normal'):
        super(ResNetGenerator32, self).__init__()
        self.num_features = num_features
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.activation = activation
        self.num_classes = num_classes
        self.distribution = distribution

        self.l1 = nn.Linear(dim_z, num_features * bottom_width ** 2)  # (_, 128*4*4)

        self.block2 = GenBlock(num_features, num_features,
                            activation=activation, upsample=True,
                            num_classes=num_classes)  # (_, 256, 8, 8)
        self.block3 = GenBlock(num_features, num_features,
                            activation=activation, upsample=True,
                            num_classes=num_classes)  # (_, 256, 16, 16)
        self.block4 = GenBlock(num_features, num_features,
                            activation=activation, upsample=True,
                            num_classes=num_classes)  # (_, 256, 32, 32)
        self.b5 = nn.BatchNorm2d(num_features)
        self.conv5 = nn.Conv2d(num_features, channel, 1, 1)  # (_, 3, 32, 32)
        # self._initialize()

    def _initialize(self):
        init.xavier_normal_(self.l1.weight.data)
        init.xavier_normal_(self.conv5.weight.data)

    def forward(self, z, y=None, **kwargs):
        h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        # print h.shape  #(_, 256, 8, 8)
        for i in range(2, 5):
            h = getattr(self, 'block{}'.format(i))(h, y, **kwargs)
        h = self.activation(self.b5(h))
        return torch.tanh(self.conv5(h))


class ResNetProjectionDiscriminator32(nn.Module):
    def __init__(self, num_features=256, channel=3, num_classes=0, activation=F.relu):
        super(ResNetProjectionDiscriminator32, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation
        # self.activation = MaxMin(num_units=num_features/2, axis=1)

        self.block1 = SVDOptimizedBlock(channel, num_features)
        self.block2 = DiscBlock(num_features, num_features, activation=self.activation, downsample=True)
        self.block3 = DiscBlock(num_features, num_features, activation=self.activation, downsample=True)
        self.block4 = DiscBlock(num_features, num_features, activation=self.activation, downsample=True)
        self.proj = SVDconv.SVDLinear(num_features, 1, bias=False)

    def forward(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.proj(h)
        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        return output

    def loss_orth(self):
        return self.block1.loss_orth() + self.block2.loss_orth() + self.block3.loss_orth() + \
               self.block4.loss_orth() + self.proj.loss_orth()


# ==============  STL ==============
class ResNetGenerator48(nn.Module):
    """Generator generates 48x48."""
    def __init__(self, num_features=64, dim_z=128, bottom_width=6,
                 activation=F.relu, num_classes=0, distribution='normal'):
        super(ResNetGenerator48, self).__init__()
        self.num_features = num_features
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.activation = activation
        self.num_classes = num_classes
        self.distribution = distribution

        self.l1 = nn.Linear(dim_z, 8 * num_features * bottom_width ** 2)  # (_, 512, 6, 6)
        self.block2 = GenBlock(num_features * 8, num_features * 4, activation=activation, upsample=True,
                            num_classes=num_classes)  # (_, 256, 12, 12)
        self.block3 = GenBlock(num_features * 4, num_features * 2, activation=activation, upsample=True,
                            num_classes=num_classes)  # (_, 128, 24, 24)
        self.block4 = GenBlock(num_features * 2, num_features, activation=activation, upsample=True,
                            num_classes=num_classes)  # (_, 64, 48, 48)
        self.b5 = nn.BatchNorm2d(num_features)
        self.conv5 = nn.Conv2d(num_features, 3, 1, 1)  # (_, 3, 48, 48)
        self._initialize()

    def _initialize(self):
        init.xavier_normal_(self.l1.weight.data)
        init.xavier_normal_(self.conv5.weight.data)

    def forward(self, z, y=None, **kwargs):
        h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        for i in range(2, 5):
            h = getattr(self, 'block{}'.format(i))(h, y, **kwargs)
        h = self.activation(self.b5(h))
        return torch.tanh(self.conv5(h))


class ResNetProjectionDiscriminator48(nn.Module):
    def __init__(self, num_features=64, num_classes=0, activation=F.relu, power_iter=1, usepolar=True, polar_iter=5, lipconst=1):
        super(ResNetProjectionDiscriminator48, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = SVDOptimizedBlock(3, num_features)
        self.block2 = DiscBlock(num_features, num_features * 2, activation=activation, downsample=True)
        self.block3 = DiscBlock(num_features * 2, num_features * 4, activation=activation, downsample=True)
        self.block4 = DiscBlock(num_features * 4, num_features * 8, activation=activation, downsample=True)
        self.block5 = DiscBlock(num_features * 8, num_features * 16, activation=activation, downsample=True)
        self.proj = SVDconv.SVDLinear(num_features*16, 1, bias=False)

    def forward(self, x, y=None):
        h = x  # (10, 3, 48, 48)
        h = self.block1(h)  # (10, 64, 24, 24)
        h = self.block2(h)  # (10, 128, 12, 12)
        h = self.block3(h)  # (10, 256, 6, 6)
        h = self.block4(h)  # (10, 512, 3, 3)
        h = self.block5(h)  # (10, 1024, 1, 1)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3)) # (10, 1024)
        output = self.proj(h)
        return output

    def loss_orth(self):
        return self.block1.loss_orth() + self.block2.loss_orth() + self.block3.loss_orth() + \
               self.block4.loss_orth() + self.block5.loss_orth() + self.proj.loss_orth()


def getGD(structure, dataset, num_Gfeatures, num_Dfeatures, image_size, dim_z=128):
    if structure == 'dcgan':
        if dataset == 'mnist':
            netG = DCGenerator32(num_features=num_Gfeatures, channel=1, dim_z=dim_z)
            netD = DCDiscriminator32(num_features=num_Dfeatures, channel=1)
        elif dataset == 'cifar':
            netG = DCGenerator32(num_features=num_Gfeatures, channel=3)
            netD = DCDiscriminator32(num_features=num_Dfeatures, channel=3)
        elif dataset == 'stl':
            netG = DCGenerator32(num_features=num_Gfeatures, channel=3, first_kernel=6)
            netD = DCDiscriminator32(num_features=num_Dfeatures, channel=3, first_kernel=6)
        elif dataset == 'lsun':
            netG = DCGenerator256YT(dim_z=dim_z, numGfeature=num_Gfeatures)
            netD = Discriminator256YT(numDfeature=num_Dfeatures)
        elif dataset == 'celeba':
            netG = DCGenerator512YT(dim_z=dim_z, numGfeature=num_Gfeatures)
            netD = Discriminator512YT(numDfeature=num_Dfeatures)
    elif structure == 'resnet':
        leaky_relu = lambda x: F.leaky_relu(x, negative_slope=0.1)
        if dataset == 'cifar':
            netG = ResNetGenerator32(num_features=num_Gfeatures)
            num_Dfeatures /= 2
            netD = ResNetProjectionDiscriminator32(num_features=num_Dfeatures, activation=leaky_relu)
        elif dataset == 'stl':
            netG = ResNetGenerator48(num_features=num_Gfeatures)
            netD = ResNetProjectionDiscriminator48(num_features=num_Dfeatures, activation=leaky_relu)
    return netG, netD
