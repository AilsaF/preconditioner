from torch.nn import init
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import Higham_norm

from higham_disc_resblocks import Block as DiscBlock
from higham_disc_resblocks import OptimizedBlock
from gen_resblocks import Block as GenBlock
from torch.nn import utils

# ================= toy =================

class generatortoy(torch.nn.Module):
    def __init__(self, dim_z, g_dim):
        super(generatortoy, self).__init__()
        self.fc1 = torch.nn.Linear(dim_z, g_dim)
        self.fc2 = torch.nn.Linear(g_dim, g_dim)
        self.fc3 = torch.nn.Linear(g_dim, 2)

    def forward(self, z):
        x = self.fc1(z)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x)
        return x


class discriminatortoy(torch.nn.Module):
    def __init__(self, d_dim, use_adaptivePC=True, pclevel=1, diter=1):
        super(discriminatortoy, self).__init__()
        self.fc1 = Higham_norm.spectral_norm(torch.nn.Linear(2, d_dim), use_adaptivePC=use_adaptivePC, pclevel=pclevel,
                                             diter=diter)
        self.fc2 = Higham_norm.spectral_norm(torch.nn.Linear(d_dim, d_dim), use_adaptivePC=use_adaptivePC,
                                             pclevel=pclevel, diter=diter)
        self.proj = Higham_norm.spectral_norm(torch.nn.Linear(d_dim, 1, bias=False), use_adaptivePC=use_adaptivePC,
                                              pclevel=pclevel, diter=diter)
        self.activation = lambda x: F.leaky_relu(x, negative_slope=0.1)

    # forward method
    def forward(self, input):
        x = input
        for i in range(1, 3):
            x = getattr(self, 'fc{}'.format(i))(x)
            x = F.leaky_relu(x, 0.1)
        x = x.view(x.size(0), -1)
        y = self.proj(x)
        return y

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

    # forward method
    def forward(self, input):
        x = self.l1(input)
        x = x.view(-1, self.num_features * 8, self.first_kernel, self.first_kernel)
        for i in range(1, 4):
            x = getattr(self, 'deconv{}'.format(i))(x)
            x = getattr(self, 'deconv{}_bn'.format(i))(x)
            x = F.relu(x)
        x = F.tanh(self.conv4(x))
        return x


class DCDiscriminator32(nn.Module):
    # initializers
    def __init__(self, num_features=64, channel=3, first_kernel=4, use_adaptivePC=True, pclevel=1, diter=1):
        super(DCDiscriminator32, self).__init__()
        self.num_features = num_features
        # self.first_kernel = first_kernel
        self.conv1 = Higham_norm.spectral_norm(nn.Conv2d(channel, num_features, 3, 1, 1),
                                               use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.conv2 = Higham_norm.spectral_norm(nn.Conv2d(num_features, num_features, 4, 2, 1),
                                               use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.conv3 = Higham_norm.spectral_norm(nn.Conv2d(num_features, num_features * 2, 3, 1, 1),
                                               use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.conv4 = Higham_norm.spectral_norm(nn.Conv2d(num_features * 2, num_features * 2, 4, 2, 1),
                                               use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.conv5 = Higham_norm.spectral_norm(nn.Conv2d(num_features * 2, num_features * 4, 3, 1, 1),
                                               use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.conv6 = Higham_norm.spectral_norm(nn.Conv2d(num_features * 4, num_features * 4, 4, 2, 1),
                                               use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.conv7 = Higham_norm.spectral_norm(nn.Conv2d(num_features * 4, num_features * 8, 3, 1, 1),
                                               use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.proj = Higham_norm.spectral_norm(nn.Linear(num_features * 8 * first_kernel * first_kernel, 1, bias=False),
                                              use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)

    # forward method
    def forward(self, input):
        x = input
        for i in range(1, 8):
            x = getattr(self, 'conv{}'.format(i))(x)
            x = F.leaky_relu(x, 0.1)
        x = x.view(x.size(0), -1)
        y = self.proj(x)
        return y


class DCGenerator64(nn.Module):
    def __init__(self, dim_z, num_features=64):
        super(DCGenerator64, self).__init__()
        self.num_features = num_features
        self.dim = 64//16  # = 4
        self.fc = nn.Linear(dim_z, num_features*8*self.dim*self.dim)
        self.fc_bn = nn.BatchNorm1d(num_features*8*self.dim*self.dim)

        self.deconv1 = nn.ConvTranspose2d(num_features*8, num_features*4, 4, 2, 1)
        self.deconv1_bn = nn.BatchNorm2d(num_features*4)
        self.deconv2 = nn.ConvTranspose2d(num_features*4, num_features*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(num_features*2)
        self.deconv3 = nn.ConvTranspose2d(num_features*2, num_features, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(num_features)
        self.deconv4 = nn.ConvTranspose2d(num_features, 3, 4, 2, 1)

    def forward(self, input):
        x = self.fc_bn(self.fc(input))
        x = x.view(-1, self.num_features*8, self.dim, self.dim)
        for i in range(1, 4):
            x = getattr(self, 'deconv{}'.format(i))(x)
            x = getattr(self, 'deconv{}_bn'.format(i))(x)
            x = F.relu(x)
        x = F.tanh(self.deconv4(x))
        return x

class Discriminator64(nn.Module):
    def __init__(self, numDfeature=64, use_adaptivePC=True, pclevel=1, diter=1):
        super(Discriminator64, self).__init__()
        self.conv1 = Higham_norm.spectral_norm(nn.Conv2d(3, numDfeature, 4, 2, 1), use_adaptivePC=use_adaptivePC,
                                               pclevel=pclevel, diter=diter)
        self.conv2 = Higham_norm.spectral_norm(nn.Conv2d(numDfeature, numDfeature * 2, 4, 2, 1),
                                               use_adaptivePC=use_adaptivePC,
                                               pclevel=pclevel, diter=diter)
        self.conv3 = Higham_norm.spectral_norm(nn.Conv2d(numDfeature * 2, numDfeature * 4, 4, 2, 1),
                                               use_adaptivePC=use_adaptivePC,
                                               pclevel=pclevel, diter=diter)
        self.conv4 = Higham_norm.spectral_norm(nn.Conv2d(numDfeature * 4, numDfeature * 8, 4, 2, 1),
                                               use_adaptivePC=use_adaptivePC,
                                               pclevel=pclevel, diter=diter)
        self.proj = Higham_norm.spectral_norm(nn.Linear(4 * 4 * numDfeature * 8, 1, bias=False),
                                              use_adaptivePC=use_adaptivePC,
                                              pclevel=pclevel, diter=diter)
    def forward(self, input):
        x = input
        for i in range(1, 5):
            x = getattr(self, 'conv{}'.format(i))(x)
            x = F.leaky_relu(x, 0.1)
        x = x.view(x.size(0), -1)
        y = self.proj(x)
        return y

class DCGenerator256(nn.Module):
    def __init__(self, dim_z, numGfeature=64):
        super(DCGenerator256, self).__init__()
        self.num_features = numGfeature
        self.dim = 128/16  # = 8
        self.fc = nn.Linear(dim_z, numGfeature*16*self.dim*self.dim)
        self.fc_bn = nn.BatchNorm1d(numGfeature*16*self.dim*self.dim)
        self.deconv1 = nn.ConvTranspose2d(numGfeature*16, numGfeature*8, 4, 2, 1)
        self.deconv1_bn = nn.BatchNorm2d(numGfeature*8)
        self.deconv2 = nn.ConvTranspose2d(numGfeature*8, numGfeature*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(numGfeature*4)
        self.deconv3 = nn.ConvTranspose2d(numGfeature*4, numGfeature*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(numGfeature*2)
        self.deconv4 = nn.ConvTranspose2d(numGfeature*2, numGfeature, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(numGfeature)
        self.deconv5 = nn.ConvTranspose2d(numGfeature, 3, 4, 2, 1)

    def forward(self, input):
        x = self.fc_bn(self.fc(input))
        x = x.view(-1, self.num_features*16, self.dim, self.dim)
        for i in range(1, 5):
            x = getattr(self, 'deconv{}'.format(i))(x)
            x = getattr(self, 'deconv{}_bn'.format(i))(x)
            x = F.relu(x)
        x = F.tanh(self.deconv5(x))
        return x

class Discriminator256(nn.Module):
    def __init__(self, numDfeature=64, use_adaptivePC=True, pclevel=1, diter=1):
        super(Discriminator256, self).__init__()
        self.conv1 = Higham_norm.spectral_norm(nn.Conv2d(3, numDfeature, 3, 1, 1), use_adaptivePC=use_adaptivePC,
                                               pclevel=pclevel, diter=diter)
        self.conv2 = Higham_norm.spectral_norm(nn.Conv2d(numDfeature, numDfeature * 2, 4, 2, 1),
                                               use_adaptivePC=use_adaptivePC,
                                               pclevel=pclevel, diter=diter)
        self.conv3 = Higham_norm.spectral_norm(nn.Conv2d(numDfeature * 2, numDfeature * 4, 4, 2, 1),
                                               use_adaptivePC=use_adaptivePC,
                                               pclevel=pclevel, diter=diter)
        self.conv4 = Higham_norm.spectral_norm(nn.Conv2d(numDfeature * 4, numDfeature * 8, 4, 2, 1),
                                               use_adaptivePC=use_adaptivePC,
                                               pclevel=pclevel, diter=diter)
        self.conv5 = Higham_norm.spectral_norm(nn.Conv2d(numDfeature * 8, numDfeature * 16, 4, 2, 1),
                                               use_adaptivePC=use_adaptivePC,
                                               pclevel=pclevel, diter=diter)
        self.proj = Higham_norm.spectral_norm(nn.Linear(16 * 16 * numDfeature * 16, 1, bias=False),
                                              use_adaptivePC=use_adaptivePC,
                                              pclevel=pclevel, diter=diter)
    def forward(self, input):
        x = input
        for i in range(1, 6):
            x = getattr(self, 'conv{}'.format(i))(x)
            x = F.leaky_relu(x, 0.1)
        x = x.view(x.size(0), -1)
        y = self.proj(x)
        return y


class DCGenerator128YT(nn.Module):
    def __init__(self, dim_z, numGfeature=512):
        super(DCGenerator128YT, self).__init__()
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
        self.conv6 = nn.Conv2d(int(numGfeature/32), int(numGfeature/32), 3, 1, 1)
        self.conv6_bn = nn.BatchNorm2d(int(numGfeature/32))
        self.conv7 = nn.Conv2d(int(numGfeature/32), 3, 3, 1, 1)

    def forward(self, input):
        x = self.fc_bn(self.fc(input))
        x = x.view(-1, self.numGfeature, 4, 4)
        for i in range(1, 6):
            x = getattr(self, 'deconv{}'.format(i))(x)
            x = getattr(self, 'deconv{}_bn'.format(i))(x)
            x = F.relu(x)
        x = F.relu(self.conv6_bn(self.conv6(x)))
        x = F.tanh(self.conv7(x))
        return x

class Discriminator128YT(nn.Module):
    def __init__(self, numDfeature=512, use_adaptivePC=True, pclevel=1, diter=1):
        super(Discriminator128YT, self).__init__()
        self.conv1 = Higham_norm.spectral_norm(nn.Conv2d(3, numDfeature / 16, 4, 2, 1), use_adaptivePC=use_adaptivePC,
                                               pclevel=pclevel, diter=diter)
        self.conv2 = Higham_norm.spectral_norm(nn.Conv2d(numDfeature / 16, numDfeature / 8, 4, 2, 1),
                                               use_adaptivePC=use_adaptivePC,
                                               pclevel=pclevel, diter=diter)
        self.conv3 = Higham_norm.spectral_norm(nn.Conv2d(numDfeature / 8, numDfeature / 4, 4, 2, 1),
                                               use_adaptivePC=use_adaptivePC,
                                               pclevel=pclevel, diter=diter)
        self.conv4 = Higham_norm.spectral_norm(nn.Conv2d(numDfeature / 4, numDfeature / 2, 4, 2, 1),
                                               use_adaptivePC=use_adaptivePC,
                                               pclevel=pclevel, diter=diter)
        self.conv5 = Higham_norm.spectral_norm(nn.Conv2d(numDfeature / 2, numDfeature, 4, 2, 1),
                                               use_adaptivePC=use_adaptivePC,
                                               pclevel=pclevel, diter=diter)
        self.proj = Higham_norm.spectral_norm(nn.Linear(4 * 4 * numDfeature, 1, bias=False),
                                              use_adaptivePC=use_adaptivePC,
                                              pclevel=pclevel, diter=diter)

    def forward(self, input):
        x = input
        for i in range(1, 6):
            x = getattr(self, 'conv{}'.format(i))(x)
            x = F.leaky_relu(x, 0.1)
        x = x.view(x.size(0), -1)
        y = self.proj(x)
        return y


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
        for i in range(1, 7):
            x = getattr(self, 'deconv{}'.format(i))(x)
            x = getattr(self, 'deconv{}_bn'.format(i))(x)
            x = F.relu(x)
        x = F.relu(self.conv7_bn(self.conv7(x)))
        x = F.tanh(self.conv8(x))
        return x

class Discriminator256YT(nn.Module):
    def __init__(self, numDfeature=512, use_adaptivePC=True, pclevel=1, diter=1):
        super(Discriminator256YT, self).__init__()
        self.conv1 = Higham_norm.spectral_norm(nn.Conv2d(3, numDfeature / 32, 4, 2, 1), use_adaptivePC=use_adaptivePC,
                                               pclevel=pclevel, diter=diter)
        self.conv2 = Higham_norm.spectral_norm(nn.Conv2d(numDfeature / 32, numDfeature / 16, 4, 2, 1),
                                               use_adaptivePC=use_adaptivePC,
                                               pclevel=pclevel, diter=diter)
        self.conv3 = Higham_norm.spectral_norm(nn.Conv2d(numDfeature / 16, numDfeature / 8, 4, 2, 1),
                                               use_adaptivePC=use_adaptivePC,
                                               pclevel=pclevel, diter=diter)
        self.conv4 = Higham_norm.spectral_norm(nn.Conv2d(numDfeature / 8, numDfeature / 4, 4, 2, 1),
                                               use_adaptivePC=use_adaptivePC,
                                               pclevel=pclevel, diter=diter)
        self.conv5 = Higham_norm.spectral_norm(nn.Conv2d(numDfeature / 4, numDfeature / 2, 4, 2, 1),
                                               use_adaptivePC=use_adaptivePC,
                                               pclevel=pclevel, diter=diter)
        self.conv6 = Higham_norm.spectral_norm(nn.Conv2d(numDfeature / 2, numDfeature, 4, 2, 1),
                                               use_adaptivePC=use_adaptivePC,
                                               pclevel=pclevel, diter=diter)
        self.proj = Higham_norm.spectral_norm(nn.Linear(4 * 4 * numDfeature, 1, bias=False),
                                              use_adaptivePC=use_adaptivePC,
                                              pclevel=pclevel, diter=diter)

    def forward(self, input):
        x = input
        for i in range(1, 7):
            x = getattr(self, 'conv{}'.format(i))(x)
            x = F.leaky_relu(x, 0.1)
        x = x.view(x.size(0), -1)
        y = self.proj(x)
        return y


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
    def __init__(self, numDfeature=1024, use_adaptivePC=True, pclevel=1, diter=1):
        super(Discriminator512YT, self).__init__()
        self.conv0 = Higham_norm.spectral_norm(nn.Conv2d(3, numDfeature / 64, 4, 2, 1), use_adaptivePC=use_adaptivePC,
                                               pclevel=pclevel, diter=diter)
        self.conv1 = Higham_norm.spectral_norm(nn.Conv2d(numDfeature / 64, numDfeature / 32, 4, 2, 1), use_adaptivePC=use_adaptivePC,
                                               pclevel=pclevel, diter=diter)
        self.conv2 = Higham_norm.spectral_norm(nn.Conv2d(numDfeature / 32, numDfeature / 16, 4, 2, 1),
                                               use_adaptivePC=use_adaptivePC,
                                               pclevel=pclevel, diter=diter)
        self.conv3 = Higham_norm.spectral_norm(nn.Conv2d(numDfeature / 16, numDfeature / 8, 4, 2, 1),
                                               use_adaptivePC=use_adaptivePC,
                                               pclevel=pclevel, diter=diter)
        self.conv4 = Higham_norm.spectral_norm(nn.Conv2d(numDfeature / 8, numDfeature / 4, 4, 2, 1),
                                               use_adaptivePC=use_adaptivePC,
                                               pclevel=pclevel, diter=diter)
        self.conv5 = Higham_norm.spectral_norm(nn.Conv2d(numDfeature / 4, numDfeature / 2, 4, 2, 1),
                                               use_adaptivePC=use_adaptivePC,
                                               pclevel=pclevel, diter=diter)
        self.conv6 = Higham_norm.spectral_norm(nn.Conv2d(numDfeature / 2, numDfeature, 4, 2, 1),
                                               use_adaptivePC=use_adaptivePC,
                                               pclevel=pclevel, diter=diter)
        self.proj = Higham_norm.spectral_norm(nn.Linear(4 * 4 * numDfeature, 1, bias=False),
                                              use_adaptivePC=use_adaptivePC,
                                              pclevel=pclevel, diter=diter)

    def forward(self, input):
        x = input
        for i in range(0, 7):
            x = getattr(self, 'conv{}'.format(i))(x)
            x = F.leaky_relu(x, 0.1)
        x = x.view(x.size(0), -1)
        y = self.proj(x)
        return y


def init_ortho_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        init.orthogonal_(m.weight)

def init_normal_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        init.xavier_normal_(m.weight)

def init_xavierunif_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        init.xavier_uniform_(m.weight)


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

    def forward(self, z, y=None, **kwargs):
        h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        for i in range(2, 5):
            h = getattr(self, 'block{}'.format(i))(h, y, **kwargs)
        h = self.activation(self.b5(h))
        return torch.tanh(self.conv5(h))


class ResNetProjectionDiscriminator32(nn.Module):
    def __init__(self, num_features=256, channel=3, num_classes=0, activation=F.relu, use_adaptivePC=True, pclevel=1, diter=1):
        super(ResNetProjectionDiscriminator32, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation
        self.block1 = OptimizedBlock(channel, num_features, use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.block2 = DiscBlock(num_features, num_features, activation=self.activation, downsample=True,
                                use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.block3 = DiscBlock(num_features, num_features, activation=self.activation, downsample=False,
                                use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.block4 = DiscBlock(num_features, num_features, activation=self.activation, downsample=False,
                                use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.proj = Higham_norm.spectral_norm(nn.Linear(num_features, 1, bias=False), use_adaptivePC=use_adaptivePC,
                                              pclevel=pclevel, diter=diter)

    def forward(self, x, y=None):
        h = x
        for i in range(1, 5):
            h = getattr(self, 'block{}'.format(i))(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.proj(h)
        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        return output

# ============== deepgan =================
from higham_disc_resblocks import DeepBlock2 as DeepDiscBlock
from gen_resblocks import DeepBlock2 as DeepGenBlock

class DeepResNetGenerator32(nn.Module):
    """Generator generates 32x32."""
    def __init__(self, num_features=256, dim_z=128, channel=3, bottom_width=4,
                 activation=F.relu, num_classes=0, distribution='normal'):
        super(DeepResNetGenerator32, self).__init__()
        self.num_features = num_features
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.activation = activation
        self.num_classes = num_classes
        self.distribution = distribution

        self.l1 = nn.Linear(dim_z, num_features * bottom_width ** 2)  # (_, 128*4*4)
        
        self.block2 = DeepGenBlock(num_features, num_features,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.block3 = DeepGenBlock(num_features, num_features,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.block4 = DeepGenBlock(num_features, num_features,
                            activation=activation, upsample=True,
                            num_classes=num_classes)  
        # self.block5 = DeepGenBlock(num_features, num_features,
        #                     activation=activation, upsample=False,
        #                     num_classes=num_classes)                 
        
        self.b5 = nn.BatchNorm2d(num_features)
        self.conv5 = nn.Conv2d(num_features, channel, 1, 1)  # (_, 3, 32, 32)

    def forward(self, z, y=None, **kwargs):
        h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        for i in range(2, 5):
            h = getattr(self, 'block{}'.format(i))(h, y, **kwargs)
        h = self.activation(self.b5(h))
        return torch.tanh(self.conv5(h))

class DeepResNetProjectionDiscriminator32(nn.Module):
    def __init__(self, num_features=256, channel=3, num_classes=0, activation=F.relu, use_adaptivePC=True, pclevel=1, diter=1):
        super(DeepResNetProjectionDiscriminator32, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation
        self.c1 = Higham_norm.spectral_norm(nn.Conv2d(channel, num_features, 3, padding=1),
                                use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.block1 = DeepDiscBlock(num_features, num_features, activation=self.activation, downsample=True,
                                use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.block2 = DeepDiscBlock(num_features, num_features, activation=self.activation, downsample=True,
                                use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.block3 = DeepDiscBlock(num_features, num_features, activation=self.activation, downsample=True,
                                use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.block4 = DeepDiscBlock(num_features, num_features, activation=self.activation, downsample=True,
                                use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)

        self.proj = Higham_norm.spectral_norm(nn.Linear(num_features, 1, bias=False), use_adaptivePC=use_adaptivePC,
                                              pclevel=pclevel, diter=diter)
        if num_classes > 0:
            self.l_y = Higham_norm.spectral_norm(nn.Embedding(num_classes, num_features), use_adaptivePC=use_adaptivePC,
                                              pclevel=pclevel, diter=diter)
        self._initialize()

    def _initialize(self):
        if hasattr(self, 'l_y'):
            init.xavier_uniform_(self.l_y.weight.data)

    def forward(self, x, y=None):
        h = x
        h = self.c1(x)
        for i in range(1, 5):
            h = getattr(self, 'block{}'.format(i))(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.proj(h)
        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        return output


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

    def forward(self, z, y=None, **kwargs):
        h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        for i in range(2, 5):
            h = getattr(self, 'block{}'.format(i))(h, y, **kwargs)
        h = self.activation(self.b5(h))
        return torch.tanh(self.conv5(h))


class ResNetProjectionDiscriminator48(nn.Module):
    def __init__(self, num_features=64, num_classes=0, activation=F.relu, use_adaptivePC=True, pclevel=1, diter=1):
        super(ResNetProjectionDiscriminator48, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = OptimizedBlock(3, num_features, use_adaptivePC=use_adaptivePC,
                                     pclevel=pclevel, diter=diter)
        self.block2 = DiscBlock(num_features, num_features * 2, activation=activation, downsample=True,
                                use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.block3 = DiscBlock(num_features * 2, num_features * 4, activation=activation, downsample=True,
                                use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.block4 = DiscBlock(num_features * 4, num_features * 8, activation=activation, downsample=True,
                                use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.block5 = DiscBlock(num_features * 8, num_features * 16, activation=activation, downsample=True,
                                use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.proj = Higham_norm.spectral_norm(nn.Linear(num_features * 16, 1, bias=False),
                                              use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)

    def forward(self, x, y=None):
        h = x
        for i in range(1, 6):
            h = getattr(self, 'block{}'.format(i))(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3)) # (10, 1024)
        output = self.proj(h)
        return output


# =======  256 * 256 ===========
class ResNetGenerator128(nn.Module):
    """Generator generates 32x32."""

    def __init__(self, num_features=256, dim_z=128, channel=3, bottom_width=4,
                 activation=F.relu, num_classes=0, distribution='normal'):
        super(ResNetGenerator128, self).__init__()
        self.num_features = num_features
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.activation = activation
        self.num_classes = num_classes
        self.distribution = distribution

        self.l1 = nn.Linear(dim_z, 32 * num_features * bottom_width ** 2)  # (_, 128*4*4)
        self.block2 = GenBlock(num_features*32, num_features*16,
                            activation=activation, upsample=True,
                            num_classes=num_classes)  # (_, 256, 8, 8)
        self.block3 = GenBlock(num_features*16, num_features*8,
                            activation=activation, upsample=True,
                            num_classes=num_classes)  # (_, 256, 16, 16)
        self.block4 = GenBlock(num_features*8, num_features*4,
                            activation=activation, upsample=True,
                            num_classes=num_classes)  # (_, 256, 32, 32)
        self.block5 = GenBlock(num_features*4, num_features*2,
                               activation=activation, upsample=True,
                               num_classes=num_classes)  # (_, 256, 64, 64)
        self.block6 = GenBlock(num_features*2, num_features,
                               activation=activation, upsample=True,
                               num_classes=num_classes)  # (_, 256, 128, 128)
        # self.block7 = GenBlock(num_features, num_features,
        #                        activation=activation, upsample=True,
        #                        num_classes=num_classes)  # (_, 256, 256, 256)
        self.b8 = nn.BatchNorm2d(num_features)
        self.conv8 = nn.Conv2d(num_features, channel, 1, 1)  # (_, 3, 256, 256)

    def forward(self, z, y=None, **kwargs):
        h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        for i in range(2, 7):
            h = getattr(self, 'block{}'.format(i))(h, y, **kwargs)
        h = self.activation(self.b8(h))
        return torch.tanh(self.conv8(h))


class ResNetProjectionDiscriminator128(nn.Module):
    def __init__(self, num_features=256, channel=3, num_classes=0, activation=F.relu, use_adaptivePC=True, pclevel=0, diter=1):
        super(ResNetProjectionDiscriminator128, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation
        # self.activation = MaxMin(num_units=num_features/2, axis=1)

        self.block1 = OptimizedBlock(channel, num_features, use_adaptivePC=use_adaptivePC,
                                     pclevel=pclevel, diter=diter)
        self.block2 = DiscBlock(num_features, num_features*2, activation=self.activation, downsample=True,
                            use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.block3 = DiscBlock(num_features*2, num_features*4, activation=self.activation, downsample=True,
                            use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.block4 = DiscBlock(num_features*4, num_features*8, activation=self.activation, downsample=True,
                            use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.block5 = DiscBlock(num_features*8, num_features*16, activation=self.activation, downsample=True,
                                use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.block6 = DiscBlock(num_features*16, num_features*32, activation=self.activation, downsample=True,
                                use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        self.proj = Higham_norm.spectral_norm(nn.Linear(num_features*32, 1, bias=False),
                            use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)

    def forward(self, x, y=None):
        h = x
        for i in range(1, 7):
            h = getattr(self, 'block{}'.format(i))(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))  # (_, 128)
        output = self.proj(h) # (_, 1)
        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        return output


def getGD(structure, dataset, num_Gfeatures, num_Dfeatures, image_size, use_adaptivePC=True, pclevel=1,
          ignoreD=False, dim_z=128, diter=1):
    if structure == 'dcgan':
        if dataset == 'toy':
            netG = generatortoy(dim_z, num_Gfeatures)
            if not ignoreD:
                netD = discriminatortoy(num_Dfeatures, use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        elif dataset == 'mnist':
            netG = DCGenerator32(num_features=num_Gfeatures, channel=1, dim_z=dim_z)
            if not ignoreD:
                netD = DCDiscriminator32(num_features=num_Dfeatures, channel=1, use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        elif dataset == 'cifar':
            netG = DCGenerator32(num_features=num_Gfeatures, channel=3)
            if not ignoreD:
                netD = DCDiscriminator32(num_features=num_Dfeatures, channel=3, use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        elif dataset == 'stl':
            netG = DCGenerator32(num_features=num_Gfeatures, channel=3, first_kernel=6)
            if not ignoreD:
                netD = DCDiscriminator32(num_features=num_Dfeatures, channel=3, first_kernel=6, use_adaptivePC=use_adaptivePC,
                                         pclevel=pclevel, diter=diter)
        elif dataset in ['celeba', 'lsun', 'tower', 'church_outdoor', 'living_room']:
            if image_size == 64:
                netG = DCGenerator64(dim_z=dim_z, num_features=num_Gfeatures)
                if not ignoreD:
                    netD = Discriminator64(numDfeature=num_Dfeatures, use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
            elif image_size == 128:
                netG = DCGenerator128YT(dim_z=dim_z, numGfeature=num_Gfeatures)
                if not ignoreD:
                    netD = Discriminator128YT(numDfeature=num_Dfeatures, use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
            elif image_size == 256:
                netG = DCGenerator256YT(dim_z=dim_z, numGfeature=num_Gfeatures)
                if not ignoreD:
                    netD = Discriminator256YT(numDfeature=num_Dfeatures, use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
            elif image_size == 512:
                netG = DCGenerator512YT(dim_z=dim_z, numGfeature=num_Gfeatures)
                if not ignoreD:
                    netD = Discriminator512YT(numDfeature=num_Dfeatures, use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)

    if structure == 'resnet':
        leaky_relu = lambda x: F.leaky_relu(x, negative_slope=0.1)
        if dataset == 'cifar':
            # netG = ResNetGenerator32(num_features=num_Gfeatures)
            # if not ignoreD:
            #     num_Dfeatures /= 2
            #     netD = ResNetProjectionDiscriminator32(num_features=num_Dfeatures, activation=leaky_relu,
            #                         use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
            netG = DeepResNetGenerator32(num_features=num_Gfeatures)
            if not ignoreD:
                netD = DeepResNetProjectionDiscriminator32(num_features=num_Dfeatures, activation=leaky_relu,
                                    use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        elif dataset == 'stl':
            netG = ResNetGenerator48(num_features=num_Gfeatures)
            if not ignoreD:
                netD = ResNetProjectionDiscriminator48(num_features=num_Dfeatures, activation=leaky_relu,
                                                       use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
        elif dataset in ['celeba', 'lsun']:
            netG = ResNetGenerator128(num_features=num_Gfeatures)
            if not ignoreD:
                netD = ResNetProjectionDiscriminator128(num_features=num_Dfeatures, activation=leaky_relu,
                                   use_adaptivePC=use_adaptivePC, pclevel=pclevel, diter=diter)
    if ignoreD:
        netD = None
    else:
        netG.apply(init_normal_weights)
        # netD.apply(init_ortho_weights)
        netD.apply(init_xavierunif_weights)
    return netG, netD




