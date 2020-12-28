from torch.nn import init
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ===========  DCGAN for 32 mnist ================
# G(z)
class DCGenerator32BN(nn.Module):
    # initializers
    def __init__(self, dim_z=128, num_features=64, channel=3, first_kernel=4):
        super(DCGenerator32BN, self).__init__()
        self.dim_z = dim_z
        self.num_features = num_features
        self.first_kernel = first_kernel
        self.l1 = nn.Linear(dim_z, num_features * 8 * first_kernel * first_kernel)
        self.deconv1 = nn.ConvTranspose2d(num_features * 8, num_features * 4, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(num_features * 4, num_features * 2, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(num_features * 2, num_features, 4, 2, 1)
        self.conv4 = nn.Conv2d(num_features, channel, 3, 1, 1)
        # self.initialize()

        self.deconv1_n = nn.BatchNorm2d(num_features * 4)
        self.deconv2_n = nn.BatchNorm2d(num_features * 2)
        self.deconv3_n = nn.BatchNorm2d(num_features)

    # forward method
    def forward(self, input):
        x = self.l1(input)
        x = x.view(-1, self.num_features * 8, self.first_kernel, self.first_kernel)
        x = F.relu(self.deconv1_n(self.deconv1(x)))
        x = F.relu(self.deconv2_n(self.deconv2(x)))
        x = F.relu(self.deconv3_n(self.deconv3(x)))
        x = F.tanh(self.conv4(x))
        return x


class DCDiscriminator32BN(nn.Module):
    # initializers
    def __init__(self, num_features=64, channel=3, first_kernel=4):
        super(DCDiscriminator32BN, self).__init__()
        self.num_features = num_features
        # self.first_kernel = first_kernel
        self.conv1 = nn.Conv2d(channel, num_features, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_features, num_features, 4, 2, 1)
        self.conv3 = nn.Conv2d(num_features, num_features * 2, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_features * 2, num_features * 2, 4, 2, 1)
        self.conv5 = nn.Conv2d(num_features * 2, num_features * 4, 3, 1, 1)
        self.conv6 = nn.Conv2d(num_features * 4, num_features * 4, 4, 2, 1)
        self.conv7 = nn.Conv2d(num_features * 4, num_features * 8, 3, 1, 1)
        self.proj = nn.Linear(num_features * 8 * first_kernel * first_kernel, 1, bias=False)

        self.conv1_n = nn.BatchNorm2d(num_features)
        self.conv2_n = nn.BatchNorm2d(num_features)
        self.conv3_n = nn.BatchNorm2d(num_features * 2)
        self.conv4_n = nn.BatchNorm2d(num_features * 2)
        self.conv5_n = nn.BatchNorm2d(num_features * 4)
        self.conv6_n = nn.BatchNorm2d(num_features * 4)
        self.conv7_n = nn.BatchNorm2d(num_features * 8)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1_n(self.conv1(input)), 0.1)  # (50, 128, 32, 32)
        x = F.leaky_relu(self.conv2_n(self.conv2(x)), 0.1)  # (50, 256, 16, 16)
        x = F.leaky_relu(self.conv3_n(self.conv3(x)), 0.1)  # (50, 512, 8, 8)
        x = F.leaky_relu(self.conv4_n(self.conv4(x)), 0.1)  # (50, 512, 8, 8)
        x = F.leaky_relu(self.conv5_n(self.conv5(x)), 0.1)
        x = F.leaky_relu(self.conv6_n(self.conv6(x)), 0.1)
        x = F.leaky_relu(self.conv7_n(self.conv7(x)), 0.1)
        x = x.view(x.size(0), -1)
        y = self.proj(x)
        return y


# ===========  DCGAN for 32 mnist ================
# G(z)
class DCGenerator32WN(nn.Module):
    # initializers
    def __init__(self, dim_z=128, num_features=64, channel=3, first_kernel=4):
        super(DCGenerator32WN, self).__init__()
        self.dim_z = dim_z
        self.num_features = num_features
        self.first_kernel = first_kernel
        self.l1 = nn.Linear(dim_z, num_features * 8 * first_kernel * first_kernel)
        self.deconv1 = nn.utils.weight_norm(nn.ConvTranspose2d(num_features * 8, num_features * 4, 4, 2, 1))
        self.deconv2 = nn.utils.weight_norm(nn.ConvTranspose2d(num_features * 4, num_features * 2, 4, 2, 1))
        self.deconv3 = nn.utils.weight_norm(nn.ConvTranspose2d(num_features * 2, num_features, 4, 2, 1))
        self.conv4 = nn.Conv2d(num_features, channel, 3, 1, 1)

    # forward method
    def forward(self, input):
        x = self.l1(input)
        x = x.view(-1, self.num_features * 8, self.first_kernel, self.first_kernel)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.tanh(self.conv4(x))
        return x


class DCDiscriminator32WN(nn.Module):
    # initializers
    def __init__(self, num_features=64, channel=3, first_kernel=4):
        super(DCDiscriminator32WN, self).__init__()
        self.num_features = num_features
        # self.first_kernel = first_kernel
        self.conv1 = nn.utils.weight_norm(nn.Conv2d(channel, num_features, 3, 1, 1))
        self.conv2 = nn.utils.weight_norm(nn.Conv2d(num_features, num_features, 4, 2, 1))
        self.conv3 = nn.utils.weight_norm(nn.Conv2d(num_features, num_features * 2, 3, 1, 1))
        self.conv4 = nn.utils.weight_norm(nn.Conv2d(num_features * 2, num_features * 2, 4, 2, 1))
        self.conv5 = nn.utils.weight_norm(nn.Conv2d(num_features * 2, num_features * 4, 3, 1, 1))
        self.conv6 = nn.utils.weight_norm(nn.Conv2d(num_features * 4, num_features * 4, 4, 2, 1))
        self.conv7 = nn.utils.weight_norm(nn.Conv2d(num_features * 4, num_features * 8, 3, 1, 1))
        self.proj = nn.utils.weight_norm(nn.Linear(num_features * 8 * first_kernel * first_kernel, 1, bias=False))

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




def orthogonal_regularization(model, beta=1e-4):
    r"""
        author: Xu Mingle
        Time: February 19, 2019 15:12:43
        input:
            model: which is the model we want to use orthogonal regularization, e.g. Generator or Discriminator
            device: cpu or gpu
            beta: hyperparameter
        output: loss
    """

    # beta * (||W^T.W * (1-I)||_F)^2 or
    # beta * (||W.W.T * (1-I)||_F)^2
    # If H < W, you can use the former. If H > W, you can use the latter, so you can reduce the memory appropriately.

    loss_orth = torch.tensor(0.).cuda()

    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad and len(param.shape) == 4:
            N, C, H, W = param.shape
            weight = param.view(N * C, H, W)
            weight_squared = torch.bmm(weight, weight.permute(0, 2, 1))  # (N * C) * H * H
            ones = torch.ones(N * C, H, H, dtype=torch.float32).cuda()  # (N * C) * H * H
            diag = torch.eye(H, dtype=torch.float32).cuda()  # (N * C) * H * H
            loss_orth += ((weight_squared * (ones - diag)) ** 2).sum()
    return loss_orth * beta



def init_ortho_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        init.orthogonal_(m.weight)

def init_normal_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        init.xavier_normal_(m.weight)

def init_xavierunif_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        init.xavier_uniform_(m.weight)




def getGD(structure, dataset, num_Gfeatures, num_Dfeatures, image_size, ignoreD=False, dim_z=128, norm='batchnorm'):
    if structure == 'dcgan':
        if norm == 'weightnorm':
            if dataset == 'cifar':
                netG = DCGenerator32WN(num_features=num_Gfeatures, channel=3)
                if not ignoreD:
                    netD = DCDiscriminator32WN(num_features=num_Dfeatures, channel=3)
            elif dataset == 'stl':
                netG = DCGenerator32WN(num_features=num_Gfeatures, channel=3, first_kernel=6)
                if not ignoreD:
                    netD = DCDiscriminator32WN(num_features=num_Dfeatures, channel=3, first_kernel=6)
        elif norm in ['batchnorm', 'orthoreg']:
            if dataset == 'cifar':
                netG = DCGenerator32BN(num_features=num_Gfeatures, channel=3)
                if not ignoreD:
                    netD = DCDiscriminator32BN(num_features=num_Dfeatures, channel=3)
            elif dataset == 'stl':
                netG = DCGenerator32BN(num_features=num_Gfeatures, channel=3, first_kernel=6)
                if not ignoreD:
                    netD = DCDiscriminator32BN(num_features=num_Dfeatures, channel=3, first_kernel=6)

    if ignoreD:
        netD = None
    else:
        netG.apply(init_normal_weights)
        netD.apply(init_ortho_weights)
        # netD.apply(init_xavierunif_weights)
    return netG, netD


