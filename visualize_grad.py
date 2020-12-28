from torch.autograd import Variable, grad
import torch
import numpy as np
import torch.nn.functional as F
import os
import sys
sys.path.append('/home/illini/rsgan')
# import networks
import datasets
import hignorm_networks

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


def get_dloss(type, dis_fake, dis_real):
    if type == 'log':
        return torch.mean(F.softplus(-dis_real)) + torch.mean(F.softplus(dis_fake))
    elif type == 'hinge':
        return torch.mean(torch.relu(1. - dis_real)) + torch.mean(torch.relu(1. + dis_fake))

def get_gloss(type, dis_fake):
    if type == 'log':
        return torch.mean(F.softplus(-dis_fake))
    elif type == 'hinge':
        return -torch.mean(dis_fake)

def getGradNorm(model):
    norms = []
    for name, p in model.named_parameters():
        param_norm = p.grad.data.cpu()
        norms.append(torch.flatten(param_norm))
    return torch.cat(norms)

def plotBars(m, name):
    means = []
    stds = []
    for i in range(m.shape[0]):
        means.append(m[i].mean().item())
        stds.append(m[i].std().item())
    # print stds
    x = range(4000, 100001, 4000)
    plt.figure()
    plt.errorbar(x, means, yerr=stds)
    plt.savefig(name+'.pdf')


G_feature = 32
D_feature = 128

vis_D = True
folder = 'ie510result/cifar_structdcgan_GfeatureNum{}_DfeatureNum{}_losslog/'.format(G_feature, D_feature)
exp_ = 'vanilla_CIFAR_size32_dlr0.0002_glr0.0002_diter1_giter1_b10.5_b20.999_Gnumfea32_Dnumfea128_batchsize64_sniter1_usepolarTrueiter3_ematricktry4/'

batch_size = 64
usepolar = 'usepolarTrue' in exp_
loss_type = 'hinge' if 'hinge' in folder else 'log'
structure = 'resnet' if 'resnet' in folder else 'dcgan'
device = torch.device('cpu')

netG, netD = hignorm_networks.getGD('dcgan', 'cifar', G_feature, D_feature, dim_z=128, image_size=256, polar_iter=3, power_iter=1, lipconst=1)
# netG.cuda()
# netD.cuda()
z = torch.randn(batch_size, 128, device=device)

loader = datasets.getDataLoader('cifar', 32, batch_size)
data_iter = iter(loader)
x_true = next(data_iter)[0]#.cuda()


if vis_D:
    save_folder = os.path.join('Dlosslanscape', structure, exp_)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    gradient_list = []
    for iteration in range(4000, 100001, 4000):
        print iteration
        # Gmodel_state = torch.load(os.path.join(folder, exp_, 'models/G_epoch{}.pth'.format(iteration)))
        Dmodel_state = torch.load(os.path.join(folder, exp_, 'models/D_epoch{}.pth'.format(iteration)))
        # netG.load_state_dict(Gmodel_state)
        netD.load_state_dict(Dmodel_state)
        # x_hat = netG(z)
        netD.zero_grad()
        # y_hat = netD(x_hat)
        y = netD(x_true)
        # d_loss = get_dloss(loss_type, y_hat, y)
        d_loss = y.mean()
        d_loss.backward(retain_graph=True)
        gradients = getGradNorm(netD)
        gradient_list.append(gradients.reshape(1, -1))
        # print torch.cat(gradient_list, dim=0).shape
        np.save(save_folder+'/Dgrad_jacob.npy'.format(), torch.cat(gradient_list, dim=0))
    gradient_list = torch.cat(gradient_list, dim=0)
    f = open(save_folder + "/Dgrad_jacob_scalar.txt", "w")
    f.write("For the whole matrix: min: {}; max: {}; std: {}\n".format(gradient_list.min(), gradient_list.max(), gradient_list.std()))
    f.write("For the last iter: min: {}; max: {}; std: {}\n".format(gradient_list[-1].min(), gradient_list[-1].max(),gradient_list[-1].std()))
    f.close()
    plotBars(gradient_list, save_folder + '/Dgrad_jacob')

    gradient_list = []
    for iteration in range(4000, 100001, 4000):
        print iteration
        Gmodel_state = torch.load(os.path.join(folder, exp_, 'models/G_epoch{}.pth'.format(iteration)))
        Dmodel_state = torch.load(os.path.join(folder, exp_, 'models/D_epoch{}.pth'.format(iteration)))
        netG.load_state_dict(Gmodel_state)
        netD.load_state_dict(Dmodel_state)
        x_hat = netG(z)
        netD.zero_grad()
        y_hat = netD(x_hat)
        y = netD(x_true)
        d_loss = get_dloss(loss_type, y_hat, y)
        d_loss.backward(retain_graph=True)
        gradients = getGradNorm(netD)
        gradient_list.append(gradients.reshape(1, -1))
        np.save(save_folder + '/Dgrad.npy'.format(), torch.cat(gradient_list, dim=0))
    gradient_list = torch.cat(gradient_list, dim=0)
    f = open(save_folder + "/Dgrad_scalar.txt", "w")
    f.write("For the whole matrix: min: {}; max: {}; std: {}\n".format(gradient_list.min(), gradient_list.max(),
                                                                       gradient_list.std()))
    f.write("For the last iter: min: {}; max: {}; std: {}\n".format(gradient_list[-1].min(), gradient_list[-1].max(),
                                                                    gradient_list[-1].std()))
    f.close()
    plotBars(gradient_list, save_folder + '/Dgrad')




if not vis_D:
    save_folder = os.path.join('Glosslanscape', structure, exp_)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    gradient_list = []

    for iteration in range(200000, 1000, -4000):
        print iteration
        Gmodel_state = torch.load(os.path.join(folder, exp_, 'models/G_epoch{}.pth'.format(iteration)))
        Dmodel_state = torch.load(os.path.join(folder, exp_, 'models/D_epoch{}.pth'.format(iteration)))
        netG.load_state_dict(Gmodel_state)
        netD.load_state_dict(Dmodel_state)
        netG.zero_grad()
        x_hat = netG(z)
        y_hat = netD(x_hat)
        # y = netD(x_true)
        # g_loss = get_gloss(loss_type, y_hat)
        g_loss = y_hat.mean()
        g_loss.backward(retain_graph=True)
        gradients = getGradNorm(netG)
        gradient_list.append(gradients.reshape(1, -1))
        print torch.cat(gradient_list, dim=0).shape
        np.save(save_folder+'/Ggrad_jacob_mean.npy'.format(), torch.cat(gradient_list, dim=0))




