import numpy as np
import torch
import hignorm_networks
import SVD_networks
import os
import shutil
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/illini/rsgan')
import datasets

def getG_condnum(netG):
    z_ = torch.randn((64, 128))
    pertubation_del=(torch.randn(z_.shape))
    eps=1
    pertu_length=torch.norm(pertubation_del, dim=1, keepdim=True)
    pertubation_del = (pertubation_del / pertu_length) * eps
    z_prime = z_ + pertubation_del
    pertube_images=netG(z_)-netG(z_prime)
    pertube_latent_var = z_ - z_prime
    Q = torch.norm(pertube_images.view(64, -1), dim=1) / torch.norm(pertube_latent_var.view(64, -1), dim=1)

    cond_num = Q.max()**2 / Q.min()**2
    log_cond_num = torch.log(cond_num).item()
    return log_cond_num

def getD_condnum(netD, x_):
    pertubation_del=(torch.randn(x_.shape))
    eps=1
    pertu_length=torch.norm(pertubation_del, dim=1, keepdim=True)
    pertubation_del = (pertubation_del / pertu_length) * eps
    x_prime = x_ + pertubation_del
    pertube_images=netD(x_)-netD(x_prime)
    pertube_latent_var = x_ - x_prime
    Q = torch.norm(pertube_images.view(64, -1), dim=1) / torch.norm(pertube_latent_var.view(64, -1), dim=1)

    cond_num = Q.max()**2 / Q.min()**2
    log_cond_num = torch.log(cond_num).item()
    return log_cond_num



device = torch.device('cpu')

folder = 'ie510result/cifar_structdcgan_GfeatureNum64_DfeatureNum64_losslog'
exp_ = 'SVDGAN_CIFAR_size32_dlr0.0002_glr0.0002_diter1_giter1_b10.5_b20.999_Gnumfea64_Dnumfea64_batchsize64_ematrickpushsptodeisirecurve0.01'

structure= 'resnet' if 'resnet' in folder else 'dcgan'
# dataset = re.search('_(.*)_size', exp_).group(1).lower()
dataset = exp_.split('_')[1].lower()
img_sizes = {'mnist':32, 'cifar':32, 'stl':48, 'celeba':256, 'lsun':256}
G_feature = int(re.search('GfeatureNum(.*)_DfeatureNum', folder).group(1))
D_feature = int(re.search('DfeatureNum(.*)_loss', folder).group(1))

#usepolar = re.search('usepolar(.*)iter', exp_).group(1) == 'True'
#if usepolar:
#    polar_iter = int(re.search('usepolarTrueiter(.*)_', exp_).group(1))
#else:
#    polar_iter = 1

save_folder = os.path.join('conditionNum', structure, exp_)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)


#netG, netD = hignorm_networks.getGD(structure, dataset, G_feature, D_feature, img_sizes[dataset], usepolar=usepolar, polar_iter=polar_iter)
netG, netD = SVD_networks.getGD(structure, dataset, G_feature, D_feature, img_sizes[dataset])
loader = datasets.getDataLoader(dataset, img_sizes[dataset], batch_size=64)
data_iter = iter(loader)

gap = 4000
iters = range(gap, 100001, gap)
# iters = range(4000, 200001, 8000)

g_condnums = []
dx_condnums = []
dxh_condnums = []

for iter_num in iters:
    print iter_num
    path = os.path.join(folder, exp_, 'models/G_epoch{}.pth'.format(iter_num))
    netG.load_state_dict(torch.load(path, map_location=device))
    log_cond_num = getG_condnum(netG)
    g_condnums.append(log_cond_num)

    path = os.path.join(folder, exp_, 'models/D_epoch{}.pth'.format(iter_num))
    netD.load_state_dict(torch.load(path, map_location=device))
    z = torch.randn(64, 128)
    x = netG(z)
    log_cond_num = getD_condnum(netD, x)
    dxh_condnums.append(log_cond_num)

    x = next(data_iter)[0]
    log_cond_num = getD_condnum(netD, x)
    dx_condnums.append(log_cond_num)

plt.figure()
plt.plot(iters, g_condnums, label='G log condition number')
plt.legend()
plt.savefig(os.path.join(save_folder, 'G_condnum.pdf'))

plt.figure()
plt.plot(iters, dx_condnums, label='D(true x) log condition number')
plt.legend()
plt.savefig(os.path.join(save_folder, 'Dx_condnum.pdf'))

plt.figure()
plt.plot(iters, dxh_condnums, label='D(fake x) log condition number')
plt.legend()
plt.savefig(os.path.join(save_folder, 'Dhx_condnum.pdf'))


