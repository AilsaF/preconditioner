import numpy as np
import torch
import hignorm_networks
import os
import shutil
import re
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


device = torch.device('cpu')

folder = 'ie510result/cifar_structdcgan_GfeatureNum64_DfeatureNum64_losslog'
exp_ = 'vanilla_CIFAR_size32_dlr0.0002_glr0.0002_diter1_giter1_b10.5_b20.999_Gnumfea64_Dnumfea64_batchsize64_sniter1_usepolarTrueiter1_ematrickparapanelty/'

structure = 'resnet' if 'resnet' in folder else 'dcgan'
# dataset = re.search('_(.*)_size', exp_).group(1).lower()
dataset = exp_.split('_')[1].lower()
img_sizes = {'mnist':32, 'cifar':32, 'stl':48, 'celeba':256, 'lsun':256}
G_feature = int(re.search('GfeatureNum(.*)_DfeatureNum', folder).group(1))
D_feature = int(re.search('DfeatureNum(.*)_loss', folder).group(1))

usepolar = re.search('usepolar(.*)iter', exp_).group(1) == 'True'
if usepolar:
    polar_iter = int(re.search('usepolarTrueiter(.*)_', exp_).group(1))
else:
    polar_iter = 1

save_folder = os.path.join('spectrum_plots', structure, exp_)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)


netG, netD = hignorm_networks.getGD(structure, dataset, G_feature, D_feature, img_sizes[dataset], usepolar=usepolar, polar_iter=polar_iter)

iters = [2000, 10000, 50000, 60000, 70000, 80000, 90000, 100000]
#iters = [4000, 12000, 52000, 100000]

img = torch.randn(10, 3, img_sizes[dataset], img_sizes[dataset])
for iter_num in iters:
    print iter_num
    # save weights
    path = os.path.join(folder, exp_, 'models/D_epoch{}.pth'.format(iter_num))
    netD.load_state_dict(torch.load(path, map_location=device))
    y = netD(img)

    # plot spec
    count = len(glob.glob1('save_weights', "weights*.npy"))
    plt.figure()
    for j in range(count):
        w = np.load('save_weights/weights{}.npy'.format(j))
        w = w.reshape(w.shape[0], -1)
        print w.shape
        _, vals, _ = np.linalg.svd(w)
        plt.plot(np.linspace(0, 1, num=len(vals)), vals[::-1], label='layer {}'.format(j))
        os.remove('save_weights/weights{}.npy'.format(j))
    plt.legend()
    plt.savefig(os.path.join(save_folder, 'spectrum_iter{}.pdf'.format(iter_num)))
    plt.clf()






