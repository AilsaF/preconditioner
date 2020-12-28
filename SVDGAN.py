import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad
import numpy as np
import argparse
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader, TensorDataset
import SVD_networks
import copy
import os
directory_path = os.path.abspath(os.getcwd())
import sys
if 'tf6' in directory_path:
    sys.path.append('/home/tf6/slicedGAN')
elif 'illini' in directory_path:
    sys.path.append('/home/illini/rsgan')
else:
    raise ValueError("invalid server")
import datasets
import utils
import moving_average



def getSavePath():
    if 'tf6' in directory_path:
        dir_name = '/data01/tf6/'
    elif 'illini' in directory_path:
        dir_name = '/home/illini/pngan/'
    dir_name += 'ie510result/{}_struct{}_GfeatureNum{}_DfeatureNum{}_loss{}/'.format(args.dataset,
                                 args.structure, args.Gnum_features, args.Dnum_features, args.losstype)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    dir_name = dir_name + 'SVDGAN_{}_size{}_dlr{}_glr{}_diter{}_giter{}_b1{}_b2{}_Gnumfea{}_Dnumfea{}_batchsize{}'.format(
                            args.dataset.upper(), args.image_size, args.d_lr, args.g_lr, args.d_freq, args.g_freq,
                            args.beta1, args.beta2, args.Gnum_features, args.Dnum_features, args.batch_size)

    dir_name += '_ematrick' if args.ema_trick else ''
    dir_name += (args.specname + '/')

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    save_path = dir_name + 'models/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return dir_name, save_path


def get_gloss(dis_fake, dis_real):
    if args.gantype == 'vanilla':
        if args.losstype == 'log':
            return torch.mean(F.softplus(-dis_fake))
        elif args.losstype == 'hinge':
            return -torch.mean(dis_fake)
    else:
        scalar = torch.FloatTensor([0]).cuda()
        z = dis_fake - dis_real
        z_star = torch.max(z, scalar.expand_as(z))
        return (z_star + torch.log(torch.exp(z - z_star) + torch.exp(0 - z_star))).mean()


def get_dloss(dis_fake, dis_real):
    if args.gantype == 'vanilla':
        if args.losstype == 'log':
            return torch.mean(F.softplus(-dis_real)) + torch.mean(F.softplus(dis_fake))
        elif args.losstype == 'hinge':
            return torch.mean(torch.relu(1. - dis_real)) + torch.mean(torch.relu(1. + dis_fake))
    else:
        scalar = torch.FloatTensor([0]).cuda()
        z = dis_real - dis_fake
        z_star = torch.max(z, scalar.expand_as(z))
        return (z_star + torch.log(torch.exp(z - z_star) + torch.exp(0 - z_star))).mean()


def train():
    dir_name, save_path = getSavePath()
    netG, netD = SVD_networks.getGD(args.structure, args.dataset, args.Gnum_features, args.Dnum_features, args.image_size)
    if args.ema_trick:
        ema_netG_9999 = copy.deepcopy(netG)
        
    netG.cuda()
    netD.cuda()

    # netG = torch.nn.DataParallel(netG, device_ids=[0])
    # netD = torch.nn.DataParallel(netD, device_ids=[1])
    # netG.to(torch.device("cuda:0"))
    # netD.to(torch.device("cuda:1"))
    
    g_optimizer = torch.optim.Adam(netG.parameters(), lr=args.g_lr, betas=(args.beta1, args.beta2))
    d_optimizer = torch.optim.Adam(netD.parameters(), lr=args.d_lr, betas=(args.beta1, args.beta2))

    g_losses, d_losses = [], []
    grad_normD, grad_normG = [], []

    loader = datasets.getDataLoader(args.dataset, args.image_size, batch_size=args.batch_size, shuffle=not args.fixz, balancedbatch=args.balancedbatch)
    data_iter = iter(loader)

    for i in range(1, args.num_iters+1):
        if i >= args.lr_decay_start:
            utils.decay_lr(g_optimizer, args.num_iters, args.lr_decay_start, args.g_lr)
            utils.decay_lr(d_optimizer, args.num_iters, args.lr_decay_start, args.d_lr)

        # G-step
        for _ in range(args.g_freq):
            if args.gantype == 'rsgan':
                try:
                    x = next(data_iter)[0].cuda()
                except StopIteration:
                    data_iter = iter(loader)
                    x = next(data_iter)[0].cuda()
                y = netD(x)

            g_optimizer.zero_grad()
            z = torch.randn(args.batch_size, args.input_dim, device=device)
            x_hat = netG(z)
            y_hat = netD(x_hat)

            if args.gantype == 'vanilla':
                g_loss = get_gloss(y_hat, None)
            else:
                g_loss = get_gloss(y_hat, y)
            g_losses.append(g_loss.item())
            g_loss.backward()
            g_optimizer.step()
            # grad_normG.append(utils.getGradNorm(netG))

            if args.ema_trick:
                moving_average.soft_copy_param(ema_netG_9999, netG, 0.9999)

        for _ in range(args.d_freq):
            try:
                x = next(data_iter)[0].cuda()
            except StopIteration:
                data_iter = iter(loader)
                x = next(data_iter)[0].cuda()

            d_optimizer.zero_grad()
            z = torch.randn(args.batch_size, args.input_dim, device=device)
            x_hat = netG(z).detach()
            y_hat = netD(x_hat)
            y = netD(x)
            d_loss = get_dloss(y_hat, y) + netD.loss_orth()*10
            d_losses.append(d_loss.item())

            d_loss.backward()
            d_optimizer.step()
            # grad_normD.append(utils.getGradNorm(netD))

        if i % args.print_freq == 0:
            print('Iteration: {}; G-Loss: {}; D-Loss: {};'.format(i, g_loss, d_loss))

        # if i == 1:
        #     save_image((x / 2. + 0.5)[:36], os.path.join(dir_name, 'real.pdf'))

        if i > 0 and i % args.save_freq == 0:
            torch.save(netG.state_dict(), save_path + 'G_epoch{}.pth'.format(i))
            torch.save(netD.state_dict(), save_path + 'D_epoch{}.pth'.format(i))
            if args.ema_trick:
                torch.save(ema_netG_9999.state_dict(), save_path + 'emaG0.9999_epoch{}.pth'.format(i))

        if i % args.plot_freq == 0:
            plot_x = netG(torch.randn(36, args.input_dim, device=device)).data
            plot_x = plot_x / 2. + 0.5
            save_image(plot_x, os.path.join(dir_name, 'fake_images-{}.pdf'.format(i + 1)))
            utils.plot_losses(g_losses, d_losses, grad_normG, grad_normD, dir_name)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', choices=['toy', 'mnist', 'cifar', 'stl', 'celeba', 'lsun'])
    parser.add_argument('--structure', type=str, default='dcgan', choices=['resnet', 'dcgan'])
    parser.add_argument('--losstype', type=str, default='log', choices=['log', 'hinge'])
    # parser.add_argument('--norm', type=str, default='bn', choices=['sn', 'bn'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--input_dim', type=int, default=128)
    parser.add_argument('--num_iters', type=int, default=100000)
    parser.add_argument('--fixz', action='store_true')
    parser.add_argument('--Gnum_features', type=int, default=64)
    parser.add_argument('--Dnum_features', type=int, default=64)
    parser.add_argument('--power_iter', type=int, default=1)
    parser.add_argument('--usepolar', action='store_true')
    parser.add_argument('--polar_iter', type=int, default=5)
    parser.add_argument('--lipconst', type=float, default=1)
    parser.add_argument('--gantype', type=str, default='vanilla')

    parser.add_argument('--g_lr', type=float, default=2e-4)
    parser.add_argument('--d_lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--g_freq', type=int, default=1)
    parser.add_argument('--d_freq', type=int, default=1)
    parser.add_argument('--lr_decay_start', type=int, default=50000)

    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--plot_freq', type=int, default=500)
    parser.add_argument('--save_freq', type=int, default=500)
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--specname', type=str, default='') # give a specific name for the folder
    parser.add_argument('--ema_trick', action='store_true')
    parser.add_argument('--balancedbatch', action='store_true')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda')

    train()




# mnist
# CUDA_VISIBLE_DEVICES=4 python SVDGAN.py --dataset mnist --beta1 0.5 --beta2 0.999 --batch_size 64 --g_lr 2e-4 --d_lr 2e-4 --d_freq 1 --g_freq 1 --lr_decay_start 0 --losstype log --num_iters 30000  --usepolar --specname deltaorho+maxmin &


# cifar
# CUDA_VISIBLE_DEVICES=0 python SVDGAN.py --dataset cifar --g_lr 1e-3 --d_lr 1e-3 --d_freq 5 --g_freq 1 --beta1 0.9 --beta2 0.999 --save_freq 2000 --specname settingF  &
# export MKL_NUM_THREADS=4 && CUDA_VISIBLE_DEVICES=2 python SVDGAN.py --dataset cifar --beta1 0.5 --beta2 0.999 --batch_size 64 --structure resnet --Gnum_features 256 --Dnum_features 256 --g_lr 2e-4 --d_lr 2e-4 --d_freq 1 --g_freq 1 --losstype hinge --save_freq 2000 --num_iters 200000 --lr_decay_start 100000 --ema_trick &

# CUDA_VISIBLE_DEVICES=3 python SVDGAN.py --dataset cifar --beta1 0. --beta2 0.9 --batch_size 64 --structure resnet --Gnum_features 256 --Dnum_features 256 --g_lr 2e-4 --d_lr 2e-4 --d_freq 5 --g_freq 1 --lr_decay_start 0 --losstype hinge --save_freq 2000 --ema_trick &


# stl
# CUDA_VISIBLE_DEVICES=0 python SVDGAN.py --dataset stl --image_size 48 --beta1 0.5 --beta2 0.999 --batch_size 64 --g_lr 2e-4 --d_lr 2e-4 --d_freq 1 --g_freq 1 --lr_decay_start 0 --losstype log  &
# export MKL_NUM_THREADS=4 && CUDA_VISIBLE_DEVICES=4 python SVDGAN.py --dataset stl --image_size 48 --beta1 0.5 --beta2 0.999 --batch_size 64 --structure resnet --Gnum_features 64 --Dnum_features 64 --g_lr 2e-4 --d_lr 2e-4 --d_freq 1 --g_freq 1 --losstype hinge --save_freq 2000 --num_iters 200000 &


# 256 celeba
# export MKL_NUM_THREADS=8 && CUDA_VISIBLE_DEVICES=1 python SVDGAN.py --dataset celeba --image_size 512 --beta1 0.5 --beta2 0.999 --batch_size 64 --structure dcgan --Gnum_features 1024 --Dnum_features 1024 --g_lr 2e-4 --d_lr 2e-4 --losstype log  --save_freq 4000 --ema_trick --num_iters 200000 &


# 256 lsun
# export MKL_NUM_THREADS=8 && CUDA_VISIBLE_DEVICES=4  python SVDGAN.py --dataset living_room --image_size 256 --beta1 0.5 --beta2 0.999 --batch_size 64 --structure dcgan --Gnum_features 1024 --Dnum_features 1024 --g_lr 2e-4 --d_lr 2e-4  --losstype log --save_freq 4000 --ema_trick --num_iters 200000 --gantype vanilla  &



# dummy
# CUDA_VISIBLE_DEVICES=0 python SVDGAN.py --dataset lsun --image_size 256 --beta1 0.5 --beta2 0.999 --batch_size 64 --structure dcgan --Gnum_features 1600 --Dnum_features 800 --g_lr 2e-4 --d_lr 2e-4 --lr_decay_start 0 --losstype log --polar_iter 3 --usepolar --save_freq 40000 --ema_trick --num_iters 200000 --gantype vanilla --specname xarunifD &