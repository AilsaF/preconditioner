import torch
import torch.nn.functional as F
import numpy as np
import argparse
from torchvision.utils import save_image
import hignorm_networks
import othernorm_networks
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
    dir_name = dir_name + 'ie510result/{}_struct{}_GfeatureNum{}_DfeatureNum{}_loss{}_deep7/'.format(args.dataset,
                                 args.structure, args.Gnum_features, args.Dnum_features, args.losstype)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    dir_name = dir_name + '{}_{}_size{}_dlr{}_glr{}_diter{}_giter{}_b1{}_b2{}_Gnumfea{}_Dnumfea{}_batchsize{}_useadappolar{}iter{}'.format(args.gantype,
                            args.dataset.upper(), args.image_size, args.d_lr, args.g_lr, args.d_freq, args.g_freq,
                            args.beta1, args.beta2, args.Gnum_features, args.Dnum_features, args.batch_size,
                            args.apc, args.pclevel)

    dir_name += (args.specname + '/')

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    save_path = dir_name + 'models/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return dir_name, save_path


def get_gloss(dis_fake, dis_real):
    if args.gantype != 'rsgan':
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
    if args.gantype != 'rsgan':
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
    if args.gantype == 'vanilla' or args.gantype == 'rsgan':
        netG, netD = hignorm_networks.getGD(args.structure, args.dataset, args.Gnum_features, args.Dnum_features,
                                            args.image_size, use_adaptivePC=args.apc, dim_z=args.input_dim,
                                            pclevel=args.pclevel, diter=args.d_freq)
        ema_netG_9999 = copy.deepcopy(netG)
    elif args.gantype in ['weightnorm', 'batchnorm', 'orthoreg']:
        netG, netD = othernorm_networks.getGD(args.structure, args.dataset, args.Gnum_features, args.Dnum_features,
                                              args.image_size, norm=args.gantype)
        ema_netG_9999 = copy.deepcopy(netG)
    else:
        netG, netD, ema_netG_9999 = None, None, None
        print ("GAN type is not supported")

    netG.cuda()
    netD.cuda()

    # netG = torch.nn.DataParallel(netG, device_ids=[0])
    # netD = torch.nn.DataParallel(netD, device_ids=[1])
    # netG.to(torch.device("cuda:0"))
    # netD.to(torch.device("cuda:1"))

    if args.reload > 0:
        netG.load_state_dict(torch.load(save_path + 'G_epoch{}.pth'.format(args.reload)))
        netD.load_state_dict(torch.load(save_path + 'D_epoch{}.pth'.format(args.reload)))
        ema_netG_9999.load_state_dict(
            torch.load(save_path + 'emaG0.9999_epoch{}.pth'.format(args.reload), map_location=torch.device('cpu')))

    g_optimizer = torch.optim.Adam(netG.parameters(), lr=args.g_lr, betas=(args.beta1, args.beta2))
    d_optimizer = torch.optim.Adam(netD.parameters(), lr=args.d_lr, betas=(args.beta1, args.beta2))

    g_losses, d_losses = [], []
    grad_normD, grad_normG = [], []

    loader = datasets.getDataLoader(args.dataset, args.image_size, batch_size=args.batch_size, shuffle=not args.fixz,
                                    balancedbatch=args.balancedbatch)
    data_iter = iter(loader)

    for i in range(1, args.num_iters+1):
        if i >= args.lr_decay_start:
            utils.decay_lr(g_optimizer, args.num_iters, args.lr_decay_start, args.g_lr)
            utils.decay_lr(d_optimizer, args.num_iters, args.lr_decay_start, args.d_lr)
        if i <= args.reload:
            continue

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
            del x_hat
            del z
            torch.cuda.empty_cache()

            if args.gantype != 'rsgan':
                g_loss = get_gloss(y_hat, None)
            else:
                g_loss = get_gloss(y_hat, y)
            g_losses.append(g_loss.item())
            g_loss.backward()
            g_optimizer.step()
            # grad_normG.append(utils.getGradNorm(netG))
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
            del x_hat
            del z
            torch.cuda.empty_cache()
            d_loss = get_dloss(y_hat, y)
            if args.gantype == 'orthoreg':
                d_loss += othernorm_networks.orthogonal_regularization(netD)
            d_losses.append(d_loss.item())

            d_loss.backward()
            d_optimizer.step()
            # grad_normD.append(utils.getGradNorm(netD))

        if i % args.print_freq == 0:
            print('Iteration: {}; G-Loss: {}; D-Loss: {};'.format(i, g_loss, d_loss))
            #save_image((x / 2. + 0.5)[:36], os.path.join(dir_name, 'real.pdf'))

        if i > 0 and i % args.save_freq == 0:
            torch.save(netG.state_dict(), save_path + 'G_epoch{}.pth'.format(i))
            torch.save(netD.state_dict(), save_path + 'D_epoch{}.pth'.format(i))
            torch.save(ema_netG_9999.state_dict(), save_path + 'emaG0.9999_epoch{}.pth'.format(i))

        if i % args.plot_freq == 0:
            plot_x = netG(torch.randn(36, args.input_dim, device=device)).data
            plot_x = plot_x / 2. + 0.5
            save_image(plot_x, os.path.join(dir_name, 'fake_images-{}.pdf'.format(i + 1)))
            utils.plot_losses(g_losses, d_losses, grad_normG, grad_normD, dir_name)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--structure', type=str, default='dcgan', choices=['resnet', 'dcgan'])
    parser.add_argument('--losstype', type=str, default='log', choices=['log', 'hinge'])
    # parser.add_argument('--norm', type=str, default='sn', choices=['sn', 'bn', 'wn'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--input_dim', type=int, default=128)
    parser.add_argument('--num_iters', type=int, default=100000)
    parser.add_argument('--fixz', action='store_true')
    parser.add_argument('--Gnum_features', type=int, default=64)
    parser.add_argument('--Dnum_features', type=int, default=64)
    parser.add_argument('--apc', action='store_true')
    parser.add_argument('--pclevel', type=int, default=5)
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
    parser.add_argument('--balancedbatch', action='store_true')
    parser.add_argument('--reload', type=int, default=0)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda')

    train()




# cifar
# CUDA_VISIBLE_DEVICES=1 python GAN.py --dataset cifar --batch_size 64 --pclevel 0 --save_freq 1000 --g_lr 0.0001 --d_lr 0.0001 --beta1 0.5 --beta2 0.999 --d_freq 1 --specname settingB & 
# export MKL_NUM_THREADS=4 && CUDA_VISIBLE_DEVICES=2 python GAN.py --dataset cifar --batch_size 64 --structure resnet --Gnum_features 512 --Dnum_features 256 --num_iters 100000  --losstype hinge --pclevel 0 --save_freq 2000 --specname deepblock_new --d_freq 5 --apc &

# export MKL_NUM_THREADS=4 && CUDA_VISIBLE_DEVICES=3 python GAN.py --dataset cifar --batch_size 64 --beta1 0. --beta2 0.9 --structure resnet --Gnum_features 256 --Dnum_features 256 --d_freq 5 --losstype hinge --pclevel 0 --apc --save_freq 1000 --seed 2 --specname adaptivefitreluV2+strategy7+unifD+seed2 &


# stl
# export MKL_NUM_THREADS=4 && CUDA_VISIBLE_DEVICES=4 python GAN.py --dataset stl --image_size 48  --batch_size 64 --pclevel 0 --apc --save_freq 1000 --beta1 0.9 --beta2 0.999 &

# export MKL_NUM_THREADS=4 && CUDA_VISIBLE_DEVICES=1 python GAN.py --dataset stl --image_size 48 --batch_size 64 --structure resnet --losstype hinge --pclevel 1 --save_freq 2000 --num_iters 200000 --lr_decay_start 150000 --specname fitreluV2+PN1+unifD &

# export MKL_NUM_THREADS=4 && CUDA_VISIBLE_DEVICES=0 python GAN.py --dataset stl --image_size 48 --batch_size 64 --beta1 0. --beta2 0.9 --structure resnet --d_freq 5 --losstype hinge --pclevel 0 --apc --save_freq 1000 --seed 2 --specname adaptivefitreluV2+strategy7+unifD+seed2  &

# 256 celeba
# CUDA_VISIBLE_DEVICES=2 python GAN.py --dataset celeba --image_size 256 --beta1 0.5 --beta2 0.999 --batch_size 64 --structure dcgan --Gnum_features 512 --Dnum_features 512 --g_freq 1 --g_lr 5e-4 --d_lr 2e-4 --lr_decay_start 0 --losstype log --pclevel 1 --save_freq 4000 --num_iters 100000 --gantype rsgan --specname orthoD &
# export MKL_NUM_THREADS=8 &&  CUDA_VISIBLE_DEVICES=3 python GAN.py --dataset celeba --image_size 512 --beta1 0.5 --beta2 0.999 --batch_size 64 --d_freq 1 --structure dcgan --Gnum_features 1024 --Dnum_features 512 --pclevel 0 --save_freq 4000 --num_iters 200000 &


# 256 lsun

# export MKL_NUM_THREADS=8 &&  CUDA_VISIBLE_DEVICES=1  python GAN.py --dataset lsun --image_size 128 --beta1 0.5 --beta2 0.999 --batch_size 64 --structure dcgan --Gnum_features 512 --Dnum_features 512 --pclevel 1 --save_freq 4000 --num_iters 200000 --specname PN1+orthoD  &


# dummy
# export MKL_NUM_THREADS=8 &&  CUDA_VISIBLE_DEVICES=4  python GAN.py --dataset lsun --image_size 256 --beta1 0.5 --beta2 0.999 --batch_size 64 --structure dcgan --Gnum_features 32 --Dnum_features 32 --pclevel 0 --apc --save_freq 4000 --num_iters 200000 --specname adaptivefitreluV2+strategy6+orthoD+newstruct  &