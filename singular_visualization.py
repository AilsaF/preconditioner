import torch
from torch.nn.functional import normalize
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

folder = '/data01/tf6/ie510result/cifar_structresnet_GfeatureNum512_DfeatureNum256_losshinge_deep7/'
exp_ = 'vanilla_CIFAR_size32_dlr0.0002_glr0.0002_diter5_giter1_b10.5_b20.999_Gnumfea512_Dnumfea256_batchsize64_useadappolarFalseiter0deepblock/'
eps = 1e-12
end = 100000
gap = 10000
iters = range(gap, end+1, gap)

model = 'D'
weight_keyword = 'weight_orig' if model=='D' else 'weight'


if 'TOY' in exp_:
    dataset = 'TOY'
elif 'CIFAR' in exp_:
    dataset = 'CIFAR'
elif 'STL' in exp_:
    dataset = 'STL'
elif 'LSUN' in exp_:
    dataset = 'LSUN'
elif 'CELEBA' in exp_:
    dataset = 'celeba'

structure = 'dcgan' if 'dcgan' in folder else 'resnet'
save_folder = os.path.join('Ten_Singular_dist', structure, dataset, exp_)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)


def plot_cn_v1():
    sing_num = 5
    order = ['1st', '2nd', '3rd', '4th', '5th']
    usepolar = 'useadappolarTrue' in exp_
    # usepolar = True
    if structure == 'dcgan':
        layer_num = 7
    elif dataset == 'CIFAR':
        layer_num = 35
    elif dataset == 'STL':
        layer_num = 15
    elif dataset == 'celeba':
        layer_num = 7

    singular_dict = {}
    condi_num0 = torch.empty(layer_num, len(iters))

    for iter in range(len(iters)):
        path = os.path.join(folder, exp_, 'models/{}_epoch{}.pth'.format(model, gap * (iter + 1)))
        # print iter
        m = torch.load(path, map_location=torch.device('cpu'))
        j = 0
        for k in m.keys():
            if weight_keyword in k and 'proj' not in k:
                if k not in singular_dict:
                    singular_dict[k] = torch.zeros(sing_num * 2, len(iters))
                weight = m[k]
                weight_mat = weight.view(weight.shape[0], -1)
                u = m[k[:-4] + 'u']
                v = m[k[:-4] + 'v']
                sigma = torch.dot(u, torch.mv(weight_mat, v))
                weight_mat /= sigma

                if usepolar:
                    piter = m[k[:-4] + 'pcleval']
                    # piter = 0
                    weight_mat = preconditioner(weight_mat, piter)
                _, S, _ = torch.svd(weight_mat)
                singular_dict[k][:min(sing_num, S.shape[0]), iter] = S[:sing_num]
                # if S.shape[0] > 5:
                singular_dict[k][-min(sing_num, S.shape[0]):, iter] = S[-sing_num:]
                length1 = max(1, int(S.shape[0] * 0.1))
                condi_num0[j, iter] = S[0] / S[-length1:].mean()
                j += 1

    if sing_num>1:
        fig, axes = plt.subplots(1, sing_num, sharey=True, figsize=(15,4))
        for i, ax in enumerate(axes.flatten()):
            for l, k in enumerate(sorted(singular_dict.keys())):
                ax.plot(singular_dict[k][i], label="layer {}".format(l+1))
            ax.title.set_text('{} largest singular value'.format(order[i]))
            ax.set_xticks(range(len(iters)))
            ax.set_xticklabels(range(2, 21, 2))
            ax.set_xlabel('x10000 iteration', fontsize=15)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.ylim(ymin=0)
        # plt.setp(axes[0], ylabel="Singular Value")
        fig.text(0.09, 0.5, 'Singular Value', va='center', rotation='vertical', fontsize=15)
        plt.rcParams['xtick.labelsize'] = 15
        plt.rcParams['ytick.labelsize'] = 15
        plt.savefig(save_folder+'/{} largest singulars.pdf'.format(model), bbox_inches='tight')

        fig, axes = plt.subplots(1, sing_num, sharey=True, figsize=(15,4))
        for i, ax in enumerate(axes.flatten()):
            for l, k in enumerate(sorted(singular_dict.keys())):
                if not singular_dict[k][-i-1].sum() == 0:
                    ax.plot(singular_dict[k][-i-1], label="layer {}".format(l+1))
            ax.title.set_text('{} smallest singular value'.format(order[i]))
            ax.set_xticks(range(len(iters)))
            ax.set_xticklabels(range(2, 21, 2))
            ax.set_xlabel('x10000 iteration', fontsize=15)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.ylim(ymin=0)
        # plt.setp(axes[0], ylabel="Singular Value")
        fig.text(0.09, 0.5, 'Singular Value', va='center', rotation='vertical', fontsize=15)
        plt.rcParams['xtick.labelsize'] = 15
        plt.rcParams['ytick.labelsize'] = 15
        plt.savefig(save_folder+'/{} smallest singulars.pdf'.format(model), bbox_inches='tight')

    else:
        fig = plt.figure()
        for k in sorted(singular_dict.keys()):
            plt.plot(singular_dict[k][0], label=k)
        plt.title('singular value')
        plt.legend(loc='best')
        plt.savefig(save_folder + '/{} singulars.pdf'.format(model))

    fig = plt.figure()
    for i in range(layer_num):
        plt.plot(condi_num0[i], label='layer {}'.format(i+1))
    plt.legend()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(range(len(iters)), range(1, 11, 1), fontsize=18)
    plt.xlabel("x10000 iteration", fontsize=18)
    plt.ylabel("Condition Number", fontsize=18)
    plt.savefig(save_folder+'/{} condition number mode {}.pdf'.format(model, 0), bbox_inches='tight')



def plot_cn_v2():
    if structure == 'dcgan':
        layer_num = 6
    elif dataset == 'CIFAR':
        layer_num = 10
    elif dataset == 'STL':
        layer_num = 15
    elif dataset == 'celeba':
        layer_num = 7
    piters = torch.empty(layer_num, len(iters))
    condi_num = torch.empty(layer_num, len(iters))

    for iter in range(len(iters)):
        path = os.path.join(folder, exp_, 'models/{}_epoch{}.pth'.format(model, gap * (iter + 1)))
        # print iter
        j = 0
        m = torch.load(path, map_location=torch.device('cpu'))
        for key in m.keys():
            if '_cns' in key and 'proj' not in key:
                cn = m[key][-1]
                piter = m[key[:-3] + 'pcleval']
                piters[j, iter] = piter
                smallest_singular = 1 / cn
                for _ in range(piter):
                    smallest_singular = 1.5 * smallest_singular - 0.5 * smallest_singular ** 3
                condi_num[j, iter] = 1 / smallest_singular
                j += 1

    fig = plt.figure()
    for i in range(layer_num):
        plt.plot(condi_num[i], label='layer {}'.format(i + 1))
    plt.legend()
    plt.xticks(range(len(iters)), iters)
    plt.xlabel("Iteration")
    plt.ylabel("Condition Number")
    plt.savefig(save_folder + '/{} condition number.pdf'.format(model))

    fig = plt.figure()
    for i in range(layer_num):
        plt.plot(piters[i], label='layer {}'.format(i + 1))
    plt.legend()
    plt.xticks(range(len(iters)), iters)
    plt.xlabel("Iteration")
    plt.ylabel("Preconditioner Iteration")
    plt.savefig(save_folder + '/{} precond iteration.pdf'.format(model))


def plot_cn_v3():
    if structure == 'dcgan':
        layer_num = 7
    elif dataset == 'CIFAR':
        layer_num = 10
    elif dataset == 'STL':
        layer_num = 15

    sing_num = 10
    singular_dict = {}
    condi_num = torch.empty(layer_num, len(iters))
    piters = torch.empty(layer_num, len(iters))

    for iter in range(len(iters)):
        path = os.path.join(folder, exp_, 'models/{}_epoch{}.pth'.format(model, gap * (iter + 1)))
        # print iter
        m = torch.load(path)
        j = 0
        for k in m.keys():
            if weight_keyword in k and 'proj' not in k:
                if k not in singular_dict:
                    singular_dict[k] = torch.zeros(sing_num * 2, len(iters))
                weight = m[k]
                weight_mat = weight.view(weight.shape[0], -1)
                u = m[k[:-4] + 'u']
                v = m[k[:-4] + 'v']
                sigma = torch.dot(u, torch.mv(weight_mat, v))
                weight_mat /= sigma

                if m[k[:-4] + 'cns'].shape[0]:
                    piter = m[k[:-4] + 'piter']
                else:
                    piter = 1

                weight_mat = preconditioner(weight_mat, piter)
                _, S, _ = torch.svd(weight_mat)
                singular_dict[k][:min(sing_num, S.shape[0]), iter] = S[:sing_num]
                singular_dict[k][-min(sing_num, S.shape[0]):, iter] = S[-sing_num:]
                condi_num[j, iter] = S[0] / S[-5:].mean()
                piters[j, iter] = piter
                j += 1

    fig, axes = plt.subplots(1, sing_num, sharey=True, figsize=(int(1.5 * sing_num), 2))
    for i, ax in enumerate(axes.flatten()):
        for k in sorted(singular_dict.keys()):
            ax.plot(singular_dict[k][i], label=k)
        ax.title.set_text('{} largest singular value'.format(i + 1))
        ax.set_xticks(range(len(iters)))
        ax.set_xticklabels(iters)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(save_folder + '/{} largest singulars.pdf'.format(model))

    fig, axes = plt.subplots(1, sing_num, sharey=True, figsize=(int(1.5 * sing_num), 2))
    for i, ax in enumerate(axes.flatten()):
        for k in sorted(singular_dict.keys()):
            ax.plot(singular_dict[k][-i - 1], label=k)
        ax.title.set_text('{} smallest singular value'.format(i + 1))
        ax.set_xticks(range(len(iters)))
        ax.set_xticklabels(iters)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim(ymin=0)
    plt.setp(axes[0], ylabel="Singular Value")
    plt.savefig(save_folder + '/{} smallest singulars.pdf'.format(model))

    fig = plt.figure()
    for i in range(layer_num):
        plt.plot(condi_num[i], label='layer {}'.format(i + 1))
    plt.legend()
    plt.xticks(range(len(iters)), iters)
    plt.xlabel("Iteration")
    plt.ylabel("Condition Number")
    plt.savefig(save_folder + '/{} condition number.pdf'.format(model))

    fig = plt.figure()
    for i in range(layer_num):
        plt.plot(piters[i], label='layer {}'.format(i + 1))
    plt.legend()
    plt.xticks(range(len(iters)), iters)
    plt.xlabel("Iteration")
    plt.ylabel("Preconditioner Iteration")
    plt.savefig(save_folder + '/{} precond iteration.pdf'.format(model))


# def preconditioner(weight, piter):
#     if piter == 0:
#         return weight
#     elif piter == 1:
#         weight = -0.30 * weight.mm(weight.t()).mm(weight).mm(weight.t()).mm(weight) + 1.31 * weight
#     elif piter == 2:
#         weight = 0.45 * weight.mm(weight.t()).mm(weight).mm(weight.t()).mm(weight) \
#                  - 1.49 * weight.mm(weight.t()).mm(weight) + 2.03 * weight
#     elif piter == 3:
#         weight = -1.54 * weight.mm(weight.t()).mm(weight).mm(weight.t()).mm(weight).mm(weight.t()).mm(weight)\
#                  + 4.66 * weight.mm(weight.t()).mm(weight).mm(weight.t()).mm(weight) \
#                  - 5.12 * weight.mm(weight.t()).mm(weight) + 3.0 * weight
#     elif piter == 4:
#         # weight = 11.50 * weight.mm(weight.t()).mm(weight).mm(weight.t()).mm(weight).mm(weight.t()).mm(weight).mm(weight.t()).mm(weight)\
#         #          - 35.58 * weight.mm(weight.t()).mm(weight).mm(weight.t()).mm(weight).mm(weight.t()).mm(weight) \
#         #          + 39.71 * weight.mm(weight.t()).mm(weight).mm(weight.t()).mm(weight) \
#         #          - 19.64 * weight.mm(weight.t()).mm(weight) + 4.97 * weight
#         weight = 4.64 * weight.mm(weight.t()).mm(weight).mm(weight.t()).mm(weight).mm(weight.t()).mm(weight).mm(weight.t()).mm(weight) \
#                  - 15.56 * weight.mm(weight.t()).mm(weight).mm(weight.t()).mm(weight).mm(weight.t()).mm(weight) \
#                  + 19.51 * weight.mm(weight.t()).mm(weight).mm(weight.t()).mm(weight) \
#                  - 11.53 * weight.mm(weight.t()).mm(weight) + 3.93 * weight
#     else:
#         raise ValueError("No pre-conditioner provided")
#     # elif piter == 5:
#     #     weight = 2.727 * weight.mm(weight.t()).mm(weight).mm(weight.t()).mm(weight) - 5.256 * weight.mm(weight.t()).mm(weight) + 3.446 * weight
#     # elif piter > 5:
#     #     weight = 3.260 * weight.mm(weight.t()).mm(weight).mm(weight.t()).mm(weight) - 6.101 * weight.mm(weight.t()).mm(weight) + 3.740 * weight
#     return weight


def preconditioner(weight, piter):  # v2
    if piter == 0:
        return weight
    elif piter == 1:
        weight = 1.507 * weight - 0.507 * weight.mm(weight.t()).mm(weight)
    elif piter == 2:
        weight = 0.560 * weight.mm(weight.t()).mm(weight).mm(weight.t()).mm(weight) \
                 - 1.643 * weight.mm(weight.t()).mm(weight) + 2.083 * weight
    elif piter == 3:
        weight = - 1.283 * weight.mm(weight.t()).mm(weight).mm(weight.t()).mm(weight).mm(weight.t()).mm(weight) \
                 + 4.023 * weight.mm(weight.t()).mm(weight).mm(weight.t()).mm(weight) \
                 - 4.649 * weight.mm(weight.t()).mm(weight) + 2.909 * weight
    elif piter == 4:
        weight = 2.890 * weight.mm(weight.t()).mm(weight).mm(weight.t()).mm(weight).mm(weight.t()).mm(weight).mm(
            weight.t()).mm(weight) \
                 - 10.351 * weight.mm(weight.t()).mm(weight).mm(weight.t()).mm(weight).mm(weight.t()).mm(weight) \
                 + 14.097 * weight.mm(weight.t()).mm(weight).mm(weight.t()).mm(weight) \
                 - 9.261 * weight.mm(weight.t()).mm(weight) + 3.625 * weight
    return weight



plot_cn_v1()


# export MKL_NUM_THREADS=4 && python singular_visualization.py &


def fa(m):
    kappas = []
    for k in m.keys():
        if 'weight_orig' in k:
            weight = m[k]
            weight_mat = weight.view(weight.shape[0], -1)
            # print weight_mat.shape
            u = m[k[:-4]+'u']
            v = m[k[:-4] + 'v']
            sigma = torch.dot(u, torch.mv(weight_mat, v))
            weight_mat /= sigma
            piter = m[k[:-4] + 'piter']
            weight_mat = preconditioner(weight_mat, piter)
            S = torch.svd(weight_mat)[1]
            length1 = max(1, int(S.shape[0]*0.1))
            kappa = (S[0] / S[-length1:].mean()).item()
            print "{}, smallest singular {}, condition num {}".format(k, S[-1].item(), kappa)
            kappas.append(kappa)
    print "avg condition number {}".format(np.mean(kappas))


def f0(m):
    kappas = []
    for k in m.keys():
        if 'weight_orig' in k:
            weight = m[k]
            weight_mat = weight.view(weight.shape[0], -1)
            # print weight_mat.shape
            u = m[k[:-4]+'u']
            v = m[k[:-4] + 'v']
            sigma = torch.dot(u, torch.mv(weight_mat, v))
            weight_mat /= sigma
            # weight_mat = 2.05 * weight_mat - 1.25 * weight_mat.mm(weight_mat.t()).mm(weight_mat)
            for j in range(0):
                weight_mat = 1.5 * weight_mat - 0.5 * weight_mat.mm(weight_mat.t()).mm(weight_mat)
            # weight_mat = preconditioner(weight_mat, 3)
            try:
                S = torch.svd(weight_mat)[1]
                length1 = max(1, int(S.shape[0]*0.1))
                kappa = (S[0] / S[-length1:].mean()).item()
                print "{}, smallest singular {}, condition num {}".format(k, S[-1].item(), kappa)
                kappas.append(kappa)
            except:
                continue
    print "avg condition number {}".format(np.mean(kappas))



def g(m):
    for k in m.keys():
        if 'weight_orig' in k:
            weight = m[k]
            weight_mat =  weight.view(weight.shape[0], -1)
            u = m[k[:-4]+'u']
            v = m[k[:-4] + 'v']
            sigma = torch.dot(u, torch.mv(weight_mat, v))
            weight_mat /= sigma
            piter = m[k[:-4] + 'piter']
            # avg_cn = (m[k[:-4] + 'cns'][-4:]).mean()
            # if avg_cn <= 5:
            #     piter = 1
            # elif avg_cn <= 10:
            #     piter = 2
            # elif avg_cn <= 20:
            #     piter = 3
            # else:
            #     piter = 4
            print torch.svd(weight_mat)[1][-1]
            for j in range(piter):
                weight_mat = 1.5 * weight_mat - 0.5 * weight_mat.mm(weight_mat.t()).mm(weight_mat)
            S = torch.svd(weight_mat)[1]
            print piter
            print "{}, smallest singular {}, condition num {}".format(k, S[-1].item(), (S[0] / S[-1]).item())



def svdcond(m):
    kappas = []
    for k in m.keys():
        if '.U' in k and 'proj' not in k and 'fc' not in k:
            U = m[k]
            V = m[k[:-1]+'V']
            D = m[k[:-1]+'D']
            weight_mat = torch.matmul(U.t() * D, V)
            S = torch.svd(weight_mat)[1]
            length1 = max(1, int(S.shape[0] * 0.1))
            kappa = (S[0] / S[-length1:].mean()).item()
            print "{}, smallest singular {}, condition num {}".format(k, S[-1].item(), kappa)
            kappas.append(kappa)
    print "avg condition number {}".format(np.mean(kappas))


# from torch import _weight_norm, norm_except_dim
# def wncond(m):
#     kappas = []
#     for k in m.keys():
#         if 'weight_v' in k and 'proj' not in k and 'fc' not in k:
#             v = m[k]
#             g = m[k[:-1]+'g']
#             w = _weight_norm(v, g)
#             weight_mat = w.view(w.shape[0], -1)
#             try:
#                 S = torch.svd(weight_mat)[1]
#                 length1 = max(1, int(S.shape[0] * 0.1))
#                 kappa = (S[0] / S[-length1:].mean()).item()
#                 print "{}, smallest singular {}, condition num {}".format(k, S[-1].item(), kappa)
#                 kappas.append(kappa)
#             except:
#                 continue
#     print "avg condition number {}".format(np.mean(kappas))
#
# def bncond(m):
#     kappas = []
#     for k in m.keys():
#         if 'weight' in k and 'proj' not in k and 'fc' not in k:
#             w = m[k]
#             weight_mat = w.view(w.shape[0], -1)
#             try:
#                 S = torch.svd(weight_mat)[1]
#                 length1 = max(1, int(S.shape[0] * 0.1))
#                 kappa = (S[0] / S[-length1:].mean()).item()
#                 print "{}, smallest singular {}, condition num {}".format(k, S[-1].item(), kappa)
#                 kappas.append(kappa)
#             except:
#                 continue
#     print "avg condition number {}".format(np.mean(kappas))


def plot_singular():
    if structure == 'dcgan':
        layer_num = 8
    elif dataset == 'CIFAR':
        layer_num = 11
    elif dataset == 'STL':
        layer_num = 16
    true_singular = torch.empty(layer_num, len(iters))
    estimate_singular = torch.empty(layer_num, len(iters))

    for iter in range(len(iters)):
        path = os.path.join(folder, exp_, 'models/{}_epoch{}.pth'.format(model, gap * (iter + 1)))
        # print iter
        m = torch.load(path)
        j = 0
        for k in m.keys():
            if weight_keyword in k :
                weight = m[k]
                weight_mat = weight.view(weight.shape[0], -1)
                _, S, _ = torch.svd(weight_mat)
                true_singular[j, iter] = S[0]

                u = m[k[:-4] + 'u']
                v = m[k[:-4] + 'v']
                sigma = torch.dot(u, torch.mv(weight_mat, v))
                estimate_singular[j, iter] = sigma
                j+=1

    np.save(save_folder+'/estimate_singular.npy', estimate_singular)
    np.save(save_folder+'/true_singular.npy', true_singular)


# >>> for _ in range(10):
# ...     d = torch.randn(100, 1024).cuda()
# ...     print "bad"
# ...     getcond(m1, d)
# ...     print "good 1"
# ...     getcond(m2, d)
# ...     print "good 2"
# ...     get cond(m3, d)
#   File "<stdin>", line 8
#     get cond(m3, d)
#            ^
# SyntaxError: invalid syntax
# >>> for _ in range(10):
# ...     d = torch.randn(100, 1024).cuda()
# ...     print "bad"
# ...     getcond(m1, d)
# ...     print "good 1"
# ...     getcond(m2, d)
# ...     print "good 2"
# ...     getcond(m3, d)
# ...     print ">>>>>>>>>>>>>"


# m = torch.load('/data01/tf6/ie510result/stl_structresnet_GfeatureNum64_DfeatureNum64_losshinge/vanilla_STL_size48_dlr0.0002_glr0.0002_diter5_giter1_b10.0_b20.9_Gnumfea64_Dnumfea64_batchsize64_sniter1_useadappolarFalseiter1_ematrickfitrelu1+strategy4+unifD/models/D_epoch100000.pth')
# weight = m['block5.c2.weight_orig']
# w3 = weight.view(1024, -1)
# s = torch.svd(w3)[1]
# w3 /= s[0]
#
# s1 = torch.svd(w1)[1]
# s2 = torch.svd(w2)[1]
# s3 = torch.svd(w3)[1]
# sinweight = (torch.linspace(0, 1, 1024)**0.5).cuda()
# revsinweight = (torch.linspace(1, 0, 1024)**0.5).cuda()
#
# sinweight = torch.linspace(1, 1024, 1024).cuda()
# revsinweight = torch.linspace(1024, 1, 1024).cuda()
#
# cond1 = (s1 * sinweight).sum() / (s1 * revsinweight).sum()
# cond2 = (s2 * sinweight).sum() / (s2 * revsinweight).sum()
# cond3 = (s3 * sinweight).sum() / (s3 * revsinweight).sum()
# cond1
# cond2
# cond3

def goodfellowconditionnumber(G):
    batch_size = 64
    z_dim = 128
    eps = 1
    eig_min = 1
    eig_max = 20
    pertubation_del=(torch.randn(batch_size, z_dim))
    pertu_length=torch.norm(pertubation_del, dim=1, keepdim=True)
    pertubation_del = (pertubation_del / pertu_length) * eps
    z_ = torch.randn(batch_size, z_dim)
    z_prime = z_ + pertubation_del
    pertube_images = G(z_)-G(z_prime)
    pertube_latent_var = z_ - z_prime
    Q = torch.norm(pertube_images.view(batch_size, -1), dim=1) / torch.norm(pertube_latent_var.view(batch_size, -1), dim=1)
    print("Q", Q)
    L_max = 0.0
    L_min = 0.0
    count_max=0
    count_min=0
    for i in range(batch_size):
        if Q[i] > eig_max:
            L_max += (Q[i] - eig_max) ** 2
            count_max+=1
        if Q[i] < eig_min:
            L_min += (Q[i] - eig_min) ** 2
            count_min+=1
    L = L_max+L_min
    print("L", L)

# netG, _ = hignorm_networks.getGD('resnet', 'stl', 64, 64, dim_z=128, image_size=48, ignoreD=True)
# for i in range(4000, 200001, 4000):
#     path = '/data01/tf6/ie510result/stl_structresnet_GfeatureNum64_DfeatureNum64_losshinge/vanilla_STL_size48_dlr0.0002_glr0.0002_diter1_giter1_b10.5_b20.999_Gnumfea64_Dnumfea64_batchsize64_sniter1_useadappolarFalseiter0_ematrickSN+unifD/models/G_epoch{}.pth'.format(i)
#     netG.load_state_dict(torch.load(path))
#     print(i)
#     goodfellowconditionnumber(netG)
