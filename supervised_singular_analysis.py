import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

scaled_largest_singular = np.zeros((200, 110))
scaled_smallest_singular = np.zeros((200, 110))
scaled_condition_number = np.zeros((200, 110))
modified_scaled_condition_number = np.zeros((200, 110))

folder = '/home/tf6/preconditioner/cifar10_classifiction_results_normdontscaleback+morescaler/cifar_fixup_resnet110_pc3_lr0.1_bs128_epoch200_cutmixprob0_steplr_noBN_withscalar_init_kaiming_seed0_multiscalerlrislr+kaiminginit+pconlinear_guangzongyaozua'
for i in range(200):
    m = torch.load(folder+'/checkpoint{}.th'.format(i))['state_dict']
    j = 0
    for k in m.keys():
        if 'weight' in k:
            weight = m[k]
            # print(k, weight.shape)
            weight = weight.view(weight.shape[0], -1)
            sigma = torch.norm(weight)
            weight = weight / sigma
            singulars = torch.svd(weight)[1]
            scaled_largest_singular[i,j] = singulars[0]
            scaled_smallest_singular[i,j] = singulars[-1]
            scaled_condition_number[i,j] = 1./singulars[-1]
            sin_num = max(1, int(singulars.shape[0] * 0.1))
            modified_scaled_condition_number[i,j] = 1. / (singulars[-sin_num:]).mean()
            j += 1

def plot(array, name, log=False):
    plt.figure()
    for j in range(37):
        plt.plot(array[:,j], label='layer{}'.format(j))
    plt.legend()
    if log:
        plt.yscale('log')
    plt.ylabel(name)
    plt.savefig(name+'1.pdf')

    plt.figure()
    for j in range(37, 37+36):
        plt.plot(array[:,j], label='layer{}'.format(j))
    plt.legend()
    if log:
        plt.yscale('log')
    plt.ylabel(name)
    plt.savefig(name+'2.pdf')

    plt.figure()
    for j in range(37+36, 110):
        plt.plot(array[:,j], label='layer{}'.format(j))
    plt.legend()
    if log:
        plt.yscale('log')
    plt.ylabel(name)
    plt.savefig(name+'3.pdf')


plot(scaled_condition_number, 'scaled_condition_number', True)
plot(modified_scaled_condition_number, 'modified_scaled_condition_number', True)
plot(scaled_smallest_singular, 'scaled_smallest_singular')
plot(scaled_largest_singular, 'scaled_largest_singular')
np.save('scaled_condition_number.npy', scaled_condition_number)
np.save('modified_scaled_condition_number.npy', modified_scaled_condition_number)
np.save('scaled_smallest_singular.npy', scaled_smallest_singular)
np.save('scaled_largest_singular.npy', scaled_largest_singular)
    
    
