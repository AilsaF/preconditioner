import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

layernum = 1202
scaled_largest_singular = np.zeros((200, layernum))
scaled_smallest_singular = np.zeros((200, layernum))
scaled_condition_number = np.zeros((200, layernum))
modified_scaled_condition_number = np.zeros((200, layernum))

folder = '/home/tf6/preconditioner/cifar10_classifiction_results_normdontscaleback+morescaler/cifar_fixup_resnet1202_pc0.0_lr0.1_bs128_epoch200_cutmixprob0_steplr_noBN_withscalar_init_fixup_seed0'
for i in range(20):
    m = torch.load(folder+'/checkpoint{}.th'.format(i))['state_dict']
    j = 0
    for k in m.keys():
        # if 'scale' in k:
        #     scaled_largest_singular[i,j] = m[k]
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
    for j in range(layernum//3+1):
        plt.plot(array[:,j], label='layer{}'.format(j))
    plt.legend()
    if log:
        plt.yscale('log')
    plt.ylabel(name)
    plt.savefig(name+'1.pdf')

    plt.figure()
    for j in range(layernum//3+1, (layernum//3)*2+1):
        plt.plot(array[:,j], label='layer{}'.format(j))
    plt.legend()
    if log:
        plt.yscale('log')
    plt.ylabel(name)
    plt.savefig(name+'2.pdf')

    plt.figure()
    for j in range((layernum//3)*2+1, layernum):
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
    
    
