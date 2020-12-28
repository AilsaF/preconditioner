from PIL import Image
import torch
import torch.nn.functional as F
import torchvision
import os
import functools
import numpy as np
import time
import itertools
import hignorm_networks
import sliced_wasserstein
import sys
import os
directory_path = os.path.abspath(os.getcwd())
if 'tf6' in directory_path:
    sys.path.append('/home/tf6/slicedGAN')
elif 'illini' in directory_path:
    sys.path.append('/home/illini/rsgan')
else:
    raise ValueError("invalid server")
import datasets


h_dim = 128
eval_fid = True

if 'tf6' in directory_path:
    folder = '/data01/tf6/'
elif 'illini' in directory_path:
    folder = '/home/illini/pngan/'

if 1:
    db = 'cifar'
    img_size = 32
    netG, _ = hignorm_networks.getGD('dcgan', db, 64, 64, dim_z=h_dim, image_size=img_size, ignoreD=True)
    folder += 'ie510result/cifar_structdcgan_GfeatureNum64_DfeatureNum64_losslog/'

    # netG, _ = hignorm_networks.getGD('resnet', db, 256, 256, dim_z=h_dim, image_size=img_size, ignoreD=True)
    # folder = 'ie510result/cifar_structresnet_GfeatureNum256_DfeatureNum256_losshinge'
    exp_ = 'vanilla_CIFAR_size32_dlr0.0002_glr0.0002_diter1_giter1_b10.5_b20.999_Gnumfea64_Dnumfea64_batchsize64_sniter1_usepolarFalseiter0_ematrickSN+unifD'
    end = 100000
    start = 1000
    gap = 1000
    ema = 0
if 0:
    db = 'stl'
    img_size = 48
    #netG, _ = hignorm_networks.getGD('dcgan', db, 64, 64, dim_z=h_dim, image_size=img_size, ignoreD=True)
    #folder += 'ie510result/stl_structdcgan_GfeatureNum64_DfeatureNum64_losslog'

    netG, _ = hignorm_networks.getGD('resnet', db, 64, 64, dim_z=h_dim, image_size=img_size, ignoreD=True)
    folder += 'ie510result/stl_structresnet_GfeatureNum64_DfeatureNum64_losshinge/'
    exp_ = 'vanilla_STL_size48_dlr0.0002_glr0.0002_diter1_giter1_b10.5_b20.999_Gnumfea64_Dnumfea64_batchsize64_sniter1_usepolarTrueiter3_ematrickPN3'

    end = 100000
    start = 10
    gap = 2000
    ema = 0

if 0:
    db = 'mnist'
    img_size = 32
    netG, _ = hignorm_networks.getGD('dcgan', db, 64, 64)
    folder += 'ie510result/mnist_structdcgan_featureNum64_losslog'
    exp_ = 'Vanilla_MNIST_dlr0.0001_glr0.0001_diter1_giter1_b10.5_b20.999_numfea64_batchsize64_sniter1_usepolarFalseiter5deltaorho+maxmin'
    end = 30000
    start = 28000
    gap = 500

if 0:
    db = 'lsun'
    img_size = 256
    netG, _ = hignorm_networks.getGD('dcgan', db, 1024, 512, dim_z=h_dim, image_size=256, ignoreD=True)
    folder += 'ie510result/lsun_structdcgan_GfeatureNum1024_DfeatureNum768_losslog/'
    exp_ = 'SVDGAN_LSUN_size256_dlr0.0002_glr0.0002_diter1_giter1_b10.5_b20.999_Gnumfea1024_Dnumfea768_batchsize64_ematrick/'
    end = 100000
    start = 1000
    gap = 4000
    ema = 0

if 0:
    db = 'tower'
    img_size = 256
    netG, _ = hignorm_networks.getGD('dcgan', db, 1024, 512, dim_z=h_dim, image_size=256, ignoreD=True)
    folder += 'ie510result/tower_structdcgan_GfeatureNum1024_DfeatureNum1024_losslog/'
    exp_ = 'vanilla_TOWER_size256_dlr0.0002_glr0.0002_diter1_giter1_b10.5_b20.999_Gnumfea1024_Dnumfea1024_batchsize64_sniter1_useadappolarTrueiter0_ematrickadaptivefitrelu+strategy4+orthoD/'
    end = 100000
    start = 1000
    gap = 4000
    ema = 0

if 0:
    db = 'celeba'
    img_size = 512
    netG, _ = hignorm_networks.getGD('dcgan', db, 1024, 512, dim_z=h_dim, image_size=img_size, ignoreD=True)
    folder += 'ie510result/celeba_structdcgan_GfeatureNum1024_DfeatureNum1024_losslog/'
    # folder += 'ie510result/celeba_structdcgan_GfeatureNum1024_DfeatureNum512_losslog/'
    exp_ = 'vanilla_CELEBA_size512_dlr0.0002_glr0.0002_diter1_giter1_b10.5_b20.999_Gnumfea1024_Dnumfea1024_batchsize64_useadappolarTrueiter0/'
    end = 172000
    start = 1000
    gap = 4000
    ema = 0

num_images = 16384 # 8192 
minibatch_size = int(np.clip(8192 // img_size, 4, 256))
loader = datasets.getDataLoader(db, img_size, batch_size=minibatch_size, train=True)
data_iter = iter(loader)

# Feed in reals.
api = sliced_wasserstein.API(num_images, [3,img_size,img_size], np.uint8, minibatch_size)
api.begin('reals')
for _ in range(0, num_images, minibatch_size):
    images = data_iter.next()[0]
    if db == 'mnist':
        images = images.repeat(3, axis=1)
    images = np.array(((images / 2 + 0.5) * 255)).astype(np.uint8)
    # print(images.min(), images.max())
    api.feed('reals', images)
results = api.end('reals')
# print('%-12s' % misc.format_time(time.time() - time_begin), end='')
for val, fmt in zip(results, api.get_metric_formatting()):
    print(fmt % val, end='')
print("======")


swd_results = []

for iter in range(end, start-1, -gap):
    api.begin('fakes')
    if ema:
        model_path = os.path.join(folder, exp_, 'models/emaG0.9999_epoch{}.pth'.format(iter))
    else:
        model_path = os.path.join(folder, exp_, 'models/G_epoch{}.pth'.format(iter))
    print(model_path)
    netG.load_state_dict(torch.load(model_path))
    netG.cuda()

    for _ in range(0, num_images, minibatch_size):
        z = torch.randn(minibatch_size, h_dim).cuda()
        with torch.no_grad():
            images = netG(z).cpu()
            if db == 'mnist':
                images = images.repeat(3, axis=1)
        images = np.array(((images / 2 + 0.5) * 255)).astype(np.uint8)
        # print(images.min(), images.max())
        api.feed('fakes', images)
    results = api.end('fakes')
    # print('%-12s' % misc.format_time(time.time() - time_begin), end='')
    sw = []
    for val, fmt in zip(results, api.get_metric_formatting()):
        print(fmt % val, end='')
        sw.append(fmt % val)
    print()

    swd_results.append(sw)
    if ema:
        np.save(os.path.join(folder, exp_, 'swd_ema.npy'), np.array(swd_results))
    else:
        np.save(os.path.join(folder, exp_, 'swd.npy'), np.array(swd_results))




# Colocations handled automatically by placer.
# Dataset shape = [3, 32, 32]
# Dynamic range = [0, 255]
# Label size    = 0
# minibatch size 256
# dataset_obj <dataset.TFRecordDataset object at 0x7f2d4d8c3b90>
# Initializing metrics.sliced_wasserstein.API...
# image_shape [3, 32, 32]
# obj <metrics.sliced_wasserstein.API object at 0x7f2d4d8c3b10>
# Snapshot  Time_eval   SWDx1e3_32   SWDx1e3_16   SWDx1e3_avg  
