'''
From https://github.com/tsc2017/Inception-Score
Code derived from https://github.com/openai/improved-gan/blob/master/inception_score/model.py and https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py

Usage:
    Call get_inception_score(images, splits=10)
Args:
    images: A numpy array with values ranging from 0 to 255 and shape in the form [N, 3, HEIGHT, WIDTH] where N, HEIGHT and WIDTH can be arbitrary. A dtype of np.uint8 is recommended to save CPU memory.
    splits: The number of splits of the images, default is 10.
Returns:
    Mean and standard deviation of the Inception Score across the splits.
'''

import tensorflow as tf
import os
import functools
import numpy as np
import time
from tensorflow.python.ops import array_ops
import torch
import itertools
import hignorm_networks

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
import fid_tf


h_dim = 128
eval_fid = True

if 1:
    db = 'cifar'
    img_size = 32
    # netG, _ = hignorm_networks.getGD('dcgan', db, 64, 64, dim_z=h_dim, image_size=img_size, ignoreD=True)
    # folder = 'ie510result/cifar_structdcgan_GfeatureNum64_DfeatureNum64_losslog/'

    netG, _ = hignorm_networks.getGD('resnet', db, 512, 256, dim_z=h_dim, image_size=img_size, ignoreD=True)
    folder = '/data01/tf6/ie510result/cifar_structresnet_GfeatureNum512_DfeatureNum128_losshinge_deep1'
    exp_ = 'vanilla_CIFAR_size32_dlr0.0002_glr0.0002_diter5_giter1_b10.5_b20.999_Gnumfea512_Dnumfea128_batchsize64_useadappolarTrueiter0deepblock'
    end = 100000
    start = 1000
    gap = 2000
    ema = 0
    # eval_fid = False

if 0:
    db = 'stl'
    img_size = 48
    # netG, _ = hignorm_networks.getGD('resnet', db, 64, 64, dim_z=h_dim, image_size=img_size, ignoreD=True)
    # folder = 'ie510result/stl_structresnet_GfeatureNum64_DfeatureNum64_losshinge/'
    netG, _ = hignorm_networks.getGD('dcgan', db, 64, 64, dim_z=h_dim, image_size=img_size, ignoreD=True)
    folder = 'ie510result/stl_structdcgan_GfeatureNum64_DfeatureNum64_losslog'
    exp_='vanilla_STL_size48_dlr0.0002_glr0.0002_diter1_giter1_b10.0_b20.9_Gnumfea64_Dnumfea64_batchsize64_useadappolarTrueiter0'
    end = 100000
    start = 80000
    gap = 2000
    ema = 0
    eval_fid = False

if 0:
    db = 'mnist'
    img_size = 32
    netG, _ = hignorm_networks.getGD('dcgan', db, 64, 64)
    folder = '/scratch/tf6/ie510result/mnist_structdcgan_featureNum64_losslog'
    exp_ = 'Vanilla_MNIST_dlr0.0001_glr0.0001_diter1_giter1_b10.5_b20.999_numfea64_batchsize64_sniter1_usepolarFalseiter5deltaorho+maxmin'
    end = 30000
    start = 28000
    gap = 500

if 0:
    db = 'lsun'
    img_size = 256
    netG, _ = hignorm_networks.getGD('dcgan', db, 1024, 512, dim_z=h_dim, image_size=256, ignoreD=True)
    folder = 'ie510result/lsun_structdcgan_GfeatureNum1024_DfeatureNum768_losslog/'
    exp_ = 'SVDGAN_LSUN_size256_dlr0.0002_glr0.0002_diter1_giter1_b10.5_b20.999_Gnumfea1024_Dnumfea768_batchsize64_ematrick/'
    end = 200000
    start = 1000
    gap = 4000

if 0:
    db = 'celeba'
    img_size = 256
    netG, _ = hignorm_networks.getGD('dcgan', db, 1024, 512, dim_z=h_dim, image_size=img_size, ignoreD=True)
    folder = '/home/illini/pngan/ie510result/celeba_structdcgan_GfeatureNum1024_DfeatureNum512_losslog/'
    # folder = '/scratch/tf6/ie510result/celeba_structdcgan_GfeatureNum1024_DfeatureNum512_losslog/'
    exp_ = 'vanilla_CELEBA_size256_dlr0.0002_glr0.0002_diter1_giter1_b10.5_b20.999_Gnumfea1024_Dnumfea512_batchsize64_sniter1_usepolarFalseiter1_ematrick/'
    end = 100000
    start = 1000
    gap = 4000

if not eval_fid:
    tfgan = tf.contrib.gan

    session=tf.compat.v1.InteractiveSession()

    # A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
    BATCH_SIZE = 64
    INCEPTION_URL = 'http://download.tensorflow.org/models/frozen_inception_v1_2015_12_05.tar.gz'
    INCEPTION_FROZEN_GRAPH = 'inceptionv1_for_inception_score.pb'

    # Run images through Inception.
    inception_images = tf.compat.v1.placeholder(tf.float32, [None, 3, None, None])
    def inception_logits(images = inception_images, num_splits = 1):
        images = tf.transpose(images, [0, 2, 3, 1])
        size = 299
        images = tf.compat.v1.image.resize_bilinear(images, [size, size])
        generated_images_list = array_ops.split(images, num_or_size_splits = num_splits)
        logits = tf.map_fn(
            fn = functools.partial(
                 tfgan.eval.run_inception,
                 default_graph_def_fn = functools.partial(
                 tfgan.eval.get_graph_def_from_url_tarball,
                 INCEPTION_URL,
                 INCEPTION_FROZEN_GRAPH,
                 os.path.basename(INCEPTION_URL)),
                 output_tensor = 'logits:0'),
            elems = array_ops.stack(generated_images_list),
            parallel_iterations = 1,
            back_prop = False,
            swap_memory = True,
            name = 'RunClassifier')
        logits = array_ops.concat(array_ops.unstack(logits), 0)
        return logits

    logits=inception_logits()

    def get_inception_probs(inps):
        n_batches = int(np.ceil(float(inps.shape[0]) / BATCH_SIZE))
        preds = np.zeros([inps.shape[0], 1000], dtype = np.float32)
        for i in range(n_batches):
            inp = inps[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] / 255. * 2 - 1
            preds[i * BATCH_SIZE : i * BATCH_SIZE + min(BATCH_SIZE, inp.shape[0])] = session.run(logits,{inception_images: inp})[:, :1000]
        preds = np.exp(preds) / np.sum(np.exp(preds), 1, keepdims=True)
        return preds

    def preds2score(preds, splits=10):
        scores = []
        for i in range(splits):
            part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
            q = np.expand_dims(np.mean(part, 0), 0)
            kl = part * (np.log(part / q)) + (1 - part) * np.log((1 - part) / (1 - q))
            kl = np.mean(kl)
            scores.append(np.exp(kl))
        return np.mean(scores), np.std(scores)

    def preds2score(preds, splits=10):
        scores = []
        for i in range(splits):
            part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        return np.mean(scores), np.std(scores)

    def get_inception_score(images, splits=10):
        assert(type(images) == np.ndarray)
        assert(len(images.shape) == 4)
        assert(images.shape[1] == 3)
        assert(np.min(images[0]) >= 0 and np.max(images[0]) > 10), 'Image values should be in the range [0, 255]'
        print('Calculating Inception Score with %i images in %i splits' % (images.shape[0], splits))
        start_time=time.time()
        preds = get_inception_probs(images)
        mean, std = preds2score(preds, splits)
        print('Inception Score calculation time: %f s' % (time.time() - start_time))
        return mean, std  # Reference values: 11.38 for 50000 CIFAR-10 training set images, or mean=11.31, std=0.10 if in 10 splits.



if eval_fid:
    loader = datasets.getDataLoader(db, img_size, batch_size=10000, train=False)
    data_iter = iter(loader)
    realdata = data_iter.next()[0]
    realdata = np.array(realdata)
    if db == 'mnist':
        realdata = realdata.repeat(3, axis=1)
    realdata = (realdata / 2 + 0.5) * 255
    print(realdata.min(), realdata.max())
    realdata = realdata.astype(np.uint8)

# m, s = g(realdata, splits=10)
# print (m, s)


is_results = []
fid_results = []

## ========== fid ==============

for iter in range(end, start-1, -gap):
# for iter in range(start, end+1, 1000):
    if ema:
        model_path = os.path.join(folder, exp_, 'models/emaG0.9999_epoch{}.pth'.format(iter))
    else:
        model_path = os.path.join(folder, exp_, 'models/G_epoch{}.pth'.format(iter))
    print(model_path)
    try:
        netG.load_state_dict(torch.load(model_path))
    except:
        continue
    netG.cuda()

    data = []
    sample_num = 10000 if eval_fid else 50000
    for _ in range(0, sample_num, 100):
        z = torch.randn(100, h_dim).cuda()
        with torch.no_grad():
            x = netG(z).cpu().numpy()
            if db == 'mnist':
                # x = x.reshape(-1, 1, 28, 28)
                x = x.repeat(3, axis=1)
        data.append(x)
    data = np.concatenate(data)
    data = (data / 2 + 0.5) * 255
    data = data.astype(np.uint8)
    if not eval_fid:
        # get IS
        m, s = get_inception_score(data, splits=10)
        print (m, s)
        is_results.append([m, s])
        if ema:
            np.save(os.path.join(folder, exp_, 'InceptionScore_ema.npy'), np.array(is_results))
        else:
            np.save(os.path.join(folder, exp_, 'InceptionScore.npy'), np.array(is_results))
    if eval_fid:
        # get FID
        fid = fid_tf.get_fid(realdata, data)
        fid_results.append(fid)
        print(fid)
        if ema:
            np.save(os.path.join(folder, exp_, "fids_ema.npy"), fid_results)
        else:
            np.save(os.path.join(folder, exp_, "fids.npy"), fid_results)


# loader = datasets.getDataLoader('stl', 48, batch_size=50000)
# data_iter = iter(loader)
# data = data_iter.next()[0]
# data = np.array(data)
# # data = data.repeat(3, axis=1)
# data = (data / 2 + 0.5) * 255
# data = data.astype(np.uint8)
# m, s = get_inception_score(data, splits=10)
# print (m, s)


# ### =========== precision recall ================
# import prd_score as prd
# import pickle
# ref_emb = fid_tf.get_inception_activations(realdata)
# prds = []
# exp_names = []
# # folder = '/scratch/tf6/maxsliced_results/cpgan_results_rebuttal/cifar_structdcgan_featureNum64_imgsize32_losslog_withbatchnorm/'
#
# folders = ['real', 'CPGAN_STL_randz_unsortz_dlr0.0001_glr0.0004_diter2_giter2_b10.5_b20.999_numfea64_batchsize64_geps1.0_deps1.0_slack1.0/', 'Vanilla_STL_randz_dlr0.0002_glr0.0002_diter1_giter1_b10.5_b20.999_numfea64_batchsize64/', 'wgan_stl_dlr0.0001_glr0.0001_diter1_giter5_b10.5_b20.9_Gnumfea64_Dnumfea64/']
# # for lr in ['real', 5e-07, 1e-06, 5e-06, 1e-05, 5e-05, 1e-04]:
# for exp, name, epoch in zip(folders, ['Real DATA', 'RSGAN', 'JS-GAN', 'WGAN-GP']):
#     print(exp)
#     # if model_type == 'Vanilla':
#     #     model_path = folder + "Vanilla_MNIST_randz_dlr{}_glr{}_diter1_giter1_b10.5_b20.999_numfea64_batchsize128_perturbpart1.0/models/G_epoch30000.pth".format(lr, lr)
#     # elif model_type == 'CPGAN':
#     #     model_path = folder + "CPGAN_MNIST_randz_unsortz_dlr{}_glr{}_diter1_giter1_b10.5_b20.999_numfea64_batchsize128_perturbpart1.0/models/G_epoch30000.pth".format(
#     #         lr, lr)
#     model_path = folder + exp + "models/G_epoch100000.pth"
#     if exp == 'real':
#         loader = datasets.getDataLoader(db, img_size, batch_size=10000, train=True)
#         data_iter = iter(loader)
#         data = data_iter.next()[0]
#         data = np.array(data)
#         if db == 'mnist':
#             data = data.repeat(3, axis=1)
#         data = (data / 2 + 0.5) * 255
#         data = data.astype(np.uint8)
#     else:
#         netG.load_state_dict(torch.load(model_path))
#         netG.cuda()
#         data = []
#         for _ in range(0, 10000, 500):
#             z = (torch.rand(500, h_dim)*2-1).cuda()
#             with torch.no_grad():
#                 x = netG(z).cpu().numpy()
#                 if db == 'mnist':
#                     # x = x.reshape(-1, 1, 28, 28)
#                     x = x.repeat(3, axis=1)
#             data.append(x)
#         data = np.concatenate(data)
#         data = (data / 2 + 0.5) * 255
#         data = data.astype(np.uint8)
#
#     eval_emb = fid_tf.get_inception_activations(data)
#     prd_res = prd.compute_prd_from_embedding(eval_data=eval_emb, ref_data=ref_emb)
#     prds.append(prd_res)
#     # exp_names.append(str(lr))
#     exp_names.append(name)
#
# prd.plot(prds, labels= exp_names, out_path='stl_prd.pdf')
#
# # prd.plot(prds, labels= exp_names, out_path= model_type+'_mnist_perturb_prd.png')
# with open('stl_prd.pickle', 'wb') as f:
#     pickle.dump((exp_names, prds), f)

# with open('data.pickle', 'rb') as f:
#     data = pickle.load(f)





