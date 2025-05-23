"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from multiprocessing import cpu_count

import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from inception_pytorch import InceptionV3
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


def get_activations_statistics_for_gendata(G_model, incep_model, total_img=10000, batch_size=50, dims=2048, device='cpu'):
    incep_model.eval()
    G_model.to(device).eval()

    pred_arr = np.empty((total_img, dims))
    start_idx = 0
    for i in range(0, total_img, batch_size):
        z = torch.randn(batch_size, 128).to(device)
        batch = G_model(z)
        with torch.no_grad():
            pred = incep_model(batch)[0]

        # If incep_model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)
    return mu, sigma


def get_activations_statistics_for_realdata(dataset_name, image_size, incep_model, total_img=10000, batch_size=50, dims=2048, device='cpu'):
    """Calculates the activations of the pool_3 layer for all images.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    print("generating real data activation statistics")
    incep_model.eval()

    data_loader = datasets.getDataLoader(dataset_name, image_size, batch_size=batch_size, shuffle=False, noise=False)
    data_iter = iter(data_loader)
    pred_arr = np.empty((total_img, dims))

    start_idx = 0

    for i in range(0, total_img, batch_size):
        batch = data_iter.next()
        batch = batch[0].to(device)

        with torch.no_grad():
            pred = incep_model(batch)[0]

        # If incep_model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)
    os.makedirs('inception_stats')
    np.savez('inception_stats/'+dataset_name+'_inception_statistics.npz', **{'mu' : mu, 'sigma' : sigma})
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_fid(G_model, dataset_name, image_size, device, dims=2048):
    """Calculates the FID of two paths"""
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    incep_model = InceptionV3([block_idx]).to(device)

    my_file = pathlib.Path('inception_stats/'+dataset_name+'_inception_statistics.npz')
    if my_file.is_file():
        print("yeah! we've already had statistics!")
        m1 = np.load('inception_stats/'+dataset_name+'_inception_statistics.npz')['mu']
        s1 = np.load('inception_stats/'+dataset_name+'_inception_statistics.npz')['sigma']
    else:
        m1, s1 = get_activations_statistics_for_realdata(dataset_name, image_size, incep_model, dims=dims, device=device)

    m2, s2 = get_activations_statistics_for_gendata(G_model, incep_model, dims=dims, device=device)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def main(dataset_name, folder, exp, image_size, start_iter, end_iter, gap, ema):
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')

    # ==========  load net G ==================
    if dataset_name == 'cifar':
        netG, _ = hignorm_networks.getGD('dcgan', dataset_name, 64, 64, dim_z=128, image_size=image_size, ignoreD=True)
        # netG, _ = hignorm_networks.getGD('resnet', dataset_name, 256, 256, dim_z=128, image_size=image_size, ignoreD=True)
        
    elif dataset_name in ['lsun', 'celeba']:
        if image_size in [128, 256, 512]:
            feature_num = 512 if image_size == 128 else 1024
            netG, _ = hignorm_networks.getGD('dcgan', dataset_name, feature_num, feature_num, dim_z=128, image_size=image_size, ignoreD=True)
        else:
            raise ValueError("invalid image size")
    
    else:
        raise ValueError("xiaciyiding")

    # ==========  get fid ==============
    fid_results = []
    for iter in range(end_iter, start_iter-1, -gap):
        if ema:
            model_path = os.path.join(folder, exp, 'models/emaG0.9999_epoch{}.pth'.format(iter))
        else:
            model_path = os.path.join(folder, exp, 'models/G_epoch{}.pth'.format(iter))
        print(model_path)
        netG.load_state_dict(torch.load(model_path)) 
        fid_value = calculate_fid(netG, dataset_name, image_size, device)
        print('FID: ', fid_value)
        fid_results.append(fid_value)
        if ema:
            np.save(os.path.join(folder, exp, "fids_ema_pytorch.npy"), fid_results)
        else:
            np.save(os.path.join(folder, exp, "fids_pytorch.npy"), fid_results)


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_name', type=str, default='cifar')
    parser.add_argument('--folder', type=str, default='cifar_structdcgan_GfeatureNum64_DfeatureNum64_losslog')
    parser.add_argument('--exp', type=str, default='rsgan_CIFAR_size32_dlr0.0002_glr0.0002_diter1_giter1_b10.5_b20.999_Gnumfea64_Dnumfea64_batchsize64_useadappolarTrueiter0')
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--end_iter', type=int, default=100000)
    parser.add_argument('--gap', type=int, default=2000)
    parser.add_argument('--ema', action='store_true')
    args = parser.parse_args()

    main(args.dataset_name, args.folder, args.exp, args.image_size, args.start_iter, args.end_iter, args.gap, args.ema)