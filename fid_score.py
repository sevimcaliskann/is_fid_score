#!/usr/bin/env python

import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
from scipy import linalg
from scipy.misc import imread, imresize
from torch.nn.functional import adaptive_avg_pool2d
import pickle
import posixpath

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from inception import InceptionV3

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('path', type=str, nargs=2,
                    help=('Path to the generated images or '
                          'to .npz statistic files'))
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('--is_classwise', type=int, default=0,
                    help=('Calculation of fid scores are done by each class or not, if 1, else if 0 no'))
parser.add_argument('-c', '--gpu', default='', type=str,
                    help='GPU to use (leave blank for CPU only)')


def get_activations(files, model, batch_size=50, dims=2048,
                    cuda=False, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if len(files) % batch_size != 0:
        print(('Warning: number of images is not a multiple of the '
               'batch size. Some samples are going to be ignored.'))
    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    n_batches = len(files) // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))

    for i in tqdm(range(n_batches)):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches))
        start = i * batch_size
        end = start + batch_size

        images = [imread(str(f)).astype(np.float32)
                           for f in files[start:end]]

        single_channel_images = [np.stack((img,)*3, axis=-1)
                           for img in images if len(img.shape)==2]
        images.extend(single_channel_images)

        images = np.array([imresize(img, (299, 299)).astype(np.float32)
                           for img in images if len(img.shape)>2 and img.shape[2]==3])

        # Reshape to (n_images, 3, height, width)
        images = images.transpose((0, 3, 1, 2))
        images /= 255

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        if cuda:
            batch = batch.cuda()

        pred = model(batch)[0]


        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(batch_size, -1)

    if verbose:
        print(' done')

    files = [f.as_posix().split('/')[1] for f in files]
    activation_dict = dict(zip(files[:n_used_imgs], pred_arr))
    return pred_arr, activation_dict


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

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=50,
                                    dims=2048, cuda=False, verbose=True, is_classwise = False, emos = None):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    if not is_classwise:
        act, _ = get_activations(files, model, batch_size, dims, cuda, verbose)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
    else:
        _, act_dict = get_activations(files, model, batch_size, dims, cuda, verbose)
        mu = dict()
        sigma = dict()
        for emo in emos:
            ids = emos[emo]
            ids = list(set(ids).intersection(set(act_dict.keys())))
            print('len of ids: ', len(ids))
            if not ids:
                continue
            act = np.array([act_dict[id] for id in ids])
            m = np.mean(act,axis=0)
            s = np.cov(act, rowvar=False)
            mu[emo] = m
            sigma[emo] = s
    return mu, sigma



def _compute_statistics_of_path(path, model, batch_size=50, dims=2048, cuda=True, is_classwise=False):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        emos = None
        if is_classwise==True:
            pickle_file= list(path.glob('*.pkl'))
            emos = pickle.load(open(pickle_file[0].absolute().as_posix(), 'rb'))
            emos = create_emo_lists(emos)
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, cuda, is_classwise=is_classwise, emos=emos)

    return m, s


def calculate_fid_given_paths(paths, batch_size, cuda=True, dims=2048, is_classwise=False):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()

    m1, s1 = _compute_statistics_of_path(paths[0], model, batch_size,
                                         dims, cuda, is_classwise)
    m2, s2 = _compute_statistics_of_path(paths[1], model, batch_size,
                                         dims, cuda, is_classwise)
    if is_classwise:
        emos = set(m1.keys()).intersection(m2.keys())
        fid_value = dict()
        for i in emos:
            fid = calculate_frechet_distance(m1[i], s1[i], m2[i], s2[i])
            fid_value[i] = fid
    else:
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value

def create_emo_lists(emos):
    max_emo = max(emos.values())
    min_emo = min(emos.values())
    emos_ = dict()
    for i in range(min_emo, max_emo+1):
        ids = [id for id in emos.keys() if emos[id]==i]
        if len(ids)>0:
            emos_[i] = ids

    return emos_



if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu



    fid_value = calculate_fid_given_paths(args.path,
                                          args.batch_size,
                                          args.gpu != '',
                                          args.dims,
                                          args.is_classwise==1)

    if args.is_classwise==1:
        labels = {0:'Neutral', 1:'Happy', 2:'Sadness', 3:'Surprise', 4:'Fear', 5:'Disgust', 6:'Anger', 7:'Contempt'}
        print(fid_value)
        for i in fid_value.keys():
            print('for %s, the FID value is %.6f' % (labels[i], fid_value[i]))
    else:
        print('FID: ', fid_value)
