import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

from inception import InceptionV3
import torchvision.datasets as dset
import torchvision.transforms as transforms
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import pathlib
from tqdm import tqdm
from scipy.misc import imread, imresize


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
parser.add_argument('-c', '--gpu', default='', type=str,
                    help='GPU to use (leave blank for CPU only)')



def get_pred(x, model):
    tmp = model.model(x)
    tmp = model.emo_layer(tmp)
    return F.softmax(tmp).data.cpu().numpy()


def get_scores(files, model, batch_size=50, dims=8,
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
    model.model.eval()
    model.emo_layer.eval()

    if len(files) % batch_size != 0:
        print(('Warning: number of images is not a multiple of the '
               'batch size. Some samples are going to be ignored.'))
    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    n_batches = len(files) // batch_size
    n_used_imgs = n_batches * batch_size
    N = len(files)

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

        pred = get_pred(batch, model)
        pred_arr[start:end] = pred.reshape(batch_size, -1)

        # Now compute the mean kl-div
    split_scores = []
    splits = 8

    for k in range(splits):
        part = pred_arr[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    if verbose:
        print(' done')

    return np.mean(split_scores), np.std(split_scores)

if __name__ == '__main__':

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = InceptionV3()
    if args.gpu != '':
        model.cuda()

    for p in args.path:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

        path = pathlib.Path(p)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        m, s = get_scores(files, model, batch_size=50, dims=8,
                            cuda=args.gpu != '', verbose=True)
        print('For path -> %s , the inception scores are : mean: %.3f, STD: %.3f ' % (p, m, s))
