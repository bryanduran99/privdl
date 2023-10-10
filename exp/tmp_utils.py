# -*- coding: UTF-8 -*-
import torch as tc
import cv2
import numpy as np
from tqdm import tqdm

import exp_utils
exp_utils.setup_import_path()
import block

def resize_guiyihua_transpose(img, size=None):
    '''H*W*C*uint8(0\~255) -> C*height*width*float32(-1\~1)\n
    size=None -> height=img.shape[0], width=img.shape[1]\n
    size=integer -> height=width=integer\n
    size=(integer_a, integer_b) -> height=integer_a, width=integer_b'''
    if size is not None:
        if isinstance(size, int):
            height = width = size
        else:
            height, width = size
        img = cv2.resize(img, (width, height),
            interpolation=cv2.INTER_AREA) # height*width*C*uint8
    img = img.astype(np.float32)  # height*width*C*float32
    img = (img - 128) / 128 # norm to -1 ~ 1
    img = np.transpose(img, [2, 0, 1]) # HWC to CHW
    return img

def calcu_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = tc.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = tc.zeros(3)
    std = tc.zeros(3)
    print(f'==> Computing mean and std...')
    for input, target in tqdm(dataloader):
        input = resize_guiyihua_transpose(input[0].numpy(), 112)
        input = tc.from_numpy(input)
        for i in range(3):
            mean[i] += input[ i, :, :].mean()
            std[i] += input[ i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def get_mean_std(dataset_name):
    ''' according input compute mean and std '''
    if dataset_name == 'celeba':
        print('==>Load celeba......')
        dataset = block.dataset.hubble.xnn_paper.celeba()
        mean, std = calcu_mean_and_std(dataset)
        print('celeba mean and std: ', mean, std)
    elif dataset_name == 'webface':
        print('==>Load webface......')
        dataset = block.dataset.hubble.xnn_paper.webface()
        mean, std = calcu_mean_and_std(dataset)
        print('webface mean and std: ', mean, std)
    elif dataset_name == 'msra':
        print('==>Load msra......')
        dataset = block.dataset.hubble.xnn_paper.msra()
        mean, std = calcu_mean_and_std(dataset)
        print('msra mean and std: ', mean, std)
    elif dataset_name == 'vggface2':
        print('==>Load vggface2......')
        dataset = block.dataset.hubble.xnn_paper.vggface2()
        mean, std = calcu_mean_and_std(dataset)
        print('vggface2 mean and std: ', mean, std)
    elif dataset_name == 'imdb':
        print('==>Load imdb......')
        dataset = block.dataset.hubble.xnn_paper.imdb()
        mean, std = calcu_mean_and_std(dataset)
        print('imdb mean and std: ', mean, std)
    elif dataset_name == 'facescrub':
        print('==>Load facescrub......')
        dataset = block.dataset.hubble.xnn_paper.facescrub()
        mean, std = calcu_mean_and_std(dataset)
        print('facescrub mean and std: ', mean, std)
    else:
        raise ValueError('dataset_name must be celeba, webface or msra')

if __name__ == '__main__':
    # get_mean_std('webface')
    # get_mean_std('msra')
    # get_mean_std('facescrub')
    get_mean_std('vggface2')
    get_mean_std('imdb')
    