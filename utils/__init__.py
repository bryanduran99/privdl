'''常用小工具'''

import numpy as np

from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from itertools import chain
from tqdm import tqdm
from os import makedirs

from .imdecode import imdecode
from .init_hpman import init_hpman
from .box import Box
from .clock import strftime, Clock, interval, TrainLoopClock
from .checkpoint import load_checkpoint
from .state import State
from .file_io import json_save, json_load, pickle_save, pickle_load, torch_save, torch_load
from .create_dpflow import create_dpflow
from .move_tensor import batch_to_cuda, batch_to_numpy, batch_to_rank
from .transformed_dataset import TransformedDataset
from .monitor import Monitor
from .ddp_sampler import DistributedWeightedSampler
import time

def identity(x):
    return x


def step(optimizer, loss, lr_scheduler=None):
    '''optim.zero_grad -> loss.backward -> optim.step -> (scheduler.step)'''
    optimizer.zero_grad()
    # t0 = time.time()
    loss.backward()
    # t1 = time.time()
    # print("loss_back:{}".format(t1-t0))
    optimizer.step()
    if lr_scheduler is not None:
        lr_scheduler.step()


def accuracy(scores, labels):
    '''the rate that scores.argmax(dim=1) == labels'''
    accurate = (scores.argmax(dim=1) == labels).sum()
    return int(accurate) / len(labels)


def sub_state_dict(Dict, prefix):
    '''取 Dict 的子集，要求 key 前缀为 prefix，并删去 prefix 作为新的 key'''
    return {k[len(prefix):]: v
        for k, v in Dict.items() if k.startswith(prefix)}


def input_option(name, options):
    '''让用户输入选项的 index，返回相应的选项\n
    name: 选项的名称, options: 选项的内容'''
    print(f'input_option: {name}')
    for index, option in enumerate(options):
        print(f'┃  {index}: {option}')
    index = int(input(f'┗ {name}: '))
    return options[index]


def choice(array, size):
    '''从 array 里无放回地随机取 size 个元素'''
    return np.random.choice(array, size, replace=False)
