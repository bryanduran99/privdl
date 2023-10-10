import numpy as np


def batch_to_cuda(batch_data):
    '''return batch_data (on cuda) \n
    pytorch DataLoader 迭代时，依据 dataset 的 \_\_getitem\_\_
    返回形式，batch_data 可能是 list of tensor、dict of tensor、tensor，
    本函数将这些类型的数据中的 tensor 放到 cuda 上'''
    if isinstance(batch_data, list):
        return [data.cuda() for data in batch_data]
    elif isinstance(batch_data, dict):
        return {name: data.cuda() for name, data in batch_data.items()}
    else:
        return batch_data.cuda()

def batch_to_rank(batch_data, local_rank):
    '''return batch_data (on cuda) \n
    pytorch DataLoader 迭代时，依据 dataset 的 \_\_getitem\_\_
    返回形式，batch_data 可能是 list of tensor、dict of tensor、tensor，
    本函数将这些类型的数据中的 tensor 放到 cuda 上'''
    if isinstance(batch_data, list):
        return [data.to(local_rank) for data in batch_data]
    elif isinstance(batch_data, dict):
        return {name: data.to(local_rank) for name, data in batch_data.items()}
    else:
        return batch_data.to(local_rank)

def batch_to_numpy(batch_data):
    '''return batch_data (on cpu) \n
    batch_data 可以是 list of tensor、dict of tensor、tensor'''
    if isinstance(batch_data, list):
        return [np.array(data.detach().cpu()) for data in batch_data]
    elif isinstance(batch_data, dict):
        return {name: np.array(data.detach().cpu()) for name, data in batch_data.items()}
    else:
        return np.array(batch_data.detach().cpu())