import os
import torch as tc


def load_checkpoint(*args, path='checkpoint.tar'):
    '''
    load checkpoint for args from given path if path exists

    return save_checkpoint function for saving args to given path

    all args need to have state_dict and load_state_dict methods
    '''
    path = os.path.realpath(path)
    if os.path.exists(path):
        print(f'load_checkpoint: {path}')
        ckpt = tc.load(path)
        for i, x in enumerate(args):
            x.load_state_dict(ckpt[i])
    else:
        print(f'load_checkpoint: checkpoint not found, skip')
    def save_checkpoint():
        '''save checkpoint of predefined args to predefined path'''
        print(f'save_checkpoint: {path}')
        ckpt = {i: x.state_dict() for i, x in enumerate(args)}
        tc.save(ckpt, path)
    return save_checkpoint
