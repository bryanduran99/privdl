import os
import pickle
import json
from refile import smart_open
import torch as tc


def smart_realpath(path):
    if path.startswith('s3://'):
        return path
    else:
        return os.path.realpath(path)


def json_save(obj, path):
    '''save obj to path as json with indent=2'''
    path = smart_realpath(path)
    print(f'json_save: {path}')
    with smart_open(path, 'w') as file:
        json.dump(obj, file, indent=2)


def json_load(path):
    '''load json from path'''
    path = smart_realpath(path)
    print(f'json_load: {path}')
    with smart_open(path, 'r') as file:
        return json.load(file)


def pickle_save(obj, path):
    '''save obj to path as pickle file'''
    path = smart_realpath(path)
    print(f'pickle_save: {path}')
    with smart_open(path, 'wb') as file:
        pickle.dump(obj, file)


def pickle_load(path):
    '''load pickle file from path'''
    path = smart_realpath(path)
    print(f'pickle_load: {path}')
    with smart_open(path, 'rb') as file:
        return pickle.load(file)


def torch_save(obj, path):
    '''save obj to path as torch file'''
    path = smart_realpath(path)
    print(f'torch_save: {path}')
    tc.save(obj, path)


def torch_load(path):
    '''load torch file from path'''
    path = smart_realpath(path)
    print(f'torch_load: {path}')
    return tc.load(path)
