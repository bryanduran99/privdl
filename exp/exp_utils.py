import os
import sys


def privdl_path():
    '''从 exp_utils.py 向上两层找出 privdl 所在 path，
    例如 /data/jupyter/privdl/privdl'''
    realpath = os.path.realpath(__file__)
    return os.path.dirname(os.path.dirname(realpath))


def data_path():
    '''privdl 下的 data 文件夹'''
    return f'{privdl_path()}/data'


def result_path(exp_file):
    '''exp_file 所对应的 result 文件夹'''
    return f'{privdl_path()}/result/{exp_file}'


def setup_import_path():
    '''让 utils 和 block 可以被 import'''
    path = privdl_path()
    if path not in sys.path:
        sys.path.append(path)


def setup_result_path(exp_file):
    '''返回 exp_file 所对应的 result 文件夹路径，
    如果路径不存在则创建'''
    path = result_path(exp_file)
    os.makedirs(path, exist_ok=True)
    return path
