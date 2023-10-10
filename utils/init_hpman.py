from hpman.m import hpm
import os
import argparse
import hpargparse


def init_hpman(file_path):
    '''parse hpman for files in dirname(file_path)'''
    path = os.path.realpath(file_path)
    path = os.path.dirname(path)
    print(f'init_hpman: parse file in {path}')
    hpm.parse_file(path)
    parser = argparse.ArgumentParser()
    hpargparse.bind(parser, hpm)
    parser.parse_args()
