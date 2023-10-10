import torch as tc


def one_hot(labels, class_num):
    '''len(labels) -> len(labels) * class_num'''
    shape = len(labels), class_num
    ret = tc.zeros(shape, device=labels.device)
    index = labels.view(-1, 1)
    ret.scatter_(dim=1, index=index, value=1)
    return ret
