import torch as tc
import torch.nn.functional as F
from . import loss_utils


class AMSoftmax(tc.nn.Module):
    '''features -> scores'''

    def __init__(self, in_features, class_num, margin=0.35, scale=32):
        super().__init__()
        self.class_num = class_num
        self.margin = margin
        self.scale = scale
        weight = tc.Tensor(class_num, in_features)
        tc.nn.init.xavier_uniform_(weight)
        self.weight = tc.nn.Parameter(weight)

    def forward(self, features, labels):
        features = F.normalize(features)
        weight = F.normalize(self.weight)
        cosine = F.linear(features, weight)
        one_hot_labels = loss_utils.one_hot(labels, self.class_num)
        scores = cosine - self.margin * one_hot_labels
        scores *= self.scale
        return scores
