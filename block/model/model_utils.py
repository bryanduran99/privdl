import cv2
import numpy as np
import torch as tc


def resize_norm_transpose(img, size=None):
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


def conv_block(in_channels, out_channels, kernel_size,
    stride=1, padding=0, depthwise=False, nonlinear=True):
    '''Conv + BatchNorm + PReLU'''
    return tc.nn.Sequential(
        tc.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
            groups=(in_channels if depthwise else 1), bias=False),
        tc.nn.BatchNorm2d(out_channels),
        tc.nn.PReLU(out_channels) if nonlinear else tc.nn.Identity())


class Bottleneck(tc.nn.Module):
    '''in_channels -> in_channels*expansion -> out_channels'''

    def __init__(self, in_channels, out_channels, expansion, stride=1):
        super().__init__()
        self.shortcut = stride == 1 and in_channels == out_channels
        exp_channels = in_channels * expansion
        self.conv = tc.nn.Sequential(
            conv_block(in_channels, exp_channels, kernel_size=1),
            conv_block(exp_channels, exp_channels, kernel_size=3,
                stride=stride, padding=1, depthwise=True),
            conv_block(exp_channels, out_channels, kernel_size=1, nonlinear=False))

    def forward(self, x):
        return self.conv(x) + x if self.shortcut else self.conv(x)


def conv_blockT(in_channels, out_channels, kernel_size,
    stride=1, padding=0, output_padding=0, depthwise=False, nonlinear=True):
    '''Convt + BatchNorm + PReLU'''
    return tc.nn.Sequential(
        tc.nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
            stride, padding, output_padding,
            groups=(in_channels if depthwise else 1), bias=False),
        tc.nn.BatchNorm2d(out_channels),
        tc.nn.PReLU(out_channels) if nonlinear else tc.nn.Identity())


class BottleneckT(tc.nn.Module):
    '''in_channels -> in_channels*expansion -> out_channels'''

    def __init__(self, in_channels, out_channels, expansion, stride=1):
        super().__init__()
        self.shortcut = stride == 1 and in_channels == out_channels
        exp_channels = in_channels * expansion
        self.conv = tc.nn.Sequential(
            conv_blockT(in_channels, exp_channels, kernel_size=1),
            conv_blockT(exp_channels, exp_channels, kernel_size=3,
                stride=stride, padding=1, output_padding=stride-1, depthwise=True),
            conv_blockT(exp_channels, out_channels, kernel_size=1, nonlinear=False))

    def forward(self, x):
        return self.conv(x) + x if self.shortcut else self.conv(x)
