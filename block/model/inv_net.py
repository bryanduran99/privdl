from . import model_utils
import torch as tc


conv_block = model_utils.conv_block
Bottleneck = model_utils.Bottleneck
conv_blockT = model_utils.conv_blockT
BottleneckT = model_utils.BottleneckT


class InvNet(tc.nn.Module):
    '''N*3*112*112 -> N*3*112*112 \n
    N*64*56*56 -> N*3*112*112 \n
    N*64*28*28 -> N*3*112*112 \n
    N*128*14*14 -> N*3*112*112 \n
    N*128*7*7 -> N*3*112*112 \n
    '''

    def __init__(self):
        '''mobile face net 的逆向网络'''
        super().__init__()

        layers = [] # N*128*7*7
        layers += [BottleneckT(128, 128, 2) for _ in range(2)] # 2-layer
        layers += [BottleneckT(128, 128, 4, 2)] # N*128*14*14 # 3-layer
        layers += [BottleneckT(128, 128, 2) for _ in range(6)] # 9-layer
        layers += [BottleneckT(128, 64, 4, 2)] # N*64*28*28 # 10-layer
        layers += [BottleneckT(64, 64, 2) for _ in range(4)] # 14-layer
        layers += [BottleneckT(64, 64, 2, 2)] # N*64*56*56 # 15-lyer
        layers += [conv_blockT(64, 64, 3, 1, 1, depthwise=True)] # 16-layer
        layers += [conv_blockT(64, 3, 3, 2, 1, 1,
            nonlinear=False)] # N*3*112*112 # 17-layer
        self.conv7 = tc.nn.Sequential(*layers[:3])
        self.conv14 = tc.nn.Sequential(*layers[3:10])
        self.conv28 = tc.nn.Sequential(*layers[10:15])
        self.conv56 = tc.nn.Sequential(*layers[15:])

        self.down_sample = tc.nn.ModuleList([ # N*3*112*112
            conv_block(3, 64, 3, 2, 1), # N*64*56*56
            Bottleneck(64, 64, 2, 2), # N*64*28*28
            Bottleneck(64, 128, 4, 2), # N*128*14*14
            Bottleneck(128, 128, 4, 2)]) # N*128*7*7

        self.merge = tc.nn.ModuleList([
            conv_block(6, 3, 1, nonlinear=False),
            conv_block(128, 64, 1, nonlinear=False),
            conv_block(128, 64, 1, nonlinear=False),
            conv_block(256, 128, 1, nonlinear=False)])

    def forward(self, x):
        input_scale = x.shape[-1]
        if input_scale >= 112: # N*3*112*112
            x, x112 = self.down_sample[0](x), x
        if input_scale >= 56: # N*64*56*56
            x, x56 = self.down_sample[1](x), x
        if input_scale >= 28: # N*64*28*28
            x, x28 = self.down_sample[2](x), x
        if input_scale >= 14: # N*128*14*14
            x, x14 = self.down_sample[3](x), x
        # N*128*7*7
        x = self.conv7(x) # N*128*14*14
        if input_scale >= 14:
            x = tc.cat([x, x14], dim=1) # N*256*14*14
            x = self.merge[3](x) # N*128*14*14
        x = self.conv14(x) # N*64*28*28
        if input_scale >= 28:
            x = tc.cat([x, x28], dim=1) # N*128*28*28
            x = self.merge[2](x) # N*64*28*28
        x = self.conv28(x) # N*64*56*56
        if input_scale >= 56:
            x = tc.cat([x, x56], dim=1) # N*128*56*56
            x = self.merge[1](x) # N*64*56*56
        x = self.conv56(x) # N*3*112*112
        if input_scale >= 112:
            x = tc.cat([x, x112], dim=1) # N*6*112*112
            x = self.merge[0](x) # N*3*112*112
        return x


class InvNet_2(tc.nn.Module):
    '''N*3*112*112 -> N*3*112*112 \n
    N*64*56*56 -> N*3*112*112 \n
    N*64*28*28 -> N*3*112*112 \n
    N*128*14*14 -> N*3*112*112 \n
    N*128*7*7 -> N*3*112*112 \n
    '''

    def __init__(self):
        '''mobile face net 的逆向网络'''
        super().__init__()

        layers = [] # N*128*7*7
        layers += [BottleneckT(128, 128, 2) for _ in range(2)] # 2-layer
        layers += [BottleneckT(128, 128, 4, 2)] # N*128*14*14 # 3-layer
        layers += [BottleneckT(128, 128, 2) for _ in range(6)] # 9-layer
        layers += [BottleneckT(128, 64, 4, 2)] # N*64*28*28 # 10-layer
        layers += [BottleneckT(64, 64, 2) for _ in range(4)] # 14-layer
        layers += [BottleneckT(64, 64, 2, 2)] # N*64*56*56 # 15-lyer
        layers += [conv_blockT(64, 64, 3, 1, 1, depthwise=True)] # 16-layer
        layers += [conv_blockT(64, 3, 3, 2, 1, 1,
            nonlinear=False)] # N*3*112*112 # 17-layer
        self.conv7 = tc.nn.Sequential(*layers[:3])
        self.conv14 = tc.nn.Sequential(*layers[3:10])
        self.conv28 = tc.nn.Sequential(*layers[10:15])
        self.conv56 = tc.nn.Sequential(*layers[15:])

        self.down_sample = tc.nn.ModuleList([ # N*3*112*112
            conv_block(3, 64, 3, 2, 1), # N*64*56*56
            Bottleneck(64, 64, 2, 2), # N*64*28*28
            Bottleneck(64, 128, 4, 2), # N*128*14*14
            Bottleneck(128, 128, 4, 2)]) # N*128*7*7

        self.merge = tc.nn.ModuleList([
            conv_block(6, 3, 1, nonlinear=False),
            conv_block(128, 64, 1, nonlinear=False),
            conv_block(128, 64, 1, nonlinear=False),
            conv_block(256, 128, 1, nonlinear=False)])

    def forward(self, x):
        input_scale = x.shape[-1]
        if input_scale >= 112: # N*3*112*112
            x, x112 = self.down_sample[0](x), x
        if input_scale >= 56: # N*64*56*56
            x, x56 = self.down_sample[1](x), x
        if input_scale >= 28: # N*64*28*28
            x, x28 = self.down_sample[2](x), x
        if input_scale >= 14: # N*128*14*14
            x, x14 = self.down_sample[3](x), x
        # N*128*7*7
        x = self.conv7(x) # N*128*14*14
        if input_scale >= 14:
            x = tc.cat([x, x14], dim=1) # N*256*14*14
            x = self.merge[3](x) # N*128*14*14
        x = self.conv14(x) # N*64*28*28
        if input_scale >= 28:
            x = tc.cat([x, x28], dim=1) # N*128*28*28
            x = self.merge[2](x) # N*64*28*28
        x = self.conv28(x) # N*64*56*56
        if input_scale >= 56:
            x = tc.cat([x, x56], dim=1) # N*128*56*56
            x = self.merge[1](x) # N*64*56*56
        x = self.conv56(x) # N*3*112*112
        # if input_scale >= 112:
        #     x = tc.cat([x, x112], dim=1) # N*6*112*112
        #     x = self.merge[0](x) # N*3*112*112
        return x
