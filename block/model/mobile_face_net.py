from . import model_utils 
# import model_utils
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F
import torch as tc
from torchsummary import summary

conv_block = model_utils.conv_block
Bottleneck = model_utils.Bottleneck

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def l2_norm(input,axis=1):
    norm = tc.norm(input,2,axis,True)
    output = tc.div(input, norm)
    return output

class Conv_block(tc.nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class Linear_block(tc.nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Depth_Wise(tc.nn.Module):
     def __init__(self, in_c, out_c, residual = False, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
        super(Depth_Wise, self).__init__()
        self.conv = Conv_block(in_c, out_c=groups, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.conv_dw = Conv_block(groups, groups, groups=groups, kernel=kernel, padding=padding, stride=stride)
        self.project = Linear_block(groups, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
     def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class Residual(tc.nn.Module):
    def __init__(self, c, num_block, groups, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Residual, self).__init__()
        modules = []
        for _ in range(num_block):
            modules.append(Depth_Wise(c, c, residual=True, kernel=kernel, padding=padding, stride=stride, groups=groups))
        self.model = Sequential(*modules)
    def forward(self, x):
        return self.model(x)

class MobileFaceNet_SOTA(tc.nn.Module):
    def __init__(self, embedding_size):
        super(MobileFaceNet_SOTA, self).__init__()
        self.conv1 = Conv_block(3, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_dw = Conv_block(64, 64, kernel=(3, 3), stride=(1, 1), padding=(1, 1), groups=64)
        self.conv_23 = Depth_Wise(64, 64, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=128)
        self.conv_3 = Residual(64, num_block=4, groups=128, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_34 = Depth_Wise(64, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=256)
        self.conv_4 = Residual(128, num_block=6, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_45 = Depth_Wise(128, 128, kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=512)
        self.conv_5 = Residual(128, num_block=2, groups=256, kernel=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_6_sep = Conv_block(128, 512, kernel=(1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_6_dw = Linear_block(512, 512, groups=512, kernel=(7,7), stride=(1, 1), padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)

    def img_transform(self, img):
        '''H*W*3*uint8(0\~255) -> 3*112*112*float32(-1\~1)'''
        return model_utils.resize_norm_transpose(img, size=112)
    
    def forward(self, x):
        out = self.conv1(x)

        out = self.conv2_dw(out)

        out = self.conv_23(out)

        out = self.conv_3(out)
        
        out = self.conv_34(out)

        out = self.conv_4(out)
        return out

        # out = self.conv_45(out)

        # out = self.conv_5(out)

        # out = self.conv_6_sep(out)

        # out = self.conv_6_dw(out)

        # out = self.conv_6_flatten(out)

        # out = self.linear(out)

        # out = self.bn(out)
        # return l2_norm(out)

class MobileFaceNet(tc.nn.Module):
    '''N*3*112*112 -> N*128'''

    def __init__(self):
        super().__init__()
        self.feat_size = 128
        layers = [] # N*3*112*112
        layers += [conv_block(3, 64, 3, 2, 1)] # N*64*56*56 # 1-layer
        layers += [conv_block(64, 64, 3, 1, 1, depthwise=True)] # 2-layer
        layers += [Bottleneck(64, 64, 2, 2)] # N*64*28*28 # 3-layer
        layers += [Bottleneck(64, 64, 2) for _ in range(4)] # 7-layer
        layers += [Bottleneck(64, 128, 4, 2)] # N*128*14*14 # 8-layer
        layers += [Bottleneck(128, 128, 2) for _ in range(6)] # 14-layer
        layers += [Bottleneck(128, 128, 4, 2)] # N*128*7*7 # 15-layer
        layers += [Bottleneck(128, 128, 2) for _ in range(2)] # 17-layer
        layers += [conv_block(128, 512, 1)] # N*512*7*7 # 18-layer
        layers += [conv_block(512, 512, (7, 7),
            depthwise=True, nonlinear=False)] # N*512*1*1 # 19-layer
        layers += [conv_block(512, self.feat_size, 1,
            nonlinear=False)] # N*128*1*1 # 20-layer
        self.conv = tc.nn.Sequential(*layers)

    def forward(self, x): # N*3*112*112
        return self.conv(x).view(-1, self.feat_size) # N*128

    def img_transform(self, img):
        '''H*W*3*uint8(0\~255) -> 3*112*112*float32(-1\~1)'''
        return model_utils.resize_norm_transpose(img, size=112)

    def ml_params(self):
        '''metric learning parameters'''
        return self.conv[-1].parameters()

    def prelu_params(self):
        '''PReLU parameters'''
        ml_param_ids = set(map(id, self.ml_params()))
        for module in self.modules():
            if isinstance(module, tc.nn.PReLU):
                for param in module.parameters():
                    if id(param) not in ml_param_ids:
                        yield param

    def base_params(self):
        '''除 metric learning parameters 和 PReLU parameters 之外的 parameters'''
        ml_param_ids = set(map(id, self.ml_params()))
        prelu_params_ids = set(map(id, self.prelu_params()))
        non_base_ids = ml_param_ids | prelu_params_ids
        return (p for p in self.parameters() if id(p) not in non_base_ids)


class MobileFaceNetHead(MobileFaceNet):
    '''N*3*112*112 -> mid_feats'''

    def __init__(self, layer_num):
        '''MobileFaceNet 的 head 截取，layer_num 为正整数时，
        表示保留 MobileFaceNet 的前 layer_num 层；
        layer_num 为负整数时，表示删去 MobileFaceNet 的后 abs(layer_num) 层'''
        super().__init__()
        if layer_num < 0:
            layer_num = len(self.conv) + layer_num
            
        assert layer_num != len(self.conv), '请至少删去一层'
        # 请至少删去一层，否则 forward 和 ml_params 无法正常工作
        self.conv = self.conv[:layer_num]

    def forward(self, x):  # N*3*112*112
        return self.conv(x)

    def ml_params(self):
        return []


class MobileFaceNetTail(MobileFaceNet):
    '''mid_feats -> N*128'''

    def __init__(self, layer_num, extend_num=0):
        '''MobileFaceNet 的 tail 截取，layer_num 为正整数时，
        表示保留 MobileFaceNet 的后 layer_num 层；
        layer_num 为负整数时，表示删去 MobileFaceNet 的前 abs(layer_num) 层'''
        super().__init__()
        if layer_num < 0:
            layer_num = len(self.conv) + layer_num
        elif layer_num == 0:
            layer_num = len(self.conv)
        ml_layer_num = 3 # 度量学习部分的层数
        assert layer_num >= ml_layer_num # 度量学习的层必须保留

        self.feat_size = 128
        layers = [] # N*3*112*112
        layers += [conv_block(3, 64, 3, 2, 1)] # N*64*56*56 # 1-layer
        layers += [conv_block(64, 64, 3, 1, 1, depthwise=True)] # 2-layer
        layers += [Bottleneck(64, 64, 2, 2)] # N*64*28*28 # 3-layer
        layers += [Bottleneck(64, 64, 2) for _ in range(4)] # 7-layer
        layers += [Bottleneck(64, 128, 4, 2)] # N*128*14*14 # 8-layer
        layers += [Bottleneck(128, 128, 2) for _ in range(6)] # 14-layer
        layers += [Bottleneck(128, 128, 4, 2)] # N*128*7*7 # 15-layer
        layers += [Bottleneck(128, 128, 2) for _ in range(2)] # 17-layer
        layers += [conv_block(128, 512, 1)] # N*512*7*7 # 18-layer
        layers += [conv_block(512, 512, (7, 7),
            depthwise=True, nonlinear=False)] # N*512*1*1 # 19-layer
        layers += [conv_block(512, self.feat_size, 1,
            nonlinear=False)] # N*128*1*1 # 20-layer

        layers = layers[-layer_num:]
        if extend_num > 0:
            extend = [Bottleneck(128, 128, 2) for _ in range(extend_num)]
            layers = layers[:-ml_layer_num] + extend + layers[-ml_layer_num:]
        self.conv = tc.nn.Sequential(*layers)


class MobileFaceNetDeepTail(MobileFaceNetTail):
    '''mid_feats -> N*128'''

    def __init__(self, cut_layer_num, append_layer_num):
        '''MobileFaceNet 的 tail 截取，并在截取部分之前添加 Bottleneck，
        layer_num 为正整数时，表示保留 MobileFaceNet 的后 layer_num 层；
        layer_num 为负整数时，表示删去 MobileFaceNet 的前 abs(layer_num) 层；
        append_layer_num 表示添加的 Bottleneck 数量
        '''
        super().__init__(cut_layer_num)
        assert 3 <= len(self.conv) <= 12 # 其他长度的 cut 暂不支持
        self.bottlenecks = tc.nn.Sequential(*[
            Bottleneck(128, 128, 2) for _ in range(append_layer_num)])

    def forward(self, x): # mid_feats
        x = self.bottlenecks(x)
        return self.conv(x).view(-1, self.feat_size) # N*128
if __name__ == '__main__':
    net = MobileFaceNet()
    summary(net, input_size=(3, 112, 112), device="cpu")