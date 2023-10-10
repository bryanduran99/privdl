'''Restore obufuscated faeture map based row/column-wise simliraty,
攻击阶段，通过离线存储 first-epoch的feature map供后续使用，来加速训练过程'''

from distutils.command.build_scripts import first_line_re
import os
import time 
import numpy as np
import torch as tc
import cv2

import exp_utils
exp_utils.setup_import_path()
import block
import utils
from einops import rearrange
import PIL
import matplotlib.pyplot as plt

def recover_feats(feats):
    B, C, _, _ = feats.shape # shape of feats : B C H W 
    for i in range(B): # 82s
        for j in range(C): # 0.64s
            feats[i][j] = recover_rows(feats[i][j]) # 2e-3s
            feats[i][j] = recover_rows(rearrange(feats[i][j], 'h w -> w h')) # 2e-3s
            feats[i][j] = rearrange(feats[i][j].clone(), 'w h -> h w') # 8e-5s
    return feats

def recover_rows(x):
    # l2 distance given two rows
    H = x.shape[0]
    s = tc.linalg.norm(x[:, None]-x, ord=2, dim=2)
 
    remain = list(range(1, H))
    cur = 0
    rst = x[cur].unsqueeze(0)

    while len(remain):
        next = min(remain, key=lambda i: s[cur, i])
        # concatenates a sequence of tensors along a new dimension
        next_tensor = x[next].unsqueeze(0)
        rst = tc.cat((rst, next_tensor), dim=0) 
        remain.remove(next)
        cur = next

    return rst

class MobileFaceNetLit(block.model.light.ModelLight):
    '''MobileFaceNet + AMSoftmax + cross_entropy'''

    def __init__(self, class_num):
        super().__init__()
        self.model = block.model.mobile_face_net.MobileFaceNet()
        self.softmax = block.loss.amsoftmax.AMSoftmax(
            in_features=self.model.feat_size,
            class_num=class_num)
        self.config_optim()

    def config_optim(self):
        ml_params = utils.chain(
            self.softmax.parameters(), self.model.ml_params())
        prelu_params = self.model.prelu_params()
        base_params = self.model.base_params()

        param_groups = [
            dict(params=base_params, weight_decay=4e-5),
            dict(params=prelu_params, weight_decay=0),
            dict(params=ml_params, weight_decay=4e-4)]

        self.optimizer = tc.optim.SGD(param_groups,
            lr=0.1, momentum=0.9, nesterov=True)
        self.lr_scheduler = tc.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[6, 7, 8], gamma=0.1)

        self.monitor.add('lr', # 监控学习率
            get=lambda: self.optimizer.param_groups[0]['lr'],
            to_str=lambda x: f'{x:.1e}')

    def sample_transform(self, sample):
        img, label = sample
        img = self.model.img_transform(img)
        return img, label

    def forward(self, imgs, labels):
        feats = self.model(imgs)
        scores = self.softmax(feats, labels)
        return feats, scores

    def train_step(self, batch_data):
        imgs, labels = batch_data
        feats, scores = self(imgs, labels)
        loss = tc.nn.functional.cross_entropy(scores, labels)
        utils.step(self.optimizer, loss)

        self.monitor.add('loss', lambda: float(loss),
            to_str=lambda x: f'{x:.2e}') # 监控 loss
        self.monitor.add('batch_acc', lambda: utils.accuracy(scores, labels),
            to_str=lambda x: f'{x*100:.2f}%') # 监控 batch 准确率

    def on_epoch_end(self):
        self.lr_scheduler.step()

    def inference(self, batch_input):
        return self.model(batch_input)


class XNN_Single_Lit(block.model.light.ModelLight):
    '''单客户场景下的 XNN'''

    def __init__(self, xnn_parts, class_num):
        super().__init__()
        self.extractor = xnn_parts.extractor()
        self.obfuscate = xnn_parts.obfuscate()
        self.tail = xnn_parts.tail()
        self.softmax = block.loss.amsoftmax.AMSoftmax(
            in_features=self.tail.feat_size, class_num=class_num)
        self.config_optim()

    def config_optim(self):
        ml_params = utils.chain(
            self.softmax.parameters(), self.tail.ml_params())
        prelu_params = self.tail.prelu_params()
        base_params = self.tail.base_params()

        param_groups = [ # 每类参数设置不同的 weight_decay
            dict(params=base_params, weight_decay=4e-5),
            dict(params=prelu_params, weight_decay=0),
            dict(params=ml_params, weight_decay=4e-4)]
        self.optimizer = tc.optim.AdamW(param_groups, lr=0.1)

        self.monitor.add('lr', # 监控学习率
            get=lambda: self.optimizer.param_groups[0]['lr'],
            to_str=lambda x: f'{x:.1e}')

    def sample_transform(self, sample):
        img, label = sample
        img = self.extractor.img_transform(img)
        return img, label

    def train(self):
        self.extractor.eval()
        self.obfuscate.eval()
        self.tail.train()
        self.softmax.train()

    def eval(self):
        self.extractor.eval()
        self.obfuscate.eval()
        self.tail.eval()
        self.softmax.eval()

    def forward(self, mid_feats, labels):
        rec_feats = self.tail(mid_feats)
        return self.softmax(rec_feats, labels)

    def train_step(self, batch_data):
        imgs, labels = batch_data
        with tc.no_grad():
            mid_feats = self.extractor(imgs)
            mid_feats = self.obfuscate(mid_feats)
        scores = self(mid_feats, labels)
        loss = tc.nn.functional.cross_entropy(scores, labels)
        utils.step(self.optimizer, loss)

        self.monitor.add('loss', lambda: float(loss),
            to_str=lambda x: f'{x:.2e}') # 监控 loss
        self.monitor.add('batch_acc', lambda: utils.accuracy(scores, labels),
            to_str=lambda x: f'{x*100:.2f}%') # 监控 batch 准确率

    def inference(self, batch_input):
        mid_feats = self.extractor(batch_input)
        mid_feats = self.obfuscate(mid_feats)
        return self.tail(mid_feats)


class ERN_lit(XNN_Single_Lit):
    '''期望识别模型，强行在期望混淆后的数据上训练识别模型'''
    def __init__(self, xnn_parts, class_num):
        super().__init__(xnn_parts, class_num)
        self.extractor = xnn_parts.extractor()
        self.obfuscate = xnn_parts.obfuscate()
        self.tail = xnn_parts.tail()
        self.softmax = block.loss.amsoftmax.AMSoftmax(
            in_features=self.tail.feat_size, class_num=class_num)
        self.config_optim()
        self.feats_count = 0
        # self.train_feats_dir = '/home/liukaixin/privdl/privdl/data/feats/028_attack_train/'
        self.head_layers = xnn_parts.get_head_layers() ###
        self.train_feats_dir = '/data/feats/028_attack_train_{}/'.format(self.head_layers)
        if not os.path.exists(self.train_feats_dir):
            os.makedirs(self.train_feats_dir)

    def train_step(self, batch_data):
        imgs, labels = batch_data
        with tc.no_grad():
            mid_feats1 = self.extractor(imgs)
            mid_feats2 = self.obfuscate(mid_feats1, self.feats_count)
            mid_feats = recover_feats(mid_feats2.cpu()) ### restore feature map
        if self.feats_count < 1:
            tc.save((mid_feats1.cpu(), labels.cpu()), self.train_feats_dir + f'raw_feats_{self.feats_count}.pth')
            tc.save((mid_feats2.cpu(), labels.cpu()), self.train_feats_dir + f'permuted_feats_{self.feats_count}.pth')
        tc.save((mid_feats, labels.cpu()), self.train_feats_dir + f'rec_feats_{self.feats_count}.pth')
        self.feats_count += 1
        
        # from mid features
        scores = self(mid_feats.cuda(), labels)
        loss = tc.nn.functional.cross_entropy(scores, labels)
        utils.step(self.optimizer, loss)

        self.monitor.add('loss', lambda: float(loss),
            to_str=lambda x: f'{x:.2e}') # 监控 loss
        self.monitor.add('batch_acc', lambda: utils.accuracy(scores, labels),
            to_str=lambda x: f'{x*100:.2f}%') # 监控 batch 准确率
    
    def train_step_on_feats(self, batch_data):
        mid_feats, labels = batch_data
        
        # from mid features
        scores = self(mid_feats, labels)
        loss = tc.nn.functional.cross_entropy(scores, labels)
        utils.step(self.optimizer, loss)

        self.monitor.add('loss', lambda: float(loss),
            to_str=lambda x: f'{x:.2e}') # 监控 loss
        self.monitor.add('batch_acc', lambda: utils.accuracy(scores, labels),
            to_str=lambda x: f'{x*100:.2f}%') # 监控 batch 准确率
    
    def inference(self, batch_input):
        mid_feats = self.extractor(batch_input)
        mid_feats = self.obfuscate(mid_feats, self.feats_count)
        mid_feats = recover_feats(mid_feats.cpu()) ###
        self.feats_count += 1
        return self.tail(mid_feats.cuda())


class INN_Lit(block.model.light.ModelLight):
    '''XNN 逆重建模型'''

    def __init__(self, xnn_parts, pretrain_ckpt):
        super().__init__()
        self.extractor = xnn_parts.extractor()
        self.obfuscate = xnn_parts.obfuscate()
        self.inn = xnn_parts.inn()
        self.config_optim()
        self.config_recognition_model(pretrain_ckpt)

    def config_optim(self):
        self.optimizer = tc.optim.AdamW(self.inn.parameters())
        self.monitor.add('lr', # 监控学习率
            get=lambda: self.optimizer.param_groups[0]['lr'],
            to_str=lambda x: f'{x:.1e}')

    def config_recognition_model(self, pretrain_ckpt):
        print('load recognition model...')
        model = block.model.mobile_face_net.MobileFaceNet()
        params = utils.torch_load(pretrain_ckpt)
        params = utils.sub_state_dict(params, 'model.')
        model.load_state_dict(params, strict=True)
        self.recognition_model = model

    def sample_transform(self, sample):
        img, label = sample
        img = self.extractor.img_transform(img)
        return img, label

    def train(self):
        self.extractor.eval()
        self.obfuscate.eval()
        self.inn.train()

    def eval(self):
        self.extractor.eval()
        self.obfuscate.eval()
        self.inn.eval()

    def forward(self, mid_feats):
        return self.inn(mid_feats)

    def train_step(self, batch_data):
        imgs, labels = batch_data
        with tc.no_grad():
            mid_feats = self.extractor(imgs)
            self.obfuscate.reset_parameters() # 期望攻击
            mid_feats = self.obfuscate(mid_feats)
        restores = self(mid_feats)
        loss = block.loss.ssim.neg_ssim(restores, imgs)
        utils.step(self.optimizer, loss)
        self.monitor.add('loss', lambda: float(loss),
            to_str=lambda x: f'{x:.2e}') # 监控 loss

    def inference(self, batch_input):
        mid_feats = self.extractor(batch_input)
        mid_feats = self.obfuscate(mid_feats)
        restores = self(mid_feats)
        Min, Max = restores.min(), restores.max()
        restores = (restores - Min) / (Max - Min)
        restores = restores*2 - 1
        return self.recognition_model(restores)


class XnnParts_template:

    def __init__(self, pretrain_ckpt):
        self.pretrain_ckpt = pretrain_ckpt

    def extractor(self):
        print('load extractor...')
        model = block.model.mobile_face_net.MobileFaceNetHead(layer_num=2)
        params = utils.torch_load(self.pretrain_ckpt)
        params = utils.sub_state_dict(params, 'model.')
        model.load_state_dict(params, strict=False)
        return model

    def obfuscate(self):
        return tc.nn.Conv2d(64, 64, 1, 1, 0, bias=False)

    def tail(self):
        return block.model.mobile_face_net.MobileFaceNetTail(layer_num=-2)

    def inn(self):
        return block.model.inv_net.InvNet_2()


class XnnParts_mat_full:
    '''CHW 维度的矩阵混淆'''

    def __init__(self):
        pass

    def extractor(self):
        return block.model.mobile_face_net.MobileFaceNetHead(layer_num=0)

    def obfuscate(self):
        class Obfuscate(tc.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim
                self.mat = tc.nn.Linear(dim, dim, bias=False)
            def forward(self, imgs):
                img_vectors = imgs.view(-1, self.dim)
                img_vectors = self.mat(img_vectors)
                return img_vectors.view(imgs.shape)
            def reset_parameters(self):
                self.mat.reset_parameters()
        return Obfuscate(3 * 112 * 112)

    def tail(self):
        return block.model.mobile_face_net.MobileFaceNet()

    def inn(self):
        return block.model.inv_net.InvNet_2()


class XnnParts_mat:
    '''HW 维度的矩阵混淆'''

    def __init__(self):
        pass

    def extractor(self):
        return block.model.mobile_face_net.MobileFaceNetHead(layer_num=0)

    def obfuscate(self):
        class Obfuscate(tc.nn.Module):
            def __init__(self, height, width):
                super().__init__()
                self.dim = dim = height * width
                self.mat = tc.nn.Linear(dim, dim, bias=False)
            def forward(self, imgs):
                img_vectors = imgs.view(-1, self.dim)
                img_vectors = self.mat(img_vectors)
                return img_vectors.view(imgs.shape)
            def reset_parameters(self):
                self.mat.reset_parameters()
        return Obfuscate(112, 112)

    def tail(self):
        return block.model.mobile_face_net.MobileFaceNet()

    def inn(self):
        return block.model.inv_net.InvNet_2()


class XnnParts_img_conv:

    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def extractor(self):
        return block.model.mobile_face_net.MobileFaceNetHead(layer_num=0)

    def obfuscate(self):
        padding = (self.kernel_size - 1) // 2
        return tc.nn.Conv2d(3, 3, self.kernel_size, 1, padding, bias=False)

    def tail(self):
        return block.model.mobile_face_net.MobileFaceNet()

    def inn(self):
        return block.model.inv_net.InvNet_2()


class XnnParts_mid_conv:

    def __init__(self, pretrain_ckpt, head_layers, mid_layers, tail_extend_layers=0):
        self.pretrain_ckpt = pretrain_ckpt
        self.head_layers = head_layers
        self.mid_layers = mid_layers
        self.tail_extend_layers = tail_extend_layers

    def extractor(self):
        print('load pretrain_ckpt for XnnParts_mid_conv...')
        head = block.model.mobile_face_net.MobileFaceNetHead(
            layer_num=self.head_layers)
        params = utils.torch_load(self.pretrain_ckpt)
        params = utils.sub_state_dict(params, 'model.')
        head.load_state_dict(params, strict=False)
        return head

    def obfuscate(self):

        class Obfuscate(tc.nn.Module):
            def __init__(self, channels, mid_layers):
                super().__init__()
                layers = []
                for _ in range(mid_layers):
                    layers.append(tc.nn.Conv2d(channels, channels, 1, 1, 0, bias=False))
                    layers.append(tc.nn.LeakyReLU(negative_slope=0.25))
                self.conv = tc.nn.Sequential(*layers)
            def forward(self, x):
                return self.conv(x)
            def reset_parameters(self):
                for m in self.conv.modules():
                    if isinstance(m, tc.nn.Conv2d):
                        m.reset_parameters()

        channels = 64 if self.head_layers < 8 else 128
        return Obfuscate(channels, self.mid_layers)

    def tail(self):
        return block.model.mobile_face_net.MobileFaceNetTail(
            layer_num=-self.head_layers,
            extend_num=self.tail_extend_layers)

    def inn(self):
        return block.model.inv_net.InvNet_2()

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

class XnnParts_mid_mat:
    def __init__(self, pretrain_ckpt, head_layers, mat_type, tail_extend_layers=0):
        self.pretrain_ckpt = pretrain_ckpt
        self.head_layers = head_layers
        self.mat_type = mat_type
        self.tail_extend_layers = tail_extend_layers
        
    def extractor(self):
        class identity_layer(tc.nn.Identity):
            def __init__(self):
                super().__init__()
            def img_transform(self, img):
                '''H*W*3*uint8(0\~255) -> 3*112*112*float32(-1\~1)'''
                return resize_norm_transpose(img, size=112)
        
        if self.head_layers == 0:
            return identity_layer()
        print('load pretrain_ckpt for XnnParts_mid_conv...')
        head = block.model.mobile_face_net.MobileFaceNetHead(
            layer_num=self.head_layers)
        params = utils.torch_load(self.pretrain_ckpt)
        params = utils.sub_state_dict(params, 'model.')
        head.load_state_dict(params, strict=False)
        return head

    def obfuscate(self):

        class FullMatObf(tc.nn.Module):
            def __init__(self, feat_shape):
                super().__init__()
                C, H, W = feat_shape
                self.dim = C * H * W
                self.mat = tc.nn.Linear(self.dim, self.dim, bias=False)
            def forward(self, x):
                x_vectors = x.view(-1, self.dim)
                x_vectors = self.mat(x_vectors)
                return x_vectors.view(x.shape)
            def reset_parameters(self):
                self.mat.reset_parameters()

        class PermuteMatObf(tc.nn.Module):
            def __init__(self, feat_shape):
                super().__init__()
                self.feat_shape = feat_shape
                C, H, W = feat_shape
                self.h_indexes = tc.randperm(H)
                self.w_indexes = tc.randperm(W)
                self.c_indexes = tc.randperm(C)
            def forward(self, fmap, feats_count):
                fmap = fmap[:, :, :, self.w_indexes]
                fmap = fmap[:, :, self.h_indexes, :]
                if feats_count > 0: # 第一个batch不打乱通道，可视化时方便对齐
                    fmap = fmap[:, self.c_indexes, :, :]
                return fmap  
            def reset_parameters(self):
                C, H, W = self.feat_shape
                self.h_indexes = tc.randperm(H)
                self.w_indexes = tc.randperm(W)
                self.c_indexes = tc.randperm(C)

        class NonLinearObf(tc.nn.Module):
            def __init__(self, feat_shape):
                super().__init__()
                C, H, W = feat_shape
                self.dim = C * H * W
                self.mat = tc.nn.Linear(self.dim, self.dim, bias=False)
                self.m = tc.nn.LeakyReLU(0.1)

            def forward(self, x, feats_count):
                x_vectors = x.view(-1, self.dim)
                x_vectors = self.mat(x_vectors)
                x_vectors = self.m(x_vectors)
                return x_vectors.view(x.shape)
            def reset_parameters(self):
                self.mat.reset_parameters()

        head_layers = self.head_layers
        if head_layers == 0:
            feat_shape = 3, 112, 112
        elif 1 <= head_layers <= 2:
            feat_shape = 64, 56, 56
        elif 3 <= head_layers <= 7:
            feat_shape = 64, 28, 28
        elif 8 <= head_layers <= 14:
            feat_shape = 128, 14, 14
        elif 15 <= head_layers <= 17:
            feat_shape = 128, 7, 7
        if self.mat_type == 'full':
            return FullMatObf(feat_shape)
        elif self.mat_type == 'permute':
            return PermuteMatObf(feat_shape)
        elif self.mat_type == 'nonlinear':
            return NonLinearObf(feat_shape)

    def tail(self):

        return block.model.mobile_face_net.MobileFaceNetTail(
            layer_num=-self.head_layers,
            extend_num=self.tail_extend_layers)

    def inn(self):
        return block.model.inv_net.InvNet_2()
    
    def get_head_layers(self):
        return self.head_layers


def split_dataset(dataset, debug=False):
    # id, img_per_id = 1000, 50
    id, img_per_id = 100, 10

    hubble_utils = block.dataset.hubble.hubble_utils
    trainset, testset_by_person = hubble_utils.split_dataset_by_person(
        dataset, test_id_num=id, test_img_per_id=img_per_id)
    trainset, testset_by_img = hubble_utils.split_dataset_by_img(
        trainset, test_id_num=id, test_img_per_id=img_per_id, train_scale=1.5)
    return trainset, testset_by_img, testset_by_person


def pretrain_model_test(pretrain_ckpt, testers, work_dir):
    '''无脱敏数据回流情况下，pretrain_model在客户数据上的准确性测试'''

    class PretrainModelLit(block.model.light.ModelLight):

        def __init__(self, pretrain_ckpt):
            super().__init__()
            model = block.model.mobile_face_net.MobileFaceNet()
            params = utils.torch_load(pretrain_ckpt)
            params = utils.sub_state_dict(params, 'model.')
            model.load_state_dict(params)
            self.model = model

        def sample_transform(self, sample):
            img, label = sample
            img = self.model.img_transform(img)
            return img, label

        def inference(self, batch_input):
            return self.model(batch_input)

    print('pretrain_model_test:')
    model_lit = PretrainModelLit(pretrain_ckpt).cuda()
    results = {tester.name: tester.test(model_lit) for tester in testers}
    utils.json_save(results, f'{work_dir}/pretrain_model_test.json')


def main():
    # 配置实验设置
    result_path = exp_utils.setup_result_path(__file__)
    pretrain_ckpt = f'{exp_utils.data_path()}/model/celeba_pretrain.tar'
    batch_size = 128
    num_workers = 16
    debug = utils.input_option('bebug or not', [True, False])
    gen_feats = utils.input_option('gen_feats or not', [True, False])

    obfuscation_type = 'mid_mat'
    # utils.input_option('obfuscation_type', ['no', 'mat', 'mat_full', 'img_conv', 'mid_conv', 'mid_mat'])
    # obfuscation_type = 'mid_conv'
    result_path = f'{result_path}/{obfuscation_type}'

    if obfuscation_type == 'no':
        pass
    elif obfuscation_type == 'mat':
        xnn_parts = XnnParts_mat()
    elif obfuscation_type == 'mat_full':
        xnn_parts = XnnParts_mat_full()
        batch_size = 56
    elif obfuscation_type == 'img_conv':
        obf_kernel_size = utils.input_option('obf_kernel_size',
            [3, 5, 7, 9, 11, 33, 55, 77, 99, 111])
        result_path = f'{result_path}/kernel_size_{obf_kernel_size}'
        xnn_parts = XnnParts_img_conv(obf_kernel_size)
    elif obfuscation_type == 'mid_conv':
        mid_layers = utils.input_option('mid_layers', list(range(1, 10)))
        head_layers = utils.input_option('head_layers', list(range(1, 18)))
        result_path = f'{result_path}/Head{head_layers}_Mid{mid_layers}'
        xnn_parts = XnnParts_mid_conv(pretrain_ckpt, head_layers, mid_layers, tail_extend_layers=head_layers)
    elif obfuscation_type == 'mid_mat':
        mat_type = 'permute'
        # utils.input_option('mat_type', ['full', 'permute'])
        head_layers = utils.input_option('head_layers', list(range(0, 18)))
        result_path = f'{result_path}/Head{head_layers}_Mat{mat_type}'
        xnn_parts_ERN = XnnParts_mid_mat(pretrain_ckpt, head_layers, mat_type, tail_extend_layers=head_layers)
    else:
        raise TypeError(f'obfuscation_type {obfuscation_type} not supported')

    set_info = 'setting: {debug}_{gen}_{head_layer}'.format(debug='debug' if debug else 'notdebug', \
        gen='genfeats' if gen_feats else 'notgen', head_layer=head_layers)
    print("\n|",set_info.center(50,"*"),"|\n") 

    work_dir = result_path
    # 配置数据集 和 相应的预训练模型
    t0 = time.time()
    client_dataset = block.dataset.hubble.xnn_paper.msra()
    trainset, testset_by_img, testset_by_person = split_dataset(client_dataset, debug=debug)
    attacker_dataset = block.dataset.hubble.xnn_paper.celeba()
    if debug: # debug
        id_list = attacker_dataset.class_ids()[:100]
        attacker_dataset = attacker_dataset.subset_by_class_id(id_list)
    print(f'time of prepare dataset:{time.time() - t0}')

    def attack_xnn(first_train): # 期望攻击
        model_lit = ERN_lit(xnn_parts_ERN, class_num=attacker_dataset.class_num())
        if not first_train:
            print(f'torch_load:{work_dir}/inn/model_lit.tar')
            model_lit.load_state_dict(tc.load(f'{work_dir}/inn/model_lit.tar'))
        trainer = block.train.standard.Trainer(
            dataset=attacker_dataset, total_epochs=100, work_dir=f'{work_dir}/inn')
        RIA = block.test.top1_test.RestoreIdentificationAccuracy(
            dataset=testset_by_person, name='RestoreIdentificationAccuracy')
        RIA.config_dataloader(batch_size=batch_size, num_workers=num_workers)
        trainer.config_dataloader(batch_size=batch_size, num_workers=num_workers)
        
        if debug:
            m = 10
        else:
            m = 60

        if first_train:
            trainer.config_logger(log_interval=2*60)
            trainer.config_tester([RIA], interval=m*60)
            model_lit = trainer.fit(model_lit)
        else:
            print('enter trainer')
            trainer.config_logger(log_interval=2*60)
            trainer.config_tester([RIA], interval=20*60)
            trainer.fit_on_feats(model_lit)
    
    if gen_feats:
        attack_xnn(first_train=True) # generate feats in 1st train
    attack_xnn(first_train=False) # train on feats
    
if __name__ == '__main__':
    main()