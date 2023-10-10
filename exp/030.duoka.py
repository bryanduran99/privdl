# -*- coding: UTF-8 -*-
import math
import os
import random
import PIL.Image as Image
import numpy as np
import torch as tc
from simpleViT import SimpleViT
import exp_utils
from refile import smart_open
exp_utils.setup_import_path()
import block
import utils

import time
import json
from matplotlib import pyplot as plt
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from scipy import linalg

# 初始化随机数种子，保证每次训练结果一致
def init_seed(seed=0, deter=False, bench=False):
    random.seed(seed)
    np.random.seed(seed)
    tc.manual_seed(seed)
    tc.cuda.manual_seed(seed)
    tc.cuda.manual_seed_all(seed)
    if deter:
        tc.backends.cudnn.deterministic = True
    if bench:
        tc.backends.cudnn.benchmark = True
init_seed(11)

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
        param_groups = [  # 每类参数设置不同的 weight_decay
            dict(params=self.tail.parameters(), weight_decay=4e-5),
            dict(params=self.softmax.parameters(), weight_decay=4e-4)]
        self.optimizer = tc.optim.SGD(param_groups, lr=0.05)

        self.monitor.add('lr',  # 监控学习率
                         get=lambda: self.optimizer.param_groups[0]['lr'],
                         to_str=lambda x: f'{x:.1e}')

    def sample_transform(self, sample):
        img, label = sample
        img = block.model.mobile_face_net.MobileFaceNetHead(layer_num=0).img_transform(img)
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
            # mid_feats = mid_feats.reshape(mid_feats.shape[0], self.c, self.w, self.h)
            mid_feats = self.obfuscate(mid_feats)
        scores = self(mid_feats, labels)
        loss = tc.nn.functional.cross_entropy(scores, labels)
        utils.step(self.optimizer, loss)

        self.monitor.add('loss', lambda: float(loss),
                         to_str=lambda x: f'{x:.2e}')  # 监控 loss
        self.monitor.add('batch_acc', lambda: utils.accuracy(scores, labels),
                         to_str=lambda x: f'{x * 100:.2f}%')  # 监控 batch 准确率

    def inference(self, batch_input):
        mid_feats = self.extractor(batch_input)
        # mid_feats = mid_feats.reshape(mid_feats.shape[0], self.c, self.w, self.h)
        mid_feats = self.obfuscate(mid_feats)
        return self.tail(mid_feats)
    
    def get_tail(self):
        return self.tail
    
    def set_ext(self, ext):
        self.extractor.load_state_dict(ext, strict=False)

class ERN_lit(XNN_Single_Lit):
    '''期望识别模型，强行在期望混淆后的数据上训练识别模型'''

    def train_step(self, batch_data):
        self.obfuscate.reset_parameters()
        super().train_step(batch_data)


class XnnParts_ViT_train:

    def __init__(self):

        class Obfuscate(tc.nn.Module):

            def __init__(self):
                super().__init__()

            def reset_parameters(self):
                pass

            def forward(self, img):
                return img
        self.obf = Obfuscate()

    def extractor(self):
        head = block.model.mobile_face_net.MobileFaceNetHead(layer_num=0)
        return head

    def obfuscate(self):
        return self.obf

    def tail(self):
        return SimpleViT(
            image_size=112,
            patch_size=14,
            dim=512,
            depth=3,
            heads=16,
            mlp_dim=2048,
            channels=3
        )

    def inn(self):
        return block.model.inv_net.InvNet_2()

    
class XnnParts_ViT:

    def __init__(self, isobf, ext, is_identity=False):
        self.isobf = isobf
        self.ext = ext
        self.is_identity = is_identity

        class Obfuscate(tc.nn.Module):

            def __init__(self, patch_num, martix_size, obf):
                super().__init__()
                self.patch_num = patch_num
                self.martix_size = martix_size
                self.reset_parameters()
                self.obf = obf

            def key_to_permutation(self, key: int, n: int):
                fac = math.factorial(n)
                key %= fac
                v = []
                for i in range(n):
                    fac = math.factorial(n - i - 1)
                    loc = key // fac
                    key %= fac
                    count = 0
                    cur = -1
                    while True:
                        cur += 1
                        if cur not in v:
                            count += 1
                        if count == loc + 1:
                            break
                    v.append(cur)
                return v

            def reset_parameters(self):
                self.K1 = random.randint(0, math.factorial(self.patch_num))
                self.V1 = self.key_to_permutation(self.K1, self.patch_num)

                self.x = np.random.randn(self.martix_size, self.martix_size)
                self.x = tc.FloatTensor(self.x)
                self.x /= pow(self.martix_size, 0.5)
                
            def reset_parameters_identity(self):
                self.K1 = 0
                self.V1 = self.key_to_permutation(self.K1, self.patch_num)

                self.x = np.random.randn(self.martix_size, self.martix_size)
                self.x = tc.FloatTensor(self.x)
                self.x /= pow(self.martix_size, 0.5)

            def forward(self, img):
                if self.obf:
                    self.x = self.x.to(img.device)
                    tmp_img = img.clone()
                    for w in range(self.patch_num):
                        img[:, w, :] = tmp_img[:, self.V1[w], :]
                    
                    img = img.matmul(self.x)             
                
                return img
        self.obf = Obfuscate(64, 512, self.isobf)
        if self.is_identity:
            self.obf.reset_parameters_identity()

    def extractor(self):
        
        model = SimpleViT(
            image_size=112,
            patch_size=14,
            dim=512,
            depth=3,
            heads=16,
            mlp_dim=2048,
            channels=3
        )
        
        model.mode_feature(2)
        return model


    def obfuscate(self):
        return self.obf

    def tail(self):
        # input size = (64, 512)
        model = SimpleViT(
            image_size=128,
            patch_size=16,
            dim=512,
            depth=3,
            heads=16,
            mlp_dim=2048,
            channels=2
        )
        model.mode_no_embedding()
        return model

    def inn(self):
        return block.model.inv_net.InvNet_2()


def split_dataset(dataset, person_num):
    hubble_utils = block.dataset.hubble.hubble_utils
    trainset, testset_by_person = hubble_utils.split_dataset_by_person(
        dataset, test_id_num=person_num, test_img_per_id=20)
    trainset, testset_by_img = hubble_utils.split_dataset_by_img(
        trainset, test_id_num=person_num, test_img_per_id=20, train_scale=1.5)
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

def get_curve(path, type):
    with open(path, 'r') as file:
        data = json.load(file)
    x = range(len(data))
    y1 = []
    y2 = []
    y = []
    for i in data:
        if type == 1:
            y.append(i['monitor']['loss'])

        elif type == 2:
            y1.append(i['results']['testset_by_img']['rate'])
            y2.append(i['results']['testset_by_person']['rate'])
        else:
            y.append(i['results']['RestoreIdentificationAccuracy']['rate'])
    plt.figure(figsize=(20, 8), dpi=240)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.weight'] = 'bold'

    legend_font = {
        'family': 'serif',
        'weight': 'bold',
        'size': 20
    }
    if type == 1:
        plt.xlabel("Time", fontdict=legend_font)
        plt.ylabel("Loss", fontdict=legend_font)
        plt.title("Time-Loss", fontdict=legend_font)
        plt.grid(alpha=0.8)
        plt.plot(x, y, color='red')
        plt.legend(loc="upper right")
        plt.savefig(path + '.png')
    elif type == 2:
        plt.xlabel("Time", fontdict=legend_font)
        plt.ylabel("Accuracy", fontdict=legend_font)
        plt.title("Time-Accuracy", fontdict=legend_font)
        plt.grid(alpha=0.8)
        plt.plot(x, y1, color='red', label='testset_by_img')
        plt.plot(x, y2, color='blue', label='testset_by_person')
        plt.legend(loc="upper right")
        plt.savefig(path + '.png')
    else:
        plt.xlabel("Time", fontdict=legend_font)
        plt.ylabel("RestoreIdentificationAccuracy", fontdict=legend_font)
        plt.title("Time-RestoreIdentificationAccuracy", fontdict=legend_font)
        plt.grid(alpha=0.8)
        plt.plot(x, y, color='red')
        plt.legend(loc="upper right")
        plt.savefig(path + '.png')
        


def main():
    # 配置实验设置
    
    parser = argparse.ArgumentParser()

    # 1. 从外部读取local_rank
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--train_dataset", default='celeba', type=str)
    parser.add_argument("--client_dataset", default='celeba', type=str)
    parser.add_argument("--person", default=1000, type=int)
    parser.add_argument("--debug", default=False, type=bool)
    FLAGS = parser.parse_args()
    
    local_rank = FLAGS.local_rank

    # 2. 配置后端初始化对线程
    proc_count = 1
    if local_rank != -1:
        dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端
        proc_count = tc.distributed.get_world_size() #获取全局并行数，即并行训练的显卡的数量
    
#     train_dataset = utils.input_option(
#         'data_used', ['celeba', 'imdb', 'facescrub', 'msra', 'webface'])
    
#     client_dataset = utils.input_option(
#         'data_used', ['celeba', 'imdb', 'facescrub', 'msra', 'webface'])
    train_dataset = FLAGS.train_dataset
    client_dataset = FLAGS.client_dataset
    
    print("train_dataset:", train_dataset)
    print("client_dataset:", client_dataset)
    
    result_path = exp_utils.setup_result_path(os.path.basename(__file__))
    
    batch_size = 128 // proc_count
    num_workers = 4
    result_path = f'{result_path}/{train_dataset}/{client_dataset}/{proc_count}_gpus'
    train_epoch = 10
    work_dir = result_path
    
    if train_dataset == 'celeba':
        train_dataset = block.dataset.hubble.xnn_paper.celeba()
    elif train_dataset == 'imdb':
        train_dataset = block.dataset.hubble.xnn_paper.imdb()
    elif train_dataset == 'facescrub':
        train_dataset = block.dataset.hubble.xnn_paper.facescrub()
    elif train_dataset == 'msra':
        train_dataset = block.dataset.hubble.xnn_paper.msra()
    else:
        train_dataset = block.dataset.hubble.xnn_paper.webface()

    
    if client_dataset == 'celeba':
        client_dataset = block.dataset.hubble.xnn_paper.celeba()
    elif client_dataset == 'imdb':
        client_dataset = block.dataset.hubble.xnn_paper.imdb()
    elif client_dataset == 'facescrub':
        client_dataset = block.dataset.hubble.xnn_paper.facescrub()
    elif client_dataset == 'msra':
        client_dataset = block.dataset.hubble.xnn_paper.msra()
    else:
        client_dataset = block.dataset.hubble.xnn_paper.webface()
    

    xnn_parts = XnnParts_ViT_train()
    # 配置数据集 和 相应的预训练模型
    trainset, testset_by_img, testset_by_person = split_dataset(train_dataset, FLAGS.person)
    # 配置测试器
    Tester = block.test.top1_test.Top1_Tester
    testers = [Tester(dataset=testset_by_img, name='testset_by_img'),
               Tester(dataset=testset_by_person, name='testset_by_person')]
    for tester in testers:
        tester.config_dataloader(batch_size=batch_size, num_workers=num_workers)

    def train_vit_ext():  # 脱敏回流训练
        model_lit = XNN_Single_Lit(xnn_parts, class_num=trainset.class_num())
        # 3. 把模型放到各自线程所在的GPU上；
        model_lit = model_lit.to(local_rank)
        # 4. 将各GPU上的模型用DDP包裹，从此之后，调用model()就是用DDP模式的前向传播了，要使用原始的前向传播，需要model.module()
        '''
        very important: 
        - 只有训练集才要用Sampler！在train之后，经常使用验证集对数据集进行验证得到validation_loss，此时没有必要使用多卡，只需要在一个进程上进行验证。
        - 在多卡模式下要进行只在一个进程上的操作，通过model.module(inputs)而不是model(inputs)来调用forward()前向传播，而其他进程通过torch.distributed.barrier()来等待主进程完成validate操作.
        '''
        model_ddp = DDP(model_lit, device_ids=[local_rank], output_device=local_rank)
        model_lit = model_lit.module # raw model
        trainer = block.train.standard.Trainer(
            dataset=trainset, total_epochs=train_epoch, work_dir=f'{work_dir}/train_ext/')
        trainer.config_dataloader(batch_size=batch_size, num_workers=num_workers)
        trainer.config_logger(log_interval=60)
        # trainer.config_tester(testers, interval=10 * 60)
        model_ddp = trainer.fit_ViT(model_ddp, local_rank=local_rank)
        return model_ddp.module().get_tail().state_dict()
        
    ext_loc = train_vit_ext()
    
    vit = XnnParts_ViT(True, ext_loc)
    trainset, testset_by_img, testset_by_person = split_dataset(client_dataset, FLAGS.person)
    # 配置测试器
    Tester = block.test.top1_test.Top1_Tester
    testers = [Tester(dataset=testset_by_img, name='testset_by_img'),
               Tester(dataset=testset_by_person, name='testset_by_person')]
        
    def train_xnn():  # 脱敏回流训练
        model_lit = XNN_Single_Lit(vit, class_num=trainset.class_num())
        
        model_lit = model_lit.to(local_rank)
        model_lit = DDP(model_lit, device_ids=[local_rank], output_device=local_rank)
        model_lit = model_lit.module
        model_lit.set_ext(ext_loc)
        trainer = block.train.standard.Trainer(
            dataset=trainset, total_epochs=train_epoch, work_dir=f'{work_dir}/test_ext/obf')
        trainer.config_dataloader(batch_size=batch_size, num_workers=num_workers)
        trainer.config_logger(log_interval=60)
        # trainer.config_tester(testers, interval=30 * 60)
        trainer.fit_ViT(model_lit, local_rank=local_rank)
        
    train_xnn()
    
    vit_identity = XnnParts_ViT(True, ext_loc, is_identity=True)
    def train_xnn_identity():  # 脱敏回流训练
        model_lit = XNN_Single_Lit(vit_identity, class_num=trainset.class_num())
        
        model_lit = model_lit.to(local_rank)
        model_lit = DDP(model_lit, device_ids=[local_rank], output_device=local_rank)
        model_lit = model_lit.module # raw model
        model_lit.set_ext(ext_loc)
        trainer = block.train.standard.Trainer(
            dataset=trainset, total_epochs=train_epoch, work_dir=f'{work_dir}/test_ext/identity')
        trainer.config_dataloader(batch_size=batch_size, num_workers=num_workers)
        trainer.config_logger(log_interval=60)
        # trainer.config_tester(testers, interval=30 * 60)
        trainer.fit_ViT(model_lit, local_rank=local_rank) 
    
    train_xnn_identity()


if __name__ == '__main__':
    main()
