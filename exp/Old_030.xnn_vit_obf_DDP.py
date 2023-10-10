# 该版本代码第一部分跑完之后出现线程死锁问题
'''
note:
因为多卡训练传入fit的是model_lit.module(即raw_model)，而不是多线程模型model_lit,所以在fit中各线程并没有融合梯度，而是各自为战。
多卡训练结果不可信（比实际的ACC要低），单卡执行脚本的结果不受影响，仍然可信
'''
# -*- coding: UTF-8 -*-
import math
import random
import PIL.Image as Image
import numpy as np
import torch as tc
import exp_utils
exp_utils.setup_import_path()

# from simpleViT import SimpleViT
from block.model.simpleViT import SimpleViT
from refile import smart_open
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

    def __init__(self, xnn_parts, class_num, c, h, w):
        super().__init__()
        self.extractor = xnn_parts.extractor()
        self.obfuscate = xnn_parts.obfuscate()
        self.tail = xnn_parts.tail()
        self.softmax = block.loss.amsoftmax.AMSoftmax(
            in_features=self.tail.feat_size, class_num=class_num)
        self.config_optim()
        self.c = c
        self.w = w
        self.h = h

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
            mid_feats = mid_feats.reshape(mid_feats.shape[0], self.c, self.w, self.h)
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
        mid_feats = mid_feats.reshape(mid_feats.shape[0], self.c, self.w, self.h)
        mid_feats = self.obfuscate(mid_feats)
        return self.tail(mid_feats)
    
    def get_tail(self):
        return self.tail

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

            def __init__(self, patch_size, img_size, obf):
                super().__init__()
                self.patch_size = patch_size
                self.img_size = img_size
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
                self.wb = int(self.img_size / self.patch_size)
                self.hb = int(self.img_size / self.patch_size)
                self.K1 = random.randint(0, math.factorial(self.wb * self.hb))
                self.K2 = random.randint(0, math.factorial(self.patch_size * self.patch_size))
                self.K3 = random.randint(0, math.factorial(self.patch_size * self.patch_size))
                self.V1 = self.key_to_permutation(self.K1, self.wb * self.hb)
                self.V2 = self.key_to_permutation(self.K2, self.patch_size * self.patch_size)
                self.V3 = self.key_to_permutation(self.K3, self.patch_size * self.patch_size)
                self.V3 = self.V3[0: int(len(self.V3) / 2) + 1]

                self.x = np.random.randn(self.patch_size * self.patch_size, self.patch_size * self.patch_size)
                U, s, Vh = linalg.svd(self.x)
                CN = s[0]/s[-1]
                lg = math.log2(CN)
                self.lg = lg
                self.x = tc.FloatTensor(self.x)
                self.x /= self.patch_size
                
            def reset_parameters_identity(self):
                self.wb = int(self.img_size / self.patch_size)
                self.hb = int(self.img_size / self.patch_size)
                self.K1 = 0
                self.K2 = 0
                self.K3 = 0
                self.V1 = self.key_to_permutation(self.K1, self.wb * self.hb)
                self.V2 = self.key_to_permutation(self.K2, self.patch_size * self.patch_size)
                self.V3 = self.key_to_permutation(self.K3, self.patch_size * self.patch_size)
                self.V3 = self.V3[0: int(len(self.V3) / 2) + 1]

                self.x = np.random.randn(self.patch_size * self.patch_size, self.patch_size * self.patch_size)
                U, s, Vh = linalg.svd(self.x)
                CN = s[0]/s[-1]
                lg = math.log2(CN)
                self.lg = lg
                self.x = tc.FloatTensor(self.x)
                self.x /= self.patch_size

            def get_lg(self):
                return self.lg

            def forward(self, img):
                if self.obf:
                    self.x = self.x.to(img.device)
                    tmp_img = img.clone()
                    for w in range(self.wb):
                        for h in range(self.hb):
                            new_w = self.V1[w * self.hb + h] // self.hb
                            new_h = self.V1[w * self.hb + h] % self.hb
                            img[:, :, self.patch_size * w: self.patch_size * (w + 1),
                            self.patch_size * h: self.patch_size * (h + 1)] \
                                = tmp_img[:, :, self.patch_size * new_w: self.patch_size * (new_w + 1),
                                  self.patch_size * new_h: self.patch_size * (new_h + 1)]
                    for w in range(self.wb):
                        for h in range(self.hb):
                            t = img[:,:,w*self.patch_size:(w+1)*self.patch_size,h*self.patch_size:(h+1)*self.patch_size]
                            t = t.reshape(t.shape[0], t.shape[1], 1, self.patch_size * self.patch_size).matmul(self.x).reshape(
                                t.shape[0], t.shape[1], self.patch_size, self.patch_size)
                            img[:,:,w*self.patch_size:(w+1)*self.patch_size,h*self.patch_size:(h+1)*self.patch_size] = t
                    
                

                return img
        self.obf = Obfuscate(16, 128, self.isobf)
        if self.is_identity:
            self.obf.reset_parameters_identity()

    def extractor(self):
        
        model = self.ext
        model.mode_feature(3)
        return model


    def obfuscate(self):
        return self.obf

    def tail(self):
        return SimpleViT(
            image_size=128,
            patch_size=16,
            dim=512,
            depth=3,
            heads=16,
            mlp_dim=2048,
            channels=2
        )

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
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--train_dataset", default='celeba', type=str)
    parser.add_argument("--client_dataset", default='celeba', type=str)
    parser.add_argument("--person", default=1000, type=int)
    FLAGS = parser.parse_args()
    local_rank = FLAGS.local_rank
    dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端
    
    
#     train_dataset = utils.input_option(
#         'data_used', ['celeba', 'imdb', 'facescrub', 'msra', 'webface'])
    
#     client_dataset = utils.input_option(
#         'data_used', ['celeba', 'imdb', 'facescrub', 'msra', 'webface'])
    train_dataset = FLAGS.train_dataset
    client_dataset = FLAGS.client_dataset
    
    print("train_dataset:", train_dataset)
    print("client_dataset:", client_dataset)
    
    result_path = exp_utils.setup_result_path(__file__)
    batch_size = 128
    num_workers = 8
    result_path = f'{result_path}/{train_dataset}/{client_dataset}'
    train_epoch = 1 #####
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
        model_lit = XNN_Single_Lit(xnn_parts, class_num=trainset.class_num(), c=3, w=112, h=112)
        
        model_lit = model_lit.to(local_rank)
        model_lit = DDP(model_lit, device_ids=[local_rank], output_device=local_rank)
        model_lit = model_lit.module
        trainer = block.train.standard.Trainer(
            dataset=trainset, total_epochs=train_epoch, work_dir=f'{work_dir}/train_ext/xnn')
        trainer.config_dataloader(batch_size=batch_size, num_workers=num_workers)
        trainer.config_logger(log_interval=60)
        trainer.config_tester(testers, interval=10 * 60)
        trainer.fit_ViT(model_lit, local_rank=local_rank)
        return model_lit    
    ext = train_vit_ext()
    

    vit = XnnParts_ViT(True, ext.get_tail())
    trainset, testset_by_img, testset_by_person = split_dataset(client_dataset, FLAGS.person)
    # 配置测试器
    Tester = block.test.top1_test.Top1_Tester
    testers = [Tester(dataset=testset_by_img, name='testset_by_img'),
               Tester(dataset=testset_by_person, name='testset_by_person')]
    def train_xnn():  # 脱敏回流训练
        model_lit = XNN_Single_Lit(vit, class_num=trainset.class_num(), c=2, w=128, h=128)
        model_lit = model_lit.to(local_rank)
        model_lit = DDP(model_lit, device_ids=[local_rank], output_device=local_rank)
        model_lit = model_lit.module
        trainer = block.train.standard.Trainer(
            dataset=trainset, total_epochs=train_epoch, work_dir=f'{work_dir}/test_ext/obf')
        trainer.config_dataloader(batch_size=batch_size, num_workers=num_workers)
        trainer.config_logger(log_interval=60)
        trainer.config_tester(testers, interval=30 * 60)
        trainer.fit_ViT(model_lit, local_rank=local_rank) 
    train_xnn()
    

    vit_identity = XnnParts_ViT(True, ext.get_tail(), is_identity=True)
    def train_xnn_identity():  # 脱敏回流训练
        model_lit = XNN_Single_Lit(vit_identity, class_num=trainset.class_num(), c=2, w=128, h=128)
        
        model_lit = model_lit.to(local_rank)
        model_lit = DDP(model_lit, device_ids=[local_rank], output_device=local_rank)
        model_lit = model_lit.module
        trainer = block.train.standard.Trainer(
            dataset=trainset, total_epochs=train_epoch, work_dir=f'{work_dir}/test_ext/identity')
        trainer.config_dataloader(batch_size=batch_size, num_workers=num_workers)
        trainer.config_logger(log_interval=60)
        trainer.config_tester(testers, interval=30 * 60)
        trainer.fit_ViT(model_lit, local_rank=local_rank) 
    train_xnn_identity()

if __name__ == '__main__':
    main()
