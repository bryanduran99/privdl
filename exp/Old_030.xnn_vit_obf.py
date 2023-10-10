# 单卡版本
# -*- coding: UTF-8 -*-
import math
import os
import random
import PIL.Image as Image
import numpy as np
import torch as tc
import exp_utils
exp_utils.setup_import_path()

import block
from block.model.simpleViT import SimpleViT
from refile import smart_open
from torchvision import datasets, transforms

import utils
import time
import json
from scipy import linalg
from matplotlib import pyplot as plt

from torchsummary import summary


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

        self.monitor.add('lr',  # 监控学习率
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
                         to_str=lambda x: f'{x:.2e}')  # 监控 loss
        self.monitor.add('batch_acc', lambda: utils.accuracy(scores, labels),
                         to_str=lambda x: f'{x * 100:.2f}%')  # 监控 batch 准确率

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
        param_groups = [  # 每类参数设置不同的 weight_decay
            dict(params=self.tail.parameters(), weight_decay=4e-5),
            dict(params=self.softmax.parameters(), weight_decay=4e-4)]
        self.optimizer = tc.optim.SGD(param_groups, lr=0.05)

        self.monitor.add('lr',  # 监控学习率
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
            mid_feats = mid_feats.reshape(mid_feats.shape[0], 2, 112, 112)
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
        mid_feats = mid_feats.reshape(mid_feats.shape[0], 2, 112, 112)
        mid_feats = self.obfuscate(mid_feats)
        return self.tail(mid_feats)


class ERN_lit(XNN_Single_Lit):
    '''期望识别模型，强行在期望混淆后的数据上训练识别模型'''

    def train_step(self, batch_data):
        self.obfuscate.reset_parameters()
        super().train_step(batch_data)


class XnnParts_ViT:

    def __init__(self, isobf, depth_num, pretrain_ckpt='s3://xionghuixin/privdl/data/model/celeba_pretrain.tar', head_layers=10):
        self.pretrain_ckpt = pretrain_ckpt
        self.head_layers = head_layers
        self.isobf = isobf
        self.depth_num = depth_num

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
                self.mat = [[0] * self.patch_size] * self.patch_size
                for i in range(self.patch_size):
                    for j in range(self.patch_size):
                        if self.V2[i * self.patch_size + j] in self.V3:
                            self.mat[i][j] = 1
                        else:
                            self.mat[i][j] = -1
                self.mat = tc.tensor(self.mat).cuda()
                self.mat = self.mat.repeat(self.wb, self.hb)
                self.x = np.random.randn(self.patch_size * self.patch_size, self.patch_size * self.patch_size)
                U, s, Vh = linalg.svd(self.x)
                CN = s[0]/s[-1]
                lg = math.log2(CN)
                self.lg = lg
                self.x = tc.FloatTensor(self.x).cuda()

            def get_lg(self):
                return self.lg

            def forward(self, img):
                img /= 3
                
                if self.obf:
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
                            # print('t', t.shape)
                            img[:,:,w*self.patch_size:(w+1)*self.patch_size,h*self.patch_size:(h+1)*self.patch_size] = t
                return img
        self.obf = Obfuscate(14, 112, self.isobf)

    def extractor(self):
        
        model = block.model.mobile_face_net.MobileFaceNet_SOTA(embedding_size=512)
        param_path = f'{exp_utils.data_path()}/model/model_mobilefacenet.pth'
        params = utils.torch_load(param_path)
        model.load_state_dict(params)
        return model

    def obfuscate(self):
        return self.obf

    def tail(self):
        return SimpleViT(
            image_size=112,
            patch_size=14,
            dim=512,
            depth=self.depth_num,
            heads=16,
            mlp_dim=2048,
            channels=2
        )

    def inn(self):
        return block.model.inv_net.InvNet_2()


def split_dataset(dataset, img_per_id=50):
    hubble_utils = block.dataset.hubble.hubble_utils
    trainset, testset_by_person = hubble_utils.split_dataset_by_person(
        dataset, test_id_num=1000, test_img_per_id=img_per_id) # origin = 1000
    trainset, testset_by_img = hubble_utils.split_dataset_by_img(
        trainset, test_id_num=1000, test_img_per_id=img_per_id, train_scale=1.5)
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

    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(path, 'r') as file:
        data = json.load(file)
    x = range(len(data))
    y1 = []
    y2 = []
    y = []
    best_acc = 0
    for i in data:
        if type == 1:
            y.append(i['monitor']['loss'])

        elif type == 2:
            y1.append(i['results']['testset_by_img']['rate'])
            y2.append(i['results']['testset_by_person']['rate'])
            if best_acc < i['results']['testset_by_person']['rate']:
                best_acc = i['results']['testset_by_person']['rate']
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
        plt.plot(x, y1, color='red', label='testset_by_img=' + str(sum(y1[-10:])/10))
        plt.plot(x, y2, color='blue', label='testset_by_person=' + str(sum(y2[-10:])/10))
        plt.legend(loc="upper right")
        plt.savefig(path + '.png')
        best_acc = sum(y1[-10:])/10
        with open(path + '_' + str(best_acc), 'w') as f:
            f.write(str(best_acc))
            f.close()
        
    else:
        plt.xlabel("Time", fontdict=legend_font)
        plt.ylabel("RestoreIdentificationAccuracy", fontdict=legend_font)
        plt.title("Time-RestoreIdentificationAccuracy", fontdict=legend_font)
        plt.grid(alpha=0.8)
        plt.plot(x, y, color='red')
        plt.legend(loc="upper right")
        plt.savefig(path + '.png')
    return best_acc
        


def main():
    # 配置实验设置
    result_path = exp_utils.setup_result_path(__file__)
    batch_size = 128
    num_workers = 8
    depth_num = 3
    obfuscation_type = utils.input_option(
        'obfuscation_type', ['no', 'ViT_Obf'])
    result_path = f'{result_path}/{obfuscation_type}'
    train_epoch = 15
    attack_epoch = 15

    if obfuscation_type == 'no':
        xnn_parts = XnnParts_ViT(False, depth_num)
    elif obfuscation_type == 'ViT_Obf':
        xnn_parts = XnnParts_ViT(True, depth_num)
        # summary(xnn_parts.extractor(), input_size=(3, 112, 112), device="cpu")
    else:
        raise TypeError(f'obfuscation_type {obfuscation_type} not supported')
    work_dir = result_path
    
    # 配置数据集 和 相应的预训练模型
    dataset_type = utils.input_option(
        'datasets', ['msra', 'webface'])
    if dataset_type == 'msra':
        client_dataset = block.dataset.hubble.xnn_paper.msra()
        trainset, testset_by_img, testset_by_person = split_dataset(client_dataset)
    elif dataset_type == 'webface':
        client_dataset = block.dataset.hubble.xnn_paper.webface()
        trainset, testset_by_img, testset_by_person = split_dataset(client_dataset,img_per_id=40)
        train_epoch = 200
    # 配置测试器
    Tester = block.test.top1_test.Top1_Tester
    testers = [Tester(dataset=testset_by_img, name='testset_by_img'),
               Tester(dataset=testset_by_person, name='testset_by_person')]
    for tester in testers:
        tester.config_dataloader(batch_size=batch_size, num_workers=num_workers)

    def train_xnn():  # 脱敏回流训练
        model_lit = XNN_Single_Lit(xnn_parts, class_num=trainset.class_num())
        trainer = block.train.standard.Trainer(
            dataset=trainset, total_epochs=train_epoch, work_dir=f'{work_dir}/xnn/{dataset_type}')
        trainer.config_dataloader(batch_size=batch_size, num_workers=num_workers)
        log_path = trainer.config_logger(log_interval=60)
        test_log_path = trainer.config_tester(testers, interval=20 * 60)
        trainer.fit(model_lit)
        return log_path, test_log_path

    log, test_log = train_xnn()

    

    get_curve(log, 1)
    get_curve(test_log, 2)




if __name__ == '__main__':
    main()
