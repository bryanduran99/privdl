# -*- coding: UTF-8 -*-
'''
上一版中k=1的逻辑不对，这一版增加了对于k=1这种不混淆情况的支持
note:
因为多卡训练传入fit的是model_lit.module(即raw_model)，而不是多线程模型model_lit,所以在fit中各线程并没有融合梯度，而是各自为战。
多卡训练结果不可信（比实际的ACC要低），单卡执行脚本的结果不受影响，仍然可信
'''
import math
import os
import random
import PIL.Image as Image
import numpy as np
from sklearn.cluster import k_means
import torch as tc
import torch.nn.functional as F
import exp_utils
exp_utils.setup_import_path()

import block
from simpleViT import SimpleViT
from refile import smart_open
from torchvision import datasets, transforms

import utils
import time
import json
from matplotlib import pyplot as plt
from torchsummary import summary

import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


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



def batch_to_cuda_insta(batch_data):
    '''return batch_data (on cuda) \n
    pytorch DataLoader 迭代时，依据 dataset 的 \_\_getitem\_\_
    返回形式，batch_data 可能是 list of tensor、dict of tensor、tensor，
    本函数将这些类型的数据中的 tensor 放到 cuda 上'''
    if isinstance(batch_data, list):
        ret = []
        for data in batch_data:
            if isinstance(data, list):
                ret.append([item.cuda() for item in data])
            else:
                ret.append(data.cuda())
        return ret
    elif isinstance(batch_data, dict):
        return {name: data.cuda() for name, data in batch_data.items()}
    else:
        return batch_data.cuda()

def batch_to_rank_insta(batch_data, local_rank):
    '''return batch_data (on cuda) \n
    pytorch DataLoader 迭代时，依据 dataset 的 \_\_getitem\_\_
    返回形式，batch_data 可能是 list of tensor、dict of tensor、tensor，
    本函数将这些类型的数据中的 tensor 放到 cuda 上'''
    if isinstance(batch_data, list):
        ret = []
        for data in batch_data:
            if isinstance(data, list):
                ret.append([item.to(local_rank) for item in data])
            else:
                ret.append(data.to(local_rank))
        return ret
    elif isinstance(batch_data, dict):
        return {name: data.to(local_rank) for name, data in batch_data.items()}
    else:
        return batch_data.to(local_rank)


def vec_mul_ten(vec, tensor):
    tensor = tensor.float()
    size = list(tensor.size())
    size[0] = -1
    size_rs = [1 for i in range(len(size))]
    size_rs[0] = -1
    vec = vec.reshape(size_rs).expand(size)
    res = vec * tensor
    return res


def batch_mixup_sign_flip(batch_data, k):
    '''batch mixup. 
    Output: (mixed_x, mixed_y, lam)
    mixed_x size:(batch_size, 3, 112, 112)
    mixed_y size: (k, batch_size)
    lam size: (k, batch_size)'''

    imgs, labels = batch_data
    batch_size = imgs.size()[0] // k
    lambs = np.random.normal(0, 1, size=(batch_size, k)) 
    lambs = tc.from_numpy(lambs).float()

    mixed_x = vec_mul_ten(lambs[:, 0], imgs[:batch_size])
    y = [labels[:batch_size]]
    for i in range(1, k):
        mixed_x += vec_mul_ten(lambs[:, i], imgs[i*batch_size:(i+1)*batch_size])
        y.append(labels[i*batch_size:(i+1)*batch_size])

    sign = tc.randint(2, size=list(mixed_x.shape)) * 2.0 - 1
    mixed_x *= sign.float()

    return [mixed_x, y, lambs]

class InstaTrainer(block.train.standard.Trainer):
    """Trainer的子类，重写其中的fit函数"""
    def __init__(self, dataset, total_epochs, val_set=None, work_dir=None, k=4):
        super().__init__(dataset, total_epochs, val_set, work_dir)
        self.k = k
        self.best_acc = 0
        self.best_epoch = 0

    def fit_DDP(self, model_lit, local_rank=-1):
        '''训练模型并返回练完成的模型'''
        dataloader = self.get_dataloader(model_lit.sample_transform) # 加载训练集
        # DDP：设置sampler的epoch，
        # DistributedSampler需要这个来指定shuffle方式，
        # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果。
        clock = utils.TrainLoopClock(dataloader, self.total_epochs) # 配置训练循环
        for batch_data in clock: # 训练循环
            if self.k > 1: # mixup
                batch_data = batch_mixup_sign_flip(batch_data, self.k) 
            else: # do not mixup, then Uniform form
                img = batch_data[0]
                y = [batch_data[1]]
                lambs = tc.ones(0)
                batch_data = [img, y, lambs]

            if local_rank != -1:
                batch_data = batch_to_rank_insta(batch_data, local_rank=local_rank)
            else:
                batch_data = batch_to_cuda_insta(batch_data)
            
            model_lit.train() # 设置模型为训练模式
            model_lit.train_step(batch_data) # 训练一个 batch
                
            self.add_log(clock, model_lit.monitor) # 添加 log
            
            if clock.epoch_end(): # 当 epoch 结束时需要执行的操作
                model_lit.on_epoch_end()
                # 当没有初始化多线程时，因为短路原则，也不会因为get_rank而报错
                if local_rank == -1 or dist.get_rank() == 0: 
                    self.call_testers(clock, model_lit) 
                    # 只保留精度最佳的ckpt，节省时间
                    if self.current_accuracy > self.best_acc:
                        self.best_acc = self.current_accuracy
                        self.best_epoch = clock.epoch
                        best_model_state_dict = {k:v.to('cpu') for k, v in model_lit.state_dict().items()}
                        utils.torch_save(best_model_state_dict, os.path.join(self.work_dir, 'best_model.tar'))
        
        return model_lit

class InstaAMSoftmax(block.loss.amsoftmax.AMSoftmax):
    '''features -> scores'''

    def __init__(self, in_features, class_num, margin=0.35, scale=32, k=4):
        super().__init__(in_features, class_num, margin=0.35, scale=32)
        self.k = k
        self.vec_mul_ten = vec_mul_ten

    def one_hot(self, labels, class_num):
        '''len(labels) -> len(labels) * class_num'''
        shape = len(labels), class_num
        ret = tc.zeros(shape, device=labels.device)
        index = labels.view(-1, 1)
        ret.scatter_(dim=1, index=index, value=1)
        return ret

    def forward(self, features, k_labels, lambs):
        features = F.normalize(features)
        weight = F.normalize(self.weight)
        cosine = F.linear(features, weight)
        k_labels_one_hot = [self.one_hot(labels, self.class_num) for labels in k_labels]
        # print('len(lam)', len(lambs),'len(lams[1])', lambs[1] ,'len(k_labels_one_hot)', k_labels_one_hot)
        if self.k > 1:
            mixed_labels_one_hot = self.vec_mul_ten(lambs[:, 0], k_labels_one_hot[0])
            for i in range(1, self.k):
                mixed_labels_one_hot += self.vec_mul_ten(lambs[:, i], k_labels_one_hot[i])
        else:
            mixed_labels_one_hot = k_labels_one_hot[0]

        
        scores = cosine - self.margin * mixed_labels_one_hot
        scores *= self.scale

        # one_hot to label
        mixed_labels = tc.topk(mixed_labels_one_hot, 1)[1].squeeze(1)
        return scores, mixed_labels


''' 
根据state_dict中键的前缀，获取指定的模块的state_dict，比如
state_dict = {'ext.conv1.weight': 1, 'ext.conv2.weight': 2}
return {'conv1.weight': 1, 'conv2.weight': 2}
'''
def get_state_dict_by_prefix(state_dict, prefix):
    ret = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            ret[k[len(prefix)+1:]] = v
    return ret

class XNN_Single_Lit(block.model.light.ModelLight):
    '''单客户场景下的 XNN'''

    def __init__(self, xnn_parts, class_num, lr, k=4, param_path=None):
        super().__init__()
        self.k = k
        self.lr = lr
        self.extractor = xnn_parts.extractor()
        self.obfuscate = xnn_parts.obfuscate()
        self.tail = xnn_parts.tail()
        self.softmax = InstaAMSoftmax(
            in_features=self.tail.feat_size, class_num=class_num, k=self.k)
        self.config_optim(self.lr)
        if param_path is not None:
            params_dict = utils.torch_load(param_path)

            self.extractor.load_state_dict(get_state_dict_by_prefix(params_dict, prefix='extractor'), strict=False)
            # self.obfuscate.load_state_dict(params_dict, strict=False)
            self.tail.load_state_dict(get_state_dict_by_prefix(params_dict, prefix='tail'), strict=False)
            self.softmax.load_state_dict(get_state_dict_by_prefix(params_dict, prefix='softmax'), strict=False)
            print("load params Done!")

    def config_optim(self, lr):
        param_groups = [  # 每类参数设置不同的 weight_decay
            dict(params=self.extractor.parameters(), weight_decay=4e-5),
            dict(params=self.tail.parameters(), weight_decay=4e-5),
            dict(params=self.softmax.parameters(), weight_decay=4e-4)]
        self.optimizer = tc.optim.SGD(param_groups, lr=lr)

        self.monitor.add('lr',  # 监控学习率
                         get=lambda: self.optimizer.param_groups[0]['lr'],
                         to_str=lambda x: f'{x:.1e}')

    def sample_transform(self, sample):
        img, label = sample
        img = block.model.mobile_face_net.MobileFaceNetHead(layer_num=0).img_transform(img)
        return img, label

    def train(self):
        self.extractor.train()
        self.obfuscate.eval()
        self.tail.train()
        self.softmax.train()

    def eval(self):
        self.extractor.eval()
        self.obfuscate.eval()
        self.tail.eval()
        self.softmax.eval()

    def forward(self, feats, labels, lambs):
        templates = self.tail(feats)
        return self.softmax(templates, labels, lambs)

    def train_step(self, batch_data):
        imgs, labels, lambs = batch_data
        with tc.no_grad():
            mid_feats = self.extractor(imgs)
            # mid_feats = mid_feats.reshape(mid_feats.shape[0], 2, 112, 112)
            mid_feats = self.obfuscate(mid_feats)
        scores, mixed_labels = self(mid_feats, labels, lambs)
        loss = tc.nn.functional.cross_entropy(scores, mixed_labels)
        utils.step(self.optimizer, loss)

        self.monitor.add('loss', lambda: float(loss),
                         to_str=lambda x: f'{x:.2e}')  # 监控 loss
        self.monitor.add('batch_acc', lambda: utils.accuracy(scores, mixed_labels),
                         to_str=lambda x: f'{x * 100:.2f}%')  # 监控 batch 准确率

    def inference(self, batch_input):
        mid_feats = self.extractor(batch_input)
        # mid_feats = mid_feats.reshape(mid_feats.shape[0], 2, 112, 112)
        mid_feats = self.obfuscate(mid_feats)
        return self.tail(mid_feats)

    def get_tail(self):
        return self.tail

class XnnParts_ViT:

    def __init__(self, isobf=False, ext=None, is_identity=False):
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
        dataset, test_id_num=person_num, test_img_per_id=20) # origin = 1000
    trainset, testset_by_img = hubble_utils.split_dataset_by_img(
        trainset, test_id_num=person_num, test_img_per_id=20, train_scale=1.5)
    return trainset, testset_by_img, testset_by_person


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--train_dataset", default='celeba', type=str)
    parser.add_argument("--k", default=4, type=int)
    # if k == 1, don't mixup
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--lr", default=5e-2, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--person", default=1000, type=int)
    parser.add_argument("--load_params", default=False, type=bool)
    # parser.add_argument("--client_dataset", default='celeba', type=str)
    # parser.add_argument("--person", default=1000, type=int)
    FLAGS = parser.parse_args()
    
    local_rank = FLAGS.local_rank
    proc_count = 1
    if local_rank != -1:
        dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端
        proc_count = tc.distributed.get_world_size() #获取全局并行数，即并行训练的显卡的数量
    train_dataset = FLAGS.train_dataset
    print("train_dataset:", train_dataset)
    batch_size = FLAGS.batch_size // proc_count
    num_workers = FLAGS.num_workers
    train_epoch = 100 # 训练到收敛，过拟合没关系，因为保存了最优模型
    result_dir = exp_utils.setup_result_path(os.path.basename(__file__))
    result_dir = f'{result_dir}/isnta/{train_dataset}/k_{FLAGS.k}/{proc_count}_gpu'
    print('result_path:', result_dir)

    param_path = None
    if FLAGS.load_params:
        param_path = '/home/liukaixin/privdl/privdl/result/031.InstaHide_OnXNN3_DDP.py/isnta/celeba/k_1/1_gpu/best_model.tar'

    
    # 配置数据集
    if train_dataset == 'msra':
        client_dataset = block.dataset.hubble.xnn_paper.msra()
    elif train_dataset == 'webface':
        client_dataset = block.dataset.hubble.xnn_paper.webface()
    elif train_dataset == 'celeba':
        client_dataset = block.dataset.hubble.xnn_paper.celeba()
    trainset, testset_by_img, testset_by_person = split_dataset(client_dataset, person_num=FLAGS.person)
    # if k > 1, then mixup dataset
    if FLAGS.k > 1:        
        trainset = trainset.mixup_multi_dataset(k=FLAGS.k)  # ToDo,修改InstaClock
    
    # 配置模型
    vit = XnnParts_ViT(isobf=False)

    # 配置训练器并训练
    Tester = block.test.top1_test.Top1_Tester
    testers = [Tester(dataset=testset_by_img, name='testset_by_img'),
               Tester(dataset=testset_by_person, name='testset_by_person')]
    for tester in testers:
        tester.config_dataloader(batch_size=batch_size, num_workers=num_workers)

    def train_xnn():  # 脱敏回流训练
        model_lit = XNN_Single_Lit(vit, class_num=trainset.class_num(), lr=FLAGS.lr, k=FLAGS.k, param_path=param_path)
        if local_rank != -1:
            model_lit = model_lit.to(local_rank)
            model_lit = DDP(model_lit, device_ids=[local_rank], output_device=local_rank)
            model_lit = model_lit.module
        else:
            device = tc.device('cuda' if tc.cuda.is_available else 'cpu')
            model_lit = model_lit.to(device)

        trainer = InstaTrainer(
            dataset=trainset, total_epochs=train_epoch, work_dir=result_dir, k=FLAGS.k)
        # temporary batch_size = k * batch_size, for loading (k-1) distraction batches
        trainer.config_dataloader(batch_size=batch_size * FLAGS.k, num_workers=num_workers) 
        log_path = trainer.config_logger(log_interval=60)
        test_log_path = trainer.config_tester(testers)
        trainer.fit_DDP(model_lit, local_rank=local_rank)

        return log_path, test_log_path


    log, test_log = train_xnn()

    get_curve(log, 1)
    get_curve(test_log, 2)

if __name__ == '__main__':
    main()

    # DDP: 使用torch.distributed.launch启动DDP模式; 使用CUDA_VISIBLE_DEVICES，来决定使用哪些GPU
    # CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 -m torch.distributed.launch --nproc_per_node 8 031.InstaHide_OnXNN2_DDP.py --train_dataset celeba