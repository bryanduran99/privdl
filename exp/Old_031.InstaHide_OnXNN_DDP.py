# -*- coding: UTF-8 -*-
'''
train batch和dist batch采用两个dataloader
无放回采样得到distractor batch, 并且每个train batch的load都附带一次distractor batch的dataloader对象的定义，耗时主要是因为这个
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

def vec_mul_ten(vec, tensor):
    tensor = tensor.float()
    size = list(tensor.size())
    size[0] = -1
    size_rs = [1 for i in range(len(size))]
    size_rs[0] = -1
    vec = vec.reshape(size_rs).expand(size)
    res = vec * tensor
    return res
    
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


class InstaClock(utils.TrainLoopClock):

    def __init__(self, dataloader, total_epochs, k=4, dis_dataset=None, transform=None):
        super().__init__(dataloader, total_epochs)
        self.vec_mul_ten = vec_mul_ten
        self.k = k

        self.dis_dataset = dis_dataset
        self.transform = transform
        self.batch_size = dataloader.batch_size
        self.num_workers = dataloader.num_workers


    def __iter__(self):
        self.time0 = time.time()
        for self.epoch in range(self.total_epochs):
            if dist.is_available() and dist.is_initialized():
                self.dataloader.sampler.set_epoch(self.epoch)

            for self.batch, (imgs, y) in enumerate(self.dataloader):
                
                lambs = np.random.normal(0, 1, size=(self.batch_size, self.k)) 
                # 每个batch随机生成 lambdas
                # 源码采用np中的lams = np.random.normal(0, 1, size=(x.size()[0], args.klam))
                # Q：不需要保证lambda为正值吗 
                lambs = tc.from_numpy(lambs).float()
                mixed_imgs = self.vec_mul_ten(lambs[:, 0], imgs)
                ys = [y]

                # # 严格按照论文中的方法，每个dis_batch是相互独立的有放回采样
                # for i in range(1, self.k): # (k-1)次相互独立的采样，生成(k-1)个干扰batch
                #     indices = tc.randperm(len(self.dis_dataset))[:self.batch_size]
                #     # 从full_trainset中，无放回地随机采样batch_size
                #     dis_batch = self.dis_dataset.newset_by_indices(indices)
                #     if self.transform is not None:
                #         dis_batch = utils.TransformedDataset(dis_batch, self.transform)
                #     distractor_imgs, distractor_y = next(iter(utils.DataLoader(dis_batch, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True)))
                #     mixed_imgs += self.vec_mul_ten(lambs[:, i], distractor_imgs)
                #     ys.append(distractor_y)

                # 有放回采样耗时巨大，对每个dis_batch无放回采样 k-1 次，生成 k 个干扰batch
                # 无放回采样的dataloader() = 13s
                indices = tc.randperm(len(self.dis_dataset))[:(self.k-1) * self.batch_size] # (k-1)个batch一起采
                dis_batches = self.dis_dataset.newset_by_indices(indices)

                if self.transform is not None:
                    dis_batches = utils.TransformedDataset(dis_batches, self.transform) # 0.5s
                    # pass
                if dist.is_available() and  dist.is_initialized():
                    dis_sampler = DistributedSampler(dis_batches)
                    # print('proc1',dist.get_rank(), 'self.batch_size', self.batch_size, 'len(dis_batches)', len(dis_batches),'len(imgs)', len(imgs))
                    dis_dataloader = utils.DataLoader(dis_batches, batch_size=self.batch_size, num_workers=self.num_workers, sampler=dis_sampler) # 11s
                else:
                    dis_dataloader = utils.DataLoader(dis_batches, batch_size=self.batch_size, num_workers=self.num_workers,drop_last=True) 
                
                if dist.is_available() and dist.is_initialized():
                    dis_dataloader.sampler.set_epoch(self.batch)
                for i, (distractor_imgs, distractor_y) in enumerate(dis_dataloader):
                    # print(distractor_y) # 未执行
                    mixed_imgs += self.vec_mul_ten(lambs[:, i+1], distractor_imgs)
                    ys.append(distractor_y)

                # sign flip = 0.1s
                sign = tc.randint(2, size=list(mixed_imgs.shape)) * 2.0 - 1
                mixed_imgs *= sign.float()

                # print(len(ys)) # 1
                # print(ys[0].size()) # torch.Size([16])
                
                yield [mixed_imgs, ys, lambs]

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
        clock = InstaClock(dataloader, self.total_epochs, self.k, self.dataset, model_lit.sample_transform) # 配置训练循环
        for batch_data_cpu in clock: # 训练循环
            if local_rank != -1:
                # print(batch_data_cpu[1])
                batch_data = batch_to_rank_insta(batch_data_cpu, local_rank=local_rank)
            else:
                batch_data = batch_to_cuda_insta(batch_data_cpu)
            model_lit.train() # 设置模型为训练模式
            model_lit.train_step(batch_data) # 训练一个 batch
            if clock.epoch_end(): # 当 epoch 结束时需要执行的操作
                model_lit.on_epoch_end()
            self.add_log(clock, model_lit.monitor) # 添加 log
            # 控制clock的时间，保证每三个epoch进行一次test

            if local_rank == -1 or dist.get_rank() == 0: 
                self.call_testers(clock, model_lit) # 进行测试
            
            # 当没有初始化多线程时，因为短路原则，也不会因为get_rank而报错
            if local_rank == -1 or dist.get_rank() == 0: 
                if self.current_accuracy > self.best_acc:
                    self.best_acc = self.current_accuracy
                    self.best_epoch = clock.epoch
                    best_model_state_dict = {k:v.to('cpu') for k, v in model_lit.state_dict().items()}
                    utils.torch_save(best_model_state_dict, os.path.join(self.work_dir, 'best_model.tar'))
        
        return model_lit # 返回训练完成的模型

class InstaAMSoftmax(block.loss.amsoftmax.AMSoftmax):
    '''features -> scores'''

    def __init__(self, in_features, class_num, margin=0.35, scale=32,k=4):
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
        mixed_labels_one_hot = self.vec_mul_ten(lambs[:, 0], k_labels_one_hot[0])
        for i in range(1, self.k):
            mixed_labels_one_hot += self.vec_mul_ten(lambs[:, i], k_labels_one_hot[i])
        
        scores = cosine - self.margin * mixed_labels_one_hot
        scores *= self.scale

        mixed_labels = tc.topk(mixed_labels_one_hot, 1)[1].squeeze(1)
        return scores, mixed_labels


class XNN_Single_Lit(block.model.light.ModelLight):
    '''单客户场景下的 XNN'''

    def __init__(self, xnn_parts, class_num, k=4):
        super().__init__()
        self.k = k
        self.extractor = xnn_parts.extractor()
        self.obfuscate = xnn_parts.obfuscate()
        self.tail = xnn_parts.tail()
        self.softmax = InstaAMSoftmax(
            in_features=self.tail.feat_size, class_num=class_num, k=self.k)
        self.config_optim()

    def config_optim(self):
        param_groups = [  # 每类参数设置不同的 weight_decay
            dict(params=self.extractor.parameters(), weight_decay=4e-5),
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

class ERN_lit(XNN_Single_Lit):
    '''期望识别模型，强行在期望混淆后的数据上训练识别模型'''

    def train_step(self, batch_data):
        self.obfuscate.reset_parameters()
        super().train_step(batch_data)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--train_dataset", default='celeba', type=str)
    parser.add_argument("--k", default=4, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    # parser.add_argument("--client_dataset", default='celeba', type=str)
    # parser.add_argument("--person", default=1000, type=int)
    FLAGS = parser.parse_args()
    local_rank = FLAGS.local_rank

    proc_count = 1
    if local_rank != -1:
        dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端
        proc_count = tc.distributed.get_world_size() #获取全局并行数，即并行训练的显卡的数量
    
    train_dataset = FLAGS.train_dataset
    # client_dataset = FLAGS.client_dataset
    
    print("train_dataset:", train_dataset)
    # print("client_dataset:", client_dataset)

    result_path = exp_utils.setup_result_path(__file__)
    batch_size = FLAGS.batch_size // proc_count
    num_workers = FLAGS.num_workers
    result_path = f'{result_path}/isnta'
    train_epoch = 100
    # attack_epoch = 200

    
    # 配置数据集 和 相应的训练模型
    if train_dataset == 'msra':
        client_dataset = block.dataset.hubble.xnn_paper.msra()
        trainset, testset_by_img, testset_by_person = split_dataset(client_dataset)
        train_epoch = 100
    elif train_dataset == 'webface':
        client_dataset = block.dataset.hubble.xnn_paper.webface()
        trainset, testset_by_img, testset_by_person = split_dataset(client_dataset,img_per_id=40)
    elif train_dataset == 'celeba':
        client_dataset = block.dataset.hubble.xnn_paper.celeba()
        trainset, testset_by_img, testset_by_person = split_dataset(client_dataset, img_per_id=20)
    
    # 配置模型
    vit = XnnParts_ViT()
    work_dir = result_path


    # 配置训练器并训练
    Tester = block.test.top1_test.Top1_Tester
    testers = [Tester(dataset=testset_by_img, name='testset_by_img'),
               Tester(dataset=testset_by_person, name='testset_by_person')]
    for tester in testers:
        tester.config_dataloader(batch_size=batch_size, num_workers=num_workers)

    def train_xnn():  # 脱敏回流训练
        model_lit = XNN_Single_Lit(vit, class_num=trainset.class_num(), k=FLAGS.k)
        if local_rank != -1:
            model_lit = model_lit.to(local_rank)
            model_lit = DDP(model_lit, device_ids=[local_rank], output_device=local_rank)
            model_lit = model_lit.module
        else:
            device = tc.device('cuda' if tc.cuda.is_available else 'cpu')
            model_lit = model_lit.to(device)

        trainer = InstaTrainer(
            dataset=trainset, total_epochs=train_epoch, work_dir=f'{work_dir}/{train_dataset}/{FLAGS.k}', k=FLAGS.k)
        trainer.config_dataloader(batch_size=batch_size, num_workers=num_workers)
        log_path = trainer.config_logger(log_interval=60)
        test_log_path = trainer.config_tester(testers, interval=20 * 60)
        trainer.fit_DDP(model_lit, local_rank=local_rank)

        return log_path, test_log_path


    log, test_log = train_xnn()

    get_curve(log, 1)
    get_curve(test_log, 2)

if __name__ == '__main__':
    main()

    # DDP: 使用torch.distributed.launch启动DDP模式; 使用CUDA_VISIBLE_DEVICES，来决定使用哪些GPU
    # CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 -m torch.distributed.launch --nproc_per_node 8 031.InstaHide_OnXNN_DDP.py --train_dataset celeba