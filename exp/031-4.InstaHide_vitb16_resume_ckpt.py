# -*- coding: UTF-8 -*-
'''
使用ViT_B/16作为backbone, ExtNet使用moco v3自监督学习在ImageNet上预训练
'''
import datetime
import math
import os
import random
import PIL.Image as Image
import numpy as np
import requests
from sklearn.cluster import k_means
import torch as tc
import torch.nn.functional as F
import exp_utils
import cv2
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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Normalize():
    '''Normalize an image  by channel with mean and standard deviation
    Args:
        img (numpy.ndarray): Image to be normalized, (C, H, W) in BGR order.
        mean (float): Mean for image channel-wise.
        std (float): Standard deviation for image channel-wise.
    Returns:
        numpy.ndarray: Normalized image.
    '''
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """Call function.
        Args:
            img (numpy.ndarray): Image to be normalized.
        Returns:
            numpy.ndarray: Normalized image.
        """
        img = np.transpose(img, (1,2,0)) # c h w -> h w c
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2,0,1)) # c h w -> h w c
        return img.astype(np.float32)


celeba_normalize = Normalize((-0.2883, -0.1692,  0.0652), (0.4247, 0.4464, 0.5122))
webface_normalize = Normalize((-0.3301, -0.2387, -0.0671), (0.4285, 0.4491, 0.5067))
msra_normalize = Normalize((-0.2671, -0.1706,  0.0268), (0.4198, 0.4398, 0.4995))
facescrub_normalize = Normalize((-0.2811, -0.1634,  0.0716), (0.4295, 0.4508, 0.5237))
imdb_normalize = Normalize((-0.4216, -0.3377, -0.1655),(0.4212, 0.4602, 0.5462))
vggface2_normalize = Normalize((-0.3786, -0.3168, -0.1795),(0.4852, 0.5100, 0.5807))


def resize_norm_transpose_insta(img, size=None):
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


def batch_mixup_sign_flip(batch_data, k, upper=0.65):
    '''batch mixup. 
    Output: (mixed_x, mixed_y, lam)
    mixed_x size:(batch_size, 3, 112, 112)
    mixed_y size: (k, batch_size)
    lam size: (k, batch_size)'''

    imgs, labels = batch_data
    batch_size = imgs.size()[0] // k
    lambs = np.random.normal(0, 1, size=(batch_size, k)) 
    for i in range(batch_size):
        lambs[i] = np.abs(lambs[i]) / np.sum(np.abs(lambs[i]))
        if k > 1:
            while lambs[i].max() > upper:     # upper bounds a single lambda
                lambs[i] = np.random.normal(0, 1, size=(1, k))
                lambs[i] = np.abs(lambs[i]) / np.sum(np.abs(lambs[i]))

    lambs = tc.from_numpy(lambs).float()

    mixed_x = vec_mul_ten(lambs[:, 0], imgs[:batch_size])
    y = [labels[:batch_size]]
    for i in range(1, k):
        mixed_x += vec_mul_ten(lambs[:, i], imgs[i*batch_size:(i+1)*batch_size])
        y.append(labels[i*batch_size:(i+1)*batch_size])

    sign = tc.randint(2, size=list(mixed_x.shape)) * 2.0 - 1
    mixed_x *= sign.float()

    return [mixed_x, y, lambs]


class Insta_Top1_Tester(block.test.top1_test.Top1_Tester):
    '''对模型进行 top1 准确率测试'''

    def __init__(self, dataset, name='', sample_transform=None):
        '''dataset 的 sample 由 (input, label) 组成， name 为此 tester 的名称，
        可用 config 系列函数对测试配置进行修改'''
        super().__init__(dataset, name='')
        self.dataset = dataset
        self.name = name
        self.sample_transform = sample_transform
        self.config_dataloader()

    def extract_feats(self, model_lit):
        '''用 model_lit 抽取测试集的特征并返回'''
        dataloader = self.get_dataloader(self.sample_transform)
        feats, labels = [], []
        model_lit.eval()
        with tc.no_grad():
            for batch_data in utils.tqdm(dataloader, 'extract_feats'):
                batch_data = utils.batch_to_cuda(batch_data)
                batch_inputs, batch_labels = batch_data
                batch_feats = model_lit(batch_inputs, batch_labels, lambs=None, is_inference=True)
                feats.append(utils.batch_to_numpy(batch_feats))
                labels.append(utils.batch_to_numpy(batch_labels))
        return np.concatenate(feats), np.concatenate(labels)



class InstaTrainer(block.train.standard.Trainer):
    """Trainer的子类，重写其中的fit函数"""
    def __init__(self, dataset, total_epochs, scheduler=None, val_set=None, work_dir=None, ckpt_dir=None, k=4, sample_transform=None, optimizer=None, monitor=None):
        super().__init__(dataset, total_epochs, val_set, work_dir, ckpt_dir)
        self.k = k
        self.best_loss = 1.0
        self.best_acc = 0
        self.best_epoch = 0
        self.start_epoch = 0
        self.sample_transform = sample_transform
        self.optimizer = optimizer 
        self.scheduler = scheduler
        self.monitor = monitor
        assert self.sample_transform is not None, 'sample_transform is None'
    
    def config_load_checkpoint(self, is_load=False, local_rank=-1):
        '''interval: 保存 checkpoint 的间隔(单位秒)'''
        self.is_load = is_load
        if self.is_load:
            print(f'config:rank{local_rank}_torch_load: {self.ckpt_dir}/best_ckpt.pth')
            checkpoint_data = tc.load(f'{self.ckpt_dir}/best_ckpt.pth', map_location=tc.device(local_rank))
            self.best_acc = checkpoint_data['best_acc']
            self.best_epoch = checkpoint_data['best_epoch']
            self.start_epoch = checkpoint_data['best_epoch']
            self.best_loss = checkpoint_data['best_loss'] if 'best_loss' in checkpoint_data.keys() else 1.0
            del checkpoint_data

    def fit_ViT_Insta(self, model_lit, local_rank=-1):
        '''训练模型并返回练完成的模型'''
        dataloader = self.get_dataloader(self.sample_transform) # 加载训练集
        # DDP：设置sampler的epoch，
        # DistributedSampler需要这个来指定shuffle方式，
        # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果。
        clock = utils.TrainLoopClock(dataloader, self.total_epochs, start_epoch=self.start_epoch+1) # 配置训练循环
        for batch_data in clock: # 训练循环
            # print('self.best_acc:', self.best_acc, 'self.best_epoch:', self.best_epoch, 'clock.epoch:', clock.epoch)
            if self.k > 1: # mixup
                # t0 = time.time()
                batch_data = batch_mixup_sign_flip(batch_data, self.k) 
                # print('batch_mixup time:', time.time()-t0)
            else: # do not mixup, then Uniform form
                img = batch_data[0]
                y = [batch_data[1]]
                lambs = tc.ones(0)
                batch_data = [img, y, lambs]

            if local_rank != -1:
                batch_data = batch_to_rank_insta(batch_data, local_rank=local_rank)
            else:
                batch_data = batch_to_cuda_insta(batch_data)

            imgs, labels, lambs = batch_data
            
            
            # train step 
            model_lit.train() # 设置模型为训练模式
            scores, mixed_labels = model_lit(imgs, labels,lambs, is_inference=False)
            loss = tc.nn.functional.cross_entropy(scores, mixed_labels)
            
            # print(self.optimizer.param_groups[0]['lr']) # 这里就有问题
            utils.step(self.optimizer, loss)

            self.monitor.add('loss', lambda: float(loss),
                            to_str=lambda x: f'{x:.2e}')  # 监控 loss
            self.monitor.add('batch_acc', lambda: utils.accuracy(scores, mixed_labels),
                            to_str=lambda x: f'{x * 100:.2f}%')  # 监控 batch 准确率
            # print('time: ', time.time() - t0)
            if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0: 
                self.add_log(clock, self.monitor) # 添加 log
            if dist.is_available() and dist.is_initialized():
                dist.barrier()


            if loss < self.best_loss:
                if local_rank == -1 or dist.get_rank() == 0: 
                    self.call_testers(clock, model_lit) 
                    # 只保留精度最佳的ckpt，节省时间
                    if self.current_accuracy > self.best_acc:
                        self.best_acc = self.current_accuracy
                        self.best_epoch = clock.epoch
                        # 保存模型
                        if local_rank == -1:
                            best_model_state_dict = {k:v.to('cpu') for k, v in model_lit.state_dict().items()}
                        else:
                            best_model_state_dict = {k:v.to('cpu') for k, v in model_lit.module.state_dict().items()}
                        ckpt = {'best_epoch': self.best_epoch,
                                'best_acc': self.best_acc,
                                'best_loss': self.best_loss,
                                'model_state_dict': best_model_state_dict,
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'scheduler_state_dict': self.scheduler.state_dict()
                                }
                        utils.torch_save(ckpt, f'{self.ckpt_dir}/best_ckpt.pth')
                        del ckpt

                    if local_rank != -1:
                        # 在多卡模式下在主进程测试并save，其他进程通过torch.distributed.barrier()来等待主进程完成validate操作
                        # barrier()函数的使用要非常谨慎，如果只有单个进程包含了这条语句，那么程序就会陷入无限等待
                        dist.barrier()

            if clock.epoch_end():
            # if True:
                self.scheduler.step()
                if (clock.epoch + 1) % 1 == 0: # 每5个epoch test一次
                # if True:
                    # 当没有初始化多线程时，因为短路原则，也不会因为get_rank而报错
                    if local_rank == -1 or dist.get_rank() == 0: 
                        self.call_testers(clock, model_lit) 
                        # 只保留精度最佳的ckpt，节省时间
                        if self.current_accuracy > self.best_acc:
                            self.best_acc = self.current_accuracy
                            self.best_epoch = clock.epoch
                            # 保存模型
                            if local_rank == -1:
                                best_model_state_dict = {k:v.to('cpu') for k, v in model_lit.state_dict().items()}
                            else:
                                best_model_state_dict = {k:v.to('cpu') for k, v in model_lit.module.state_dict().items()}
                            ckpt = {'best_epoch': self.best_epoch,
                                    'best_acc': self.best_acc,
                                    'best_loss': self.best_loss,
                                    'model_state_dict': best_model_state_dict,
                                    'optimizer_state_dict': self.optimizer.state_dict(),
                                    'scheduler_state_dict': self.scheduler.state_dict()
                                    }
                            utils.torch_save(ckpt, f'{self.ckpt_dir}/best_ckpt.pth')
                            del ckpt
                    if local_rank != -1:
                        # 在多卡模式下在主进程测试并save，其他进程通过torch.distributed.barrier()来等待主进程完成validate操作
                        # barrier()函数的使用要非常谨慎，如果只有单个进程包含了这条语句，那么程序就会陷入无限等待
                        dist.barrier()
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


class InstaHide_Single_Lit(block.model.light.ModelLight):
    '''单客户场景a下的 XNN'''

    def __init__(self, class_num, k, debug_stop_grad=True):
        super().__init__()
        self.class_num = class_num
        self.k=k
        self.stop_grad = debug_stop_grad

        self.backbone = block.model.moco_v3_vits.vit_base_nomlp(stop_grad=self.stop_grad)

        self.softmax = InstaAMSoftmax(
            in_features=self.backbone.embed_dim, class_num=self.class_num, k=self.k)


    def forward(self, images, labels, lambs, is_inference=False):
        rec_feats = self.backbone(images)

        if is_inference:   
            return rec_feats    
        out = self.softmax(rec_feats, labels, lambs)
        return out



def split_dataset(dataset, person_num):
    hubble_utils = block.dataset.hubble.hubble_utils
    trainset, testset_by_person = hubble_utils.split_dataset_by_person(
        dataset, test_id_num=person_num, test_img_per_id=20) # origin = 1000
    trainset, testset_by_img = hubble_utils.split_dataset_by_img(
        trainset, test_id_num=person_num, test_img_per_id=20, train_scale=1.5)
    return trainset, testset_by_img, testset_by_person
    

def main():

    # 配置实验设置
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--train_dataset", default='celeba', type=str)
    parser.add_argument("--k", default=2, type=int)
    parser.add_argument("--debug_stop_grad", default="True", type=str)


    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr", default=5e-2, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--person", default=1000, type=int)
    parser.add_argument("--is_attack", default="False", type=str)
    parser.add_argument("--attacker_dataset", default="facescrub", type=str)
    parser.add_argument("--schedule", default="True", type=str)

    parser.add_argument("--subset_ratio", default=1.0, type=float)

    FLAGS = parser.parse_args()
    bool_type = lambda x: x.lower() == 'true'
    FLAGS.is_attack = bool_type(FLAGS.is_attack)
    FLAGS.schedule = bool_type(FLAGS.schedule)
    FLAGS.debug_stop_grad = bool_type(FLAGS.debug_stop_grad)

    local_rank = FLAGS.local_rank
    proc_count = 1
    if local_rank != -1:
        # nccl是GPU设备上最快、最推荐的后端
        dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=18000))
        proc_count = tc.distributed.get_world_size() #获取全局并行数，即并行训练的显卡的数量
    batch_size = FLAGS.batch_size // proc_count

    train_dataset = FLAGS.train_dataset
    num_workers = FLAGS.num_workers
    train_epoch = 5000 #

    set_ratio = '' if FLAGS.subset_ratio == 1.0 else f'{FLAGS.subset_ratio}'
    result_dir = exp_utils.setup_result_path(os.path.basename(__file__))
    result_dir = f'{result_dir}/insta/k_{FLAGS.k}/{FLAGS.is_attack}/{train_dataset}/{set_ratio}{FLAGS.attacker_dataset}/{proc_count}_gpu'
    print('result_path:', result_dir)
    
    alert_info = f'031-4_k{FLAGS.k}_{FLAGS.is_attack}_{train_dataset}_{set_ratio}{FLAGS.attacker_dataset}_{proc_count}gpu'

    ckpt_dir = f'/data/ckpt/insta/k{FLAGS.k}_{FLAGS.is_attack}_{train_dataset}_{set_ratio}{FLAGS.attacker_dataset}_{proc_count}gpu'
    print('ckpt_dir:', ckpt_dir)
    is_load = False
    if local_rank == -1 or dist.get_rank() == 0:
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
    if os.path.exists(os.path.join(ckpt_dir, 'best_ckpt.pth')):
        is_load = True

    # 配置数据集
    log_inter = 60
    if train_dataset == 'msra':
        client_dataset = block.dataset.hubble.xnn_paper.msra()
        normalize = msra_normalize
    elif train_dataset == 'webface':
        client_dataset = block.dataset.hubble.xnn_paper.webface()
        normalize = webface_normalize
    elif train_dataset == 'celeba':
        client_dataset = block.dataset.hubble.xnn_paper.celeba()
        normalize = celeba_normalize
    elif train_dataset == 'facescrub':
        client_dataset = block.dataset.hubble.xnn_paper.facescrub() # 10w
        normalize = facescrub_normalize
        FLAGS.person = 100 # person_num of facescrub = 530 < 1000
        log_inter = 10
    elif train_dataset == 'vggface2':
        client_dataset = block.dataset.hubble.xnn_paper.vggface2() # 200w+
        normalize = vggface2_normalize
    else:
        raise ValueError('dataset do not exist')

    trainset, testset_by_img, testset_by_person = split_dataset(client_dataset, person_num=FLAGS.person)
    if FLAGS.is_attack:
        if FLAGS.attacker_dataset == 'facescrub':
            attack_dataset = block.dataset.hubble.xnn_paper.facescrub() # 10w
            normalize = facescrub_normalize
            trainset = attack_dataset
        elif FLAGS.attacker_dataset == 'imdb':
            attack_dataset = block.dataset.hubble.xnn_paper.imdb() # 200w+
            normalize = imdb_normalize
            # 按比例选择攻击集的子集， 只针对imdb
            if 0 < FLAGS.subset_ratio < 1.0:
                subset_size = len(attack_dataset) * FLAGS.subset_ratio
                train_dataset = attack_dataset.subset_by_size(subset_size)
            else:
                trainset = attack_dataset

        

    # preprocess
    def sample_transform(sample):
        '''H*W*3*uint8(0\~255) -> 3*224*224*float32(-1\~1)'''
        img, label = sample
        np_img = resize_norm_transpose_insta(img, size=224)
        transformed_img = normalize(np_img)
        return transformed_img, label
    
    # if k > 1, then mixup dataset
    if FLAGS.k > 1:        
        trainset = trainset.mixup_multi_dataset(k=FLAGS.k)  
    
    # 配置训练器并训练
    Tester = Insta_Top1_Tester
    testers = [Tester(dataset=testset_by_img, name='testset_by_img',sample_transform=sample_transform),
               Tester(dataset=testset_by_person, name='testset_by_person', sample_transform=sample_transform)]
    for tester in testers:
        tester.config_dataloader(batch_size=batch_size, num_workers=num_workers)

    # 脱敏回流训练
    def train_xnn():  
        # 定义模型
        # print(f'local_rank:{local_rank}, is_load:{is_load}',is_load)
        cls = attack_dataset.class_num() if FLAGS.attacker_dataset == 'imdb' else trainset.class_num()
        model_lit = InstaHide_Single_Lit(class_num=cls, k=FLAGS.k, debug_stop_grad=FLAGS.debug_stop_grad)
        if is_load:
            print(f'rank{local_rank}_torch_load: {ckpt_dir}/best_ckpt.pth')
            checkpoint_data = tc.load(f'{ckpt_dir}/best_ckpt.pth', map_location=tc.device(local_rank))
            model_lit.load_state_dict(checkpoint_data['model_state_dict'])

        # DDP
        if local_rank != -1:
            model_lit = model_lit.to(local_rank)
            model_lit = DDP(model_lit, device_ids=[local_rank], output_device=local_rank)
        else:
            device = tc.device('cuda' if tc.cuda.is_available else 'cpu')
            model_lit = model_lit.to(device)
        
        # 定义优化器 & lr变化策略
        optimizer = tc.optim.SGD(filter(lambda p: p.requires_grad, model_lit.parameters()), lr=FLAGS.lr, momentum=0.9)

        warm_up_iter = 3 # 不使用warm_up
        T_max = 33 # warm_up + cosAnnealLR 的周期
        lr_max = FLAGS.lr
        lr_min = 1e-3
        if FLAGS.schedule:
            lambda_lr = lambda cur_iter: max(1e-3/FLAGS.lr, cur_iter/warm_up_iter) if cur_iter < warm_up_iter else \
                (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi))) / FLAGS.lr
        else:
            lambda_lr = lambda cur_iter: 1
        scheduler = tc.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

        # load optimizer & lr_scheduler
        if is_load:
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            scheduler.last_epoch = checkpoint_data['best_epoch']  # 将last_epoch设置为已经训练的epoch数
            # print(scheduler.get_last_lr(), scheduler.last_epoch)
            del checkpoint_data
        # print(optimizer.param_groups[0]['lr'])

        monitor = utils.Monitor()
        monitor.add('lr', get=lambda: optimizer.param_groups[0]['lr'], to_str=lambda x: f'{x:.1e}')
        

        # train
        trainer = InstaTrainer(
            dataset=trainset, total_epochs=train_epoch, scheduler=scheduler, work_dir=result_dir, \
                 ckpt_dir=ckpt_dir, k=FLAGS.k, sample_transform=sample_transform, optimizer=optimizer, monitor=monitor)
        trainer.config_dataloader(batch_size=batch_size * FLAGS.k, num_workers=num_workers) 
        log_path = trainer.config_logger(log_interval=log_inter) # default log_interval = 60, while facescrub = 10
        test_log_path = trainer.config_tester(testers)
        trainer.config_load_checkpoint(is_load=is_load,local_rank=local_rank)
        
        if dist.is_available() and dist.is_initialized():
                dist.barrier()
        trainer.fit_ViT_Insta(model_lit, local_rank=local_rank)
        
        return log_path, test_log_path
    
    try:
        train_xnn()
    except Exception as e:
        if local_rank == -1 or dist.get_rank() == 0:
            # note: DDP的timeout异常捕捉不到。。。
            print("Exception")
            requests.get(f'https://api.day.app/ocZjTT3y2AhsowCaiKtqLC/break_down/{alert_info}_{time.strftime("%y%m%d%H%M%S",time.localtime())}')
            raise e
        else:
            pass
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
        

if __name__ == '__main__':
    main()
    print('done')

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 -m torch.distributed.launch --nproc_per_node 8 031-4.InstaHide_vitb16_resume_ckpt.py  