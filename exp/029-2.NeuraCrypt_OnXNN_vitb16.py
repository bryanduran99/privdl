# -*- coding: UTF-8 -*-
'''
1. task: 
    face recognition
2. network archetecture:
    following the impl of NeuraCrypt_Repo(vit_head as encoder and vit_tail as clf), except that mlp_head -> softmax_head
'''
import time
import datetime
import os
import math
import random
import cv2
import numpy as np
import requests
import torch as tc
from simpleViT import SimpleViT
import exp_utils
from refile import smart_open
exp_utils.setup_import_path()
import block
import utils
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from typing import TypeVar
from torch.nn.modules import Module

T = TypeVar('T', bound='Module')

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
init_seed(int(time.time()))
# init_seed(11)

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

class Ext_Top1_Tester(block.test.top1_test.Top1_Tester):
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
                batch_feats = model_lit(batch_inputs, batch_labels, is_inference=True)
                feats.append(utils.batch_to_numpy(batch_feats))
                labels.append(utils.batch_to_numpy(batch_labels))
        return np.concatenate(feats), np.concatenate(labels)

def cdist(feat0, feat1):
    '''L2norm'''
    return np.linalg.norm(feat0 - feat1)

class RestoreIdentificationAccuracy(Ext_Top1_Tester):
    '''对期望攻击模型进行重建识别准确率测试'''

    def test(self, model_lit, return_top1_dict=False):
        '''重建识别准确率测试，print 测试结果，return 测试的 log dict'''
        # 待检索的泄露 query 和攻击者的 base 要用不同的混淆 key
        if dist.is_available() and dist.is_initialized():
            model_lit = model_lit.module

        base_feats, labels = self.extract_feats(model_lit)
        query_feats, labels = self.extract_feats(model_lit)

        person_dict = utils.defaultdict(list) # 按 label 对 sample 分类
        for person_id, label in enumerate(labels):
            person_dict[label].append(person_id)
        # 每类取一个 sample 作为 base
        base_dict = {label: ids.pop() for label, ids in person_dict.items()}
        top1_dict = utils.defaultdict(list) # {query_label: [(query_id, top1_id, top1_label), ...]}

        acc = total = 0 # top1 准确数，总测试数
        for query_label, query_ids in utils.tqdm(person_dict.items(), 'RIA test'):
            for query_id in query_ids: # 对每类的每个 query 进行测试
                top1_dist, top1_label = min( # 取该 query 的 top1 的 label
                    (cdist(base_feats[base_id], query_feats[query_id]), base_label)
                    for base_label, base_id in base_dict.items())
                top1_dict[query_label].append((query_id, base_dict[top1_label], top1_label))
                # 看 top1 的 label 和该 query 的 label 是否一致
                acc += int(top1_label == query_label)
                total += 1 # 总测试数 +1

        print(f'{self.name} acc={acc}/{total}={acc/total*100:.2f}%')

        ret = dict(acc=acc, total=total, rate=acc/total)
        if return_top1_dict:
            ret['top1_dict'] = top1_dict
            ret['base_dict'] = base_dict
        return ret

class NC_Trainer(block.train.standard.Trainer):
    """Trainer的子类，重写其中的fit函数"""
    def __init__(self, dataset, total_epochs, scheduler=None, val_set=None, work_dir=None,ckpt_dir=None, sample_transform=None, optimizer=None, monitor=None):
        super().__init__(dataset, total_epochs, val_set, work_dir, ckpt_dir)
        self.best_acc = 0
        self.best_epoch = 0
        self.sample_transform = sample_transform
        self.optimizer = optimizer
        # self.scheduler = CosineAnnealingLR(self.optimizer, T_max=15, eta_min=1e-3)
        self.scheduler = scheduler
        self.monitor = monitor
        assert self.sample_transform is not None # 为保证效果，必须要有sample_transform
    

    def fit_NC(self, model_lit, local_rank=-1):
        '''训练模型并返回练完成的模型'''
        # 数据加载
        # self.call_testers(clock, model_lit)
        dataloader = self.get_dataloader(self.sample_transform) # 加载训练集
        clock = utils.TrainLoopClock(dataloader, self.total_epochs) # 配置训练循环

        # 模型训练
        # catch_nan = len(dataloader)
        for batch_data in clock: # 训练循环
            if local_rank != -1:
                batch_data = utils.batch_to_rank(batch_data, local_rank=local_rank)
            else:
                batch_data = utils.batch_to_cuda(batch_data)
            imgs, labels = batch_data

            # train step
            model_lit.train()
            scores = model_lit(imgs, labels, is_inference=False)
            loss = tc.nn.functional.cross_entropy(scores, labels)
            utils.step(self.optimizer, loss)

            self.monitor.add('loss', lambda: float(loss),
                            to_str=lambda x: f'{x:.2e}')  # 监控 loss
            self.monitor.add('batch_acc', lambda: utils.accuracy(scores, labels),
                            to_str=lambda x: f'{x * 100:.2f}%')  # 监控 batch 准确率
            
            # log, test, save ckpt  
            self.add_log(clock, self.monitor) # 添加 log
            if dist.is_available() and dist.is_initialized():
                dist.barrier() # 等待所有进程都执行到这里

            if clock.epoch_end():
                if self.scheduler is not None:
                    self.scheduler.step()
                if local_rank == -1 or dist.get_rank() == 0: 
                # 当没有初始化多线程时，因为短路原则，也不会因为get_rank而报错
                    self.call_testers(clock, model_lit) 
                    # 只保留精度最佳的ckpt，节省时间
                    if self.current_accuracy > self.best_acc:
                        self.best_acc = self.current_accuracy
                        self.best_epoch = clock.epoch
                        if local_rank == -1:
                                best_model_state_dict = {k:v.to('cpu') for k, v in model_lit.state_dict().items()}
                        else:
                            best_model_state_dict = {k:v.to('cpu') for k, v in model_lit.module.state_dict().items()}
                        
                        sche_dict = self.scheduler.state_dict() if self.scheduler is not None else None
                        ckpt = {'best_epoch': self.best_epoch,
                                'best_acc': self.best_acc,
                                'model_state_dict': best_model_state_dict,
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'scheduler_state_dict': sche_dict
                                }
                        utils.torch_save(ckpt, f'{self.ckpt_dir}/best_ckpt.pth')
                        del ckpt
                        # best_model_state_dict = {k:v.to('cpu') for k, v in model_lit.state_dict().items()}
                        # utils.torch_save(best_model_state_dict, os.path.join(self.work_dir, 'best_model.tar'))
                if local_rank != -1:
                    # 在多卡模式下在主进程测试并save，其他进程通过torch.distributed.barrier()来等待主进程完成validate操作
                    # barrier()函数的使用要非常谨慎，如果只有单个进程包含了这条语句，那么程序就会陷入无限等待
                    dist.barrier()
        return model_lit

class ERA_Trainer(NC_Trainer):
    '''强行在期望混淆后的数据上训练识别模型'''
    def fit_clf(self, model_lit, local_rank=-1):
        '''训练模型并返回练完成的模型'''
        # 数据加载
        dataloader = self.get_dataloader(self.sample_transform) # 加载训练集
        clock = utils.TrainLoopClock(dataloader, self.total_epochs) # 配置训练循环

        # 模型训练
        for batch_data in clock: # 训练循环
            if local_rank != -1:
                batch_data = utils.batch_to_rank(batch_data, local_rank=local_rank)
            else:
                batch_data = utils.batch_to_cuda(batch_data)
            imgs, labels = batch_data

            # train step
            if dist.is_available() and dist.is_initialized():
                model_lit.module.obfuscate.reset_parameters()
            else:
                model_lit.obfuscate.reset_parameters()
            model_lit.train()
            scores = model_lit(imgs, labels, is_inference=False)
            loss = tc.nn.functional.cross_entropy(scores, labels)
            utils.step(self.optimizer, loss)

            self.monitor.add('loss', lambda: float(loss),
                            to_str=lambda x: f'{x:.2e}')  # 监控 loss
            self.monitor.add('batch_acc', lambda: utils.accuracy(scores, labels),
                            to_str=lambda x: f'{x * 100:.2f}%')  # 监控 batch 准确率
            
            # log, test, save ckpt    
            self.add_log(clock, self.monitor) # 添加 log
            
            if clock.epoch_end(): # 当 epoch 结束时需要执行的操作
                # 当没有初始化多线程时，因为短路原则，也不会因为get_rank而报错
                if local_rank == -1 or dist.get_rank() == 0: 
                    self.call_testers(clock, model_lit) 
                    # 只保留精度最佳的ckpt，节省时间
                    if self.current_accuracy > self.best_acc:
                        self.best_acc = self.current_accuracy
                        self.best_epoch = clock.epoch
                        if local_rank == -1:
                                best_model_state_dict = {k:v.to('cpu') for k, v in model_lit.state_dict().items()}
                        else:
                            best_model_state_dict = {k:v.to('cpu') for k, v in model_lit.module.state_dict().items()}
                        ckpt = {'best_epoch': self.best_epoch,
                                'best_acc': self.best_acc,
                                'model_state_dict': best_model_state_dict,
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'scheduler_state_dict': self.scheduler.state_dict()
                                }
                        utils.torch_save(ckpt, f'{self.ckpt_dir}/best_ckpt.pth')
                        del ckpt
                        # best_model_state_dict = {k:v.to('cpu') for k, v in model_lit.state_dict().items()}
                        # utils.torch_save(best_model_state_dict, os.path.join(self.work_dir, 'best_model.tar'))
                if local_rank != -1:
                    # 在多卡模式下在主进程测试并save，其他进程通过torch.distributed.barrier()来等待主进程完成validate操作
                    # barrier()函数的使用要非常谨慎，如果只有单个进程包含了这条语句，那么程序就会陷入无限等待
                    dist.barrier()
        return model_lit

# 
def load_checkpoint(model, pretrained_path):
    print("=> loading checkpoint '{}'".format(pretrained_path))
    checkpoint = tc.load(pretrained_path, map_location="cpu")
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('module.base_encoder.'):
            state_dict[k[len('module.base_encoder.'):]] = state_dict[k]
            del state_dict[k]
    msg = model.load_state_dict(state_dict, strict=False)
    # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    print("=> loaded checkpoint")
    return model


class NC_Single_Lit(block.model.light.ModelLight):
    '''单客户场景下的 XNN'''

    def __init__(self, class_num, obf, ckpt_path=None, local_rank=-1):
        super().__init__()
        self.class_num = class_num
        
        self.encoder = block.model.moco_v3_vits.vit_base_privencoder(obf=obf,layer=5)
        # summary(self.ext, input_size=(3, 224, 224), device="cpu")
        if ckpt_path is not None:
            ckpt = tc.load(ckpt_path,map_location=tc.device(local_rank))
            self.encoder.load_state_dict(ckpt['model_state_dict'], strict=False)
            for p in self.encoder.parameters():
                p.requires_grad = False
            del ckpt

        
        self.clf = block.model.moco_v3_vits.vit_base_clf(layer=6)

        self.softmax = block.loss.amsoftmax.AMSoftmax(
            in_features=self.clf.embed_dim, class_num=self.class_num)


    def forward(self, images, labels, is_inference=False):
        feats = self.encoder(images)
        logits = self.clf(feats)

        if is_inference:   
            return logits    
        out = self.softmax(logits, labels)
        return out


def split_dataset(dataset, person_num):
    hubble_utils = block.dataset.hubble.hubble_utils
    trainset, testset_by_person = hubble_utils.split_dataset_by_person(
        dataset, test_id_num=person_num, test_img_per_id=20)
    trainset, testset_by_img = hubble_utils.split_dataset_by_img(
        trainset, test_id_num=person_num, test_img_per_id=20, train_scale=1.5)
    return trainset, testset_by_img, testset_by_person

def main():
    # 配置实验设置
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--train_dataset", default='celeba', type=str)
    parser.add_argument("--attacker_dataset", default='facescrub',type=str)
    
    parser.add_argument("--mode", default="NC", type=str, help="NC/Attack/Debug_mode")
    parser.add_argument("--obf", default="patch", type=str, help="dim/dim_equal/patch/no")
    
    parser.add_argument("--lr", default=5e-2, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--subset_ratio", default=1.0, type=float)

    
    FLAGS = parser.parse_args()
    bool_type = lambda x: x.lower() == 'true'
    local_rank = FLAGS.local_rank

    proc_count = 1
    if local_rank != -1:
        dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=18000))  # nccl是GPU设备上最快、最推荐的后端
        proc_count = dist.get_world_size() #获取全局并行数，即并行训练的显卡的数量
    
    batch_size = FLAGS.batch_size // proc_count
    num_workers = 8
    train_epoch = 1000
    
    train_dataset = FLAGS.train_dataset
    attacker_dataset = FLAGS.attacker_dataset
    print("train_dataset:", train_dataset)


    script_file_name = os.path.basename(__file__)
    result_path = exp_utils.setup_result_path(script_file_name)
    result_path = f'{result_path}/{FLAGS.mode}/{FLAGS.obf}/{train_dataset}/{attacker_dataset}/{proc_count}_gpus'
    work_dir = result_path
    print('result_path:',result_path)
    
    alert_info = f'029-2_{FLAGS.mode}_{FLAGS.obf}_{train_dataset}_{attacker_dataset}_{proc_count}gpu'
    ckpt_dir = f'/data/ckpt/NC/mode{FLAGS.mode}_obf{FLAGS.obf}_{train_dataset}_{attacker_dataset}_{proc_count}gpu'
    print('ckpt_dir:', ckpt_dir)
    if FLAGS.mode == 'Attack':
        load_ckpt_path = f'/data/ckpt/NC/modeNC_obf{FLAGS.obf}_{train_dataset}_facescrub_{proc_count}gpu/best_ckpt.pth'
        assert os.path.exists(load_ckpt_path)


    if local_rank == -1 or dist.get_rank() == 0:
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

    if train_dataset == 'celeba':
        train_dataset = block.dataset.hubble.xnn_paper.celeba() # 20w
        normalize = celeba_normalize
    elif train_dataset == 'msra':
        train_dataset = block.dataset.hubble.xnn_paper.msra() # 300w+, 8w id[s
        normalize = msra_normalize
    elif train_dataset == 'webface':
        train_dataset = block.dataset.hubble.xnn_paper.webface() # 50w
        normalize = webface_normalize
    elif train_dataset == 'vggface2':
        train_dataset = block.dataset.hubble.xnn_paper.vggface2() # 200w+
        normalize = vggface2_normalize
    else:
        raise ValueError('dataset do not exist')
    
    trainset, testset_by_img, testset_by_person = split_dataset(train_dataset,1000)
    if 0 < FLAGS.subset_ratio < 1.0:
        subset_size = len(train_dataset) * FLAGS.subset_ratio
        train_dataset = train_dataset.subset_by_size(subset_size)
    
    if attacker_dataset == 'facescrub':
        attacker_dataset = block.dataset.hubble.xnn_paper.facescrub()
        normalize = facescrub_normalize
    elif attacker_dataset == 'imdb':
        attacker_dataset = block.dataset.hubble.xnn_paper.imdb()
        normalize = imdb_normalize
    else: 
        raise Exception("Invalid Dataset:", attacker_dataset)

    if FLAGS.mode == 'Debug_mode':
        trainset = attacker_dataset

    def sample_transform(sample):
        '''H*W*3*uint8(0\~255) -> 3*224*224*float32(-1\~1)'''
        img, label = sample
        np_img = resize_norm_transpose_insta(img, size=224)
        transformed_img = normalize(np_img)
        return transformed_img, label


    Tester = Ext_Top1_Tester
    testers = [Tester(dataset=testset_by_img, name='testset_by_img', sample_transform=sample_transform),
               Tester(dataset=testset_by_person, name='testset_by_person', sample_transform=sample_transform)]
    for tester in testers:
        tester.config_dataloader(batch_size=batch_size, num_workers=num_workers)

    def train_NC_vit(): 

        model_lit = NC_Single_Lit(class_num=trainset.class_num(), obf=FLAGS.obf)
        # summary(model_lit.encoder, input_size=(3, 224, 224), device="cpu")
        # summary(model_lit.clf, input_size=(197, 768), device="cpu")

        if local_rank == -1:
            device = tc.device('cuda' if tc.cuda.is_available else 'cpu')
            model_lit = model_lit.to(device)
        else:
            print("move model to", local_rank)
            model_lit = model_lit.to(local_rank)
            model_lit = DDP(model_lit, device_ids=[local_rank], output_device=local_rank)
            
        optimizer = tc.optim.SGD(
                                filter(lambda p: p.requires_grad, model_lit.parameters()), 
                                lr=FLAGS.lr, 
                                momentum=0.9)
        monitor = utils.Monitor()
        monitor.add('lr',  # 监控学习率
                         get=lambda: optimizer.param_groups[0]['lr'],
                         to_str=lambda x: f'{x:.1e}')
        
        warm_up_iter = 3
        T_max = 13 # warm_up + cosAnnealLR 的周期
        lr_max = FLAGS.lr
        lr_min = 1e-3
        
        lambda_lr = lambda cur_iter: max(1e-3/lr_max, cur_iter/warm_up_iter) if cur_iter < warm_up_iter else \
            (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos((cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi))) / lr_max
        # if FLAGS.train_dataset == 'vggface2': # vggface2使用大学习率过拟合，尝试全程使用1e-3
            # lambda_lr = lambda cur_iter: lr_min/lr_max
        scheduler = tc.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

        trainer = NC_Trainer(
                            dataset=trainset, 
                            total_epochs=train_epoch, 
                            scheduler=scheduler, 
                            work_dir=f'{work_dir}', 
                            ckpt_dir=ckpt_dir,
                            sample_transform=sample_transform, 
                            optimizer=optimizer, 
                            monitor=monitor)
        
        trainer.config_dataloader(batch_size=batch_size, num_workers=num_workers)
        log_interval = 60
        trainer.config_logger(log_interval=log_interval)
        trainer.config_tester(testers)

        if dist.is_available() and dist.is_initialized():
            dist.barrier() # 等待所有进程都执行到这里
        trainer.fit_NC(model_lit, local_rank=local_rank)


    def attack_NC(): # 期望攻击
        model_lit = NC_Single_Lit(class_num=attacker_dataset.class_num(), obf=FLAGS.obf, ckpt_path=load_ckpt_path, local_rank=local_rank)
        
        if local_rank == -1:
            device = tc.device('cuda' if tc.cuda.is_available else 'cpu')
            model_lit = model_lit.to(device)
        else:
            model_lit = model_lit.to(local_rank)
            model_lit = DDP(model_lit, device_ids=[local_rank], output_device=local_rank)

        optimizer = tc.optim.SGD(filter(lambda p: p.requires_grad, model_lit.parameters()), lr=FLAGS.lr)
        monitor = utils.Monitor()
        monitor.add('lr',  # 监控学习率
                         get=lambda: optimizer.param_groups[0]['lr'],
                         to_str=lambda x: f'{x:.1e}')

        RIA = RestoreIdentificationAccuracy(
            dataset=testset_by_person, name='RestoreIdentificationAccuracy', sample_transform=sample_transform)
        RIA.config_dataloader(batch_size=batch_size, num_workers=num_workers)

        trainer = ERA_Trainer(
            dataset=attacker_dataset, total_epochs=train_epoch, work_dir=work_dir, \
            ckpt_dir=ckpt_dir, sample_transform=sample_transform,optimizer=optimizer, monitor=monitor)
        trainer.config_dataloader(batch_size=batch_size, num_workers=num_workers)                 
        trainer.config_logger(log_interval=10)
        trainer.config_tester([RIA])

        trainer.fit_NC(model_lit, local_rank=local_rank)
    
    if FLAGS.mode == 'NC':
            train_NC_vit()
    elif FLAGS.mode == 'Attack':
            attack_NC()
    # try:
    #     if FLAGS.mode == 'NC':
    #         train_NC_vit()
    #     elif FLAGS.mode == 'Attack':
    #         attack_NC()
    # except Exception as e:
    #     print("Exception:", e)
    #     if local_rank == -1 or dist.get_rank() == 0:
    #         requests.get(f'https://api.day.app/ocZjTT3y2AhsowCaiKtqLC/break_down/{alert_info}_{time.strftime("%y%m%d%H%M%S",time.localtime())}')

if __name__ == '__main__':
    main()

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 -m torch.distributed.launch --nproc_per_node 8 029-2.NeuraCrypt_OnXNN_vitb16.py