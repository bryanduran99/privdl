# -*- coding: UTF-8 -*-
'''
Implementation slightly adapted from "033".
Optimizing strategy: SGD + (large lr + large momentum)  + (warm_up + CosineAnnealingLR), 这套组合很猛！
feature encoder LN not train
noise   encoder train
paper version
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
import subprocess

from functools import partial, reduce

def configure_nccl():
    """Configure multi-machine environment variables.
 
    It is required for multi-machine training.
    """
    os.environ["NCCL_LAUNCH_MODE"] = "PARALLEL"
    # os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_IB_HCA"] = subprocess.getoutput(
       "cd /sys/class/infiniband/ > /dev/null; for i in mlx5_*; "
       "do cat $i/ports/1/gid_attrs/types/* 2>/dev/null "
    "| grep v >/dev/null && echo $i ; done; > /dev/null"
    )
    # os.environ["NCCL_IB_HCA"] = "mlx5_0,mlx5_1,mlx5_2,mlx5_5"
    # os.environ["NCCL_IB_HCA"] = "mlx4_0"    # for cx3 ib hca
    os.environ["NCCL_IB_GID_INDEX"] = "3"
    os.environ["NCCL_IB_TC"] = "106"
    os.environ["NCCL_IB_DISABLE"] = "1"
    os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
    os.environ["GLOO_SOCKET_IFNAME"] = "eth0"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["NCCL_TREE_THRESHOLD"] = "0"
configure_nccl()

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
# init_seed(int(time.time()))
init_seed(11)

def loss_fn_kd(outputs, labels, teacher_outputs):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    
    alpha = 0.3
    T = 4

    KD_loss = tc.nn.KLDivLoss(reduction='sum')(tc.nn.functional.log_softmax(outputs/T, dim=1),
                             tc.nn.functional.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) / outputs.shape[0]
    fc_loss = tc.nn.functional.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss,fc_loss


def loss_fn_l2(outputs, labels, teacher_outputs):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = 1.0
    distill_loss = tc.nn.MSELoss(reduction='mean')(outputs , teacher_outputs) * alpha
    fc_loss = tc.nn.functional.cross_entropy(outputs, labels) * (1. - alpha)

    return distill_loss,fc_loss



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
        model_lit.obfuscate.reset_parameters()
        base_feats, labels = self.extract_feats(model_lit)
        model_lit.obfuscate.reset_parameters()
        query_feats, labels = self.extract_feats(model_lit)     # query_feats_shape: (20000,197,768)    label:(20000)

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

def load_checkpoint(model, pretrained_path,strict = False):
    print("=> loading checkpoint '{}'".format(pretrained_path))
    checkpoint = tc.load(pretrained_path, map_location="cpu")

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint['model_state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('module.base_encoder.'):
            state_dict[k[len('module.base_encoder.'):]] = state_dict[k]
            del state_dict[k]
    msg = model.load_state_dict(state_dict, strict=strict)
    # print(msg)
    # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    print("=> loaded checkpoint")
    return model

class ExtTrainer_face_adv(block.train.standard.Trainer):
    """Trainer的子类，重写其中的fit函数"""
    def __init__(self, dataset, total_epochs, val_set=None, work_dir=None, sample_transform=None,\
                  optimizer= None,optimizer_encoder=None, optimizer_encoder_obf=None,optimizer_server_tail=None, 
                  scheduler = None,scheduler_encoder = None,scheduler_server_tail = None,scheduler_encoder_obf = None,
                    monitor=None, tb_log_dir=None, ckpt_dir=None):
        super().__init__(dataset, total_epochs, val_set, work_dir, ckpt_dir)
        self.best_acc = 0
        self.best_epoch = 0
        self.sample_transform = sample_transform
        self.optimizer = optimizer
        self.optimizer_encoder = optimizer_encoder
        self.optimizer_encoder_obf = optimizer_encoder_obf
        self.optimizer_server_tail = optimizer_server_tail

        # self.scheduler = CosineAnnealingLR(self.optimizer, T_max=15, eta_min=1e-3)
        self.scheduler = scheduler
        self.scheduler_encoder = scheduler_encoder
        self.scheduler_server_tail = scheduler_server_tail
        self.scheduler_encoder_obf = scheduler_encoder_obf

        self.monitor = monitor
        self.tb_writer = SummaryWriter(log_dir=tb_log_dir)
        assert self.sample_transform is not None # 为保证效果，必须要有sample_transform
    

    def config_logger(self, log_interval=60, save_interval=10*60,local_rank = -1):
        '''log_interval: 打印并记录 log 的间隔(单位秒)，
        save_interval: 保存 log 文件的间隔(单位秒)'''
        logs = []
        log_path = f'{self.work_dir}/logs_{time.strftime("%y%m%d%H%M%S",time.localtime())}.json'
       

        def add_log(clock, monitor):
            '''把 clock 和 monitor 的信息打印并记到 log 中'''
            if local_rank == -1 or dist.get_rank() == 0: 
                logs.append(dict(clock=clock.check(), monitor=monitor.check()))
        
        def save_log():
            '''保存 log 到 work_dir 目录下'''
            if local_rank == -1 or dist.get_rank() == 0: 
                utils.json_save(logs, log_path)
        self.add_log = add_log
        self.save_log = save_log
        return log_path

    def fit_ViT(self, composed_model,local_rank=-1):
        '''训练模型并返回练完成的模型'''
        # 数据加载
        # self.call_testers(clock, model_lit)
        dataloader = self.get_dataloader(self.sample_transform) # 加载训练集
        clock = utils.TrainLoopClock(dataloader, self.total_epochs) # 配置训练循环
        

        # # before_train 
        # if local_rank == -1 or dist.get_rank() == 0: 
        #     self.call_testers(clock, composed_model) 

        # if local_rank != -1:
        #     # 在多卡模式下在主进程测试并save，其他进程通过torch.distributed.barrier()来等待主进程完成validate操作
        #     # barrier()函数的使用要非常谨慎，如果只有单个进程包含了这条语句，那么程序就会陷入无限等待
        #     dist.barrier()

        # 模型训练
        # catch_nan = len(dataloader)
        for batch_data in clock: # 训练循环
            if local_rank != -1:
                batch_data = utils.batch_to_rank(batch_data, local_rank=local_rank)
            else:
                batch_data = utils.batch_to_cuda(batch_data)
            imgs, labels = batch_data

            # train step
            composed_model.train()
            
            
            scores = composed_model(imgs, labels)
            loss = tc.nn.functional.cross_entropy(scores, labels)
            
            utils.step(self.optimizer, loss)

            if local_rank == -1 or dist.get_rank() == 0: 
                self.tb_writer.add_scalar('train_loss', loss, clock.batch)
                self.monitor.add('loss', lambda: float(loss),
                                to_str=lambda x: f'{x:.2e}')  # 监控 loss
                self.monitor.add('batch_acc', lambda: utils.accuracy(scores, labels),
                                to_str=lambda x: f'{x * 100:.2f}%')  # 监控 batch 准确率
                
                # log, test, save ckpt  
            
            

            if clock.epoch_end(): # 当 epoch 结束时需要执行的操作
                self.scheduler.step()

                if local_rank == -1 or dist.get_rank() == 0: 
                    self.add_log(clock, self.monitor) # 添加 log
                    self.save_log()


                if  (clock.epoch + 1) % 1 == 0 and (clock.epoch + 1) > 1: # 不要每个epoch 都测试，减少测试所需要的时间
                    time.sleep(0.003)
                    # 当没有初始化多线程时，因为短路原则，也不会因为get_rank而报错
                    if local_rank == -1 or dist.get_rank() == 0: 
                        self.call_testers(clock, composed_model) 
                        # 只保留精度最佳的ckpt，节省时间
                        self.tb_writer.add_scalar('test_by_person', self.current_accuracy, clock.epoch)
                        if True:
                            self.best_acc = self.current_accuracy
                            self.best_epoch = clock.epoch
                            if local_rank == -1:
                                best_model_state_dict = {k:v.to('cpu') for k, v in composed_model.server_tail.state_dict().items()}
                            else:
                                best_model_state_dict = {k:v.to('cpu') for k, v in composed_model.module.server_tail.state_dict().items()}
                            ckpt = {'best_epoch': self.best_epoch,
                                    'best_acc': self.best_acc,
                                    'model_state_dict': best_model_state_dict,
                                    'optimizer_state_dict': self.optimizer.state_dict(),
                                    'scheduler_state_dict': self.scheduler.state_dict()
                                    }
                            utils.torch_save(ckpt, f'{self.ckpt_dir}'+ '/restart_stage1_tail_epoch_{}.pth'.format(clock.epoch))
                            del ckpt


                            if local_rank == -1:
                                best_model_state_dict = {k:v.to('cpu') for k, v in composed_model.encoder.state_dict().items()}
                            else:
                                best_model_state_dict = {k:v.to('cpu') for k, v in composed_model.module.encoder.state_dict().items()}
                            ckpt = {'best_epoch': self.best_epoch,
                                    'best_acc': self.best_acc,
                                    'model_state_dict': best_model_state_dict,
                                    'optimizer_state_dict': self.optimizer.state_dict(),
                                    'scheduler_state_dict': self.scheduler.state_dict()
                                    }
                            utils.torch_save(ckpt, f'{self.ckpt_dir}'+ '/restart_stage1_encoder_epoch_{}.pth'.format(clock.epoch))
                            del ckpt

                            # best_model_state_dict = {k:v.to('cpu') for k, v in model_lit.state_dict().items()}
                            # utils.torch_save(best_model_state_dict, os.path.join(self.work_dir, 'best_model.tar'))
                    
                    if local_rank == -1 or dist.get_rank() == 0: 
                        self.tb_writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], clock.epoch)       
                    if local_rank != -1:
                        # 在多卡模式下在主进程测试并save，其他进程通过torch.distributed.barrier()来等待主进程完成validate操作
                        # barrier()函数的使用要非常谨慎，如果只有单个进程包含了这条语句，那么程序就会陷入无限等待
                        dist.barrier()
        self.tb_writer.close()
        return composed_model

class ExtTrainer_face_adv_new_stage2(block.train.standard.Trainer):
    """Trainer的子类，重写其中的fit函数"""
    def __init__(self, dataset, total_epochs, val_set=None, work_dir=None, sample_transform=None,\
                  optimizer_aux_server_tail = None, optimizer_obf_feature_generator=None,optimizer_aux_server_tail_distill = None, \
                  scheduler_obf_feature_generator = None, scheduler_aux_server_tail = None ,scheduler_aux_server_tail_distill = None, \
                    monitor=None, tb_log_dir=None, ckpt_dir=None):
        super().__init__(dataset, total_epochs, val_set, work_dir, ckpt_dir)
        self.best_acc = 0
        self.best_epoch = 0
        self.sample_transform = sample_transform

        self.optimizer_aux_server_tail = optimizer_aux_server_tail
        self.optimizer_obf_feature_generator = optimizer_obf_feature_generator
        self.optimizer_aux_server_tail_distill = optimizer_aux_server_tail_distill


        self.scheduler_obf_feature_generator = scheduler_obf_feature_generator
        self.scheduler_aux_server_tail = scheduler_aux_server_tail
        self.scheduler_aux_server_tail_distill = scheduler_aux_server_tail_distill


        self.monitor = monitor
        self.tb_writer = SummaryWriter(log_dir=tb_log_dir)
        assert self.sample_transform is not None # 为保证效果，必须要有sample_transform
    
    def fit_Adv_Train(self,obf_feature_generator,aux_server_tail,aux_server_tail_distill,aux_teacher_tail,test_model,test_model2,local_rank=-1):
        '''训练模型并返回练完成的模型'''
        # 数据加载
        # self.call_testers(clock, model_lit)
        dataloader = self.get_dataloader(self.sample_transform) # 加载训练集
        clock = utils.TrainLoopClock(dataloader, self.total_epochs) # 配置训练循环


        # # before_train 
        # if local_rank == -1 or dist.get_rank() == 0: 
        #     self.call_testers(clock, test_model) 

        # if local_rank != -1:
        #     # 在多卡模式下在主进程测试并save，其他进程通过torch.distributed.barrier()来等待主进程完成validate操作
        #     # barrier()函数的使用要非常谨慎，如果只有单个进程包含了这条语句，那么程序就会陷入无限等待
        #     dist.barrier()



        # 模型训练
        # catch_nan = len(dataloader)
        for batch_data in clock: # 训练循环
            if local_rank != -1:
                batch_data = utils.batch_to_rank(batch_data, local_rank=local_rank)
            else:
                batch_data = utils.batch_to_cuda(batch_data)
            imgs, labels = batch_data

            # train step
            # test_model.train() # test model 中有obf_feature_generator aux_server_tail
            obf_feature_generator.train()
            aux_server_tail.train()

            # generator
            self.optimizer_obf_feature_generator.zero_grad()
                #forward
            obf_features = obf_feature_generator(imgs)
            scores = aux_server_tail(obf_features,labels)  
                #loss
            loss_G = -tc.nn.functional.cross_entropy(scores, labels)
            utils.step(self.optimizer_obf_feature_generator, loss_G)


            # aux_classifier
            self.optimizer_aux_server_tail.zero_grad()
                #forward
            obf_features = obf_feature_generator(imgs.detach())
            scores = aux_server_tail(obf_features,labels)
                #loss
            loss_D = tc.nn.functional.cross_entropy(scores, labels)
            utils.step(self.optimizer_aux_server_tail, loss_D)

            # distill
            self.optimizer_aux_server_tail_distill.zero_grad()
            mid_features = obf_feature_generator.base_encoder(imgs.detach())
            clean_score = aux_teacher_tail(mid_features,labels)

            mid_features = obf_feature_generator(imgs.detach())
            score = aux_server_tail(mid_features,labels)
            loss_KD = loss_fn_kd(clean_score,score)
            utils.step(self.optimizer_aux_server_tail_distill, loss_KD)

            

            self.tb_writer.add_scalar('loss_G', loss_G, clock.batch)
            self.tb_writer.add_scalar('loss_D', loss_D, clock.batch)
            self.tb_writer.add_scalar('loss_KD', loss_D, clock.batch)


            self.monitor.add('loss_D', lambda: float(loss_D),
                            to_str=lambda x: f'{x:.2e}')  # 监控 loss
            self.monitor.add('loss_G', lambda: float(loss_G),
                            to_str=lambda x: f'{x:.2e}')  # 监控 loss
            self.monitor.add('loss_KD', lambda: float(loss_KD),
                            to_str=lambda x: f'{x:.2e}')  # 监控 loss
            
            self.monitor.add('loss', lambda: float(loss_G + loss_D + loss_KD),
                            to_str=lambda x: f'{x:.2e}')  # 监控 loss
            self.monitor.add('batch_acc', lambda: utils.accuracy(scores, labels),
                            to_str=lambda x: f'{x * 100:.2f}%')  # 监控 batch 准确率
            
            # log, test, save ckpt  
            self.add_log(clock, self.monitor) # 添加 log
            

            if clock.epoch_end():
                self.scheduler_aux_server_tail.step()
                self.scheduler_obf_feature_generator.step()
                self.scheduler_aux_server_tail.step()

            if clock.epoch_end() and (clock.epoch + 1) % 3 == 0 : # 当 epoch 结束时需要执行的操作
                
                # 当没有初始化多线程时，因为短路原则，也不会因为get_rank而报错
                if local_rank == -1 or dist.get_rank() == 0: 
                    self.call_testers(clock, test_model) 
                    self.call_testers(clock, test_model2) 
                    # 只保留精度最佳的ckpt，节省时间
                    self.tb_writer.add_scalar('test_by_person', self.current_accuracy, clock.epoch)
                    if True:  # 永远保存最新的
                        self.best_acc = self.current_accuracy
                        self.best_epoch = clock.epoch
                        if local_rank == -1:
                            best_model_state_dict = {k:v.to('cpu') for k, v in obf_feature_generator.state_dict().items()}
                            aux_server_tail_state_dict = {k:v.to('cpu') for k, v in aux_server_tail.state_dict().items()}
                        else:
                            best_model_state_dict = {k:v.to('cpu') for k, v in obf_feature_generator.module.state_dict().items()}
                            aux_server_tail_state_dict = {k:v.to('cpu') for k, v in aux_server_tail.module.state_dict().items()}
                        ckpt = {'best_epoch': self.best_epoch,
                                'best_acc': self.best_acc,
                                'model_state_dict': best_model_state_dict,
                                'optimizer_state_dict': self.optimizer_obf_feature_generator.state_dict(),
                                'scheduler_state_dict': self.scheduler_obf_feature_generator.state_dict()
                                }
                        utils.torch_save(ckpt, f'{self.ckpt_dir}/best_obf_feature_generator_ckpt.pth')
                        del ckpt

                        ckpt_aux = {'best_epoch': self.best_epoch,
                                'best_acc': self.best_acc,
                                'model_state_dict': aux_server_tail_state_dict,
                                'optimizer_state_dict': self.optimizer_aux_server_tail.state_dict(),
                                'scheduler_state_dict': self.scheduler_aux_server_tail.state_dict()
                                }
                        utils.torch_save(ckpt_aux, f'{self.ckpt_dir}/best_aux_server_tail_ckpt.pth')
                        del ckpt_aux

                        # best_model_state_dict = {k:v.to('cpu') for k, v in model_lit.state_dict().items()}
                        # utils.torch_save(best_model_state_dict, os.path.join(self.work_dir, 'best_model.tar'))
                if local_rank != -1:
                    # 在多卡模式下在主进程测试并save，其他进程通过torch.distributed.barrier()来等待主进程完成validate操作
                    # barrier()函数的使用要非常谨慎，如果只有单个进程包含了这条语句，那么程序就会陷入无限等待
                    dist.barrier()
                
                self.tb_writer.add_scalar('lr', self.optimizer_obf_feature_generator.param_groups[0]['lr'], clock.epoch)
        
        self.tb_writer.close()
        return obf_feature_generator,aux_server_tail


class ExtTrainer_face_adv_stage2(block.train.standard.Trainer):
    """Trainer的子类，重写其中的fit函数"""
    def __init__(self, dataset, total_epochs, val_set=None, work_dir=None, sample_transform=None,\
                  optimizer_aux_server_tail = None, optimizer_obf_feature_generator=None, \
                  scheduler_obf_feature_generator = None, scheduler_aux_server_tail = None, \
                    monitor=None, tb_log_dir=None, ckpt_dir=None):
        super().__init__(dataset, total_epochs, val_set, work_dir, ckpt_dir)
        self.best_acc = 0
        self.best_epoch = 0
        self.sample_transform = sample_transform

        self.optimizer_aux_server_tail = optimizer_aux_server_tail
        self.optimizer_obf_feature_generator = optimizer_obf_feature_generator

        self.scheduler_obf_feature_generator = scheduler_obf_feature_generator
        self.scheduler_aux_server_tail = scheduler_aux_server_tail


        self.monitor = monitor
        self.tb_writer = SummaryWriter(log_dir=tb_log_dir)
        assert self.sample_transform is not None # 为保证效果，必须要有sample_transform
    
    def config_logger(self, log_interval=60, save_interval=10*60,local_rank = -1):
        '''log_interval: 打印并记录 log 的间隔(单位秒)，
        save_interval: 保存 log 文件的间隔(单位秒)'''
        logs = []
        log_path = f'{self.work_dir}/logs_{time.strftime("%y%m%d%H%M%S",time.localtime())}.json'
       

        def add_log(clock, monitor):
            '''把 clock 和 monitor 的信息打印并记到 log 中'''
            if local_rank == -1 or dist.get_rank() == 0: 
                logs.append(dict(clock=clock.check(), monitor=monitor.check()))
        
        def save_log():
            '''保存 log 到 work_dir 目录下'''
            if local_rank == -1 or dist.get_rank() == 0: 
                utils.json_save(logs, log_path)
        self.add_log = add_log
        self.save_log = save_log
        return log_path

    def fit_Adv_Train(self,obf_feature_generator,aux_server_tail,test_model,local_rank=-1):
        '''训练模型并返回练完成的模型'''
        # 数据加载
        # self.call_testers(clock, model_lit)
        dataloader = self.get_dataloader(self.sample_transform) # 加载训练集
        clock = utils.TrainLoopClock(dataloader, self.total_epochs) # 配置训练循环

        G_weight = 0.7
        D_weight = 0.3


        # before_train 
        # if local_rank == -1 or dist.get_rank() == 0: 
        #     self.call_testers(clock, test_model) 

        # if local_rank != -1:
        #     # 在多卡模式下在主进程测试并save，其他进程通过torch.distributed.barrier()来等待主进程完成validate操作
        #     # barrier()函数的使用要非常谨慎，如果只有单个进程包含了这条语句，那么程序就会陷入无限等待
        #     dist.barrier()



        # 模型训练
        # catch_nan = len(dataloader)
        for batch_data in clock: # 训练循环
            if local_rank != -1:
                batch_data = utils.batch_to_rank(batch_data, local_rank=local_rank)
            else:
                batch_data = utils.batch_to_cuda(batch_data)
            imgs, labels = batch_data

            # train step
            # test_model.train() # test model 中有obf_feature_generator aux_server_tail
            obf_feature_generator.train()
            aux_server_tail.train()

            # generator
            self.optimizer_obf_feature_generator.zero_grad()
                #forward
            obf_features = obf_feature_generator(imgs)
            scores = aux_server_tail(obf_features,labels)  
                #loss
            loss_G = -G_weight * tc.nn.functional.cross_entropy(scores, labels)
            utils.step(self.optimizer_obf_feature_generator, loss_G)


            # aux_classifier
            self.optimizer_aux_server_tail.zero_grad()
                #forward
            obf_features = obf_feature_generator(imgs.detach())
            scores = aux_server_tail(obf_features,labels)
                #loss
            loss_D = D_weight * tc.nn.functional.cross_entropy(scores, labels)
            utils.step(self.optimizer_aux_server_tail, loss_D)
            
            if local_rank == -1 or dist.get_rank() == 0: 
                self.tb_writer.add_scalar('loss_G', loss_G, clock.batch)
                self.tb_writer.add_scalar('loss_D', loss_D, clock.batch)


                self.monitor.add('loss_D', lambda: float(loss_D),
                                to_str=lambda x: f'{x:.2e}')  # 监控 loss
                self.monitor.add('loss_G', lambda: float(loss_G),
                                to_str=lambda x: f'{x:.2e}')  # 监控 loss
                
                self.monitor.add('loss', lambda: float(loss_G + loss_D),
                                to_str=lambda x: f'{x:.2e}')  # 监控 loss
                self.monitor.add('batch_acc', lambda: utils.accuracy(scores, labels),
                                to_str=lambda x: f'{x * 100:.2f}%')  # 监控 batch 准确率
                

            

            if clock.epoch_end():
                self.scheduler_aux_server_tail.step()
                self.scheduler_obf_feature_generator.step()

                if local_rank == -1 or dist.get_rank() == 0: 
                    # log, test, save ckpt  
                    self.add_log(clock, self.monitor) # 添加 log
                    # 每个epoch 保存一次log
                    self.save_log()

                if  (clock.epoch + 1) % 1 == 0 : # 当 epoch 结束时需要执行的操作
                    time.sleep(0.003)
                    
                    # 当没有初始化多线程时，因为短路原则，也不会因为get_rank而报错
                    if local_rank == -1 or dist.get_rank() == 0: 
                        self.call_testers(clock, test_model) 
                        # 只保留精度最佳的ckpt，节省时间
                        self.tb_writer.add_scalar('test_by_person', self.current_accuracy, clock.epoch)
                        if True:  # 永远保存最新的
                            self.best_acc = self.current_accuracy
                            self.best_epoch = clock.epoch
                            if local_rank == -1:
                                best_model_state_dict = {k:v.to('cpu') for k, v in obf_feature_generator.state_dict().items()}
                                aux_server_tail_state_dict = {k:v.to('cpu') for k, v in aux_server_tail.state_dict().items()}
                            else:
                                best_model_state_dict = {k:v.to('cpu') for k, v in obf_feature_generator.module.state_dict().items()}
                                aux_server_tail_state_dict = {k:v.to('cpu') for k, v in aux_server_tail.module.state_dict().items()}
                            ckpt = {'best_epoch': self.best_epoch,
                                    'best_acc': self.best_acc,
                                    'model_state_dict': best_model_state_dict,
                                    'optimizer_state_dict': self.optimizer_obf_feature_generator.state_dict(),
                                    'scheduler_state_dict': self.scheduler_obf_feature_generator.state_dict()
                                    }
                            utils.torch_save(ckpt, f'{self.ckpt_dir}'+'/new2_stage2_obf_LN_epoch_{}.pth'.format(clock.epoch))
                            del ckpt

                            ckpt_aux = {'best_epoch': self.best_epoch,
                                    'best_acc': self.best_acc,
                                    'model_state_dict': aux_server_tail_state_dict,
                                    'optimizer_state_dict': self.optimizer_aux_server_tail.state_dict(),
                                    'scheduler_state_dict': self.scheduler_aux_server_tail.state_dict()
                                    }
                            utils.torch_save(ckpt_aux, f'{self.ckpt_dir}/new2_stage2_tail_LN.pth')
                            del ckpt_aux

                            # best_model_state_dict = {k:v.to('cpu') for k, v in model_lit.state_dict().items()}
                            # utils.torch_save(best_model_state_dict, os.path.join(self.work_dir, 'best_model.tar'))
                    if local_rank == -1 or dist.get_rank() == 0: 
                        self.tb_writer.add_scalar('lr', self.optimizer_obf_feature_generator.param_groups[0]['lr'], clock.epoch)
                    if local_rank != -1:
                        # 在多卡模式下在主进程测试并save，其他进程通过torch.distributed.barrier()来等待主进程完成validate操作
                        # barrier()函数的使用要非常谨慎，如果只有单个进程包含了这条语句，那么程序就会陷入无限等待
                        dist.barrier()
                
        
        self.tb_writer.close()
        return obf_feature_generator,aux_server_tail


    

    def fit_Adv_Train_interval(self,obf_feature_generator,aux_server_tail,test_model,local_rank=-1):
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


            if (clock.epoch // 5 ) % 2 == 0:

                # for p in obf_feature_generator.module.base_encoder_obf.parameters():
                #     p.requires_grad = False

                # for p in aux_server_tail.module.parameters():
                #     p.requires_grad = True


                # train step
                # test_model.train() # test model 中有obf_feature_generator aux_server_tail
                aux_server_tail.train()
                obf_feature_generator.eval()
                # aux_classifier
                self.optimizer_aux_server_tail.zero_grad()
                    #forward
                obf_features = obf_feature_generator(imgs.detach())
                scores = aux_server_tail(obf_features,labels)
                    #loss
                loss_D = tc.nn.functional.cross_entropy(scores, labels)
                utils.step(self.optimizer_aux_server_tail, loss_D)

                self.tb_writer.add_scalar('loss_D', loss_D, clock.batch)
                self.monitor.add('loss_D', lambda: float(loss_D),
                            to_str=lambda x: f'{x:.2e}')  # 监控 loss

            else:
                # for p in obf_feature_generator.module.base_encoder_obf.parameters():
                #     p.requires_grad = True

                # for p in aux_server_tail.module.parameters():
                #     p.requires_grad = False

                obf_feature_generator.train()
                aux_server_tail.eval()
                # generator
                self.optimizer_obf_feature_generator.zero_grad()
                    #forward
                obf_features = obf_feature_generator(imgs)
                scores = aux_server_tail(obf_features,labels)  
                    #loss
                loss_G = -tc.nn.functional.cross_entropy(scores, labels)
                utils.step(self.optimizer_obf_feature_generator, loss_G)
                self.tb_writer.add_scalar('loss_G', loss_G, clock.batch)
                self.monitor.add('loss_G', lambda: float(loss_G),
                            to_str=lambda x: f'{x:.2e}')  # 监控 loss
            
     
            self.monitor.add('batch_acc', lambda: utils.accuracy(scores, labels),
                            to_str=lambda x: f'{x * 100:.2f}%')  # 监控 batch 准确率
            # log, test, save ckpt  
            self.add_log(clock, self.monitor) # 添加 log
            

            if clock.epoch_end():
                if (clock.epoch // 5 ) % 2 == 0:
                    self.scheduler_aux_server_tail.step()
                else:
                    self.scheduler_obf_feature_generator.step()

            if clock.epoch_end() and (clock.epoch + 1) % 5 == 0 : # 当 epoch 结束时需要执行的操作
                
                # 当没有初始化多线程时，因为短路原则，也不会因为get_rank而报错
                if local_rank == -1 or dist.get_rank() == 0: 
                    self.call_testers(clock, test_model) 
                    # 只保留精度最佳的ckpt，节省时间
                    self.tb_writer.add_scalar('test_by_person', self.current_accuracy, clock.epoch)
                    if True:  # 永远保存最新的
                        self.best_acc = self.current_accuracy
                        self.best_epoch = clock.epoch
                        if local_rank == -1:
                            best_model_state_dict = {k:v.to('cpu') for k, v in obf_feature_generator.state_dict().items()}
                            aux_server_tail_state_dict = {k:v.to('cpu') for k, v in aux_server_tail.state_dict().items()}
                        else:
                            best_model_state_dict = {k:v.to('cpu') for k, v in obf_feature_generator.module.state_dict().items()}
                            aux_server_tail_state_dict = {k:v.to('cpu') for k, v in aux_server_tail.module.state_dict().items()}
                        ckpt = {'best_epoch': self.best_epoch,
                                'best_acc': self.best_acc,
                                'model_state_dict': best_model_state_dict,
                                'optimizer_state_dict': self.optimizer_obf_feature_generator.state_dict(),
                                'scheduler_state_dict': self.scheduler_obf_feature_generator.state_dict()
                                }
                        utils.torch_save(ckpt, f'{self.ckpt_dir}/best_obf_feature_generator_ckpt.pth')
                        del ckpt

                        ckpt_aux = {'best_epoch': self.best_epoch,
                                'best_acc': self.best_acc,
                                'model_state_dict': aux_server_tail_state_dict,
                                'optimizer_state_dict': self.optimizer_aux_server_tail.state_dict(),
                                'scheduler_state_dict': self.scheduler_aux_server_tail.state_dict()
                                }
                        utils.torch_save(ckpt_aux, f'{self.ckpt_dir}/best_aux_server_tail_ckpt.pth')
                        del ckpt_aux

                        # best_model_state_dict = {k:v.to('cpu') for k, v in model_lit.state_dict().items()}
                        # utils.torch_save(best_model_state_dict, os.path.join(self.work_dir, 'best_model.tar'))
                if local_rank != -1:
                    # 在多卡模式下在主进程测试并save，其他进程通过torch.distributed.barrier()来等待主进程完成validate操作
                    # barrier()函数的使用要非常谨慎，如果只有单个进程包含了这条语句，那么程序就会陷入无限等待
                    dist.barrier()
                
                self.tb_writer.add_scalar('lr', self.optimizer_obf_feature_generator.param_groups[0]['lr'], clock.epoch)
        
        self.tb_writer.close()
        return obf_feature_generator,aux_server_tail
    

    def fit_train2end_interval(self,obf_feature_generator,aux_server_tail,test_model,local_rank=-1):
        '''训练模型并返回练完成的模型'''
        # 数据加载
        # self.call_testers(clock, model_lit)
        dataloader = self.get_dataloader(self.sample_transform) # 加载训练集
        clock = utils.TrainLoopClock(dataloader, self.total_epochs) # 配置训练循环
        flag = True
        # 模型训练
        # catch_nan = len(dataloader)
        for batch_data in clock: # 训练循环
            if local_rank != -1:
                batch_data = utils.batch_to_rank(batch_data, local_rank=local_rank)
            else:
                batch_data = utils.batch_to_cuda(batch_data)
            imgs, labels = batch_data


            if flag:

                # for p in obf_feature_generator.module.base_encoder_obf.parameters():
                #     p.requires_grad = False

                # for p in aux_server_tail.module.parameters():
                #     p.requires_grad = True


                # train step
                # test_model.train() # test model 中有obf_feature_generator aux_server_tail
                aux_server_tail.train()
                obf_feature_generator.eval()
                # aux_classifier
                self.optimizer_aux_server_tail.zero_grad()
                    #forward
                obf_features = obf_feature_generator(imgs.detach())
                scores = aux_server_tail(obf_features,labels)
                    #loss
                loss_D = tc.nn.functional.cross_entropy(scores, labels)
                utils.step(self.optimizer_aux_server_tail, loss_D)

                self.tb_writer.add_scalar('loss_D', loss_D, clock.batch)
                self.monitor.add('loss_D', lambda: float(loss_D),
                            to_str=lambda x: f'{x:.2e}')  # 监控 loss

            else:
                # for p in obf_feature_generator.module.base_encoder_obf.parameters():
                #     p.requires_grad = True

                # for p in aux_server_tail.module.parameters():
                #     p.requires_grad = False

                obf_feature_generator.train()
                aux_server_tail.eval()
                # generator
                self.optimizer_obf_feature_generator.zero_grad()
                    #forward
                obf_features = obf_feature_generator(imgs)
                scores = aux_server_tail(obf_features,labels)  
                    #loss
                loss_G = -tc.nn.functional.cross_entropy(scores, labels)
                utils.step(self.optimizer_obf_feature_generator, loss_G)
                self.tb_writer.add_scalar('loss_G', loss_G, clock.batch)
                self.monitor.add('loss_G', lambda: float(loss_G),
                            to_str=lambda x: f'{x:.2e}')  # 监控 loss
            
     
            self.monitor.add('batch_acc', lambda: utils.accuracy(scores, labels),
                            to_str=lambda x: f'{x * 100:.2f}%')  # 监控 batch 准确率
            # log, test, save ckpt  
            self.add_log(clock, self.monitor) # 添加 log
            

            if clock.epoch_end():
                if flag == True:
                    self.scheduler_aux_server_tail.step()
                else:
                    self.scheduler_obf_feature_generator.step()

            if clock.epoch_end() and (clock.epoch + 1) % 5 == 0 : # 当 epoch 结束时需要执行的操作
                
                # 当没有初始化多线程时，因为短路原则，也不会因为get_rank而报错
                if local_rank == -1 or dist.get_rank() == 0: 
                    self.call_testers(clock, test_model) 
                    # 只保留精度最佳的ckpt，节省时间
                    self.tb_writer.add_scalar('test_by_person', self.current_accuracy, clock.epoch)
                    if flag is True and self.current_accuracy > 0.3:
                        flag = False
                        print('to train G\n')
                    if flag is False and self.current_accuracy < 0.1:
                        flag = True
                        print('to train D\n')


                    if True:  # 永远保存最新的
                        self.best_acc = self.current_accuracy
                        self.best_epoch = clock.epoch
                        if local_rank == -1:
                            best_model_state_dict = {k:v.to('cpu') for k, v in obf_feature_generator.state_dict().items()}
                            aux_server_tail_state_dict = {k:v.to('cpu') for k, v in aux_server_tail.state_dict().items()}
                        else:
                            best_model_state_dict = {k:v.to('cpu') for k, v in obf_feature_generator.module.state_dict().items()}
                            aux_server_tail_state_dict = {k:v.to('cpu') for k, v in aux_server_tail.module.state_dict().items()}
                        ckpt = {'best_epoch': self.best_epoch,
                                'best_acc': self.best_acc,
                                'model_state_dict': best_model_state_dict,
                                'optimizer_state_dict': self.optimizer_obf_feature_generator.state_dict(),
                                'scheduler_state_dict': self.scheduler_obf_feature_generator.state_dict()
                                }
                        utils.torch_save(ckpt, f'{self.ckpt_dir}/best_obf_feature_generator_ckpt.pth')
                        del ckpt

                        ckpt_aux = {'best_epoch': self.best_epoch,
                                'best_acc': self.best_acc,
                                'model_state_dict': aux_server_tail_state_dict,
                                'optimizer_state_dict': self.optimizer_aux_server_tail.state_dict(),
                                'scheduler_state_dict': self.scheduler_aux_server_tail.state_dict()
                                }
                        utils.torch_save(ckpt_aux, f'{self.ckpt_dir}/best_aux_server_tail_ckpt.pth')
                        del ckpt_aux

                        # best_model_state_dict = {k:v.to('cpu') for k, v in model_lit.state_dict().items()}
                        # utils.torch_save(best_model_state_dict, os.path.join(self.work_dir, 'best_model.tar'))
                if local_rank != -1:
                    # 在多卡模式下在主进程测试并save，其他进程通过torch.distributed.barrier()来等待主进程完成validate操作
                    # barrier()函数的使用要非常谨慎，如果只有单个进程包含了这条语句，那么程序就会陷入无限等待
                    dist.barrier()
                
                self.tb_writer.add_scalar('lr', self.optimizer_obf_feature_generator.param_groups[0]['lr'], clock.epoch)
        
        self.tb_writer.close()
        return obf_feature_generator,aux_server_tail


class ExtTrainer_face_adv_stage_aux(block.train.standard.Trainer):
    """Trainer的子类，重写其中的fit函数"""
    def __init__(self, dataset, total_epochs, val_set=None, work_dir=None, sample_transform=None,\
                  optimizer_aux_server_tail = None, optimizer_obf_feature_generator=None, \
                  scheduler_obf_feature_generator = None, scheduler_aux_server_tail = None, \
                    monitor=None, tb_log_dir=None, ckpt_dir=None):
        super().__init__(dataset, total_epochs, val_set, work_dir, ckpt_dir)
        self.best_acc = 0
        self.best_epoch = 0
        self.sample_transform = sample_transform

        self.optimizer_aux_server_tail = optimizer_aux_server_tail

        self.scheduler_aux_server_tail = scheduler_aux_server_tail


        self.monitor = monitor
        self.tb_writer = SummaryWriter(log_dir=tb_log_dir)
        assert self.sample_transform is not None # 为保证效果，必须要有sample_transform

    def config_logger(self, log_interval=60, save_interval=10*60,local_rank = -1):
        '''log_interval: 打印并记录 log 的间隔(单位秒)，
        save_interval: 保存 log 文件的间隔(单位秒)'''
        logs = []
        log_path = f'{self.work_dir}/logs_{time.strftime("%y%m%d%H%M%S",time.localtime())}.json'
       

        def add_log(clock, monitor):
            '''把 clock 和 monitor 的信息打印并记到 log 中'''
            if local_rank == -1 or dist.get_rank() == 0: 
                logs.append(dict(clock=clock.check(), monitor=monitor.check()))
        
        def save_log():
            '''保存 log 到 work_dir 目录下'''
            if local_rank == -1 or dist.get_rank() == 0: 
                utils.json_save(logs, log_path)
        self.add_log = add_log
        self.save_log = save_log
        return log_path
    
    def fit_train_adv_tail(self,obf_feature_generator,aux_server_tail,test_model,local_rank=-1):
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
            aux_server_tail.train()
            obf_feature_generator.eval()


            # aux_classifier
            scores = test_model(imgs,labels)
                #loss
            loss_D = tc.nn.functional.cross_entropy(scores, labels)
            utils.step(self.optimizer_aux_server_tail, loss_D)
            

            if local_rank == -1 or dist.get_rank() == 0: 
                self.tb_writer.add_scalar('loss_D', loss_D, clock.batch)
                self.monitor.add('loss_D', lambda: float(loss_D),
                                to_str=lambda x: f'{x:.2e}')  # 监控 loss
                
                self.monitor.add('batch_acc', lambda: utils.accuracy(scores, labels),
                                to_str=lambda x: f'{x * 100:.2f}%')  # 监控 batch 准确率
            
            # log, test, save ckpt  
            
            

            if clock.epoch_end():
                self.scheduler_aux_server_tail.step()
                
                if local_rank == -1 or dist.get_rank() == 0: 
                    # log, test, save ckpt  
                    self.add_log(clock, self.monitor) # 添加 log
                    self.save_log()

                if (clock.epoch + 1) % 2 == 0 and (clock.epoch + 1) > 1: # 当 epoch 结束时需要执行的操作
                    time.sleep(0.006)
                    # 当没有初始化多线程时，因为短路原则，也不会因为get_rank而报错
                    if local_rank == -1 or dist.get_rank() == 0: 
                        self.call_testers(clock, test_model) 
                        # 只保留精度最佳的ckpt，节省时间
                        self.tb_writer.add_scalar('test_by_person', self.current_accuracy, clock.epoch)
                        if self.current_accuracy > self.best_acc:  # 永远保存最新的
                            self.best_acc = self.current_accuracy
                            self.best_epoch = clock.epoch
                            if local_rank == -1:
                                aux_server_tail_state_dict = {k:v.to('cpu') for k, v in aux_server_tail.state_dict().items()}
                            else:
                                aux_server_tail_state_dict = {k:v.to('cpu') for k, v in aux_server_tail.module.state_dict().items()}
                            
                            ckpt_aux = {'best_epoch': self.best_epoch,
                                    'best_acc': self.best_acc,
                                    'model_state_dict': aux_server_tail_state_dict,
                                    'optimizer_state_dict': self.optimizer_aux_server_tail.state_dict(),
                                    'scheduler_state_dict': self.scheduler_aux_server_tail.state_dict()
                                    }
                            utils.torch_save(ckpt_aux, f'{self.ckpt_dir}/22attacker_facescrub_ckpt.pth')
                            del ckpt_aux

                            # best_model_state_dict = {k:v.to('cpu') for k, v in model_lit.state_dict().items()}
                            # utils.torch_save(best_model_state_dict, os.path.join(self.work_dir, 'best_model.tar'))
                    
                    if local_rank == -1 or dist.get_rank() == 0: 
                        self.tb_writer.add_scalar('lr', self.optimizer_aux_server_tail.param_groups[0]['lr'], clock.epoch)

                    if local_rank != -1:
                        # 在多卡模式下在主进程测试并save，其他进程通过torch.distributed.barrier()来等待主进程完成validate操作
                        # barrier()函数的使用要非常谨慎，如果只有单个进程包含了这条语句，那么程序就会陷入无限等待
                        dist.barrier()
        
        self.tb_writer.close()
        return aux_server_tail
    

class ExtTrainer_face_adv_stage3(block.train.standard.Trainer):
    """Trainer的子类，重写其中的fit函数"""
    def __init__(self, dataset, total_epochs, val_set=None, work_dir=None, sample_transform=None,\
                  optimizer_aux_server_tail = None, optimizer_obf_feature_generator=None, \
                  scheduler_obf_feature_generator = None, scheduler_aux_server_tail = None, \
                    monitor=None, tb_log_dir=None, ckpt_dir=None):
        super().__init__(dataset, total_epochs, val_set, work_dir, ckpt_dir)
        self.best_acc = 0
        self.best_epoch = 0
        self.sample_transform = sample_transform

        self.optimizer_aux_server_tail = optimizer_aux_server_tail

        self.scheduler_aux_server_tail = scheduler_aux_server_tail


        self.monitor = monitor
        self.tb_writer = SummaryWriter(log_dir=tb_log_dir)
        assert self.sample_transform is not None # 为保证效果，必须要有sample_transform

    def config_logger(self, log_interval=60, save_interval=10*60,local_rank = -1):
        '''log_interval: 打印并记录 log 的间隔(单位秒)，
        save_interval: 保存 log 文件的间隔(单位秒)'''
        logs = []
        log_path = f'{self.work_dir}/logs_{time.strftime("%y%m%d%H%M%S",time.localtime())}.json'
       

        def add_log(clock, monitor):
            '''把 clock 和 monitor 的信息打印并记到 log 中'''
            if local_rank == -1 or dist.get_rank() == 0: 
                logs.append(dict(clock=clock.check(), monitor=monitor.check()))
        
        def save_log():
            '''保存 log 到 work_dir 目录下'''
            if local_rank == -1 or dist.get_rank() == 0: 
                utils.json_save(logs, log_path)
        self.add_log = add_log
        self.save_log = save_log
        return log_path
    
    def fit_train_aux_tail(self,obf_feature_generator,aux_server_tail,teacher_server_tail,test_model,local_rank=-1):
        '''训练模型并返回练完成的模型'''
        # 数据加载
        # self.call_testers(clock, model_lit)
        dataloader = self.get_dataloader(self.sample_transform) # 加载训练集
        clock = utils.TrainLoopClock(dataloader, self.total_epochs) # 配置训练循环

        #  before_train 
        # if local_rank == -1 or dist.get_rank() == 0: 
        #     self.call_testers(clock, test_model) 

        # if local_rank != -1:
        #     # 在多卡模式下在主进程测试并save，其他进程通过torch.distributed.barrier()来等待主进程完成validate操作
        #     # barrier()函数的使用要非常谨慎，如果只有单个进程包含了这条语句，那么程序就会陷入无限等待
        #     dist.barrier()


        # 模型训练
        # catch_nan = len(dataloader)
        for batch_data in clock: # 训练循环
            if local_rank != -1:
                batch_data = utils.batch_to_rank(batch_data, local_rank=local_rank)
            else:
                batch_data = utils.batch_to_cuda(batch_data)
            imgs, labels = batch_data

            # train step 
            aux_server_tail.train()
            teacher_server_tail.eval()
            obf_feature_generator.eval()

            with tc.no_grad():
                obf_features = obf_feature_generator(imgs)
            scores = aux_server_tail(obf_features,labels)

            with tc.no_grad():
                clean_feature = obf_feature_generator.base_encoder(imgs.detach())
                scores_clean = teacher_server_tail(clean_feature,labels)

            

            #distill loss
            KD_loss,fc_loss = loss_fn_l2(scores,labels,scores_clean)
            loss = KD_loss + fc_loss

            utils.step(self.optimizer_aux_server_tail, loss)
            

            if local_rank == -1 or dist.get_rank() == 0: 
                self.tb_writer.add_scalar('loss', loss, clock.batch)
                self.tb_writer.add_scalar('KD_loss', KD_loss, clock.batch)
                self.tb_writer.add_scalar('fc_loss', fc_loss, clock.batch)


                
                self.monitor.add('loss', lambda: float(loss),
                                to_str=lambda x: f'{x:.2e}')  # 监控 loss
                self.monitor.add('KD_loss', lambda: float(KD_loss),
                                to_str=lambda x: f'{x:.2e}')  # 监控 loss
                self.monitor.add('fc_loss', lambda: float(fc_loss),
                                to_str=lambda x: f'{x:.2e}')  # 监控 loss
                
                self.monitor.add('batch_acc', lambda: utils.accuracy(scores, labels),
                                to_str=lambda x: f'{x * 100:.2f}%')  # 监控 batch 准确率
            
            
            

            if clock.epoch_end():
                self.scheduler_aux_server_tail.step()

                if local_rank == -1 or dist.get_rank() == 0: 
                    # log, test, save ckpt  
                    self.add_log(clock, self.monitor) # 添加 log
                    self.save_log()

                if clock.epoch  % 1 == 0 : # 当 epoch 结束时需要执行的操作
                    time.sleep(0.003)
                    # 当没有初始化多线程时，因为短路原则，也不会因为get_rank而报错
                    if local_rank == -1 or dist.get_rank() == 0: 
                        self.call_testers(clock, test_model) 
                        # 只保留精度最佳的ckpt，节省时间
                        self.tb_writer.add_scalar('test_by_person', self.current_accuracy, clock.epoch)
                        if self.current_accuracy > self.best_acc:  
                            self.best_acc = self.current_accuracy
                            self.best_epoch = clock.epoch
                            if local_rank == -1:
                                aux_server_tail_state_dict = {k:v.to('cpu') for k, v in aux_server_tail.state_dict().items()}
                            else:
                                aux_server_tail_state_dict = {k:v.to('cpu') for k, v in aux_server_tail.module.state_dict().items()}
                            
                            ckpt_aux = {'best_epoch': self.best_epoch,
                                    'best_acc': self.best_acc,
                                    'model_state_dict': aux_server_tail_state_dict,
                                    'optimizer_state_dict': self.optimizer_aux_server_tail.state_dict(),
                                    'scheduler_state_dict': self.scheduler_aux_server_tail.state_dict()
                                    }
                            utils.torch_save(ckpt_aux, f'{self.ckpt_dir}/a_100_8_tail.pth')
                            del ckpt_aux

                            # best_model_state_dict = {k:v.to('cpu') for k, v in model_lit.state_dict().items()}
                            # utils.torch_save(best_model_state_dict, os.path.join(self.work_dir, 'best_model.tar'))
                    if local_rank == -1 or dist.get_rank() == 0: 
                        self.tb_writer.add_scalar('lr', self.optimizer_aux_server_tail.param_groups[0]['lr'], clock.epoch)
                    if local_rank != -1:
                        # 在多卡模式下在主进程测试并save，其他进程通过torch.distributed.barrier()来等待主进程完成validate操作
                        # barrier()函数的使用要非常谨慎，如果只有单个进程包含了这条语句，那么程序就会陷入无限等待
                        dist.barrier()
        self.tb_writer.close()
        return aux_server_tail

# 打包encode 和server_lit
class Composed_ModelLit(block.model.light.ModelLight):

    def __init__(self,encoder, server_tail,class_num = 0 ):
        super().__init__()
        self.encoder = encoder
        self.server_tail = server_tail

    def forward(self, images, labels,is_inference = False):
        mid_feature = self.encoder(images)
        out = self.server_tail(mid_feature,labels,is_inference = is_inference)
        return out



class Face_server_tail(block.model.light.ModelLight):
    '''server tail'''

    def __init__(self, class_num, isobf=True, pretrained_path='', split_layer=6, tail_layer=6, ext_test=False, is_permute=True, is_matrix=True, debug_pos=False, vis=False,is_fix = True,strict = False):
        super().__init__()
        self.isobf = isobf
        self.is_permute = is_permute
        self.is_matrix = is_matrix
        self.class_num = class_num
        self.split_layer = split_layer
        self.tail_layer = tail_layer
        self.debug_pos = debug_pos
        assert self.split_layer > 0 and self.split_layer < 12
        
        self.tail = block.model.moco_v3_vits.vit_base_tail(layer=self.tail_layer, debug_pos=self.debug_pos)
        self.softmax = block.loss.amsoftmax.AMSoftmax(
            in_features=self.tail.embed_dim, class_num=self.class_num)
        
        if pretrained_path != '':
            self.tail = load_checkpoint(model=self.tail, pretrained_path=pretrained_path,strict=strict)
        if is_fix:
            for p in self.tail.parameters():
                p.requires_grad = False

    def forward(self,mid_feats,labels,is_inference = False):
        rec_feats = self.tail(mid_feats)
        if is_inference:
            return rec_feats
        return self.softmax(rec_feats,labels)

class Face_adv_encoder(block.model.light.ModelLight):
    '''encoder'''

    def __init__(self, class_num, isobf=True, pretrained_path='', split_layer=6, tail_layer=6, ext_test=False, is_permute=True, is_matrix=True, debug_pos=False, vis=False,is_fix = True,strict = False,fix_norm = True):
        super().__init__()
        self.isobf = isobf
        self.is_permute = is_permute
        self.is_matrix = is_matrix
        self.class_num = class_num
        self.split_layer = split_layer
        self.tail_layer = tail_layer
        self.debug_pos = debug_pos
        assert self.split_layer > 0 and self.split_layer < 12
        
        self.ext = block.model.moco_v3_vits.vit_base_head(layer=self.split_layer)
        if ext_test:
            self.ext = block.model.moco_v3_vits.vit_base_5and6(layer=(self.split_layer+self.tail_layer))
        # summary(self.ext, input_size=(3, 224, 224), device="cpu")
        # assert pretrained_path != '', 'pretrained_path is empty'
        if pretrained_path != '':
            self.ext = load_checkpoint(model=self.ext, pretrained_path=pretrained_path,strict=strict)
        if is_fix:
            for name,p in self.ext.named_parameters():
                p.requires_grad = False
        self.norm = partial(tc.nn.LayerNorm, eps=1e-6,elementwise_affine = True)(768).cuda()
        if fix_norm:
            print('----------fix_norm------------------------')
            for p in self.norm.parameters():
                p.requires_grad = False

        pass


    def forward(self, images):
        mid_feats = self.ext(images)
        mid_feats = self.norm(mid_feats)
        return mid_feats

class Obf_feature_generator(block.model.light.ModelLight):
    def __init__(self, base_encoder,base_encoder_obf):
        super().__init__()
        self.base_encoder = base_encoder
        self.base_encoder_obf = base_encoder_obf
        
    def forward(self, images):
        mid_feats = self.base_encoder(images)
        obf_feats = self.base_encoder_obf(images)
        return obf_feats + mid_feats
    


# class Obf_feature_generator_single_BN(block.model.light.ModelLight):
#     def __init__(self, base_encoder,base_encoder_obf):
#         super().__init__()
#         self.base_encoder = base_encoder
#         self.base_encoder_obf = base_encoder_obf
#         self.norm1 = partial(tc.nn.LayerNorm, eps=1e-6)(768)
        

#     def forward(self, images):
#         mid_feats = self.base_encoder(images)
#         obf_feats = self.base_encoder_obf(images)
#         return self.norm1(obf_feats) + mid_feats


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
    parser.add_argument("--train_dataset", default='msra', type=str)
    parser.add_argument("--attacker_dataset", default='facescrub',type=str)
    
    parser.add_argument("--debug", default=None, type=str, help="celeba_lr/celeba_pos")
    parser.add_argument("--mode", default="Visualize", type=str, help="XNN/Test_Ext/Attack/Visualize")
    parser.add_argument("--obf", default="True", type=str)
    parser.add_argument("--is_permute", default="True", type=str)
    parser.add_argument("--is_matrix", default="True", type=str)
    parser.add_argument("--pretrained", default="/data/research/version2/privdl/data/model/vit-b-300ep.pth.tar", type=str,
                        help='path to mocov3 pretrained checkpoint')
    
    parser.add_argument("--person", default=1000, type=int)
    parser.add_argument("--split_layer", default=5, type=int)
    parser.add_argument("--tail_layer", default=6, type=int)
    parser.add_argument("--lr", default=5e-2, type=float)   #5e-2
    parser.add_argument("--batch_size", default=256, type=int)    
    parser.add_argument("--subset_ratio", default=1.0, type=float)

    # tc.cuda.set_device(local_rank)

    FLAGS = parser.parse_args()
    print(FLAGS.mode)
    bool_type = lambda x: x.lower() == 'true'
    FLAGS.obf = bool_type(FLAGS.obf)
    # FLAGS.is_attack = bool_type(FLAGS.is_attack)
    FLAGS.is_permute = bool_type(FLAGS.is_permute)
    FLAGS.is_matrix = bool_type(FLAGS.is_matrix)
    local_rank = FLAGS.local_rank

    proc_count = 1
    if local_rank != -1:
        dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=30000))  # nccl是GPU设备上最快、最推荐的后端
        proc_count = tc.distributed.get_world_size() #获取全局并行数，即并行训练的显卡的数量
    
    batch_size = FLAGS.batch_size // proc_count
    num_workers = 8
    train_epoch = 1000
    
    train_dataset = FLAGS.train_dataset
    attacker_dataset = FLAGS.attacker_dataset
    print("train_dataset:", train_dataset)


    script_file_name = os.path.basename(__file__)
    result_path = exp_utils.setup_result_path(script_file_name)
    result_path = f'{result_path}/{proc_count}_gpus/{train_dataset}/{FLAGS.subset_ratio}_trainset/{FLAGS.mode}_obf/{len(FLAGS.pretrained)!=0 and len(FLAGS.pretrained)!=4}_pretrained/{FLAGS.split_layer}_layer/{FLAGS.tail_layer}_tail_layer'
    print(result_path)
    work_dir = result_path
    
    alert_info = f'033-2_{FLAGS.mode}_{FLAGS.obf}_{FLAGS.is_permute}_{FLAGS.is_matrix}_{FLAGS.train_dataset}_{FLAGS.attacker_dataset}_{proc_count}gpu'
    tb_log_dir = f'/data/privdl_data/log/{script_file_name}/{proc_count}_gpus_{train_dataset}_{FLAGS.subset_ratio}_trainset_{FLAGS.obf}_obf_{len(FLAGS.pretrained)!=0 and len(FLAGS.pretrained)!=4}_pretrained_{FLAGS.split_layer}_layer_{FLAGS.tail_layer}_tail_layer'
    print(tb_log_dir)

    if FLAGS.mode == 'Visualize':
        ckpt_dir = f'/data/ckpt/NC_MSRA_vis/modeXNN_obf{FLAGS.obf}_{FLAGS.is_permute}_{FLAGS.is_matrix}_{train_dataset}_{attacker_dataset}_8gpu'
    else:
        ckpt_dir = f'/data/ckpt/NC_MSRA/mode_{FLAGS.mode}_obf{FLAGS.obf}_{FLAGS.is_permute}_{FLAGS.is_matrix}_{train_dataset}_{attacker_dataset}_{proc_count}gpu'
    print('ckpt_dir:', ckpt_dir)

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
    
    if 0 < FLAGS.subset_ratio < 1.0:
        subset_size = len(train_dataset) * FLAGS.subset_ratio
        train_dataset = train_dataset.subset_by_size(subset_size)
    
    if FLAGS.mode == 'Visualize':
        _, testset_by_person = block.dataset.hubble.hubble_utils.split_dataset_by_person(
            train_dataset, test_id_num=8, test_img_per_id=50)
        trainset, testset_by_img, _ = split_dataset(train_dataset, FLAGS.person)
    else:
        trainset, testset_by_img, testset_by_person = split_dataset(train_dataset, FLAGS.person)
    
    if attacker_dataset == 'facescrub':
        attacker_dataset = block.dataset.hubble.xnn_paper.facescrub()
        normalize = facescrub_normalize
    elif attacker_dataset == 'imdb':
        attacker_dataset = block.dataset.hubble.xnn_paper.imdb()
        normalize = imdb_normalize
    else: 
        raise Exception("Invalid Dataset:", attacker_dataset)
    # trainset = trainset.subset_by_class_id(trainset.class_ids()[:1000])
    
    def sample_transform(sample):
        '''H*W*3*uint8(0\~255) -> 3*224*224*float32(-1\~1)'''
        img, label = sample
        np_img = resize_norm_transpose_insta(img, size=224)
        transformed_img = normalize(np_img)
        return transformed_img, label


    Tester = Ext_Top1_Tester
    # Tester(dataset=testset_by_img, name='testset_by_img', sample_transform=sample_transform),
    testers = [Tester(dataset=testset_by_person, name='testset_by_person', sample_transform=sample_transform),]
    for tester in testers:
        tester.config_dataloader(batch_size=512, num_workers=8)

    
    def test_pretrain_ext_tail():
        # ext的输出维度为(197, 768)，所以计算feature之间的L2距离时,会比ext+tail的结构（输出维度为768）更加耗时
        model_lit = XNN_Single_Lit(class_num=trainset.class_num(), isobf=FLAGS.obf, \
            pretrained_path=FLAGS.pretrained, split_layer=FLAGS.split_layer, ext_test=True)
        # for name, param in model_lit.named_parameters():
            # print(name, param.shape)
        # summary(model_lit, input_size=(3, 224, 224 ,1), device="cpu")

        device = tc.device('cuda' if tc.cuda.is_available else 'cpu')
        model_lit = model_lit.to(device)

        for _ in range(5):
            results = {tester.name: tester.test(model_lit) for tester in testers}
        return results

        

    def train_server_tail_vit(): 

        base_encoder = Face_adv_encoder(class_num=trainset.class_num(), isobf=FLAGS.obf, \
            pretrained_path=FLAGS.pretrained, split_layer=FLAGS.split_layer, tail_layer=FLAGS.tail_layer, is_permute=FLAGS.is_permute, is_matrix=FLAGS.is_matrix, debug_pos=(FLAGS.debug=='celeba_pos'),is_fix=True, fix_norm= True)
        
        server_tail = Face_server_tail(class_num=trainset.class_num(), isobf=FLAGS.obf, \
            pretrained_path='', split_layer=FLAGS.split_layer, tail_layer=FLAGS.tail_layer, is_permute=FLAGS.is_permute, is_matrix=FLAGS.is_matrix, debug_pos=(FLAGS.debug=='celeba_pos'),is_fix = False)
        # server_tail = load_checkpoint(server_tail,'/data/ckpt/NC/modestage1_obfTrue_True_True_celeba_facescrub_4gpu/best_ckpt_server_tail.pth')
        
        if local_rank == -1 or dist.get_rank() == 0:  
            summary(base_encoder.ext, input_size=(3, 224, 224), device="cpu")
            if local_rank == -1 or dist.get_rank() == 0:
                for name, param in server_tail.named_parameters():
                    print(name, param.shape)
            summary(server_tail.tail, input_size=(197, 768), device="cpu")

        
        composed_model = Composed_ModelLit(base_encoder,server_tail,trainset.class_num())

        if local_rank == -1:
            device = tc.device('cuda' if tc.cuda.is_available else 'cpu')
            composed_model = composed_model.to(device)
            
        else:
            print("move model to", local_rank)
            composed_model = composed_model.to(local_rank)
            composed_model = DDP(composed_model, device_ids=[local_rank], output_device=local_rank)
            
        optimizer = tc.optim.SGD(filter(lambda p: p.requires_grad, composed_model.parameters()), lr=FLAGS.lr, momentum=0.9)


        monitor = utils.Monitor()
        monitor.add('lr',  # 监控学习率
                         get=lambda: optimizer.param_groups[0]['lr'],
                         to_str=lambda x: f'{x:.1e}')
        
        warm_up_iter = 3
        T_max = 13 # warm_up + cosAnnealLR 的周期
        lr_max = FLAGS.lr
        lr_min = 1e-3
        # lr_min = 5e-2
        
        lambda_lr = lambda cur_iter: max(1e-3/FLAGS.lr, cur_iter/warm_up_iter) if cur_iter < warm_up_iter else \
            (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi))) / FLAGS.lr
        if FLAGS.train_dataset == 'vggface2' or FLAGS.debug == 'celeba_lr': # vggface2使用大学习率过拟合，尝试全程使用1e-3
            lambda_lr = lambda cur_iter: lr_min/FLAGS.lr

        scheduler = tc.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

        trainer = ExtTrainer_face_adv(
            dataset=trainset, total_epochs=train_epoch,\
                 scheduler=scheduler,\
                  work_dir=f'{work_dir}', sample_transform=sample_transform,\
                  optimizer = optimizer, \
                    monitor=monitor, tb_log_dir=tb_log_dir, ckpt_dir=ckpt_dir)
        
        trainer.config_dataloader(batch_size=batch_size, num_workers=num_workers)
        log_interval = 60
        trainer.config_logger(log_interval=log_interval,local_rank = local_rank)
        trainer.config_tester(testers)

        trainer.fit_ViT(composed_model , local_rank=local_rank)

    def train_adv_vit(): 

        base_encoder = Face_adv_encoder(class_num=trainset.class_num(), isobf=FLAGS.obf, \
            pretrained_path=FLAGS.pretrained, split_layer=FLAGS.split_layer, tail_layer=FLAGS.tail_layer, is_permute=FLAGS.is_permute, is_matrix=FLAGS.is_matrix, debug_pos=(FLAGS.debug=='celeba_pos'),is_fix = True,fix_norm = True)
        
        base_encoder_obf = Face_adv_encoder(class_num=trainset.class_num(), isobf=FLAGS.obf, \
            pretrained_path=FLAGS.pretrained, split_layer=FLAGS.split_layer, tail_layer=FLAGS.tail_layer, is_permute=FLAGS.is_permute, is_matrix=FLAGS.is_matrix, debug_pos=(FLAGS.debug=='celeba_pos'),is_fix = False,fix_norm = False)
        
        aux_server_tail = Face_server_tail(class_num=trainset.class_num(), isobf=FLAGS.obf, \
            pretrained_path='', split_layer=FLAGS.split_layer, tail_layer=FLAGS.tail_layer, is_permute=FLAGS.is_permute, is_matrix=FLAGS.is_matrix, debug_pos=(FLAGS.debug=='celeba_pos'),is_fix= False)
        
        obf_feature_generator = Obf_feature_generator(base_encoder = base_encoder ,base_encoder_obf=base_encoder_obf)
        print('Obf_feature_generator')
        # obf_feature_generator = Obf_feature_generator_single_BN(base_encoder = base_encoder ,base_encoder_obf=base_encoder_obf)

        test_model = Composed_ModelLit(obf_feature_generator,aux_server_tail)
        # summary(base_encoder.ext, input_size=(3, 224, 224), device="cpu")
        # if local_rank == -1 or dist.get_rank() == 0:
        #     for name, param in model_lit.tail.named_parameters():
        #         print(name, param.shape)
        # summary(base_encoder.tail, input_size=(197, 768), device="cpu")

        if local_rank == -1:
            device = tc.device('cuda' if tc.cuda.is_available else 'cpu')
            obf_feature_generator = obf_feature_generator.to(device)
            aux_server_tail = aux_server_tail.to(device)
            test_model = test_model.to(device)
        else:
            print("move model to", local_rank)
            obf_feature_generator = obf_feature_generator.to(local_rank)
            obf_feature_generator = DDP(obf_feature_generator, device_ids=[local_rank],find_unused_parameters=False, output_device=local_rank)

            aux_server_tail = aux_server_tail.to(local_rank)
            aux_server_tail = DDP(aux_server_tail, device_ids=[local_rank],find_unused_parameters=False, output_device=local_rank)

            test_model = test_model.to(local_rank)
            test_model = DDP(test_model, device_ids=[local_rank],find_unused_parameters=False, output_device=local_rank)
            
            

            
        optimizer_obf_feature_generator = tc.optim.SGD(filter(lambda p: p.requires_grad, obf_feature_generator.parameters()), lr=FLAGS.lr, momentum=0.9)
        optimizer_aux_server_tail = tc.optim.SGD(filter(lambda p: p.requires_grad, aux_server_tail.parameters()), lr=FLAGS.lr, momentum=0.9)
        
        monitor = utils.Monitor()
        monitor.add('lr',  # 监控学习率
                         get=lambda: optimizer_obf_feature_generator.param_groups[0]['lr'],
                         to_str=lambda x: f'{x:.1e}')
        
        warm_up_iter = 3
        T_max = 13 # warm_up + cosAnnealLR 的周期
        lr_max = FLAGS.lr
        lr_min = 1e-3
        # lr_min = 5e-2
        
        lambda_lr = lambda cur_iter: max(1e-3/FLAGS.lr, cur_iter/warm_up_iter) if cur_iter < warm_up_iter else \
            (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi))) / FLAGS.lr
        if FLAGS.train_dataset == 'vggface2' or FLAGS.debug == 'celeba_lr': # vggface2使用大学习率过拟合，尝试全程使用1e-3
            lambda_lr = lambda cur_iter: lr_min/FLAGS.lr
        
        scheduler_obf_feature_generator = tc.optim.lr_scheduler.LambdaLR(optimizer_obf_feature_generator, lr_lambda=lambda_lr)
        scheduler_aux_server_tail = tc.optim.lr_scheduler.LambdaLR(optimizer_aux_server_tail, lr_lambda=lambda_lr)

        trainer = ExtTrainer_face_adv_stage2(
            dataset=trainset, total_epochs=train_epoch, \
                scheduler_obf_feature_generator = scheduler_obf_feature_generator, scheduler_aux_server_tail = scheduler_aux_server_tail ,\
                  work_dir=f'{work_dir}', sample_transform=sample_transform,\
                    optimizer_aux_server_tail = optimizer_aux_server_tail, optimizer_obf_feature_generator=optimizer_obf_feature_generator,\
                    monitor=monitor, tb_log_dir=tb_log_dir, ckpt_dir=ckpt_dir)
        
        trainer.config_dataloader(batch_size=batch_size, num_workers=num_workers)
        log_interval = 60
        trainer.config_logger(log_interval=log_interval,local_rank=local_rank)
        trainer.config_tester(testers)

        trainer.fit_Adv_Train(obf_feature_generator,aux_server_tail,test_model, local_rank=local_rank)    
    

    def train_adv_vit_tradoff(): 
        #train: base_encoder_obf,aux_server_tail,aux_server_tail_distill

        #total: obf_feature_generator,aux_server_tail,aux_server_tail_distill,aux_teacher_tail,
        base_encoder = Face_adv_encoder(class_num=trainset.class_num(), isobf=FLAGS.obf, \
            pretrained_path='/data/ckpt/NC/modestage1_obfTrue_True_True_celeba_facescrub_8gpu/best_ckpt_encoder_ext_nopretrain.pth', split_layer=FLAGS.split_layer, tail_layer=FLAGS.tail_layer, is_permute=FLAGS.is_permute, is_matrix=FLAGS.is_matrix, debug_pos=(FLAGS.debug=='celeba_pos'))
        
        base_encoder_obf = Face_adv_encoder(class_num=trainset.class_num(), isobf=FLAGS.obf, \
            pretrained_path=FLAGS.pretrained, split_layer=FLAGS.split_layer, tail_layer=FLAGS.tail_layer, is_permute=FLAGS.is_permute, is_matrix=FLAGS.is_matrix, debug_pos=(FLAGS.debug=='celeba_pos'),is_fix = False)
        
        aux_server_tail = Face_server_tail(class_num=trainset.class_num(), isobf=FLAGS.obf, \
            pretrained_path='', split_layer=FLAGS.split_layer, tail_layer=FLAGS.tail_layer, is_permute=FLAGS.is_permute, is_matrix=FLAGS.is_matrix, debug_pos=(FLAGS.debug=='celeba_pos'),is_fix= False)

        aux_server_tail_distill = Face_server_tail(class_num=trainset.class_num(), isobf=FLAGS.obf, \
            pretrained_path='', split_layer=FLAGS.split_layer, tail_layer=FLAGS.tail_layer, is_permute=FLAGS.is_permute, is_matrix=FLAGS.is_matrix, debug_pos=(FLAGS.debug=='celeba_pos'),is_fix= False)
        
        aux_teacher_tail = Face_server_tail(class_num=trainset.class_num(), isobf=FLAGS.obf, \
            pretrained_path='/data/ckpt/NC/modestage1_obfTrue_True_True_celeba_facescrub_8gpu/best_ckpt_server_tail_nopretrain.pth', split_layer=FLAGS.split_layer, tail_layer=FLAGS.tail_layer, is_permute=FLAGS.is_permute, is_matrix=FLAGS.is_matrix, debug_pos=(FLAGS.debug=='celeba_pos'),is_fix= True)
        

        obf_feature_generator = Obf_feature_generator(base_encoder = base_encoder ,base_encoder_obf=base_encoder_obf)

        test_model = Composed_ModelLit(obf_feature_generator,aux_server_tail)
        test_model2 = Composed_ModelLit(obf_feature_generator,aux_server_tail_distill)
        # summary(base_encoder.ext, input_size=(3, 224, 224), device="cpu")
        # if local_rank == -1 or dist.get_rank() == 0:
        #     for name, param in model_lit.tail.named_parameters():
        #         print(name, param.shape)
        # summary(base_encoder.tail, input_size=(197, 768), device="cpu")

        if local_rank == -1:
            device = tc.device('cuda' if tc.cuda.is_available else 'cpu')
            obf_feature_generator = obf_feature_generator.to(device)
            aux_server_tail = aux_server_tail.to(device)
            aux_server_tail_distill = aux_server_tail_distill.to(device)
            test_model = test_model.to(device)
        else:
            print("move model to", local_rank)
            obf_feature_generator = obf_feature_generator.to(local_rank)
            obf_feature_generator = DDP(obf_feature_generator, device_ids=[local_rank], output_device=local_rank)

            aux_server_tail = aux_server_tail.to(local_rank)
            aux_server_tail = DDP(aux_server_tail, device_ids=[local_rank], output_device=local_rank)

            aux_server_tail_distill = aux_server_tail_distill.to(local_rank)
            aux_server_tail_distill = DDP(aux_server_tail_distill, device_ids=[local_rank], output_device=local_rank)

            test_model = test_model.to(local_rank)
            test_model = DDP(test_model, device_ids=[local_rank], output_device=local_rank)
            
            

            
        optimizer_obf_feature_generator = tc.optim.SGD(filter(lambda p: p.requires_grad, obf_feature_generator.parameters()), lr=FLAGS.lr, momentum=0.9)
        optimizer_aux_server_tail = tc.optim.SGD(filter(lambda p: p.requires_grad, aux_server_tail.parameters()), lr=FLAGS.lr, momentum=0.9)
        optimizer_aux_server_tail_distill = tc.optim.SGD(filter(lambda p: p.requires_grad, aux_server_tail_distill.parameters()), lr=FLAGS.lr, momentum=0.9)

        
        monitor = utils.Monitor()
        monitor.add('lr',  # 监控学习率
                         get=lambda: optimizer_obf_feature_generator.param_groups[0]['lr'],
                         to_str=lambda x: f'{x:.1e}')
        
        warm_up_iter = 3
        T_max = 13 # warm_up + cosAnnealLR 的周期
        lr_max = FLAGS.lr
        lr_min = 1e-3
        # lr_min = 5e-2
        
        lambda_lr = lambda cur_iter: max(1e-3/FLAGS.lr, cur_iter/warm_up_iter) if cur_iter < warm_up_iter else \
            (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi))) / FLAGS.lr
        if FLAGS.train_dataset == 'vggface2' or FLAGS.debug == 'celeba_lr': # vggface2使用大学习率过拟合，尝试全程使用1e-3
            lambda_lr = lambda cur_iter: lr_min/FLAGS.lr
        
        scheduler_obf_feature_generator = tc.optim.lr_scheduler.LambdaLR(optimizer_obf_feature_generator, lr_lambda=lambda_lr)
        scheduler_aux_server_tail = tc.optim.lr_scheduler.LambdaLR(optimizer_aux_server_tail, lr_lambda=lambda_lr)
        scheduler_aux_server_tail_distill = tc.optim.lr_scheduler.LambdaLR(optimizer_aux_server_tail_distill, lr_lambda=lambda_lr)

        trainer = ExtTrainer_face_adv_new_stage2(
            dataset=trainset, total_epochs=train_epoch, \
                scheduler_obf_feature_generator = scheduler_obf_feature_generator, scheduler_aux_server_tail = scheduler_aux_server_tail ,scheduler_aux_server_tail_distill = scheduler_aux_server_tail_distill,\
                  work_dir=f'{work_dir}', sample_transform=sample_transform,\
                    optimizer_aux_server_tail = optimizer_aux_server_tail, optimizer_obf_feature_generator=optimizer_obf_feature_generator,optimizer_aux_server_tail_distill = optimizer_aux_server_tail_distill,\
                    monitor=monitor, tb_log_dir=tb_log_dir, ckpt_dir=ckpt_dir)
        
        trainer.config_dataloader(batch_size=batch_size, num_workers=num_workers)
        log_interval = 60
        trainer.config_logger(log_interval=log_interval)
        trainer.config_tester(testers)

        trainer.fit_Adv_Train(obf_feature_generator,aux_server_tail,aux_server_tail_distill,aux_teacher_tail,test_model,test_model2, local_rank=local_rank) 


    def train_adv_server():
        base_encoder = Face_adv_encoder(class_num=attacker_dataset.class_num(), isobf=FLAGS.obf, \
            pretrained_path=FLAGS.pretrained, split_layer=FLAGS.split_layer, tail_layer=FLAGS.tail_layer, is_permute=FLAGS.is_permute, is_matrix=FLAGS.is_matrix, debug_pos=(FLAGS.debug=='celeba_pos'),is_fix = True)
        
        base_encoder_obf = Face_adv_encoder(class_num=attacker_dataset.class_num(), isobf=FLAGS.obf, \
            pretrained_path=FLAGS.pretrained, split_layer=FLAGS.split_layer, tail_layer=FLAGS.tail_layer, is_permute=FLAGS.is_permute, is_matrix=FLAGS.is_matrix, debug_pos=(FLAGS.debug=='celeba_pos'),is_fix = True)
        
        aux_server_tail = Face_server_tail(class_num=attacker_dataset.class_num(), isobf=FLAGS.obf, \
            pretrained_path='', split_layer=FLAGS.split_layer, tail_layer=FLAGS.tail_layer, is_permute=FLAGS.is_permute, is_matrix=FLAGS.is_matrix, debug_pos=(FLAGS.debug=='celeba_pos'),is_fix = False)
        

        obf_feature_generator = Obf_feature_generator(base_encoder = base_encoder ,base_encoder_obf=base_encoder_obf)
        print('Obf_feature_generator')

        obf_feature_generator = load_checkpoint(obf_feature_generator,'/data/ckpt/NC_MSRA/mode_stage2_obfTrue_True_True_msra_imdb_16gpu/new2_stage2_obf_LN_epoch_13.pth',strict=True)

        test_model = Composed_ModelLit(obf_feature_generator,aux_server_tail)
        # summary(base_encoder.ext, input_size=(3, 224, 224), device="cpu")
        # if local_rank == -1 or dist.get_rank() == 0:
        #     for name, param in model_lit.tail.named_parameters():
        #         print(name, param.shape)
        # summary(base_encoder.tail, input_size=(197, 768), device="cpu")

        if local_rank == -1:
            device = tc.device('cuda' if tc.cuda.is_available else 'cpu')
            # obf_feature_generator = obf_feature_generator.to(device)
            # aux_server_tail = aux_server_tail.to(device)
            test_model = test_model.to(device)
        else:
            print("move model to", local_rank)
            # obf_feature_generator = obf_feature_generator.to(local_rank)
            # obf_feature_generator = DDP(obf_feature_generator, device_ids=[local_rank], output_device=local_rank)

            aux_server_tail = aux_server_tail.to(local_rank)
            aux_server_tail = DDP(aux_server_tail, device_ids=[local_rank],find_unused_parameters=False, output_device=local_rank)

            test_model = test_model.to(local_rank)
            test_model = DDP(test_model, device_ids=[local_rank],find_unused_parameters=False, output_device=local_rank)
            
            

            
        optimizer_aux_server_tail = tc.optim.SGD(filter(lambda p: p.requires_grad, aux_server_tail.parameters()), lr=FLAGS.lr, momentum=0.9)
        
        monitor = utils.Monitor()
        monitor.add('lr',  # 监控学习率
                         get=lambda: optimizer_aux_server_tail.param_groups[0]['lr'],
                         to_str=lambda x: f'{x:.1e}')
        
        warm_up_iter = 3
        T_max = 13 # warm_up + cosAnnealLR 的周期
        lr_max = FLAGS.lr
        lr_min = 1e-3
        # lr_min = 5e-2 
        # lr_min = 1e-5
        
        lambda_lr = lambda cur_iter: max(1e-3/FLAGS.lr, cur_iter/warm_up_iter) if cur_iter < warm_up_iter else \
            (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi))) / FLAGS.lr
        if FLAGS.train_dataset == 'vggface2' or FLAGS.debug == 'celeba_lr': # vggface2使用大学习率过拟合，尝试全程使用1e-3
            lambda_lr = lambda cur_iter: lr_min/FLAGS.lr
        
        scheduler_aux_server_tail = tc.optim.lr_scheduler.LambdaLR(optimizer_aux_server_tail, lr_lambda=lambda_lr)

        trainer = ExtTrainer_face_adv_stage_aux(
            dataset=attacker_dataset, total_epochs=train_epoch, \
                scheduler_aux_server_tail = scheduler_aux_server_tail ,\
                  work_dir=f'{work_dir}', sample_transform=sample_transform,\
                    optimizer_aux_server_tail = optimizer_aux_server_tail,\
                    monitor=monitor, tb_log_dir=tb_log_dir, ckpt_dir=ckpt_dir)
        
        trainer.config_dataloader(batch_size=batch_size, num_workers=num_workers)
        log_interval = 60
        trainer.config_logger(log_interval=log_interval,local_rank = local_rank)
        trainer.config_tester(testers)

        trainer.fit_train_adv_tail(obf_feature_generator,aux_server_tail,test_model, local_rank=local_rank)  

    def train_disdill():
        #FLAGS.pretrained
        base_encoder = Face_adv_encoder(class_num=trainset.class_num(), isobf=FLAGS.obf, \
            pretrained_path=FLAGS.pretrained, split_layer=FLAGS.split_layer, tail_layer=FLAGS.tail_layer, is_permute=FLAGS.is_permute, is_matrix=FLAGS.is_matrix, debug_pos=(FLAGS.debug=='celeba_pos'),is_fix = True)
        
        base_encoder_obf = Face_adv_encoder(class_num=trainset.class_num(), isobf=FLAGS.obf, \
            pretrained_path=FLAGS.pretrained, split_layer=FLAGS.split_layer, tail_layer=FLAGS.tail_layer, is_permute=FLAGS.is_permute, is_matrix=FLAGS.is_matrix, debug_pos=(FLAGS.debug=='celeba_pos'),is_fix = True)
        
        aux_server_tail = Face_server_tail(class_num=trainset.class_num(), isobf=FLAGS.obf, \
            pretrained_path='', split_layer=FLAGS.split_layer, tail_layer=FLAGS.tail_layer, is_permute=FLAGS.is_permute, is_matrix=FLAGS.is_matrix, debug_pos=(FLAGS.debug=='celeba_pos'),is_fix = False)
        
        
        teacher_server_tail = Face_server_tail(class_num=trainset.class_num(), isobf=FLAGS.obf, \
            pretrained_path='', split_layer=FLAGS.split_layer, tail_layer=FLAGS.tail_layer, is_permute=FLAGS.is_permute, is_matrix=FLAGS.is_matrix, debug_pos=(FLAGS.debug=='celeba_pos'),strict= True, is_fix = True)

        # teacher_server_tail = load_checkpoint(teacher_server_tail,'/data/ckpt/NC500/mode_stage1_obfTrue_True_True_celeba_facescrub_8gpu/stage1_tail_epoch_5.pth')
        teacher_server_tail = load_checkpoint(teacher_server_tail,'/data/ckpt/NC_MSRA/mode_stage1_obfTrue_True_True_msra_imdb_16gpu/restart_stage1_tail_epoch_8.pth',strict=True)

        print("Obf_feature_generator")
        obf_feature_generator = Obf_feature_generator(base_encoder = base_encoder ,base_encoder_obf=base_encoder_obf)
        obf_feature_generator = load_checkpoint(obf_feature_generator,'/data/ckpt/NC_MSRA/mode_stage2_obfTrue_True_True_msra_imdb_16gpu/new2_stage2_obf_LN_epoch_13.pth',strict=True)

        test_model = Composed_ModelLit(obf_feature_generator, aux_server_tail)
        # summary(base_encoder.ext, input_size=(3, 224, 224), device="cpu")
        # if local_rank == -1 or dist.get_rank() == 0:
        #     for name, param in model_lit.tail.named_parameters():
        #         print(name, param.shape)
        # summary(base_encoder.tail, input_size=(197, 768), device="cpu")

        if local_rank == -1:
            device = tc.device('cuda' if tc.cuda.is_available else 'cpu')
            obf_feature_generator = obf_feature_generator.to(device)
            aux_server_tail = aux_server_tail.to(device)
            test_model = test_model.to(device)
            teacher_server_tail = teacher_server_tail.to(device)
        else:
            print("move model to", local_rank)
            obf_feature_generator = obf_feature_generator.to(local_rank)
            # obf_feature_generator = DDP(obf_feature_generator, device_ids=[local_rank], output_device=local_rank)

            aux_server_tail = aux_server_tail.to(local_rank)
            aux_server_tail = DDP(aux_server_tail, device_ids=[local_rank], output_device=local_rank)

            teacher_server_tail = teacher_server_tail.to(local_rank)
            # teacher_server_tail = DDP(teacher_server_tail, device_ids=[local_rank], output_device=local_rank)

            test_model = test_model.to(local_rank)
            test_model = DDP(test_model, device_ids=[local_rank], output_device=local_rank)
            
            

            
        optimizer_aux_server_tail = tc.optim.SGD(filter(lambda p: p.requires_grad, aux_server_tail.parameters()), lr=FLAGS.lr, momentum=0.9)
        
        monitor = utils.Monitor()
        monitor.add('lr',  # 监控学习率
                         get=lambda: optimizer_aux_server_tail.param_groups[0]['lr'],
                         to_str=lambda x: f'{x:.1e}')
        
        warm_up_iter = 3
        T_max = 13 # warm_up + cosAnnealLR 的周期
        lr_max = FLAGS.lr
        lr_min = 1e-3
        # lr_min = 5e-2
        
        lambda_lr = lambda cur_iter: max(1e-3/FLAGS.lr, cur_iter/warm_up_iter) if cur_iter < warm_up_iter else \
            (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi))) / FLAGS.lr
        if FLAGS.train_dataset == 'vggface2' or FLAGS.debug == 'celeba_lr': # vggface2使用大学习率过拟合，尝试全程使用1e-3
            lambda_lr = lambda cur_iter: lr_min/FLAGS.lr
        
        scheduler_aux_server_tail = tc.optim.lr_scheduler.LambdaLR(optimizer_aux_server_tail, lr_lambda=lambda_lr)

        trainer = ExtTrainer_face_adv_stage3(
            dataset=trainset, total_epochs=train_epoch, \
                scheduler_aux_server_tail = scheduler_aux_server_tail ,\
                  work_dir=f'{work_dir}', sample_transform=sample_transform,\
                    optimizer_aux_server_tail = optimizer_aux_server_tail,\
                    monitor=monitor, tb_log_dir=tb_log_dir, ckpt_dir=ckpt_dir)
        
        trainer.config_dataloader(batch_size=batch_size, num_workers=num_workers)
        log_interval = 60
        trainer.config_logger(log_interval=log_interval,local_rank= local_rank)
        trainer.config_tester(testers)

        trainer.fit_train_aux_tail(obf_feature_generator,aux_server_tail,teacher_server_tail,test_model, local_rank=local_rank)  
        
    from sklearn.manifold import TSNE # 用于降维
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import seaborn as sb
    import numpy as np
    def plot_embedding(data, label, title):
        # scale data to [-1, 1]
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)

        fig = plt.figure(dpi = 200)
        ax = plt.subplot(111)

        for i in range(data.shape[0]):
            plt.scatter(data[i, 0], data[i, 1], marker='o',color=plt.cm.Set1(label[i] / 8.),label=label[i],s = 5)
            # plt.text(data[i, 0], data[i, 1], str(label[i]),
            #         color=plt.cm.Set1(label[i] / 20.),
            #         fontdict={'weight': 'bold', 'size': 9})
        
        # plt.xlim(0, 1.1)
        # plt.ylim(0, 1.1)
        plt.xticks([])
        plt.yticks([])
        plt.title(title)

        fig.savefig('/data/research/version2/privdl/exp/visualization/baseline-tsne.png')
        plt.close(fig)
        
        return fig

    def Visualize():
        base_encoder = Face_adv_encoder(class_num=trainset.class_num(), isobf=FLAGS.obf, \
            pretrained_path=FLAGS.pretrained, split_layer=FLAGS.split_layer, tail_layer=FLAGS.tail_layer, is_permute=FLAGS.is_permute, is_matrix=FLAGS.is_matrix, debug_pos=(FLAGS.debug=='celeba_pos'),is_fix = True)
        
        base_encoder_obf = Face_adv_encoder(class_num=trainset.class_num(), isobf=FLAGS.obf, \
            pretrained_path=FLAGS.pretrained, split_layer=FLAGS.split_layer, tail_layer=FLAGS.tail_layer, is_permute=FLAGS.is_permute, is_matrix=FLAGS.is_matrix, debug_pos=(FLAGS.debug=='celeba_pos'),is_fix = True)
        
        server_tail = Face_server_tail(class_num=trainset.class_num(), isobf=FLAGS.obf, \
            pretrained_path='', split_layer=FLAGS.split_layer, tail_layer=FLAGS.tail_layer, is_permute=FLAGS.is_permute, is_matrix=FLAGS.is_matrix, debug_pos=(FLAGS.debug=='celeba_pos'),is_fix = True)
        server_tail = load_checkpoint(server_tail,'/data/ckpt/NC_MSRA/mode_stage1_obfTrue_True_True_msra_imdb_16gpu/restart_stage1_tail_epoch_8.pth',strict= True)
        
        server_tail2 = Face_server_tail(class_num=trainset.class_num(), isobf=FLAGS.obf, \
            pretrained_path='', split_layer=FLAGS.split_layer, tail_layer=FLAGS.tail_layer, is_permute=FLAGS.is_permute, is_matrix=FLAGS.is_matrix, debug_pos=(FLAGS.debug=='celeba_pos'),is_fix = True)
        server_tail2 = load_checkpoint(server_tail2,'/data/ckpt/NC_MSRA/mode_stage3_obfTrue_True_True_msra_imdb_8gpu/a_100_8_tail.pth',strict= True)
            
        # obf_feature_generator = Obf_feature_generator(base_encoder = base_encoder ,base_encoder_obf=base_encoder_obf)
        # obf_feature_generator = load_checkpoint(obf_feature_generator,'/data/ckpt/NC3/mode_stage2_obfTrue_True_True_celeba_facescrub_8gpu/new_stage2_obf_LN_epoch_59.pth',strict=True)
        
        if local_rank == -1 or dist.get_rank() == 0:  
            summary(base_encoder.ext, input_size=(3, 224, 224), device="cpu")
            if local_rank == -1 or dist.get_rank() == 0:
                for name, param in server_tail.named_parameters():
                    print(name, param.shape)
            summary(server_tail.tail, input_size=(197, 768), device="cpu")

        
        model_lit = Composed_ModelLit(base_encoder,server_tail,trainset.class_num())
        for p in model_lit.parameters():
            p.requires_grad = False
        
        device = tc.device('cuda' if tc.cuda.is_available else 'cpu')
        model_lit = model_lit.to(device)
        
        # print(f'rank{local_rank}_torch_load: {ckpt_dir}/best_ckpt.pth')
        # checkpoint_data = tc.load(f'{ckpt_dir}/best_ckpt.pth', map_location=tc.device(device))
        # model_lit.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
        
        feats, labels = testers[0].test(model_lit, vis=True)
        # vis_emb_data ={
        #     'feats': feats,
        #     'labels': labels
        # }
        # tc.save(vis_emb_data, '/data/vis_midobf_feats.pth')
        # feats = feats / np.linalg.norm(feats)
        vis_data = TSNE(n_components=2,perplexity=10, init='pca', random_state=0).fit_transform(feats) 
        plot_embedding(feats, labels, 't-SNE embedding of faces')
        print('done')
        return
        
        # print('done')
        # return

    # Visualize()

    # Visualize()
    try:
        if FLAGS.mode == "stage1": # attack
            train_server_tail_vit()
        elif FLAGS.mode == "new_stage2":
            train_adv_vit_tradoff()  
        elif FLAGS.mode == "stage2":
            train_adv_vit()   
        elif FLAGS.mode == "stage_aux":
            train_adv_server()   
        elif FLAGS.mode == "stage3":
            train_disdill()    
        elif FLAGS.mode == "Test_Ext":
            test_pretrain_ext_tail()
        elif FLAGS.mode == "Visualize":
            Visualize()
        else:
            pass
    except Exception as e:
        print("Exception:", e)
        if local_rank == -1 or dist.get_rank() == 0:
            requests.get(f'https://api.day.app/ocZjTT3y2AhsowCaiKtqLC/break_down_033-2/{alert_info}_{time.strftime("%y%m%d%H%M%S",time.localtime())}')

if __name__ == '__main__':
    main()

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 -m torch.distributed.launch --nproc_per_node 8 500-FNLN_false_NELN_true_MSRA.py --mode 'stage1'
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 -m torch.distributed.launch --nproc_per_node 8 500-FNLN_false_NELN_true_MSRA.py --mode 'stage2'
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 -m torch.distributed.launch --nproc_per_node 8 500-FNLN_false_NELN_true_MSRA.py --mode 'stage_aux'
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 -m torch.distributed.launch --nproc_per_node 8 500-FNLN_false_NELN_true_MSRA.py --mode 'stage3'
    

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 -m torch.distributed.launch --nproc_per_node 8  --nnodes=4 --node_rank=0 --master_addr="100.97.132.55" --master_port=12583 500-FNLN_false_NELN_true_MSRA.py --mode 'stage3'
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 -m torch.distributed.launch --nproc_per_node 8  --nnodes=4 --node_rank=1 --master_addr="100.97.132.55" --master_port=12583 500-FNLN_false_NELN_true_MSRA.py --mode 'stage3'
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 -m torch.distributed.launch --nproc_per_node 8  --nnodes=4 --node_rank=2 --master_addr="100.97.132.55" --master_port=12583 500-FNLN_false_NELN_true_MSRA.py --mode 'stage3'
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 -m torch.distributed.launch --nproc_per_node 8  --nnodes=4 --node_rank=3 --master_addr="100.97.132.55" --master_port=12583 500-FNLN_false_NELN_true_MSRA.py --mode 'stage3'

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 -m torch.distributed.launch --nproc_per_node 8  --nnodes=2 --node_rank=0 --master_addr="100.97.132.55" --master_port=12583 500-FNLN_false_NELN_true_MSRA.py --mode 'stage3'
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 -m torch.distributed.launch --nproc_per_node 8  --nnodes=2 --node_rank=1 --master_addr="100.97.132.55" --master_port=12583 500-FNLN_false_NELN_true_MSRA.py --mode 'stage3'

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 -m torch.distributed.launch --nproc_per_node 8  --nnodes=8 --node_rank=0 --master_addr="100.96.166.167" --master_port=12583 500-FNLN_false_NELN_true_MSRA.py --mode 'stage3'
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 -m torch.distributed.launch --nproc_per_node 8  --nnodes=8 --node_rank=1 --master_addr="100.96.166.167" --master_port=12583 500-FNLN_false_NELN_true_MSRA.py --mode 'stage3'
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 -m torch.distributed.launch --nproc_per_node 8  --nnodes=8 --node_rank=2 --master_addr="100.96.166.167" --master_port=12583 500-FNLN_false_NELN_true_MSRA.py --mode 'stage3'
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 -m torch.distributed.launch --nproc_per_node 8  --nnodes=8 --node_rank=3 --master_addr="100.96.166.167" --master_port=12583 500-FNLN_false_NELN_true_MSRA.py --mode 'stage3'
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 -m torch.distributed.launch --nproc_per_node 8  --nnodes=8 --node_rank=4 --master_addr="100.96.166.167" --master_port=12583 500-FNLN_false_NELN_true_MSRA.py --mode 'stage3'
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 -m torch.distributed.launch --nproc_per_node 8  --nnodes=8 --node_rank=5 --master_addr="100.96.166.167" --master_port=12583 500-FNLN_false_NELN_true_MSRA.py --mode 'stage3'
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 -m torch.distributed.launch --nproc_per_node 8  --nnodes=8 --node_rank=6 --master_addr="100.96.166.167" --master_port=12583 500-FNLN_false_NELN_true_MSRA.py --mode 'stage3'
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 -m torch.distributed.launch --nproc_per_node 8  --nnodes=8 --node_rank=7 --master_addr="100.96.166.167" --master_port=12583 500-FNLN_false_NELN_true_MSRA.py --mode 'stage3'

