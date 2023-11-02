# -*- coding: UTF-8 -*-
'''
Implementation slightly adapted from "033".
Optimizing strategy: SGD + (large lr + large momentum)  + (warm_up + CosineAnnealingLR), 这套组合很猛！
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
# init_seed(int(time.time()))
init_seed(11)

def loss_fn_kd(outputs, labels, teacher_outputs):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    
    alpha = 0.5
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
    alpha = 0.9
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


    def extract_feats(self, model_lit):
        '''用 model_lit 抽取测试集的特征并返回'''
        dataloader = self.get_dataloader(self.sample_transform)
        feats, labels = [], []
        model_lit.eval()
        with tc.no_grad():
            for batch_data in utils.tqdm(dataloader, 'extract_feats'):
                model_lit.module.encoder.obfuscate.reset_parameters() 
                batch_data = utils.batch_to_cuda(batch_data)
                batch_inputs, batch_labels = batch_data
                batch_feats = model_lit(batch_inputs, batch_labels, is_inference=True)
                feats.append(utils.batch_to_numpy(batch_feats))
                labels.append(utils.batch_to_numpy(batch_labels))
        return np.concatenate(feats), np.concatenate(labels)



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

def cdist(feat0, feat1):
    '''L2norm'''
    return np.linalg.norm(feat0 - feat1)


def load_checkpoint(model, pretrained_path):
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
    msg = model.load_state_dict(state_dict, strict=False)
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

            self.tb_writer.add_scalar('train_loss', loss, clock.batch)

            self.monitor.add('loss', lambda: float(loss),
                            to_str=lambda x: f'{x:.2e}')  # 监控 loss
            self.monitor.add('batch_acc', lambda: utils.accuracy(scores, labels),
                            to_str=lambda x: f'{x * 100:.2f}%')  # 监控 batch 准确率
            
            # log, test, save ckpt  
            self.add_log(clock, self.monitor) # 添加 log
            

            if clock.epoch_end(): # 当 epoch 结束时需要执行的操作
                self.scheduler.step()


            if clock.epoch_end() and (clock.epoch + 1) % 1 == 0: # 不要每个epoch 都测试，减少测试所需要的时间
                
                # 当没有初始化多线程时，因为短路原则，也不会因为get_rank而报错
                if local_rank == -1 or dist.get_rank() == 0: 
                    self.call_testers(clock, composed_model) 
                    # 只保留精度最佳的ckpt，节省时间
                    self.tb_writer.add_scalar('test_by_person', self.current_accuracy, clock.epoch)
                    if self.current_accuracy > self.best_acc:
                        self.best_acc = self.current_accuracy
                        self.best_epoch = clock.epoch
                        if local_rank == -1:
                            best_model_state_dict = {k:v.to('cpu') for k, v in composed_model.server_tail.tail.state_dict().items()}
                        else:
                            best_model_state_dict = {k:v.to('cpu') for k, v in composed_model.module.server_tail.tail.state_dict().items()}
                        ckpt = {'best_epoch': self.best_epoch,
                                'best_acc': self.best_acc,
                                'model_state_dict': best_model_state_dict,
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'scheduler_state_dict': self.scheduler.state_dict()
                                }
                        utils.torch_save(ckpt, f'{self.ckpt_dir}/best_ckpt_server_tail_pretrain_debug.pth')
                        del ckpt


                        if local_rank == -1:
                            best_model_state_dict = {k:v.to('cpu') for k, v in composed_model.encoder.ext.state_dict().items()}
                        else:
                            best_model_state_dict = {k:v.to('cpu') for k, v in composed_model.module.encoder.ext.state_dict().items()}
                        ckpt = {'best_epoch': self.best_epoch,
                                'best_acc': self.best_acc,
                                'model_state_dict': best_model_state_dict,
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'scheduler_state_dict': self.scheduler.state_dict()
                                }
                        utils.torch_save(ckpt, f'{self.ckpt_dir}/best_ckpt_encoder_ext_pretrain_debug.pth')
                        del ckpt

                        # best_model_state_dict = {k:v.to('cpu') for k, v in model_lit.state_dict().items()}
                        # utils.torch_save(best_model_state_dict, os.path.join(self.work_dir, 'best_model.tar'))
                if local_rank != -1:
                    # 在多卡模式下在主进程测试并save，其他进程通过torch.distributed.barrier()来等待主进程完成validate操作
                    # barrier()函数的使用要非常谨慎，如果只有单个进程包含了这条语句，那么程序就会陷入无限等待
                    dist.barrier()
                
                self.tb_writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], clock.epoch)       
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
    
    def fit_Adv_Train(self,obf_feature_generator,aux_server_tail,test_model,local_rank=-1):
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

            obf_feature_generator.module.obfuscate.reset_parameters() 

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
            
            # log, test, save ckpt  
            self.add_log(clock, self.monitor) # 添加 log
            

            if clock.epoch_end():
                self.scheduler_aux_server_tail.step()
                self.scheduler_obf_feature_generator.step()

            if clock.epoch_end() and (clock.epoch + 1) % 10 == 0 : # 当 epoch 结束时需要执行的操作
                
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
                        utils.torch_save(ckpt, f'{self.ckpt_dir}/stage2_permute_obf.pth')
                        del ckpt

                        ckpt_aux = {'best_epoch': self.best_epoch,
                                'best_acc': self.best_acc,
                                'model_state_dict': aux_server_tail_state_dict,
                                'optimizer_state_dict': self.optimizer_aux_server_tail.state_dict(),
                                'scheduler_state_dict': self.scheduler_aux_server_tail.state_dict()
                                }
                        utils.torch_save(ckpt_aux, f'{self.ckpt_dir}/stage2_permute_server_tail_ckpt.pth')
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

            obf_feature_generator.obfuscate.reset_parameters() 


            # aux_classifier
            scores = test_model(imgs,labels)
                #loss
            loss_D = tc.nn.functional.cross_entropy(scores, labels)
            utils.step(self.optimizer_aux_server_tail, loss_D)
            

            
            self.tb_writer.add_scalar('loss_D', loss_D, clock.batch)


            
            self.monitor.add('loss_D', lambda: float(loss_D),
                            to_str=lambda x: f'{x:.2e}')  # 监控 loss
            
            self.monitor.add('batch_acc', lambda: utils.accuracy(scores, labels),
                            to_str=lambda x: f'{x * 100:.2f}%')  # 监控 batch 准确率
            
            # log, test, save ckpt  
            self.add_log(clock, self.monitor) # 添加 log
            

            if clock.epoch_end():
                self.scheduler_aux_server_tail.step()

            if clock.epoch_end() and (clock.epoch + 1) % 2 == 0 : # 当 epoch 结束时需要执行的操作
                
                # 当没有初始化多线程时，因为短路原则，也不会因为get_rank而报错
                if local_rank == -1 or dist.get_rank() == 0: 
                    self.call_testers(clock, test_model) 
                    # 只保留精度最佳的ckpt，节省时间
                    self.tb_writer.add_scalar('test_by_person', self.current_accuracy, clock.epoch)
                    if self.current_accuracy > self.best_acc:  # 永远保存最新的
                        self.best_acc = self.current_accuracy
                        self.best_epoch = clock.epoch
                        if local_rank == -1:
                            aux_server_tail_state_dict = {k:v.to('cpu') for k, v in aux_server_tail.tail.state_dict().items()}
                        else:
                            aux_server_tail_state_dict = {k:v.to('cpu') for k, v in aux_server_tail.tail.state_dict().items()}
                        
                        ckpt_aux = {'best_epoch': self.best_epoch,
                                'best_acc': self.best_acc,
                                'model_state_dict': aux_server_tail_state_dict,
                                'optimizer_state_dict': self.optimizer_aux_server_tail.state_dict(),
                                'scheduler_state_dict': self.scheduler_aux_server_tail.state_dict()
                                }
                        utils.torch_save(ckpt_aux, f'{self.ckpt_dir}/best_eve_server_tail_permute_layernorm_ckpt.pth')
                        del ckpt_aux

                        # best_model_state_dict = {k:v.to('cpu') for k, v in model_lit.state_dict().items()}
                        # utils.torch_save(best_model_state_dict, os.path.join(self.work_dir, 'best_model.tar'))
                if local_rank != -1:
                    # 在多卡模式下在主进程测试并save，其他进程通过torch.distributed.barrier()来等待主进程完成validate操作
                    # barrier()函数的使用要非常谨慎，如果只有单个进程包含了这条语句，那么程序就会陷入无限等待
                    dist.barrier()
                
                self.tb_writer.add_scalar('lr', self.optimizer_aux_server_tail.param_groups[0]['lr'], clock.epoch)
        
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
    
    def fit_train_aux_tail(self,obf_feature_generator,aux_server_tail,teacher_server_tail,test_model,local_rank=-1):
        '''训练模型并返回练完成的模型'''
        # 数据加载
        # self.call_testers(clock, model_lit)
        dataloader = self.get_dataloader(self.sample_transform) # 加载训练集
        clock = utils.TrainLoopClock(dataloader, self.total_epochs) # 配置训练循环

        test_model.encoder = obf_feature_generator.base_encoder
         # before_train 
        if local_rank == -1 or dist.get_rank() == 0: 
            self.call_testers(clock, test_model) 

        if local_rank != -1:
            # 在多卡模式下在主进程测试并save，其他进程通过torch.distributed.barrier()来等待主进程完成validate操作
            # barrier()函数的使用要非常谨慎，如果只有单个进程包含了这条语句，那么程序就会陷入无限等待
            dist.barrier()


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

            
            obf_features = obf_feature_generator(imgs)
            scores = aux_server_tail(obf_features,labels)

            with tc.no_grad():
                clean_feature = obf_feature_generator.base_encoder(imgs.detach())
                scores_clean = teacher_server_tail(clean_feature,labels)

            

            #distill loss
            KD_loss,fc_loss = loss_fn_kd(scores,labels,scores_clean)
            loss = KD_loss + fc_loss

            utils.step(self.optimizer_aux_server_tail, loss)
            

            
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
            
            # log, test, save ckpt  
            self.add_log(clock, self.monitor) # 添加 log
            

            if clock.epoch_end():
                self.scheduler_aux_server_tail.step()

            if clock.epoch_end() and (clock.epoch + 1) % 1 == 0 : # 当 epoch 结束时需要执行的操作
                
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
                        utils.torch_save(ckpt_aux, f'{self.ckpt_dir}/best_distill_server_tail_ckpt.pth')
                        del ckpt_aux

                        # best_model_state_dict = {k:v.to('cpu') for k, v in model_lit.state_dict().items()}
                        # utils.torch_save(best_model_state_dict, os.path.join(self.work_dir, 'best_model.tar'))
                if local_rank != -1:
                    # 在多卡模式下在主进程测试并save，其他进程通过torch.distributed.barrier()来等待主进程完成validate操作
                    # barrier()函数的使用要非常谨慎，如果只有单个进程包含了这条语句，那么程序就会陷入无限等待
                    dist.barrier()
                
                self.tb_writer.add_scalar('lr', self.optimizer_aux_server_tail.param_groups[0]['lr'], clock.epoch)
        
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

    def __init__(self, class_num, isobf=True, pretrained_path='', split_layer=6, tail_layer=6, ext_test=False, is_permute=True, is_matrix=True, debug_pos=False, vis=False,is_fix = True):
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
            self.tail = load_checkpoint(model=self.tail, pretrained_path=pretrained_path)
            if is_fix:
                for p in self.tail.parameters():
                    p.requires_grad = False

    def forward(self,mid_feats,labels,is_inference = False):
        rec_feats = self.tail(mid_feats)
        if is_inference:
            return rec_feats
        return self.softmax(rec_feats,labels)

class Face_server_tail_mid(block.model.light.ModelLight):
    '''server tail'''

    def __init__(self, class_num, isobf=True, pretrained_path='', split_layer=6, tail_layer=6, ext_test=False, is_permute=True, is_matrix=True, debug_pos=False, vis=False,is_fix = True):
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
        
        self.select_layer = tc.nn.Linear(394,197)
        
        if pretrained_path != '':
            self.tail = load_checkpoint(model=self.tail, pretrained_path=pretrained_path)
            if is_fix:
                for p in self.tail.parameters():
                    p.requires_grad = False

    def forward(self,mid_feats,labels,is_inference = False,):
        # 降低patchnum 的维度
        
        mid_feats = mid_feats.permute(0,2,1)
        mid_feats = self.select_layer(mid_feats)
        mid_feats = mid_feats.permute(0,2,1)
        rec_feats = self.tail(mid_feats)
        if is_inference:
            return rec_feats
        return self.softmax(rec_feats,labels)

class Face_adv_encoder(block.model.light.ModelLight):
    '''encoder'''

    def __init__(self, class_num, isobf=True, pretrained_path='', split_layer=6, tail_layer=6, ext_test=False, is_permute=True, is_matrix=True, debug_pos=False, vis=False,is_fix = True):
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
            self.ext = load_checkpoint(model=self.ext, pretrained_path=pretrained_path)
            if is_fix:
                for p in self.ext.parameters():
                    p.requires_grad = False


    def forward(self, images):
        mid_feats = self.ext(images)
        return mid_feats


class Obfuscate(tc.nn.Module):
    def __init__(self, patch_num, martix_size, is_permute, is_matrix):
        super().__init__()
        self.patch_num = patch_num
        self.martix_size = martix_size
        self.reset_parameters()  
        self.is_permute = is_permute
        self.is_matrix = is_matrix

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
            # find the (loc + 1)-th number that is not in v
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

    def forward(self, img):
        if self.is_permute:
            tmp_img = img.clone()
            for w in range(self.patch_num):
                img[:, w, :] = tmp_img[:, self.V1[w], :]
        if self.is_matrix:
            self.x = self.x.to(img.device)
            img = img.matmul(self.x)             
        return img


class Obf_feature_generator(block.model.light.ModelLight):
    def __init__(self, base_encoder,base_encoder_obf):
        super().__init__()
        self.base_encoder = base_encoder
        self.base_encoder_obf = base_encoder_obf
        self.obfuscate = Obfuscate(394, 768, True, False)
        for p in self.obfuscate.parameters():
            p.requires_grad = False

    def forward(self, images):
        mid_feats = self.base_encoder(images)
        obf_feats = self.base_encoder_obf(images)
        mix_feats = tc.cat([mid_feats,obf_feats],dim = 1)
        mix_feats = self.obfuscate(mix_feats)
        return mix_feats


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
    
    parser.add_argument("--debug", default=None, type=str, help="celeba_lr/celeba_pos")
    parser.add_argument("--mode", default="stage2", type=str, help="XNN/Test_Ext/Attack/Visualize")
    parser.add_argument("--obf", default="True", type=str)
    parser.add_argument("--is_permute", default="True", type=str)
    parser.add_argument("--is_matrix", default="True", type=str)
    parser.add_argument("--pretrained", default="/data/research/version2/privdl/data/model/vit-b-300ep.pth.tar", type=str,
                        help='path to mocov3 pretrained checkpoint')
    
    parser.add_argument("--person", default=1000, type=int)
    parser.add_argument("--split_layer", default=5, type=int)
    parser.add_argument("--tail_layer", default=6, type=int)
    parser.add_argument("--lr", default=5e-2, type=float)
    parser.add_argument("--batch_size", default=256, type=int)    
    parser.add_argument("--subset_ratio", default=1.0, type=float)

    

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
        dist.init_process_group(backend='nccl', init_method='env://', timeout=datetime.timedelta(seconds=18000))  # nccl是GPU设备上最快、最推荐的后端
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
        ckpt_dir = f'/data/ckpt/NC/modeXNN_obf{FLAGS.obf}_{FLAGS.is_permute}_{FLAGS.is_matrix}_{train_dataset}_{attacker_dataset}_8gpu'
    else:
        ckpt_dir = f'/data/ckpt/NC/mode{FLAGS.mode}_obf{FLAGS.obf}_{FLAGS.is_permute}_{FLAGS.is_matrix}_{train_dataset}_{attacker_dataset}_{proc_count}gpu'
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
            train_dataset, test_id_num=10, test_img_per_id=100)
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
    testers = [Tester(dataset=testset_by_img, name='testset_by_img', sample_transform=sample_transform),
               Tester(dataset=testset_by_person, name='testset_by_person', sample_transform=sample_transform),
               RestoreIdentificationAccuracy(dataset=testset_by_person, name='RestoreIdentificationAccuracy', sample_transform=sample_transform)]
    for tester in testers:
        tester.config_dataloader(batch_size=batch_size, num_workers=num_workers)

    
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
            pretrained_path=FLAGS.pretrained, split_layer=FLAGS.split_layer, tail_layer=FLAGS.tail_layer, is_permute=FLAGS.is_permute, is_matrix=FLAGS.is_matrix, debug_pos=(FLAGS.debug=='celeba_pos'),is_fix=True)
        
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
        trainer.config_logger(log_interval=log_interval)
        trainer.config_tester(testers)

        trainer.fit_ViT(composed_model , local_rank=local_rank)

    def train_adv_vit(): 

        base_encoder = Face_adv_encoder(class_num=trainset.class_num(), isobf=FLAGS.obf, \
            pretrained_path='/data/ckpt/NC/arkiv/stage_mid_norm/stage1_mid_norm_encoder_ext.pth', split_layer=FLAGS.split_layer, tail_layer=FLAGS.tail_layer, is_permute=FLAGS.is_permute, is_matrix=FLAGS.is_matrix, debug_pos=(FLAGS.debug=='celeba_pos'))
        
        base_encoder_obf = Face_adv_encoder(class_num=trainset.class_num(), isobf=FLAGS.obf, \
            pretrained_path=FLAGS.pretrained, split_layer=FLAGS.split_layer, tail_layer=FLAGS.tail_layer, is_permute=FLAGS.is_permute, is_matrix=FLAGS.is_matrix, debug_pos=(FLAGS.debug=='celeba_pos'),is_fix = False)
        
        aux_server_tail = Face_server_tail_mid(class_num=trainset.class_num(), isobf=FLAGS.obf, \
            pretrained_path='', split_layer=FLAGS.split_layer, tail_layer=FLAGS.tail_layer, is_permute=FLAGS.is_permute, is_matrix=FLAGS.is_matrix, debug_pos=(FLAGS.debug=='celeba_pos'),is_fix= False)
        

        obf_feature_generator = Obf_feature_generator(base_encoder = base_encoder ,base_encoder_obf=base_encoder_obf)

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
            obf_feature_generator = DDP(obf_feature_generator, device_ids=[local_rank], output_device=local_rank)

            aux_server_tail = aux_server_tail.to(local_rank)
            aux_server_tail = DDP(aux_server_tail, device_ids=[local_rank], output_device=local_rank)

            test_model = test_model.to(local_rank)
            test_model = DDP(test_model, device_ids=[local_rank], output_device=local_rank)
            
            

            
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
        trainer.config_logger(log_interval=log_interval)
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
        base_encoder = Face_adv_encoder(class_num=trainset.class_num(), isobf=FLAGS.obf, \
            pretrained_path='', split_layer=FLAGS.split_layer, tail_layer=FLAGS.tail_layer, is_permute=FLAGS.is_permute, is_matrix=FLAGS.is_matrix, debug_pos=(FLAGS.debug=='celeba_pos'),is_fix = True)
        
        base_encoder_obf = Face_adv_encoder(class_num=trainset.class_num(), isobf=FLAGS.obf, \
            pretrained_path='', split_layer=FLAGS.split_layer, tail_layer=FLAGS.tail_layer, is_permute=FLAGS.is_permute, is_matrix=FLAGS.is_matrix, debug_pos=(FLAGS.debug=='celeba_pos'),is_fix = True)
        
        aux_server_tail = Face_server_tail_mid(class_num=trainset.class_num(), isobf=FLAGS.obf, \
            pretrained_path='', split_layer=FLAGS.split_layer, tail_layer=FLAGS.tail_layer, is_permute=FLAGS.is_permute, is_matrix=FLAGS.is_matrix, debug_pos=(FLAGS.debug=='celeba_pos'),is_fix = False)
        
        # aux_server_tail = load_checkpoint(aux_server_tail,'/data/ckpt/NC/arkiv/stage2_permute/stage2_permute_server_tail_ckpt.pth')

        obf_feature_generator = Obf_feature_generator(base_encoder = base_encoder ,base_encoder_obf=base_encoder_obf)

        obf_feature_generator = load_checkpoint(obf_feature_generator,'/data/ckpt/NC/arkiv/stage2_permute/stage2_permute_obf.pth')

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

            # aux_server_tail = aux_server_tail.to(local_rank)
            # aux_server_tail = DDP(aux_server_tail, device_ids=[local_rank], output_device=local_rank)

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

        trainer = ExtTrainer_face_adv_stage_aux(
            dataset=trainset, total_epochs=train_epoch, \
                scheduler_aux_server_tail = scheduler_aux_server_tail ,\
                  work_dir=f'{work_dir}', sample_transform=sample_transform,\
                    optimizer_aux_server_tail = optimizer_aux_server_tail,\
                    monitor=monitor, tb_log_dir=tb_log_dir, ckpt_dir=ckpt_dir)
        
        trainer.config_dataloader(batch_size=batch_size, num_workers=num_workers)
        log_interval = 60
        trainer.config_logger(log_interval=log_interval)
        trainer.config_tester(testers)

        trainer.fit_train_adv_tail(obf_feature_generator,aux_server_tail,test_model, local_rank=local_rank)  

    def train_disdill():
        #FLAGS.pretrained
        base_encoder = Face_adv_encoder(class_num=trainset.class_num(), isobf=FLAGS.obf, \
            pretrained_path='', split_layer=FLAGS.split_layer, tail_layer=FLAGS.tail_layer, is_permute=FLAGS.is_permute, is_matrix=FLAGS.is_matrix, debug_pos=(FLAGS.debug=='celeba_pos'),is_fix = True)
        
        base_encoder_obf = Face_adv_encoder(class_num=trainset.class_num(), isobf=FLAGS.obf, \
            pretrained_path='', split_layer=FLAGS.split_layer, tail_layer=FLAGS.tail_layer, is_permute=FLAGS.is_permute, is_matrix=FLAGS.is_matrix, debug_pos=(FLAGS.debug=='celeba_pos'),is_fix = True)
        
        aux_server_tail = Face_server_tail(class_num=trainset.class_num(), isobf=FLAGS.obf, \
            pretrained_path='', split_layer=FLAGS.split_layer, tail_layer=FLAGS.tail_layer, is_permute=FLAGS.is_permute, is_matrix=FLAGS.is_matrix, debug_pos=(FLAGS.debug=='celeba_pos'),is_fix = False)
        
        teacher_server_tail = Face_server_tail(class_num=trainset.class_num(), isobf=FLAGS.obf, \
            pretrained_path="/data/ckpt/NC/arkiv/stage_mid_norm/stage1_mid_norm_tail_tail.pth", split_layer=FLAGS.split_layer, tail_layer=FLAGS.tail_layer, is_permute=FLAGS.is_permute, is_matrix=FLAGS.is_matrix, debug_pos=(FLAGS.debug=='celeba_pos'),is_fix = True)


        obf_feature_generator = Obf_feature_generator(base_encoder = base_encoder ,base_encoder_obf=base_encoder_obf)
        obf_feature_generator = load_checkpoint(obf_feature_generator,'/data/ckpt/NC/arkiv/gan120_permute/best_obf_feature_generator_ckpt.pth')


        test_model = Composed_ModelLit(obf_feature_generator.base_encoder,teacher_server_tail)
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
        trainer.config_logger(log_interval=log_interval)
        trainer.config_tester(testers)

        trainer.fit_train_aux_tail(obf_feature_generator,aux_server_tail,teacher_server_tail,test_model, local_rank=local_rank)  
        


    def Visualize():
        model_lit = XNN_Single_Lit(class_num=trainset.class_num(), isobf=FLAGS.obf, \
                pretrained_path=FLAGS.pretrained, split_layer=FLAGS.split_layer, \
                tail_layer=FLAGS.tail_layer, is_permute=FLAGS.is_permute, is_matrix=FLAGS.is_matrix, \
                debug_pos=(FLAGS.debug=='celeba_pos'), vis=True)
        for p in model_lit.parameters():
            p.requires_grad = False
        
        device = tc.device('cuda' if tc.cuda.is_available else 'cpu')
        model_lit = model_lit.to(device)
        
        print(f'rank{local_rank}_torch_load: {ckpt_dir}/best_ckpt.pth')
        checkpoint_data = tc.load(f'{ckpt_dir}/best_ckpt.pth', map_location=tc.device(device))
        model_lit.load_state_dict(checkpoint_data['model_state_dict'], strict=False)
        
        feats, labels = testers[1].test(model_lit, vis=True)
        vis_emb_data ={
            'feats': feats,
            'labels': labels
        }
        tc.save(vis_emb_data, '/data/vis_midobf_feats.pth')

        
        print('done')
        return

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
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" nohup python3 -m torch.distributed.launch --nproc_per_node 8 100-2-adv_Distillation.py > pretrain_aux_train.out 2>&1 &
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 -m torch.distributed.launch --nproc_per_node 8 100-2-adv_Distillation.py --mode 'stage_aux'
# nohup python3  100-1-adv_Distillation.py > version2.out 2>&1 &

# CUDA_VISIBLE_DEVICES="4,5,6,7" nohup python3 -m torch.distributed.launch --nproc_per_node 4 100-2-adv_Distillation.py --mode 'stage_aux' > pretrain_aux_train.out 2>&1 &

# CUDA_VISIBLE_DEVICES="2,3" python3 -m torch.distributed.launch --nproc_per_node 2 --master_port=25647  100-2-adv_Distillation.py --mode 'stage3' 
# CUDA_VISIBLE_DEVICES="0,1" nohup python3 -m torch.distributed.launch --nproc_per_node 2 --master_port=25646  100-2-adv_Distillation.py --mode 'stage3' > kd_fc_loss.out 2>&1 &
# CUDA_VISIBLE_DEVICES="2,3" nohup python3 -m torch.distributed.launch --nproc_per_node 2 --master_port=25648  100-2-adv_Distillation.py --mode 'stage3' > l2_loss.out 2>&1 &


# pid : 119182    4,5,6,7
# pid : 124577    0,1
# pod : 127348    2,3
# fsd  vfs 

# CUDA_VISIBLE_DEVICES="0,1,2,3" python3 -m torch.distributed.launch --nproc_per_node 4 100-2-adv_Distillation.py --mode 'stage3'

# CUDA_VISIBLE_DEVICES="0,1,2,3" nohup python3 -m torch.distributed.launch --nproc_per_node 4 --master_port=25649 100-1-adv_Distillation.py --mode 'stage1' > test1.log 2>&1 &
