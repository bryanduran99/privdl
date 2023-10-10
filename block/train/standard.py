import os
from threading import local
import torch as tc
from tracemalloc import start

from torch.optim import lr_scheduler
from torch.utils import data
from torch.utils.data import DataLoader

import utils
import time
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

class BatchLoader(object):
    def __init__(self, path):
        self.path = path
        self.files = os.listdir(path)
    def __iter__(self):
        for file in self.files:
            flag = file.split('-')[0]
            # if flag == 'raw' or 'permuted':
            #     continue
            feats = tc.load(os.path.join(self.path, file))
            yield feats
    def __len__(self):
        return len(self.files)


class Trainer:
    '''配置训练参数，训练模型'''
    def __init__(self, dataset, total_epochs, val_set=None, work_dir=None, ckpt_dir=None):
        '''设置训练集、训练epoch数和工作目录（用于保存 log 和 checkpoint，
        如果 work_dir=None 那么将不进行保存，默认 None），
        并用默认参数配置训练过程，可用 config 系列函数对训练配置进行修改。'''
        self.dataset = dataset
        self.total_epochs = total_epochs
        self.work_dir = work_dir
        self.ckpt_dir = ckpt_dir
        self.val_set = val_set
        self.current_accuracy = 0
        self.best_acc = 0
        self.best_epoch = 0
        
        if work_dir is not None:
            utils.makedirs(work_dir, exist_ok=True)

        self.config_dataloader() # self.get_dataloader
        self.config_logger() # self.add_log
        self.config_tester() # self.call_testers
        self.config_checkpoint() # self.save_checkpoint

    def config_dataloader(self, batch_size=256, num_workers=8, class_bal=False):
        '''设置训练数据集 loader 的 batch_size 和 num_workers'''
        def get_dataloader(transform=None):
            '''按设定的参数创建 DataLoader，transform 将施加在每个 sample 上'''
            dataset = self.dataset
            if transform is not None:
                dataset = utils.TransformedDataset(dataset, transform)
            
            # 判断是否分布式训练，如果是则返回适用于多进程的数据采样方法
            if dist.is_available() and  dist.is_initialized():
                # DDP：使用DistributedSampler，DDP帮我们把细节都封装起来了。
                if class_bal:
                    train_sampler = utils.DistributedWeightedSampler(dataset, weights=dataset.weights, replacement=True)
                else:
                    train_sampler = DistributedSampler(dataset)
                # 去掉参数shuffle=True,因为torch.utils.data.distributed.DistributedSampler是默认使用shuffle=True的，所以不能多次使用
                return utils.DataLoader(dataset, batch_size=batch_size,
                num_workers=num_workers, sampler=train_sampler, drop_last=True, pin_memory=True)
            if class_bal:
                train_sampler = data.sampler.WeightedRandomSampler(
                    weights=dataset.weights,
                    num_samples=len(dataset),
                    replacement=True)
                return utils.DataLoader(dataset, batch_size=batch_size,
                    num_workers=num_workers, sampler=train_sampler, drop_last=True, pin_memory=True)
            return utils.DataLoader(dataset, batch_size=batch_size,
                num_workers=num_workers, shuffle=True, drop_last=True, pin_memory=True) # shuffle=True
        self.get_dataloader = get_dataloader

    def config_logger(self, log_interval=60, save_interval=10*60):
        '''log_interval: 打印并记录 log 的间隔(单位秒)，
        save_interval: 保存 log 文件的间隔(单位秒)'''
        logs = []
        log_path = f'{self.work_dir}/logs_{time.strftime("%y%m%d%H%M%S",time.localtime())}.json'
        @utils.interval(log_interval)
        def add_log(clock, monitor):
            '''把 clock 和 monitor 的信息打印并记到 log 中'''
            logs.append(dict(clock=clock.check(), monitor=monitor.check()))
            save_log()
        @utils.interval(save_interval)
        def save_log():
            '''保存 log 到 work_dir 目录下'''
            utils.json_save(logs, log_path)
        self.add_log = add_log
        return log_path

    def config_tester(self, testers=None):
        '''testers: list of tester, tester.test(model_lit) 用于测试模型，
        interval: 测试间隔(单位秒)'''
        logs = []
        log_path = f'{self.work_dir}/test_logs_{time.strftime("%y%m%d%H%M%S",time.localtime())}.json'
        def call_testers(clock, model_lit):
            '''如果有 tester，则进行测试，并将测试时间和结果记录到 work_dir 目录下'''
            if testers is None:
                return
            results = {tester.name: tester.test(model_lit) for tester in testers}
            logs.append(dict(clock=clock.log(), results=results))
            utils.json_save(logs, log_path)
            if 'testset_by_person' in results.keys():
                self.current_accuracy = results['testset_by_person']['rate']
            elif 'RestoreIdentificationAccuracy' in results.keys():
                self.current_accuracy = results['RestoreIdentificationAccuracy']['rate']
        self.call_testers = call_testers
        return log_path

    def config_checkpoint(self):
        '''interval: 保存 checkpoint 的间隔(单位秒)'''
        def save_checkpoint(model_lit):
            '''将模型的 state_dict 保存到 work_dir 目录下'''
            utils.torch_save(model_lit.state_dict(),
                path=f'{self.work_dir}/checkpoint.tar')
        self.save_checkpoint = save_checkpoint

    def fit(self, model_lit):
        '''训练模型并返回练完成的模型'''
        model_lit = model_lit.cuda() # 加载模型到 GPU
        dataloader = self.get_dataloader(model_lit.train_sample_transform) 
        print(self.total_epochs)
        clock = utils.TrainLoopClock(dataloader, self.total_epochs) # 配置训练循环
        for batch_data in clock: # 训练循环
            batch_data = utils.batch_to_cuda(batch_data) # 加载 batch 数据到 GPU
            model_lit.train() # 设置模型为训练模式
            model_lit.train_step(batch_data) # 训练一个 batch
            if clock.epoch_end(): # 当 epoch 结束时需要执行的操作
                model_lit.on_epoch_end()
                if clock.epoch % 2 == 0:
                    self.call_testers(clock, model_lit) # 进行测试
                    self.save_checkpoint(model_lit) # 保存 checkpoint
            self.add_log(clock, model_lit.monitor) # 添加 log
        model_lit = model_lit.cpu() # 取回模型到 CPU
        utils.torch_save(model_lit.state_dict(), # 保存训练完成的模型
            path=f'{self.work_dir}/model_lit.tar')
        return model_lit # 返回训练完成的模型

    def fit_on_feats(self, model_lit):
        model_lit = model_lit.cuda() # 加载模型到 GPU
        batch_loader = BatchLoader(model_lit.train_feats_dir)

        clock = utils.TrainLoopClock(batch_loader, self.total_epochs) # 配置训练循环
        for batch_feats in clock:
            batch_feats = (batch_feats[0].cuda(), batch_feats[1].cuda())
            model_lit.train() # 设置模型为训练模式
            model_lit.train_step_on_feats(batch_feats) ### TO -- DO 
            if clock.epoch_end(): # 当 epoch 结束时需要执行的操作
                model_lit.on_epoch_end()
            self.add_log(clock, model_lit.monitor) # 添加 log
            self.call_testers(clock, model_lit) # 进行测试
            self.save_checkpoint(model_lit) # 保存 checkpoint
        model_lit = model_lit.cpu() # 取回模型到 CPU
        utils.torch_save(model_lit.state_dict(), # 保存训练完成的模型
            path=f'{self.work_dir}/model_lit.tar')
        return model_lit # 返回训练完成的模型


    def fit_ViT(self, model_lit, local_rank=-1):
        '''训练模型并返回练完成的模型'''
        # 加载训练集, DDP包裹的model_lit只能foward,没有其他函数
        dataloader = self.get_dataloader(model_lit.sample_transform)
        # DDP：设置sampler的epoch，
        # DistributedSampler需要这个来指定shuffle方式，
        # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果。
        clock = utils.TrainLoopClock(dataloader, self.total_epochs) # 配置训练循环
        for batch_data in clock: # 训练循环
            if local_rank != -1:
                batch_data = utils.batch_to_rank(batch_data, local_rank=local_rank)
            else:
                batch_data = utils.batch_to_cuda(batch_data)
            model_lit.train() # 设置模型为训练模式
            model_lit.train_step(batch_data) # 训练一个 batch
            if clock.epoch_end(): # 当 epoch 结束时需要执行的操作
                model_lit.on_epoch_end()
            self.add_log(clock, model_lit.monitor) # 添加 log

            if clock.epoch_end(): # 控制clock的时间，保证每个epoch进行一次test
                if dist.is_available() and  dist.is_initialized():
                    if dist.get_rank() == 0: # 只在主进程进行测试和save
                        self.call_testers(clock, model_lit.module) 
                        if self.current_accuracy > self.best_acc:
                            self.best_acc = self.current_accuracy
                            self.best_epoch = clock.epoch
                            best_model_state_dict = {k:v.to('cpu') for k, v in model_lit.module.state_dict().items()}
                            utils.torch_save(best_model_state_dict, os.path.join(self.work_dir, 'best_model.tar'))
                    # 在多卡模式下在主进程测试并save，其他进程通过torch.distributed.barrier()来等待主进程完成validate操作
                    # barrier()函数的使用要非常谨慎，如果只有单个进程包含了这条语句，那么程序就会陷入无限等待
                    dist.barrier()
                else:
                    print('test...')
                    self.call_testers(clock, model_lit) 
                    if self.current_accuracy > self.best_acc:
                        self.best_acc = self.current_accuracy
                        self.best_epoch = clock.epoch
                        best_model_state_dict = {k:v.to('cpu') for k, v in model_lit.state_dict().items()}
                        utils.torch_save(best_model_state_dict, os.path.join(self.work_dir, 'best_model.tar'))

        return model_lit # 返回训练完成的DDP模型, 与传入的模型保持类型一致

    def fit_ViT_CheX(self, model_lit, local_rank=-1):
        '''训练模型并返回练完成的模型'''
        if local_rank == -1:
            device = tc.device('cuda' if tc.cuda.is_available else 'cpu')
            model_lit = model_lit.to(device) # 加载模型到 GPU
        dataloader = self.get_dataloader() # 加载训练集,不用sample_trans，dataset中已 转换
        # DDP：设置sampler的epoch，
        # DistributedSampler需要这个来指定shuffle方式，
        # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果。
        clock = utils.TrainLoopClock(dataloader, self.total_epochs) # 配置训练循环
        cur_epoch = 0
        for batch_data in clock: # 训练循环
            batch_data = [batch_data['img'], batch_data['label']]
            if local_rank != -1:
                batch_data = utils.batch_to_rank(batch_data, local_rank=local_rank)
            else:
                batch_data = utils.batch_to_cuda(batch_data)
            # print('batch:{}'.format(batch_data[0][0]))
            # time.sleep(5)
            # continue
        
            model_lit.train() # 设置模型为训练模式

            lr_schedule_flag = False
            test_flag = False
            if cur_epoch != clock.epoch:
                cur_epoch = clock.epoch
                test_flag = True
                if cur_epoch % 5 == 0:
                    lr_schedule_flag = True
            model_lit.train_step(batch_data, lr_schedule_flag=lr_schedule_flag, chex_flag=True) # 训练一个 batch
            if clock.epoch_end(): # 当 epoch 结束时需要执行的操作
                model_lit.on_epoch_end()
            self.add_log(clock, model_lit.monitor) # 添加 log
            # 控制clock的时间，保证每三个epoch进行一次test
            if local_rank == -1 or dist.get_rank() == 0: 
                if test_flag:
                    # self.call_testers(clock, model_lit) # 进行测试
                    self.test_chex(model_lit, self.val_set)
            
            # 当没有初始化多线程时，因为短路原则，也不会因为get_rank而报错
            if local_rank == -1 or dist.get_rank() == 0: 
                self.save_checkpoint(model_lit)
        
        if local_rank == -1 or dist.get_rank() == 0:
            model_lit = model_lit.cpu() # 取回模型到 CPU
            utils.torch_save(model_lit.state_dict(), # 保存训练完成的模型
                path=f'{self.work_dir}/model_lit.tar')
        return model_lit # 返回训练完成的模型
    
    def test_chex(self, model_lit, dataset):
        sampler = data.sampler.WeightedRandomSampler(
                weights=dataset.weights,
                num_samples=len(dataset),
                replacement=True)
        dataloader = DataLoader(dataset, num_workers=8, sampler=sampler, batch_size=128)
        with tc.no_grad():
            total_correct_num = 0
            total_num = 0
            total_loss = 0
            fprs, tprs, thresholds = [], [], []
            for i, batch in enumerate(dataloader):
                batch = batch
                img_batch = batch['img'].cuda()
                label_batch = batch['label'].cuda()
                scores = tc.nn.functional.softmax(model_lit.inference(img_batch), dim=1)
                # print(f'pred{pred}\nlabel{label_batch}')

                # predict_idx = (pred == tc.max(pred, dim=1, keepdim=True)[0]).to(dtype=tc.float32)
                # correct = tc.mul(predict_idx, label_batch)

                correct_num = (scores.argmax(dim=1) == label_batch.argmax(dim=1)).sum()
                batch_num = label_batch.size()[0]
                loss = tc.nn.functional.cross_entropy(scores, label_batch)
                if i % 30 == 0:
                    print("@ Validate | positive/total: %d/%d | mAP %.3f | loss %.3f"%(correct_num, batch_num, correct_num / batch_num, loss))
                total_correct_num += correct_num
                total_num += batch_num
                total_loss += loss
                
                # ROC
                # pred = tc.nn.functional.softmax(scores, dim=-1)
                labels = label_batch.argmax(dim=1)
                pos_scores = tc.transpose(scores, 0, 1)[1]
                fpr, tpr, threshold = metrics.roc_curve(labels.cpu().numpy(), pos_scores.cpu().numpy())
                
            print("@ Validate | positive/total: %d/%d | mAP %.3f | cumulative loss %.3f"%(total_correct_num, total_num, total_correct_num / total_num, total_loss))
            self.roc_curve(fpr,tpr)
            
    # ROC
    def roc_curve(self, fpr, tpr):
        roc_auc = metrics.auc(fpr, tpr)

        plt.figure()
        lw = 2
        plt.figure(figsize=(10,10))
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC of NC on CheXPert')
        plt.legend(loc="lower right")
        plt.savefig(f'/home/liukaixin/privdl/privdl/exp/roc_curve_{time.strftime("%y%m%d%H%M%S",time.localtime())}.png')
        plt.close()