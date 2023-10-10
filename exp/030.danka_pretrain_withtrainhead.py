'''
XNN单卡：预训练 + xnn训练
'''
# -*- coding: UTF-8 -*-
import math
import os
import random
import PIL.Image as Image
import cv2
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


class ExtTrainer(block.train.standard.Trainer):
    """Trainer的子类，重写其中的fit函数"""
    def __init__(self, dataset, total_epochs, val_set=None, work_dir=None):
        super().__init__(dataset, total_epochs, val_set, work_dir)
        self.best_acc = 0
        self.best_epoch = 0

    def fit_ViT(self, model_lit, local_rank=-1):
        '''训练模型并返回练完成的模型'''
        dataloader = self.get_dataloader(model_lit.sample_transform) # 加载训练集
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


class XNN_Single_Lit(block.model.light.ModelLight):
    '''单客户场景下的 XNN'''

    def __init__(self, xnn_parts, class_num, normalize=None):
        super().__init__()
        self.normalize = normalize
        self.extractor = xnn_parts.extractor()
        self.obfuscate = xnn_parts.obfuscate()
        self.tail = xnn_parts.tail()
        self.softmax = block.loss.amsoftmax.AMSoftmax(
            in_features=self.tail.feat_size, class_num=class_num)
        self.config_optim()


    def config_optim(self):
        param_groups = [  # 每类参数设置不同的 weight_decay
            dict(params=self.extractor.parameters(), weight_decay=4e-5),
            dict(params=self.obfuscate.parameters(), weight_decay=4e-5),
            dict(params=self.tail.parameters(), weight_decay=4e-5),
            dict(params=self.softmax.parameters(), weight_decay=4e-4)]
        self.optimizer = tc.optim.SGD(param_groups, lr=0.05)

        self.monitor.add('lr',  # 监控学习率
                         get=lambda: self.optimizer.param_groups[0]['lr'],
                         to_str=lambda x: f'{x:.1e}')

    def sample_transform(self, sample):
        '''H*W*3*uint8(0\~255) -> 3*112*112*float32(-1\~1)'''
        img, label = sample
        np_img = resize_norm_transpose_insta(img, size=112)
        # print(type(np_img)) # numpy.ndarray
        # print(np_img.shape) # (3,112, 112)
        transformed_img = self.normalize(np_img)
        return transformed_img, label

    def train(self):
        self.extractor.train()
        self.obfuscate.train()
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
        with tc.no_grad():
            mid_feats = self.extractor(batch_input)
            # mid_feats = mid_feats.reshape(mid_feats.shape[0], self.c, self.w, self.h)
            mid_feats = self.obfuscate(mid_feats)
            out = self.tail(mid_feats)
        return out
    
    def get_tail(self):
        return self.tail
    
    def set_ext(self, ext):
        self.extractor.load_state_dict(ext, strict=False)

class XnnParts_MobileFaceNet_train:
    
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
            return block.model.mobile_face_net.MobileFaceNet()
    
        def inn(self):
            return block.model.inv_net.InvNet_2()

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

    def inn(self):
        return block.model.inv_net.InvNet_2()


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
    parser.add_argument("--ext_model", default='vit', type=str)
    parser.add_argument("--person", default=1000, type=int)
    FLAGS = parser.parse_args()
    local_rank = FLAGS.local_rank

    proc_count = 1
    if local_rank != -1:
        dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端
        proc_count = tc.distributed.get_world_size() #获取全局并行数，即并行训练的显卡的数量
    
    batch_size = 128
    num_workers = 8
    train_epoch = 100
    
    train_dataset = FLAGS.train_dataset
    print("train_dataset:", train_dataset)

    result_path = exp_utils.setup_result_path(os.path.basename(__file__))
    result_path = f'{result_path}/{FLAGS.ext_model}/{train_dataset}/{proc_count}_gpus'
    work_dir = result_path
    
    if train_dataset == 'celeba':
        train_dataset = block.dataset.hubble.xnn_paper.celeba() # 20w
        normalize = celeba_normalize
    elif train_dataset == 'imdb':
        train_dataset = block.dataset.hubble.xnn_paper.imdb() # 200w+
        normalize = imdb_normalize
    elif train_dataset == 'facescrub':
        train_dataset = block.dataset.hubble.xnn_paper.facescrub() # 10w
        FLAGS.person = 100 # person_num of facescrub = 530 < 1000
        normalize = facescrub_normalize
    elif train_dataset == 'msra':
        train_dataset = block.dataset.hubble.xnn_paper.msra() # 300w+
        normalize = msra_normalize
    elif train_dataset == 'webface':
        train_dataset = block.dataset.hubble.xnn_paper.webface() # 50w
        normalize = webface_normalize
    elif train_dataset == 'vggface2':
        train_dataset = block.dataset.hubble.xnn_paper.vggface2() # 200w+
        normalize = vggface2_normalize
    else:
        raise ValueError('dataset do not exist')
    trainset, testset_by_img, testset_by_person = split_dataset(train_dataset, FLAGS.person)
    # trainset = trainset.subset_by_class_id(trainset.class_ids()[:1000])

    Tester = block.test.top1_test.Top1_Tester
    testers = [Tester(dataset=testset_by_img, name='testset_by_img'),
               Tester(dataset=testset_by_person, name='testset_by_person')]
    for tester in testers:
        tester.config_dataloader(batch_size=batch_size, num_workers=num_workers)

    def train_vit_ext(): 
        if FLAGS.ext_model == 'vit':
            xnn_parts = XnnParts_ViT_train()
        elif FLAGS.ext_model == 'mobilefacenet':
            xnn_parts = XnnParts_MobileFaceNet_train()
        model_lit = XNN_Single_Lit(xnn_parts, class_num=trainset.class_num(), normalize=normalize)
        device = tc.device('cuda' if tc.cuda.is_available else 'cpu')
        model_lit = model_lit.to(device)
        
        trainer = ExtTrainer(
            dataset=trainset, total_epochs=train_epoch, work_dir=f'{work_dir}')
        trainer.config_dataloader(batch_size=batch_size, num_workers=num_workers)
        trainer.config_logger(log_interval=60)
        trainer.config_tester(testers)
        # trainer.fit_ViT(model_lit, local_rank=local_rank)
        trainer.fit_ViT(model_lit)

        return
    
    train_vit_ext()


if __name__ == '__main__':
    main()
