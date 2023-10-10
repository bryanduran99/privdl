'''
note:
因为多卡训练传入fit的是model_lit.module(即raw_model)，而不是多线程模型model_lit,所以在fit中各线程并没有融合梯度，而是各自为战。
多卡训练结果不可信（比实际的ACC要低），单卡执行脚本的结果不受影响，仍然可信
'''
import argparse
from collections import Counter
from enum import Flag
from pydoc import cli
from threading import local

from regex import F
from torch.optim import lr_scheduler
import exp_utils
exp_utils.setup_import_path()
import numpy as np
import torch as tc
from block.loss import loss_utils
import cv2
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import json

import block
import utils
from einops import rearrange
import PIL
import matplotlib.pyplot as plt
from torchsummary import summary

# import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

# 新增：
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# CHEXPERT_DATA_PATH = "/home/liukaixin/privdl/neuracrypt_era/data/chexpert.json"
CHEXPERT_TRAIN_PATH = '/home/liukaixin/privdl/privdl/tmp/chexpert_train.json'
CHEXPERT_VAL_PATH = '/home/liukaixin/privdl/privdl/tmp/chexpert_val.json'

class ChexpertDataset(Dataset):
    def __init__(self, access:str=None, benchmark:str="Edema", train:bool=None, max_length:int=None, data_path=None):
        '''
        Parameters
        -----
        - access(str): ["public", "private-encoded", None] divide access of data owner and adversary
        - label(str): ["Edema", ..., None] sick
        - train(bool | None): train or validate None: all
        '''
        super(ChexpertDataset, self).__init__()
        self.metadata = json.load(open(data_path)) 
        self.benchmark = benchmark
        if max_length is not None:
            self.metadata = self.metadata[0: min(max_length, len(self.metadata))]
        
        stage = 'train' if train else 'val'
        print(f'raw_{stage}data_len:{len(self.metadata)}')
        
        def filterDataset(item:object):
            isSelected = True
            if access is not None:
                isSelected = isSelected and (item['challenge_split'] == access)
            if train is not None:
                if train:
                    isSelected = isSelected and (item['split_group'] == "train")
                else:
                    isSelected = isSelected and (item['split_group'] != "train")
            isSelected = isSelected and (item['label_dict']['Edema'] == "1.0" or item['label_dict']['Edema'] == "0.0")
            return isSelected

        self.metadata = list(filter(filterDataset, self.metadata))
        labels = ["1.0" if md['label_dict']['Edema']== "1.0"  else "0.0" for md in self.metadata]
        # print(labels)
        label_counts = Counter(labels) # 两种
        weight_per_label = 1./ len(label_counts)
        label_weights = {
            label: weight_per_label/count for label, count in label_counts.items()
            }
        self.weights = [label_weights[l] for l in labels]
        # print(self.weights)




    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, idx):
        '''
        item size [3, 240, 240]
        '''
        item = {}
        #  H*W*3*uint8(0\~255) -> 3*256*256*float32(-1\~1)
        # print( (cv2.imread(self.metadata[idx]['path']) / 255 - 0.5) / 0.5 )
        item['img'] = tc.tensor(cv2.resize(
            (cv2.imread(self.metadata[idx]['path']) / 255 - 0.5) / 0.5, [256, 256], interpolation=cv2.INTER_CUBIC),
            dtype=tc.float32).permute([2, 0, 1])
        item['label'] = self.get_label(self.metadata[idx], self.benchmark)
        item['pid'] = self.metadata[idx]['pid']
        return item
    
    def get_label(self, item, benchmark):
        if benchmark in item['label_dict'] and item['label_dict'][benchmark] == "1.0":
            return tc.tensor([0.0, 1.0], dtype = tc.float32)
        return tc.tensor([1.0, 0.0], dtype = tc.float32)


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

        self.monitor.add('lr', # 监控学习率
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
            to_str=lambda x: f'{x:.2e}') # 监控 loss
        self.monitor.add('batch_acc', lambda: utils.accuracy(scores, labels),
            to_str=lambda x: f'{x*100:.2f}%') # 监控 batch 准确率

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
        ml_params = utils.chain(
            self.softmax.parameters(), self.tail.ml_params())
        prelu_params = self.tail.prelu_params()
        base_params = self.tail.base_params()

        param_groups = [ # 每类参数设置不同的 weight_decay
            dict(params=base_params, weight_decay=4e-5),
            dict(params=prelu_params, weight_decay=0),
            dict(params=ml_params, weight_decay=4e-4)]
        self.optimizer = tc.optim.AdamW(param_groups, lr=0.1)

        self.monitor.add('lr', # 监控学习率
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
            mid_feats = self.obfuscate(mid_feats)
        scores = self(mid_feats, labels)
        loss = tc.nn.functional.cross_entropy(scores, labels)
        utils.step(self.optimizer, loss)

        self.monitor.add('loss', lambda: float(loss),
            to_str=lambda x: f'{x:.2e}') # 监控 loss
        self.monitor.add('batch_acc', lambda: utils.accuracy(scores, labels),
            to_str=lambda x: f'{x*100:.2f}%') # 监控 batch 准确率

    def inference(self, batch_input):
        mid_feats = self.extractor(batch_input)
        mid_feats = self.obfuscate(mid_feats)
        return self.tail(mid_feats)


class NC_Single_Lit(block.model.light.ModelLight):
    '''单客户场景下的 NeuraCrypt'''

    def __init__(self, xnn_parts):
        super().__init__()
        self.extractor = xnn_parts.extractor()
        self.tail = xnn_parts.tail()
        self.cls_num = xnn_parts.cls_num
        self.loss = tc.nn.MSELoss()
        # self.softmax = block.loss.amsoftmax.AMSoftmax(
            # in_features=self.tail.feat_size, class_num=self.cls_num)
        self.config_optim()

    def config_optim(self, lr=1e-4):
        self.optimizer = tc.optim.AdamW([{'params': self.extractor.parameters()},
                                        {'params': self.tail.parameters()}], lr=lr)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

        self.monitor.add('lr', # 监控学习率
            get=lambda: self.optimizer.state_dict()['param_groups'][0]['lr'],
            to_str=lambda x: f'{x:.1e}')

    def sample_transform(self, sample):
        img, label = sample
        img = self.extractor.img_transform(img)
        # label = label.astype(np.float32) 
        # print("type:", type(img), type(label))
        return img, label

    def train(self):
        self.extractor.train()
        for p in self.extractor.parameters():
            p.requires_grad = False
        self.tail.train()
        # self.softmax.train()

    def eval(self):
        self.extractor.eval()
        self.tail.eval()
        # self.softmax.eval()

    def forward(self, imgs, labels):
        mid_feats = self.extractor(imgs)
        templates = self.tail(mid_feats)
        return templates

    def train_step(self, batch_data, lr_schedule_flag=False, chex_flag=False):
        imgs, labels = batch_data
        # one_hot_labels = loss_utils.one_hot(labels, self.cls_num)
        # print("logits:", logits)
        # print("label:", labels)
        if chex_flag:
            scores = self.inference(imgs)
            # scores = tc.nn.functional.softmax(scores, dim=-1)
            # loss = tc.nn.functional.mse_loss(pred, labels)
        else:
            scores = self(imgs, labels)
        print("logits:", scores[:20])
        print("label:", labels[:20])
        loss = self.loss(scores, labels)
        # print("logits and label's type:{},{}".format(logits.dtype, labels.dtype))

        lr_scheduler = None
        if lr_schedule_flag:
            lr_scheduler = self.scheduler
        utils.step(self.optimizer, loss, lr_scheduler)

        self.monitor.add('loss', lambda: float(loss),
            to_str=lambda x: f'{x:.2e}') # 监控 loss
        # print(f'scores:{scores} \nlabels:{labels}')
        self.monitor.add('batch_acc', lambda: utils.accuracy(scores, labels.argmax(dim=1)),
            to_str=lambda x: f'{x*100:.2f}%') # 监控 batch 准确率

    def inference(self, batch_input):
        mid_feats = self.extractor(batch_input)
        templates = self.tail(mid_feats)
        return templates


class XnnParts_template:

    def __init__(self, pretrain_ckpt):
        self.pretrain_ckpt = pretrain_ckpt

    def extractor(self):
        print('load extractor...')
        model = block.model.mobile_face_net.MobileFaceNetHead(layer_num=2)
        params = utils.torch_load(self.pretrain_ckpt)
        params = utils.sub_state_dict(params, 'model.')
        model.load_state_dict(params, strict=False)
        return model

    def obfuscate(self):
        return tc.nn.Conv2d(64, 64, 1, 1, 0, bias=False)

    def tail(self):
        return block.model.mobile_face_net.MobileFaceNetTail(layer_num=-2)

    def inn(self):
        return block.model.inv_net.InvNet_2()

class NC_Parts:

    def __init__(self, class_num, input_size):
        self.cls_num = class_num
        self.input_size = input_size 

    def extractor(self):
        return block.model.ViT.Encoder(depth=7, img_size=self.input_size, kernel_size=16, in_channels=3, hidden_channels=2048, channel_dilation=1, private=True)
        # 3, 256, 4, 3, 6, 1
    def obfuscate(self):
        pass
        return 

    def tail(self):
        # num_classes即templates的维度
        return block.model.ViT.DeepViT(feat_size=2, dim=2048, depth=1, heads=12, mlp_dim=8)

    def inn(self):
        return

def split_dataset(dataset, data_name, debug=False): #MSRA
    if data_name == 'celeba':
        ids, imgs_per = 1000, 20 #CelebA
    else:
        ids, imgs_per = 1000, 50 #MSRA
    
    if debug:
        ids, imgs_per = 10, 2
    hubble_utils = block.dataset.hubble.hubble_utils
    trainset, testset_by_person = hubble_utils.split_dataset_by_person(
        dataset, test_id_num=ids, test_img_per_id=imgs_per)
    trainset, testset_by_img = hubble_utils.split_dataset_by_img(
        trainset, test_id_num=ids, test_img_per_id=imgs_per, train_scale=1.5)
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

def main():
    ### 2. 初始化我们的模型、数据、各种配置  ####
    # DDP：从外部得到local_rank参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--input_size", default=256, type=int)
    parser.add_argument("--debug", default=False, type=bool)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--total_epoches", default=100, type=int)
    parser.add_argument("--data_name", default='CheX', type=str)
    FLAGS = parser.parse_args()
    local_rank = FLAGS.local_rank #当没有使用torch.dsitributed.launch启动多卡并行时，外界不传入该变量，故取默认值-1
    # print(local_rank)
    
    # DDP：DDP backend初始化
    proc_count = 1
    if local_rank != -1:
        dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端
        proc_count = tc.distributed.get_world_size() #获取全局并行数，即并行训练的显卡的数量


    # 配置实验设置
    # 单卡32， 多卡训练实际的batch_size =  32 * n_gpu
    debug = FLAGS.debug
    batch_size = FLAGS.batch_size // proc_count
    num_workers = FLAGS.num_workers
    input_size = FLAGS.input_size
    total_epoches = FLAGS.total_epoches
    data_name = FLAGS.data_name
    learning_rate = FLAGS.learning_rate

    obfuscation_type = 'neura_crypt'
    # utils.input_option('obfuscation_type', ['no', 'neura_crypt'])
    
    # 配置数据集
    trainset = ChexpertDataset(train=True, benchmark="Edema", data_path=CHEXPERT_TRAIN_PATH)
    valset = ChexpertDataset(train=False, benchmark="Edema", data_path=CHEXPERT_VAL_PATH)
    print(f'trainset_len: {len(trainset)}')
    print(f'valset_len: {len(valset)}')

    # 配置模型
    if obfuscation_type == 'no':
        pass
    elif obfuscation_type == 'neura_crypt':
        nc_parts = NC_Parts(class_num=2, input_size=input_size)
    else:
        raise TypeError(f'obfuscation_type {obfuscation_type} not supported')
    
    # 工作路径
    result_path = exp_utils.setup_result_path(__file__)
    result_path = f'{result_path}/{obfuscation_type}'
    # 如果需要调参，可以为不同参数的试验结果建立不同的工作路径
    result_path = f'{result_path}/{data_name}_{debug}_{batch_size * proc_count}_{learning_rate}_{total_epoches}_{input_size}'
    work_dir = result_path
    print(work_dir)
    
    # 配置测试器
    # Tester = block.test.top1_test.Top1_Tester
    # testers = [Tester(dataset=testset_by_img, name='testset_by_img'),
    #     Tester(dataset=testset_by_person, name='testset_by_person')]
    # for tester in testers:
    #     tester.config_dataloader(batch_size=batch_size, num_workers=num_workers)

    # 预训练NeuraCrypt模型
    def train_nc():
        # config Trainer and fit
        # DDP: Load模型要在构造DDP模型之前，且只需要在master上加载就行了
        # print(local_rank)
        model_lit = NC_Single_Lit(nc_parts)
        if local_rank != -1:
            model_lit = model_lit.to(local_rank)
            model_lit = DDP(model_lit, device_ids=[local_rank], output_device=local_rank)
            model_lit = model_lit.module
        else:
            device = tc.device('cuda' if tc.cuda.is_available else 'cpu')
            model_lit = model_lit.to(device)
        
        model_lit.config_optim(learning_rate)

        trainer = block.train.standard.Trainer(
            dataset=trainset, val_set=valset, total_epochs=total_epoches, work_dir=work_dir)
        trainer.config_dataloader(batch_size=batch_size, num_workers=num_workers, class_bal=True)
        trainer.config_logger(log_interval=30) # 30s/log, 10min/save_log
        # dont test, cause codebase is not available to CheXPert's test
        # trainer.config_tester(testers, interval=30000*60) 
        trainer.fit_ViT_CheX(model_lit, local_rank=local_rank)

        # trainer.config_tester

    train_nc()

if __name__ == '__main__':
    main()
    ################
    # # Bash运行
    # DDP: 使用torch.distributed.launch启动DDP模式
    # 使用CUDA_VISIBLE_DEVICES，来决定使用哪些GPU
    # CUDA_VISIBLE_DEVICES="0,1" python3 -m torch.distributed.launch --nproc_per_node 2 029.Xnn_NeuraCrypt_DDP.py