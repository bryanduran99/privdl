# -*- coding: UTF-8 -*-
'''
Implementation slightly adapted from "032.xnn_vit_on_mocov3_pretrain.py".
1. Archtecture purely adopts ViT-s/16， in which ExtNet and RecNet each account for half of the ViT-s/16.
2. While ExtNet load pretrained weights from ViT-s/16, RecNet is trained from scratch.
'''
import os
import math
import random
import cv2
import numpy as np
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


class ExtTrainer(block.train.standard.Trainer):
    """Trainer的子类，重写其中的fit函数"""
    def __init__(self, dataset, total_epochs, val_set=None, work_dir=None, sample_transform=None, optimizer=None, monitor=None):
        super().__init__(dataset, total_epochs, val_set, work_dir)
        self.best_acc = 0
        self.best_epoch = 0
        self.sample_transform = sample_transform
        self.optimizer = optimizer 
        self.monitor = monitor
        assert self.sample_transform is not None # 为保证效果，必须要有sample_transform

    # def train_step( self, model, batch, optimizer, criterion, device, epoch, batch_idx, total_batch_num, log_interval=10):
    #     pass

    def fit_ViT(self, model_lit, local_rank=-1):
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
                        # best_model_state_dict = {k:v.to('cpu') for k, v in model_lit.state_dict().items()}
                        # utils.torch_save(best_model_state_dict, os.path.join(self.work_dir, 'best_model.tar'))
                if local_rank != -1:
                    # 在多卡模式下在主进程测试并save，其他进程通过torch.distributed.barrier()来等待主进程完成validate操作
                    # barrier()函数的使用要非常谨慎，如果只有单个进程包含了这条语句，那么程序就会陷入无限等待
                    dist.barrier()
        return model_lit


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

    def forward(self, img):
        if self.obf:
            self.x = self.x.to(img.device)
            tmp_img = img.clone()
            for w in range(self.patch_num):
                img[:, w, :] = tmp_img[:, self.V1[w], :]
            
            img = img.matmul(self.x)             
        
        return img


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


class XNN_Single_Lit(block.model.light.ModelLight):
    '''单客户场景下的 XNN'''

    def __init__(self, class_num, isobf=True, pretrained_path='', split_layer=6):
        super().__init__()
        self.isobf = isobf
        self.class_num = class_num
        self.split_layer = split_layer
        assert self.split_layer > 0 and self.split_layer < 12

        self.ext = block.model.moco_v3_vits.vit_small_head(layer=self.split_layer)
        # summary(self.ext, input_size=(3, 224, 224), device="cpu")
        assert pretrained_path != '', 'pretrained_path is empty'
        self.ext = load_checkpoint(model=self.ext, pretrained_path=pretrained_path)
        for p in self.ext.parameters():
            p.requires_grad = False

        # self.obf = Obfuscate(64, 512, self.isobf)
        self.obf = Obfuscate(197, 384, self.isobf)
        for p in self.obf.parameters():
            p.requires_grad = False
        
        self.tail = block.model.moco_v3_vits.vit_small_tail(layer=6)

        self.softmax = block.loss.amsoftmax.AMSoftmax(
            in_features=self.tail.embed_dim, class_num=self.class_num)


    def forward(self, images, labels, is_inference=False):
        mid_feats = self.ext(images)
        obf_feats = self.obf(mid_feats)
        rec_feats = self.tail(obf_feats)

        if is_inference:   
            return rec_feats    
        out = self.softmax(rec_feats, labels)
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
    parser.add_argument("--person", default=1000, type=int)
    parser.add_argument("--isobf", default=True, type=bool)
    parser.add_argument("--split_layer", default=6, type=int)
    parser.add_argument("--pretrained", default="../data/model/vit-s-300ep.pth.tar", type=str,
                        help='path to mocov3 pretrained checkpoint')
    
    FLAGS = parser.parse_args()
    local_rank = FLAGS.local_rank

    proc_count = 1
    if local_rank != -1:
        dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端
        proc_count = tc.distributed.get_world_size() #获取全局并行数，即并行训练的显卡的数量
    
    batch_size = 128 // proc_count
    num_workers = 8
    train_epoch = 100
    
    train_dataset = FLAGS.train_dataset
    print("train_dataset:", train_dataset)

    result_path = exp_utils.setup_result_path(os.path.basename(__file__))
    result_path = f'{result_path}/{proc_count}_gpus/{train_dataset}/{FLAGS.isobf}_obf/{len(FLAGS.pretrained)!=0}_pretrained/{FLAGS.split_layer}_layer'
    print(result_path)
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
    attacker_dataset = block.dataset.hubble.xnn_paper.facescrub() # 10w
    # trainset = trainset.subset_by_class_id(trainset.class_ids()[:1000])

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


    # def test_pre_ext():

    #     model_lit = XNN_Ext_Lit()
        
    #     device = tc.device('cuda' if tc.cuda.is_available else 'cpu')
    #     model_lit = model_lit.to(device)


    #     for _ in range(5):
    #         if local_rank == -1 or dist.get_rank() == 0: 
    #             self.call_testers(clock, model_lit)


    def train_xnn_vit(): 

        # 定义模型
        model_lit = XNN_Single_Lit(class_num=trainset.class_num(), \
            isobf=FLAGS.isobf, pretrained_path=FLAGS.pretrained, split_layer=FLAGS.split_layer)
        summary(model_lit.ext, input_size=(3, 224, 224), device="cpu")
        summary(model_lit.tail, input_size=(197, 384), device="cpu")

        # move to cuda or local_rank
        if local_rank == -1:
            device = tc.device('cuda' if tc.cuda.is_available else 'cpu')
            model_lit = model_lit.to(device)
        else:
            model_lit = model_lit.to(local_rank)
            model_lit = DDP(model_lit, device_ids=[local_rank], output_device=local_rank)

        # 定义优化器
        # param_groups = [ 
        #     dict(params=model_lit.pretrain_ext.parameters(), weight_decay=4e-5),
        #     dict(params=model_lit.softmax.parameters(), weight_decay=4e-4)]
        optimizer = tc.optim.SGD(filter(lambda p: p.requires_grad, model_lit.parameters()), lr=0.05)
        monitor = utils.Monitor()
        monitor.add('lr',  # 监控学习率
                         get=lambda: optimizer.param_groups[0]['lr'],
                         to_str=lambda x: f'{x:.1e}')

        # define trainer and fit
        trainer = ExtTrainer(
            dataset=trainset, total_epochs=train_epoch, work_dir=f'{work_dir}', sample_transform=sample_transform,optimizer=optimizer, monitor=monitor)
        trainer.config_dataloader(batch_size=batch_size, num_workers=num_workers)
        log_interval = 60
        trainer.config_logger(log_interval=log_interval)
        trainer.config_tester(testers)
        trainer.fit_ViT(model_lit, local_rank=local_rank)
        return
    
    train_xnn_vit()

    # def attack_xnn(): # 期望攻击
    #     model_lit = ERN_lit(class_num=attacker_dataset.class_num(), \
    #         isobf=True, pretrained_path=FLAGS.pretrained, split_layer=FLAGS.split_layer)
    #     trainer = block.train.standard.Trainer(
    #         dataset=attacker_dataset, total_epochs=epochs, work_dir=f'{work_dir}/inn')
    #     RIA = block.test.top1_test.RestoreIdentificationAccuracy(
    #         dataset=testset_by_person, name='RestoreIdentificationAccuracy')
    #     RIA.config_dataloader(batch_size=batch_size, num_workers=num_workers)
    #     trainer.config_dataloader(batch_size=batch_size, num_workers=num_workers)
    #     # trainer.config_tester([RIA], interval=5*60)
    #     trainer.config_tester([RIA])
    #     trainer.fit(model_lit)
    # attack_xnn()




if __name__ == '__main__':
    main()
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 -m torch.distributed.launch --nproc_per_node 8 032-2.xnn_vit_backbone_vitb16_pretrain_ext.py --isobf False