'''XNN 的 identity leakage 的属性分析，
统计不同人脸属性的 identity leakage 的量'''


import numpy as np
import torch as tc

import exp_utils
exp_utils.setup_import_path()
import block
import utils

from deepface import DeepFace
import time


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


class XnnParts_mid_conv:

    def __init__(self, pretrain_ckpt, head_layers, mid_layers, tail_extend_layers=0):
        self.pretrain_ckpt = pretrain_ckpt
        self.head_layers = head_layers
        self.mid_layers = mid_layers
        self.tail_extend_layers = tail_extend_layers

    def extractor(self):
        print('load pretrain_ckpt for XnnParts_mid_conv...')
        head = block.model.mobile_face_net.MobileFaceNetHead(
            layer_num=self.head_layers)
        params = utils.torch_load(self.pretrain_ckpt)
        params = utils.sub_state_dict(params, 'model.')
        head.load_state_dict(params, strict=False)
        return head

    def obfuscate(self):

        class Obfuscate(tc.nn.Module):
            def __init__(self, channels, mid_layers):
                super().__init__()
                layers = []
                for _ in range(mid_layers):
                    layers.append(tc.nn.Conv2d(channels, channels, 1, 1, 0, bias=False))
                    layers.append(tc.nn.LeakyReLU(negative_slope=0.25))
                self.conv = tc.nn.Sequential(*layers)
            def forward(self, x):
                return self.conv(x)
            def reset_parameters(self):
                for m in self.conv.modules():
                    if isinstance(m, tc.nn.Conv2d):
                        m.reset_parameters()

        channels = 64 if self.head_layers < 8 else 128
        return Obfuscate(channels, self.mid_layers)

    def tail(self):
        return block.model.mobile_face_net.MobileFaceNetTail(
            layer_num=-self.head_layers,
            extend_num=self.tail_extend_layers)

    def inn(self):
        return block.model.inv_net.InvNet_2()


def load_xnn(class_num):
    pretrain_ckpt = f'{exp_utils.data_path()}/model/celeba_pretrain.tar'
    head_layers = 10
    mid_layers = 1
    xnn_parts = XnnParts_mid_conv(pretrain_ckpt, head_layers,
        mid_layers, tail_extend_layers=head_layers)
    model_lit = XNN_Single_Lit(xnn_parts, class_num)
    params = utils.torch_load(f'/data/jupyter/privdl/privdl/result/023_XnnPaper_msra_SimTail.py/mid_conv/Head{head_layers}_Mid{mid_layers}/xnn/model_lit.tar')
    model_lit.load_state_dict(params, strict=True)
    return model_lit.cuda()


def load_ern(class_num):
    pretrain_ckpt = f'{exp_utils.data_path()}/model/celeba_pretrain.tar'
    head_layers = 10
    mid_layers = 1
    xnn_parts = XnnParts_mid_conv(pretrain_ckpt, head_layers,
        mid_layers, tail_extend_layers=head_layers)
    model_lit = XNN_Single_Lit(xnn_parts, class_num)
    params = utils.torch_load(f'/data/jupyter/privdl/privdl/result/023_XnnPaper_msra_SimTail.py/mid_conv/Head{head_layers}_Mid{mid_layers}/inn/model_lit.tar')
    model_lit.load_state_dict(params, strict=True)
    return model_lit.cuda()


def split_dataset(dataset):
    hubble_utils = block.dataset.hubble.hubble_utils
    trainset, testset_by_person = hubble_utils.split_dataset_by_person(
        dataset, test_id_num=1000, test_img_per_id=50)
    trainset, testset_by_img = hubble_utils.split_dataset_by_img(
        trainset, test_id_num=1000, test_img_per_id=50, train_scale=1.5)
    return trainset, testset_by_img, testset_by_person


def test_ern(model_lit, testset):
    tester = block.test.top1_test.RestoreIdentificationAccuracy(
        dataset=testset, name='test_ern')
    tester.config_dataloader(num_workers=8)
    return tester.test(model_lit, return_top1_dict=True)

def test_xnn(model_lit, testset):
    tester = block.test.top1_test.Top1_Tester(
        dataset=testset, name='test_xnn')
    tester.config_dataloader(num_workers=8)
    return tester.test(model_lit, return_top1_dict=True)


def main():
    result_path = exp_utils.setup_result_path(__file__)
    client_dataset = block.dataset.hubble.xnn_paper.msra()
    attacker_dataset = block.dataset.hubble.xnn_paper.celeba()
    trainset, testset_by_img, testset_by_person = split_dataset(client_dataset)
    xnn = load_xnn(trainset.class_num())
    ern = load_ern(attacker_dataset.class_num())
    while True:
        trainset, testset_by_img, testset_by_person = split_dataset(client_dataset)
        xnn_result = test_xnn(xnn, testset_by_person)
        ern_result = test_ern(ern, testset_by_person)
        base_attr = {label: DeepFace.analyze(testset_by_person[Id][0],
            prog_bar=False, enforce_detection=False)
            for label, Id in utils.tqdm(xnn_result['base_dict'].items(), 'base_attr')}
        utils.pickle_save(dict(xnn_result=xnn_result, ern_result=ern_result, base_attr=base_attr),
            path=f'{result_path}/{time.time()}.pkl')


if __name__ == '__main__':
    main()
