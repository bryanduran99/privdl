'''https://fr-discourse.iap.wh-a.brainpp.cn/t/topic/11587/2
A：原图训练 Baseline 实验 (脱敏训练下限)
用 A 训练 mfn, 其在数据集 B 的测试结果见 007 的 pretrain_model_test'''


import torch as tc
import numpy as np

import exp_utils
exp_utils.setup_import_path()
import block
import utils


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


def prepare_dataset():
    hubble = block.dataset.hubble
    dataset = hubble.xnn_paper.celeba()
    return hubble.hubble_utils.split_dataset_by_img(
        dataset, test_id_num=1000, test_img_per_id=20, train_scale=1.5)


def main():
    trainset, testset = prepare_dataset()
    model_lit = MobileFaceNetLit(trainset.class_num())
    trainer = block.train.standard.Trainer(
        dataset=trainset, total_epochs=15,
        work_dir=exp_utils.setup_result_path(__file__))
    trainer.config_tester([block.test.top1_test.Top1_Tester(testset, name='testset_by_img')])
    trainer.fit(model_lit)


if __name__ == '__main__':
    main()
