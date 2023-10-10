'''003 实验中同 ID 回流和不同 ID 回流收益差不多，
主要是不同 ID 回流的点数已经持平原图回流，因此没有什么上涨空间。
本实验准备加大混淆层难度，看看能不能拉开差距。'''


import numpy as np
import torch as tc

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


class XNN_Parts:

    def __init__(self, obfuscate_depth, append_layer_num):
        '''obfuscate_depth: 混淆层深度，
        append_layer_num: tail 额外添加的层数'''
        self.obfuscate_depth = obfuscate_depth
        self.append_layer_num = append_layer_num

    def extractor(self):
        model = block.model.mobile_face_net.MobileFaceNetHead(layer_num=-3)
        param_path = f'{exp_utils.data_path()}/model/msr_pretrain.tar'
        params = utils.torch_load(param_path)
        params = utils.sub_state_dict(params, 'model.')
        model.load_state_dict(params, strict=False)
        return model

    def obfuscate(self):
        layers = []
        for _ in range(self.obfuscate_depth):
            layers.append(tc.nn.Conv2d(128, 128, 1, 1, 0, bias=False))
            layers.append(tc.nn.PReLU(128))
        return tc.nn.Sequential(*layers)

    def tail(self): # 注意用的 cut_layer_num=4，并不是最后 3 层
        return block.model.mobile_face_net.MobileFaceNetDeepTail(
            cut_layer_num=4, append_layer_num=self.append_layer_num)


def prepare_dataset(dataset_name):
    dataset = getattr(block.dataset.hubble.business, dataset_name)()
    hubble_utils = block.dataset.hubble.hubble_utils
    trainset, testset_by_person = hubble_utils.split_dataset_by_person(
        dataset, test_id_num=1000, test_img_per_id=50)
    trainset, testset_by_img = hubble_utils.split_dataset_by_img(
        trainset, test_id_num=1000, test_img_per_id=50, train_scale=1.5)
    return trainset, testset_by_img, testset_by_person


def pretrain_model_test(testers, work_dir):
    '''无脱敏数据回流情况下，pretrain_model在客户数据上的准确性测试'''

    class PretrainModelLit(block.model.light.ModelLight):

        def __init__(self):
            super().__init__()
            model = block.model.mobile_face_net.MobileFaceNet()
            param_path = f'{exp_utils.data_path()}/model/msr_pretrain.tar'
            params = utils.torch_load(param_path)
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
    model_lit = PretrainModelLit().cuda()
    results = {tester.name: tester.test(model_lit) for tester in testers}
    utils.json_save(results, f'{work_dir}/pretrain_model_test.json')


def main():
    # 配置实验设置
    result_path = exp_utils.setup_result_path(__file__)
    dataset_name = utils.input_option('dataset_name',
        ['pingdingshan', 'mianyang', 'tianjin'])
    obfuscate_depth = int(input('obfuscate_depth: '))
    append_layer_num = 9 # int(input('append_layer_num: '))
    work_dir = f'{result_path}/obf{obfuscate_depth}_tail{append_layer_num}/{dataset_name}'
    # 配置数据集
    trainset, testset_by_img, testset_by_person = prepare_dataset(dataset_name)
    # 配置测试器
    Tester = block.test.top1_test.Top1_Tester
    testers = [Tester(dataset=testset_by_img, name='testset_by_img'),
        Tester(dataset=testset_by_person, name='testset_by_person')]
    # 预训练模型测试
    pretrain_model_test(testers, work_dir=work_dir)

    def train_xnn(): # 脱敏回流训练
        xnn_parts = XNN_Parts(obfuscate_depth, append_layer_num)
        model_lit = XNN_Single_Lit(xnn_parts, class_num=trainset.class_num())
        trainer = block.train.standard.Trainer(
            dataset=trainset, total_epochs=9, work_dir=f'{work_dir}/xnn')
        trainer.config_tester(testers, interval=5*60)
        trainer.fit(model_lit)
    train_xnn()

    def train_mfn(): # 原图回流训练
        model_lit = MobileFaceNetLit(trainset.class_num())
        trainer = block.train.standard.Trainer(
            dataset=trainset, total_epochs=9, work_dir=f'{work_dir}/mfn')
        trainer.config_tester(testers, interval=5*60)
        trainer.fit(model_lit)
    train_mfn()


if __name__ == '__main__':
    main()
