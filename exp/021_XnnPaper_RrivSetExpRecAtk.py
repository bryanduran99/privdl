'''https://fr-discourse.iap.wh-a.brainpp.cn/t/topic/11587/23
看一下期望识别攻击在之前用的安防证件照数据集上的效果'''


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


class ERN_lit(XNN_Single_Lit):
    '''期望识别模型，强行在期望混淆后的数据上训练识别模型'''
    def train_step(self, batch_data):
        self.obfuscate.reset_parameters()
        super().train_step(batch_data)


class INN_Lit(block.model.light.ModelLight):
    '''XNN 逆重建模型'''

    def __init__(self, xnn_parts, pretrain_ckpt):
        super().__init__()
        self.extractor = xnn_parts.extractor()
        self.obfuscate = xnn_parts.obfuscate()
        self.inn = xnn_parts.inn()
        self.config_optim()
        self.config_recognition_model(pretrain_ckpt)

    def config_optim(self):
        self.optimizer = tc.optim.AdamW(self.inn.parameters())
        self.monitor.add('lr', # 监控学习率
            get=lambda: self.optimizer.param_groups[0]['lr'],
            to_str=lambda x: f'{x:.1e}')

    def config_recognition_model(self, pretrain_ckpt):
        print('load recognition model...')
        model = block.model.mobile_face_net.MobileFaceNet()
        params = utils.torch_load(pretrain_ckpt)
        params = utils.sub_state_dict(params, 'model.')
        model.load_state_dict(params, strict=True)
        self.recognition_model = model

    def sample_transform(self, sample):
        img, label = sample
        img = self.extractor.img_transform(img)
        return img, label

    def train(self):
        self.extractor.eval()
        self.obfuscate.eval()
        self.inn.train()

    def eval(self):
        self.extractor.eval()
        self.obfuscate.eval()
        self.inn.eval()

    def forward(self, mid_feats):
        return self.inn(mid_feats)

    def train_step(self, batch_data):
        imgs, labels = batch_data
        with tc.no_grad():
            mid_feats = self.extractor(imgs)
            self.obfuscate.reset_parameters() # 期望攻击
            mid_feats = self.obfuscate(mid_feats)
        restores = self(mid_feats)
        loss = block.loss.ssim.neg_ssim(restores, imgs)
        utils.step(self.optimizer, loss)
        self.monitor.add('loss', lambda: float(loss),
            to_str=lambda x: f'{x:.2e}') # 监控 loss

    def inference(self, batch_input):
        mid_feats = self.extractor(batch_input)
        mid_feats = self.obfuscate(mid_feats)
        restores = self(mid_feats)
        Min, Max = restores.min(), restores.max()
        restores = (restores - Min) / (Max - Min)
        restores = restores*2 - 1
        return self.recognition_model(restores)


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


class XnnParts_mat_full:
    '''CHW 维度的矩阵混淆'''

    def __init__(self):
        pass

    def extractor(self):
        return block.model.mobile_face_net.MobileFaceNetHead(layer_num=0)

    def obfuscate(self):
        class Obfuscate(tc.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim
                self.mat = tc.nn.Linear(dim, dim, bias=False)
            def forward(self, imgs):
                img_vectors = imgs.view(-1, self.dim)
                img_vectors = self.mat(img_vectors)
                return img_vectors.view(imgs.shape)
            def reset_parameters(self):
                self.mat.reset_parameters()
        return Obfuscate(3 * 112 * 112)

    def tail(self):
        return block.model.mobile_face_net.MobileFaceNet()

    def inn(self):
        return block.model.inv_net.InvNet_2()


class XnnParts_mat:
    '''HW 维度的矩阵混淆'''

    def __init__(self):
        pass

    def extractor(self):
        return block.model.mobile_face_net.MobileFaceNetHead(layer_num=0)

    def obfuscate(self):
        class Obfuscate(tc.nn.Module):
            def __init__(self, height, width):
                super().__init__()
                self.dim = dim = height * width
                self.mat = tc.nn.Linear(dim, dim, bias=False)
            def forward(self, imgs):
                img_vectors = imgs.view(-1, self.dim)
                img_vectors = self.mat(img_vectors)
                return img_vectors.view(imgs.shape)
            def reset_parameters(self):
                self.mat.reset_parameters()
        return Obfuscate(112, 112)

    def tail(self):
        return block.model.mobile_face_net.MobileFaceNet()

    def inn(self):
        return block.model.inv_net.InvNet_2()


class XnnParts_img_conv:

    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def extractor(self):
        return block.model.mobile_face_net.MobileFaceNetHead(layer_num=0)

    def obfuscate(self):
        padding = (self.kernel_size - 1) // 2
        return tc.nn.Conv2d(3, 3, self.kernel_size, 1, padding, bias=False)

    def tail(self):
        return block.model.mobile_face_net.MobileFaceNet()

    def inn(self):
        return block.model.inv_net.InvNet_2()


class XnnParts_mid_conv:

    def __init__(self, pretrain_ckpt, head_layers, mid_layers):
        self.pretrain_ckpt = pretrain_ckpt
        self.head_layers = head_layers
        self.mid_layers = mid_layers

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
            layer_num=-self.head_layers)

    def inn(self):
        return block.model.inv_net.InvNet_2()


def split_dataset(dataset):
    hubble_utils = block.dataset.hubble.hubble_utils
    trainset, testset_by_person = hubble_utils.split_dataset_by_person(
        dataset, test_id_num=1000, test_img_per_id=50)
    trainset, testset_by_img = hubble_utils.split_dataset_by_img(
        trainset, test_id_num=1000, test_img_per_id=50, train_scale=1.5)
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
    # 配置实验设置
    result_path = exp_utils.setup_result_path(__file__)
    pretrain_ckpt = f'{exp_utils.data_path()}/model/msr_pretrain.tar'
    batch_size = 128
    num_workers = 8

    obfuscation_type = utils.input_option(
        'obfuscation_type', ['no', 'mat', 'mat_full', 'img_conv', 'mid_conv'])
    result_path = f'{result_path}/{obfuscation_type}'

    if obfuscation_type == 'no':
        pass
    elif obfuscation_type == 'mat':
        xnn_parts = XnnParts_mat()
    elif obfuscation_type == 'mat_full':
        xnn_parts = XnnParts_mat_full()
        batch_size = 56
    elif obfuscation_type == 'img_conv':
        obf_kernel_size = utils.input_option('obf_kernel_size',
            [3, 5, 7, 9, 11, 33, 55, 77, 99, 111])
        result_path = f'{result_path}/kernel_size_{obf_kernel_size}'
        xnn_parts = XnnParts_img_conv(obf_kernel_size)
    elif obfuscation_type == 'mid_conv':
        head_layers = utils.input_option('head_layers', list(range(1, 18)))
        mid_layers = 3
        result_path = f'{result_path}/Head{head_layers}_Mid{mid_layers}'
        xnn_parts = XnnParts_mid_conv(pretrain_ckpt, head_layers, mid_layers)
    else:
        raise TypeError(f'obfuscation_type {obfuscation_type} not supported')

    work_dir = result_path
    # 配置数据集 和 相应的预训练模型
    client_dataset = block.dataset.hubble.business.pingdingshan()
    trainset, testset_by_img, testset_by_person = split_dataset(client_dataset)
    attacker_dataset = block.dataset.hubble.msr.get_dataset()
    # 配置测试器
    Tester = block.test.top1_test.Top1_Tester
    testers = [Tester(dataset=testset_by_img, name='testset_by_img'),
        Tester(dataset=testset_by_person, name='testset_by_person')]
    for tester in testers:
        tester.config_dataloader(batch_size=batch_size, num_workers=num_workers)

    def baseline():
        pretrain_model_test(pretrain_ckpt, testers, work_dir) # 预训练模型测试
        model_lit = MobileFaceNetLit(trainset.class_num())
        trainer = block.train.standard.Trainer(
            dataset=trainset, total_epochs=15, work_dir=f'{work_dir}/xnn')
        trainer.config_dataloader(batch_size=batch_size, num_workers=num_workers)
        trainer.config_logger(log_interval=30)
        trainer.config_tester(testers, interval=5*60)
        trainer.fit(model_lit)
    if obfuscation_type == 'no':
        baseline()
        return

    def train_xnn(): # 脱敏回流训练
        model_lit = XNN_Single_Lit(xnn_parts, class_num=trainset.class_num())
        trainer = block.train.standard.Trainer(
            dataset=trainset, total_epochs=15, work_dir=f'{work_dir}/xnn')
        trainer.config_dataloader(batch_size=batch_size, num_workers=num_workers)
        trainer.config_logger(log_interval=30)
        trainer.config_tester(testers, interval=5*60)
        trainer.fit(model_lit)
    train_xnn()

    def attack_xnn(): # 期望攻击
        model_lit = ERN_lit(xnn_parts, class_num=attacker_dataset.class_num())
        trainer = block.train.standard.Trainer(
            dataset=attacker_dataset, total_epochs=45, work_dir=f'{work_dir}/inn')
        RIA = block.test.top1_test.RestoreIdentificationAccuracy(
            dataset=testset_by_person, name='RestoreIdentificationAccuracy')
        RIA.config_dataloader(batch_size=batch_size, num_workers=num_workers)
        trainer.config_dataloader(batch_size=batch_size, num_workers=num_workers)
        trainer.config_tester([RIA], interval=5*60)
        trainer.fit(model_lit)
    attack_xnn()


if __name__ == '__main__':
    main()
