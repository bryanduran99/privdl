import numpy as np
import torch as tc
import time
import torch.distributed as dist

import utils


def cdist(feat0, feat1):
    '''L2norm'''
    return np.linalg.norm(feat0 - feat1)


class Top1_Tester:
    '''对模型进行 top1 准确率测试'''

    def __init__(self, dataset, name=''):
        '''dataset 的 sample 由 (input, label) 组成， name 为此 tester 的名称，
        可用 config 系列函数对测试配置进行修改'''
        self.dataset = dataset
        self.name = name
        self.config_dataloader()

    def config_dataloader(self, batch_size=256, num_workers=16):
        '''设置测试数据集 loader 的 batch_size 和 num_workers'''
        def get_dataloader(transform=None):
            '''按设定的参数创建 DataLoader，transform 将施加在每个 sample 上'''
            dataset = self.dataset
            if transform is not None:
                dataset = utils.TransformedDataset(dataset, transform)
            return utils.DataLoader(dataset, batch_size=batch_size,
                num_workers=num_workers, shuffle=False, drop_last=False)
        self.get_dataloader = get_dataloader

    def extract_feats(self, model_lit):
        '''用 model_lit 抽取测试集的特征并返回'''
        self.sample_transform = model_lit.test_sample_transform
        dataloader = self.get_dataloader(self.sample_transform)
        feats, labels = [], []
        model_lit.eval()
        with tc.no_grad():
            for batch_data in utils.tqdm(dataloader, 'extract_feats'):
                batch_data = utils.batch_to_cuda(batch_data)
                batch_inputs, batch_labels = batch_data
                batch_feats = model_lit.inference(batch_inputs)
                feats.append(utils.batch_to_numpy(batch_feats))
                labels.append(utils.batch_to_numpy(batch_labels))
        return np.concatenate(feats), np.concatenate(labels)

    def test(self, model_lit, return_top1_dict=False, vis=False):
        '''top1 准确率测试，print 测试结果，return 测试的 log dict'''
        # start_time = time.time()
        if dist.is_available() and dist.is_initialized():
            model_lit = model_lit.module
        feats, labels = self.extract_feats(model_lit) # 抽特征
        if vis:
            return feats, labels
        # t1_time = time.time()
        person_dict = utils.defaultdict(list) # 按 label 对 sample 分类
        for person_id, label in enumerate(labels):
            person_dict[label].append(person_id)
        # 每类取一个 sample 作为 base
        base_dict = {label: ids.pop() for label, ids in person_dict.items()}
        top1_dict = utils.defaultdict(list) # {query_label: [(query_id, top1_id, top1_label), ...]}

        acc = total = 0 # top1 准确数，总测试数
        # t2_time = time.time()
        # print("extract_time:{e_t:.2f}s \t util_time:{u_t:.2f}".format(e_t=t1_time-start_time,u_t=t2_time-t1_time))
        for query_label, query_ids in utils.tqdm(person_dict.items(), 'top1 test'):
            # t3_time = time.time()
            for query_id in query_ids: # 对每类的每个 query 进行测试
                # t5_time = time.time()
                # 此条语句为耗时大户（xnn_train:0.01s, test_ext:0.16s）
                top1_dist, top1_label = min( # 取该 query 的 top1 的 label 
                    (cdist(feats[base_id], feats[query_id]), base_label)
                    for base_label, base_id in base_dict.items())
                # t6_time = time.time()
                top1_dict[query_label].append((query_id, base_dict[top1_label], top1_label))
                # 看 top1 的 label 和该 query 的 label 是否一致
                acc += int(top1_label == query_label)
                total += 1 # 总测试数 +1
                # print("for_time:{0:.2f}, min_time:{1:.2f}, append_time:{2:.2f}".format(t5_time-t3_time, t6_time-t5_time, time.time()-t6_time))
                # t3_time = time.time()
            # t4_time = time.time()
            # print("load_dict_time:{l_t:.2f}s \t calculate_time:{c_t:.2f}".format(l_t=t3_time-t2_time,c_t=t4_time-t3_time))
            # t2_time = time.time()
        print(f'{self.name} acc={acc}/{total}={acc/total*100:.2f}%')

        ret = dict(acc=acc, total=total, rate=acc/total)
        if return_top1_dict:
            ret['top1_dict'] = top1_dict
            ret['base_dict'] = base_dict
        return ret


class RestoreIdentificationAccuracy(Top1_Tester):
    '''对期望攻击模型进行重建识别准确率测试'''

    def test(self, model_lit, return_top1_dict=False):
        '''重建识别准确率测试，print 测试结果，return 测试的 log dict'''
        # 待检索的泄露 query 和攻击者的 base 要用不同的混淆 key
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