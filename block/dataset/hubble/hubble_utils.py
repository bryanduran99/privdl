import pickle
import time
import torch as tc
import numpy as np
from collections import defaultdict
import utils
import nori2
nori_get = nori2.Fetcher().get


def get_info_type(class_info):
    '''class_info 是 hubble 上数据集的标准 repo 下的 info 文件内容，
    本函数判断 info 的数据类型并返回类型描述之一：\n
    (s, e)\n
    ((s, e), (s, e))'''

    def is_int_pair(x, y):
        return isinstance(x, int) and isinstance(y, int)

    a, b = class_info[0] # 取第 0 一个 class 的 info 进行类型判断
    if is_int_pair(a, b):
        return '(s, e)'
    elif is_int_pair(a[0], a[1]) and is_int_pair(b[0], b[1]):
        return '((s, e), (s, e))'
    else:
        raise TypeError('get_info_type： unknown info type')


def get_index_to_class_id(class_info, with_base=True, with_query=True):
    '''class_info 是 hubble 上数据集的标准 repo 下的 info 文件内容，
    本函数根据 info 的内容创建一个字典，将 info 所描述的 index 映射
    到一个 class_id，这个 class_id 是该 index 在 info 中所属 class
    的位置编号 \n
    对于 ((s, e), (s, e)) 类型的 class_info，设置 with_base 或者 with_query
    为 Fasle 可以去掉 base 或者 query'''

    info_type = get_info_type(class_info)

    if info_type == '(s, e)':
        index_to_class_id = {}
        for class_id, (s, e) in enumerate(class_info):
            for index in range(s, e):
                index_to_class_id[index] = class_id
        return index_to_class_id

    elif info_type == '((s, e), (s, e))':
        index_to_class_id = {}
        for class_id, ((s0, e0), (s1, e1)) in enumerate(class_info):
            if with_base:
                for index in range(s0, e0):
                    index_to_class_id[index] = class_id
            if with_query:
                for index in range(s1, e1):
                    index_to_class_id[index] = class_id
        return index_to_class_id

    else:
        raise TypeError(f'info_type {info_type} not supported')


def load_facerec_datas(repo_path, with_base=True, with_query=True):
    '''repo_path 目录下需要包含文件 align5p.nori_id 和 info，
    返回一个 FacerecData 的 list'''
    nori_ids = utils.pickle_load(f'{repo_path}/align5p.nori_id')
    class_info = utils.pickle_load(f'{repo_path}/info')
    index_to_class_id = get_index_to_class_id(
        class_info, with_base, with_query)
    return [FacerecData(nori_id=nori_ids[index], label=class_id)
        for index, class_id in index_to_class_id.items()]


class FacerecData:
    '''提供 nori_id 和 label 的绑定，以及从 nori_id 中读取内容的函数'''

    def __init__(self, nori_id, label):
        self.nori_id = nori_id
        self.label = label


def to_facerec_dict(facerec_list):
    '''return {label: [FacerecData, ...], ...} \n
    facerec_list 是一个 FacerecData 的 list。
    返回一个 dict，key 是 FacerecData 的 label，
    value 是该具有该 label 的 FacerecData。'''
    facerec_dict = defaultdict(list)
    for facerec_data in facerec_list:
        facerec_dict[facerec_data.label].append(facerec_data)
    return dict(facerec_dict)


def to_facerec_list(facerec_dict, reset_label=False):
    '''return [FacerecData, ...] \n
    facerec_dict = {label: [FacerecData, ...], ...}，
    根据 facerec_dict 的描述返回一个 FacerecData 的 list，
    label 以 dict key 的描述为准。如果 reset_label=True，
    那么返回的 FacerecData 的 label 取值范围将相应地修正为
    [0, len(facerec_dict))，否则 label 保持不变。'''
    facerec_list = []
    if reset_label:
        iterator = enumerate(facerec_dict.values())
    else:
        iterator = facerec_dict.items()
    for label, facerec_datas in iterator:
        facerec_list += [FacerecData(data.nori_id, label)
            for data in facerec_datas]
    return facerec_list


class FacerecDataset(tc.utils.data.Dataset):
    '''hubble 上人脸识别数据集一个简化的抽象类，pytorch Dataset 的子类'''

    def __init__(self, facerec_datas):
        '''facerec_datas 是 FacerecData 的 list'''
        self.nori_ids = np.array([data.nori_id for data in facerec_datas])
        self.labels = np.array([data.label for data in facerec_datas])

    def facerec_datas(self):
        '''将 self.nori_ids 和 self.labels 组合成为 FacerecData 的 list'''
        return [FacerecData(nori_id, label) for nori_id, label
            in zip(self.nori_ids, self.labels)]

    def __getitem__(self, index):
        '''return img(256*256*BGR*uint8), label(int)'''
        nori_id = self.nori_ids[index]
        img = pickle.loads(nori_get(nori_id))['img']
        img = utils.imdecode(img)
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.nori_ids)

    def facerec_dict(self):
        '''将自己的 facerec_datas() 转化为 facerec_dict 并返回'''
        return to_facerec_dict(self.facerec_datas())

    def class_ids(self):
        '''返回一个 list，里面是自己的 facerec_datas() 的 labels'''
        return list(self.facerec_dict())

    def class_num(self):
        '''返回数据集中类别的数量，也即人的数量、label的数量'''
        return len(self.facerec_dict())

    def subset_by_class_id(self, class_ids, reset_label=False):
        '''按人取子集，返回一个 FacerecDataset，只含 class_ids 中的人，
        如果 reset_label=True，那么返回的数据集的 label 取值范围将相应地修正为
        [0, len(class_ids))，否则 label 保持不变。'''
        class_id_set = set(class_ids)
        facerec_datas = [data for data in self.facerec_datas()
            if data.label in class_id_set]
        if reset_label:
            facerec_dict = to_facerec_dict(facerec_datas)
            facerec_datas = to_facerec_list(facerec_dict, reset_label=True)
        return __class__(facerec_datas)

    def newset_by_indices(self, indices):
        facerec_datas = self.facerec_datas()
        return __class__([facerec_datas[index] for index in indices])

    def subset_by_size(self, size):
        indices = np.arange(len(self))
        np.random.shuffle(indices)
        selected_indices = indices[:int(size)]
        return self.newset_by_indices(selected_indices)

    '''return a new FacerecDataset, which permutation self'''
    def permutation_idx(self):
        np.random.seed(int(time.time()))
        indices = np.random.permutation(len(self))
        return indices
    
    '''return a new FacerecDataset, which is the mixup of traidnset and the  obfuscated_set
    e.f., 
    when k = 2, 
    trainset = [1, 3, 5, 7, 9]
    obfuscated_set = [2, 4, 6, 8, 10]
    '''
    def mixup_multi_dataset(self, k = 4):
        ret_indices = np.empty(k * len(self), dtype = np.int32)
        for i in range(0, k):
            if i == 0:
                tmp_allset_indices = np.arange(len(self))
            else:
                tmp_allset_indices = self.permutation_idx()

            for j in range(0, len(tmp_allset_indices)):
                ret_indices[j * k + i] = tmp_allset_indices[j]
        
        return self.newset_by_indices(ret_indices) # return mixuped dataset

    def print_info(self, title=''):
        '''将数据集的详情 print 出来'''
        print(title or __class__.__name__)

        img_num = len(self)
        print(f'┃ 图片数量={img_num}')

        facerec_dict = self.facerec_dict()
        person_num = len(facerec_dict)
        print(f'┃ 人数={person_num}')
        print(f'┃ 人均图数={img_num/person_num:.2f}')

        img_nums = list(map(len, facerec_dict.values()))
        print(f'┃ 单人最少图数={min(img_nums)}')
        print(f'┃ 单人最多图数={max(img_nums)}')
        print(f'┗ 单人图数标准差={np.std(img_nums):.2f}')


def split_facerec_dataset(dataset, class_ids):
    '''dataset 是一个 FacerecDataset。
    返回 2 个 FacerecDataset，第一个只含 class_ids 中的人，
    第二个不含 class_ids 中的人。子集的 label 将按照子集的人数做修正。'''
    rest_ids = set(dataset.class_ids()) - set(class_ids)
    first = dataset.subset_by_class_id(class_ids, reset_label=True)
    second = dataset.subset_by_class_id(rest_ids, reset_label=True)
    return first, second


def split_dataset_by_person(dataset, test_id_num, test_img_per_id):
    '''return trainset, testset \n
    将 FacerecDataset 拆分为训练集和测试集，
    从 dataset 里选 test_id_num 个人作为测试集，
    其中每个人要求有 test_img_per_id 张图。\n
    测试集的人与训练集的人不相交'''
    dataset.print_info('split_dataset_by_person')

    # 先选取图数不低于 test_img_per_id 的 id
    test_ids = [class_id for class_id, facerec_datas
        in dataset.facerec_dict().items()
        if len(facerec_datas) >= test_img_per_id]
    # 确保满足条件的 id 数不低于 test_id_num
    assert len(test_ids) >= test_id_num

    # 从中随机选取 test_id_num 个 id 作为测试集的 id，并对 dataset 进行切分
    test_ids = utils.choice(test_ids, test_id_num)
    testset, trainset = split_facerec_dataset(dataset, test_ids)

    # 将 testset 中每个人的图数限制为 test_img_per_id (随机选取)
    facerec_dict = {class_id: utils.choice(facerec_datas, test_img_per_id)
        for class_id, facerec_datas in testset.facerec_dict().items()}
    testset = FacerecDataset(to_facerec_list(facerec_dict))

    trainset.print_info('训练集')
    testset.print_info('测试集')
    return trainset, testset


def split_dataset_by_img(dataset, test_id_num, test_img_per_id, train_scale=5):
    '''return trainset, testset \n
    将 FacerecDataset 拆分为训练集和测试集，
    从 dataset 里选 test_id_num 个人作为测试集，
    其中每个人要求有 test_img_per_id 张图。\n
    测试集的人是训练集的人的子集，但测试集的图片与训练集的图片不相交。\n
    train_scale 是为了保证待测试的 id 有足够多的图在训练集中，
    要求 testset 中每个人在原数据集中的图数不少于 test_img_per_id*train_scale'''
    dataset.print_info('split_dataset_by_img')

    # 先选取图数不低于 test_img_per_id*train_scale 的 id
    test_ids = [class_id for class_id, facerec_datas
        in dataset.facerec_dict().items()
        if len(facerec_datas) >= test_img_per_id*train_scale]
    # 确保满足条件的 id 数不低于 test_id_num
    assert len(test_ids) >= test_id_num

    # 从中随机选取 test_id_num 个 id 作为测试集的 id，并对 dataset 进行切分
    train_dict = dataset.facerec_dict()
    test_dict = {}
    for test_id in utils.choice(test_ids, test_id_num):
        facerec_datas = train_dict[test_id]
        np.random.shuffle(facerec_datas)
        test_dict[test_id] = facerec_datas[:test_img_per_id]
        train_dict[test_id] = facerec_datas[test_img_per_id:]

    trainset = FacerecDataset(to_facerec_list(train_dict))
    testset = FacerecDataset(to_facerec_list(test_dict))

    trainset.print_info('训练集')
    testset.print_info('测试集')
    return trainset, testset
