import utils
from . import hubble_utils


def get_dataset(img_range=(50, 100), person_num=8_0000):
    '''https://hubble.iap.wh-a.brainpp.cn/detail/1455/1455 \n
    s3://facerec-3255dbb1-de14-4590-816b-03da8a350e1e-data \n
    返回一个 FacerecDataset 类，详细用法请看这个类的文档 \n
    选择其中图片数量为 img_range 的 person_num 人'''
    print('load instagram dataset')
    # 读取 repo
    repo_path='/data/jupyter/privdl/privdl/data/hubble_cache/instagram'
    nori_ids = utils.pickle_load(f'{repo_path}/align5p.nori_id')
    class_info = utils.pickle_load(f'{repo_path}/info')
    # 选择其中图片数量为 img_range 的 person_num 人
    assert hubble_utils.get_info_type(class_info) == '(s, e)'
    class_info = [(s, e) for s, e in class_info
        if img_range[0] <= (e - s) < img_range[1]]
    assert len(class_info) >= person_num, f'图数范围{img_range}内的人数不足{person_num}'
    class_info = [class_info[i] for i in
        utils.choice(list(range(len(class_info))), person_num)]
    # 生成 FacerecDataset
    index_to_class_id = hubble_utils.get_index_to_class_id(class_info)
    facerec_datas = [
        hubble_utils.FacerecData(nori_id=nori_ids[index], label=class_id)
        for index, class_id in index_to_class_id.items()]
    return hubble_utils.FacerecDataset(facerec_datas)
