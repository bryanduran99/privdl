from . import hubble_utils


def get_dataset(repo_path='s3://facerec-29c35d89-8243-4c00-a9f0-ba3a72bc7c6b-data'):
    '''https://hubble.iap.wh-a.brainpp.cn/detail/862/862 \n
    返回一个 FacerecDataset 类，详细用法请看这个类的文档'''
    print('load msr dataset')
    facerec_datas = hubble_utils.load_facerec_datas(repo_path)
    return hubble_utils.FacerecDataset(facerec_datas)
