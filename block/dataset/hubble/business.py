'''业务数据'''


from . import hubble_utils


def get_dataset(repo_path): # 只要 base 图，不要 query 图
    facerec_datas = hubble_utils.load_facerec_datas(
        repo_path, with_base=True, with_query=False)
    return hubble_utils.FacerecDataset(facerec_datas)


def pingdingshan():
    '''https://hubble.iap.wh-a.brainpp.cn/detail/3899/3899 \n
    s3://facerec-84dd55e8-7b11-45df-8811-e31d1bbbbb37-data'''
    print('load pingdingshan dataset')
    return get_dataset('/data/jupyter/privdl/privdl/data/hubble_cache/pingdingshan')


def mianyang():
    '''https://hubble.iap.wh-a.brainpp.cn/detail/3917/3917 \n
    s3://facerec-e4e9ca07-01d7-4c10-af65-e1622e7c23ee-data'''
    print('load mianyang dataset')
    return get_dataset('/data/jupyter/privdl/privdl/data/hubble_cache/mianyang')


def tianjin():
    '''https://hubble.iap.wh-a.brainpp.cn/detail/3912/3912 \n
    s3://facerec-80d8b3ad-7ece-445e-900f-91b680b4b26b-data'''
    print('load tianjin dataset')
    return get_dataset('/data/jupyter/privdl/privdl/data/hubble_cache/tianjin')
