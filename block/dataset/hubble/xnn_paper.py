from . import hubble_utils


def celeba(repo_path='s3://xionghuixin/xnn_paper/celeba'):
    '''返回一个 FacerecDataset 类，详细用法请看这个类的文档'''
    print('load celeba dataset')
    facerec_datas = hubble_utils.load_facerec_datas(repo_path)
    return hubble_utils.FacerecDataset(facerec_datas)


def imdb(repo_path='s3://xionghuixin/xnn_paper/IMDB'):
    '''返回一个 FacerecDataset 类，详细用法请看这个类的文档'''
    print('load imdb dataset')
    facerec_datas = hubble_utils.load_facerec_datas(repo_path)
    return hubble_utils.FacerecDataset(facerec_datas)


def facescrub(repo_path='s3://xionghuixin/xnn_paper/facescrub'):
    '''返回一个 FacerecDataset 类，详细用法请看这个类的文档'''
    print('load facescrub dataset')
    facerec_datas = hubble_utils.load_facerec_datas(repo_path)
    return hubble_utils.FacerecDataset(facerec_datas)


def msra(repo_path='s3://xionghuixin/xnn_paper/msra'):
    '''返回一个 FacerecDataset 类，详细用法请看这个类的文档'''
    print('load msra dataset')
    facerec_datas = hubble_utils.load_facerec_datas(repo_path)
    return hubble_utils.FacerecDataset(facerec_datas)


def webface(repo_path='s3://liukaixin-datasets/webface_nori'):
    '''返回一个 FacerecDataset 类，详细用法请看这个类的文档'''
    print('load webface dataset')
    facerec_datas = hubble_utils.load_facerec_datas(repo_path)
    return hubble_utils.FacerecDataset(facerec_datas)


def vggface2(repo_path='s3://liukaixin-datasets/vggface2_nori'):
    '''返回一个 FacerecDataset 类，详细用法请看这个类的文档'''
    print('load vggface2 dataset')
    facerec_datas = hubble_utils.load_facerec_datas(repo_path)
    return hubble_utils.FacerecDataset(facerec_datas)