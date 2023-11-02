import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

def loss_fig():
    path = '/data/research/version2/privdl/result/100-2-adv_Distillation.py/8_gpus/celeba/1.0_trainset/stage2_obf/True_pretrained/5_layer/6_tail_layer/logs_231031152043.json'

    with open(path, 'r') as files:
        data = json.load(files)
    # x = range(len(data))[::5]
    # print(data)
    x = range(len(data))
    y = []
    for item in data:
        y.append(item['monitor']['loss_D'])
    # y = y[::5]
    y = y
    min_loss = min(y)

    plt.figure(figsize=(20, 8), dpi=320)
    plt.title('Loss-Time')
    plt.xlabel('Time')
    plt.ylabel('Loss')
    plt.grid(alpha=0.9)

    plt.plot(x,y,'r-',label='loss, min_loss={:.4f}'.format(min_loss))
    plt.legend(loc='best')

    # plt.show()
    plt.savefig('/data/research/version2/privdl/exp/fig/stage2_permute_loss_D.png')

def fig_acc():
    # plot Accuracy figure 
    path = '/data/research/version2/privdl/result/100-1-adv_Distillation.py/8_gpus/celeba/1.0_trainset/stage3_obf/True_pretrained/5_layer/6_tail_layer/test_logs_231027104440.json' # 031-5_k2-celeba-imdb-0

    with open(path, 'r') as files:
        data = json.load(files)
    x = range(len(data))
    y1, y2 = [], []
    for item in data:
        y1.append(item['results']['testset_by_img']['rate'])
        y2.append(item['results']['testset_by_person']['rate'])

    best_y1 = max(y1)
    best_y2 = max(y2)
    best_x1 = y1.index(best_y1)
    best_x2 = y2.index(best_y2)

    plt.figure(figsize=(20, 8), dpi=320)
    plt.title('Accuracy-Time')
    plt.xlabel('Time')
    plt.ylabel('Accuracy')
    plt.grid(alpha=0.9)

    plt.plot(x,y1,'r-',label='test_by_img (best_acc:{:.4f}, best_epoch:{})'.format(best_y1, best_x1))
    plt.plot(x,y2,'b-',label='test_by_person (best_acc:{:.4f}. best_epoch:{})'.format(best_y2, best_x2))
    plt.legend(loc='best')

    plt.savefig('/data/research/version2/privdl/exp/fig/stage3_acc.png')

loss_fig()