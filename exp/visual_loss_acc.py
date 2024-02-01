import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

def loss_fig(path,save_path):
    with open(path, 'r') as files:
        data = json.load(files)
    # x = range(len(data))[::5]
    # print(data)
    x = range(len(data))
    y = []
    for item in data:
        y.append(item['monitor']['loss'])
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
    plt.savefig(save_path)

def loss_GAN_fig(path,save_path):

    with open(path, 'r') as files:
        data = json.load(files)
    # x = range(len(data))[::5]
    # print(data)
    x = range(len(data))
    y_D = []
    y_G = []
    for item in data:
        y_D.append(item['monitor']['loss_D']/0.3)
        y_G.append(item['monitor']['loss_G']/0.7)
    # y = y[::5]
    min_loss_G = min(y_G)
    min_loss_D = min(y_D)

    plt.figure(figsize=(20, 8), dpi=320)
    plt.title('Loss-Time')
    plt.xlabel('Time')
    plt.ylabel('Loss')
    plt.grid(alpha=0.9)

    plt.plot(x,y_G,'r-',label='loss_G, min_loss={:.4f}'.format(min_loss_G))
    plt.plot(x,y_D,'b-',label='loss_D, min_loss={:.4f}'.format(min_loss_D))
    plt.legend(loc='best')

    # plt.show()
    plt.savefig(save_path)

def fig_acc(path,save_path):
    # plot Accuracy figure 

    with open(path, 'r') as files:
        data = json.load(files)
    x = []
    y1, y2 = [], []
    for item in data:
        x.append(item['clock']['epoch'])
        y1.append(item['results']['testset_by_img']['rate'])
        y2.append(item['results']['testset_by_person']['rate'])

    best_y1 = max(y1)
    best_y2 = max(y2)
    best_x1 = x[y1.index(best_y1)]
    best_x2 = x[y2.index(best_y2)]

    plt.figure(figsize=(20, 8), dpi=320)
    plt.title('Accuracy-Time')
    plt.xlabel('Time')
    plt.ylabel('Accuracy')
    plt.grid(alpha=0.9)

    plt.plot(x,y1,'r-',label='test_by_img (best_acc:{:.4f}, best_epoch:{})'.format(best_y1, best_x1))
    plt.plot(x,y2,'b-',label='test_by_person (best_acc:{:.4f}. best_epoch:{})'.format(best_y2, best_x2))
    plt.legend(loc='best')

    plt.savefig(save_path)

def fig_acc_just_by_person(path,save_path):
    # plot Accuracy figure 

    with open(path, 'r') as files:
        data = json.load(files)
    x = []
    y1, y2 = [], []
    for item in data:
        x.append(item['clock']['epoch'])
        # y1.append(item['results']['testset_by_img']['rate'])
        y2.append(item['results']['testset_by_person']['rate'])

    # best_y1 = max(y1)
    best_y2 = max(y2)
    # best_x1 = x[y1.index(best_y1)]
    best_x2 = x[y2.index(best_y2)]

    plt.figure(figsize=(20, 8), dpi=320)
    plt.title('Accuracy-Time')
    plt.xlabel('Time')
    plt.ylabel('Accuracy')
    plt.grid(alpha=0.9)

    # plt.plot(x,y1,'r-',label='test_by_img (best_acc:{:.4f}, best_epoch:{})'.format(best_y1, best_x1))
    plt.plot(x,y2,'b-',label='test_by_person (best_acc:{:.4f}. best_epoch:{})'.format(best_y2, best_x2))
    plt.legend(loc='best')

    plt.savefig(save_path)

def fig_acc_permute(path,save_path):
    # plot Accuracy figure 
    with open(path, 'r') as files:
        data = json.load(files)
    # x = range(len(data))
    x = []
    y1, y2,y3 = [], [] ,[]
    for item in data:
        x.append(item['clock']['epoch'])
        y1.append(item['results']['testset_by_img']['rate'])
        y2.append(item['results']['testset_by_person']['rate'])
        y3.append(item['results']['RestoreIdentificationAccuracy']['rate'])

    best_y1 = max(y1)
    best_y2 = max(y2)
    best_y3 = max(y3)
    best_x1 = x[y1.index(best_y1)]
    best_x2 = x[y2.index(best_y2)]
    best_x3 = x[y3.index(best_y3)]

    plt.figure(figsize=(20, 8), dpi=320)
    plt.title('Accuracy-Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(alpha=0.9)

    plt.plot(x,y1,'r-',label='test_by_img (best_acc:{:.4f}, best_epoch:{})'.format(best_y1, best_x1))
    plt.plot(x,y2,'b-',label='test_by_person (best_acc:{:.4f}. best_epoch:{})'.format(best_y2, best_x2))
    plt.plot(x,y3,'g-',label='RestoreIdentificationAccuracy (best_acc:{:.4f}. best_epoch:{})'.format(best_y3, best_x3))
    plt.legend(loc='best')

    plt.savefig(save_path)

# loss_fig('/data/research/version2/privdl/result/500-FNLN_false_NELN_true_webface.py/8_gpus/webface/1.0_trainset/stage3_obf/True_pretrained/5_layer/6_tail_layer/logs_240113002916.json',\
#          '/data/research/version2/privdl/exp/fig_500_webface/distill_loss.png')
# loss_GAN_fig('/data/research/version2/privdl/result/500-FNLN_false_NELN_true.py/8_gpus/celeba/1.0_trainset/stage2_obf/True_pretrained/5_layer/6_tail_layer/logs_240109142043.json',\
#              '/data/research/version2/privdl/exp/fig_500/fig_facecrub_attacker.png')
fig_acc_just_by_person('/data/research/version2/privdl/result/500-FNLN_false_NELN_true_webface.py/8_gpus/webface/1.0_trainset/stage_aux_obf/True_pretrained/5_layer/6_tail_layer/test_logs_240113120420.json',\
                '/data/research/version2/privdl/exp/fig_500_webface/imdb_attack_acc.png')
