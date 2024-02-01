# PrivDL

research codebase for privacy preserving deeplearning

## 代码约定
1. 一级文件夹有“data block result exp utils”，数据流动为 data→block→result，流控制由“exp”进行，utils 提供一些 block 和 exp 共用的工具。
2. 如果代码没有 BUG，原则上 repo 内的代码只增不减。
3. 不同版本的代码需显示声明并显示调用。
4. 实验代码中做各种操作的实验目的是什么，请都在注释里写上。
5. 可调节的超参数只允许出现在 exp 文件夹下的文件中，如果其他模块需要进行超参数条件，必须设置为函数参数形式（有可能需要新增一个函数版本），在 exp 文件中予以超参数设定。
6. block 内的文件 import repo 内的文件，原则上只应 import utils、本文件夹_utils，避免互相之间的引用，以免做出不兼容的函数版本修改后，需要修改一系列的函数调用。
7. exp 内的文件，允许但建议尽量少 import block 内的 xxx_utils，因为 xxx_utils 原则上是该文件夹内部的小工具，在 exp 内使用算是轻度 HACK。如果发现某工具需要大量使用，建议将其改写到通用的 utils 中。


## 新增代码
1. 300-final-add.py： xnn-d 方案代码
   - mode: stage1 是进行第一阶段的训练，训练teacher tail。
   - stage2: 对抗训练过程。
   - stage_aux: 是训练攻击的过程
   - distill: 是第三阶段蒸馏过程。
2. 500-FNLN_false_NELN_true_MSRA.py 同1，只是train_data 换成了MSRA
2. 500-FNLN_false_NELN_true_MSRA_just_for_vis.py 同1，针对可视化做了代码修改。
3. 500-FNLN_false_NELN_true_VGG_Face.py 同1，只是train_data 换成了VGG_Face
4. 500-FNLN_false_NELN_true_webface.py 同1， 只是train_data 换成了webface
5. 其余200-  400- 500- 可以忽略
