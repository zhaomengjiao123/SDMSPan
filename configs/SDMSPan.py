# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2023-Jul
# @Address  : Time Lab @ SDU
# @FileName : unlg_former.py
# @Project  : LGTEUN (Pan-sharpening), IJCAI 2023

# Local-Global Transformer Enhanced Unfolding Network for Pan-sharpening, IJCAI 2023

# ---> GENERAL CONFIG <---
name = 'SDMSPan'
# dataset = ['GF-2', 'QB', 'WV-3']
dataset = ['GF-2', 'WV-2', 'WV-3']
# ms_chans_list = [4, 4, 4]
currentPath = '/home/mjzhao/zmj/LGTEUN-main/LGTEUN-main'
# dataPath = '/home/zysong/zht/UCGAN-master/UCGAN-master/data/PSData3/Dataset/Dataset'
dataPath = '/home/mjzhao/zmj/LGTEUN-main/LGTEUN_data/Dataset_LGTEUN/DataSet'
ms_chans_list = [4, 4, 8]
index = 2

device_id = '0'

datas = dataset[index]
ms_chans = ms_chans_list[index]



model_type = 'SDMSPan'
work_dir = f'model_out/{name}'
log_dir = f'logs/{model_type.lower()}/{datas}'
log_file = f'{log_dir}/{name}.log'
log_level = 'INFO'

# 是否测试
# only_test = True  #
# only_test = False  #

# pretrained = True


# checkpoint_list = [#f'model_out/{name}/{datas}/2024-06-18_10-57/train_out/model_iter_170000.pth',
#                    #f'model_out/{name}/{datas}/2024-06-29_23-37/train_out/model_iter_170000.pth', # 消融实验 NO LR LOSS
#                    #f'model_out/{name}/{datas}/2024-06-29_21-58/train_out/model_iter_150000.pth', # 消融实验 NO PAN LR LOSS
#                    f'model_out/{name}/{datas}/2024-06-30_00-01/train_out/model_iter_170000.pth', # 消融实验 NO PAN LOSS
#                    f'model_out/{name}/{datas}/2024-03-13_18-27/train_out/model_iter_250000.pth',
#                 #    f'model_out/{name}/{datas}/2024-06-26_00-34/train_out/model_iter_90000.pth']
#                     f'model_out/{name}/{datas}/2024-07-04_15-39/train_out/model_iter_80000.pth'] # 消融实验 NO PAN LOSS

# checkpoint = checkpoint_list[index]
# checkpoint = currentPath + '/' + checkpoint  # for original pretrain.pth



# pretrained_list = [f'model_out/{name}/{datas}/2024-06-14_17-18/train_out/model_iter_129500.pth',
#                    f'model_out/{name}/{datas}/2024-07-03_00-59/train_out/model_iter_20000.pth',
#                    f'model_out/{name}/{datas}/2024-03-14_08-56/train_out/model_iter_20000.pth']
# pretrained = pretrained_list[index]
# pretrained = currentPath + '/' + pretrained  # for original pretrain.pth


# ---> DATASET CONFIG <---
aug_dict = {'lr_flip': 0.5, 'ud_flip': 0.5}

# bit_depth = 10
# LGTEUN 数据集
bit_depth = 11
train_set_cfg = dict(
    dataset=dict(
        type='PSDataset',
        image_dirs=[dataPath + f'/{datas}/train_low_res'],
        bit_depth=bit_depth),
    num_workers=4,
    batch_size=4,
    shuffle=True)
test_set0_cfg = dict(
    dataset=dict(
        type='PSDataset',
        image_dirs=[dataPath + f'/{datas}/test_full_res'],
        bit_depth=bit_depth),
    num_workers=0,
    batch_size=4,
    shuffle=False)
test_set1_cfg = dict(
    dataset=dict(
        type='PSDataset',
        image_dirs=[dataPath + f'/{datas}/test_low_res'],
        bit_depth=bit_depth),
    num_workers=0,
    batch_size=4,
    shuffle=False)
seed = 19971118
cuda = True

# 500 epoch
# max_iter_list = [129500, 126500, 113750]
# 700 epoch
max_iter_list = [181300, 177100, 159250]
# WV2 接着 50000iter的训练
# max_iter_list = [181300, 97100, 159250]
# WV2 接上次训练
# max_iter_list = [181300, 77100, 159250]

# max_iter_list = [181300, 126500, 159250]

# 测试用
# max_iter_list = [10, 97100, 159250]


# + 200epoch
# max_iter_list = [51800, 50600, 45500]


# 1000 epoch
# max_iter_list = [259000, 253000, 227500]


# max_iter_list = [2590, 253000, 35000]
# max_iter_list = [259000, 253000, 300000]
# max_iter_list = [10000, 10000, 10000]

max_iter = max_iter_list[index]  # max inter 1000 epoch for GF2 and WV2, about 130 epoch for WV3

# step_list = [25900, 25300, 30000]  # train / batch size
# step_list = [25900, 25300, 22750]
# step_list = [18130, 12650, 15925]
step_list = [18130, 17710, 15925]

# step_list = [12950, 12650, 11375]
# step_list = [5180, 5060, 4550]


# step_list = [25900, 25300, 22750]


# step_list = [25, 25, 22]
step = step_list[index]

save_freq = 10000
test_freq = 10000
eval_freq = 10000
# save_freq = 1000
# test_freq = 1000
# eval_freq = 1000
# save_freq = 10
# test_freq = 10
# eval_freq = 10


norm_input = True
# norm_input = False

# 1e-4
# ---> SPECIFIC CONFIG <---
# optim_cfg = {
#     'core_module': dict(type='Adam', betas=(0.9, 0.999), lr=1.5e-3)
# }
optim_cfg = {
    'core_module': dict(type='Adam', betas=(0.9, 0.999), lr=1e-4)
}


sched_cfg = dict(step_size=step, gamma=0.85)

loss_cfg = {
    'rec_loss': dict(type='l1', w=1.)
}

model_cfg = {
    'core_module': dict(),
}
