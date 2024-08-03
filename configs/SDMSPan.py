

# ---> GENERAL CONFIG <---
name = 'SDMSPan'
dataset = ['GF-2', 'WV-2', 'WV-3']
currentPath = '/home/mjzhao/zmj/LGTEUN-main/LGTEUN-main'
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
#                    f'model_out/{name}/{datas}/2024-03-13_18-27/train_out/model_iter_250000.pth',
#                 #    f'model_out/{name}/{datas}/2024-06-26_00-34/train_out/model_iter_90000.pth']

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


# 700 epoch
max_iter_list = [181300, 177100, 159250]

max_iter = max_iter_list[index]  


step_list = [18130, 17710, 15925]

step = step_list[index]

save_freq = 10000
test_freq = 10000
eval_freq = 10000



norm_input = True

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
