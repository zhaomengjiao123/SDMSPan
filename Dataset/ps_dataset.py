# -*- coding: utf-8 -*-
# @Auther   : Mingsong Li (lms-07)
# @Time     : 2023-May
# @Address  : Time Lab @ SDU
# @FileName : ps_dataset.py
# @Project  : LGTEUN (Pan-sharpening), IJCAI 2023

import os
import numpy as np
import torch
import torch.utils.data as data
from typing import Union
import cv2

from .builder import DATASETS
from .utils import _is_pan_image, get_image_id, load_image, data_normalize


# use the registry to manage the module
@DATASETS.register_module()
class PSDataset(data.Dataset):
    def __init__(self, image_dirs, bit_depth, norm_input=False):
        r""" Build dataset from folders

        Args:
            image_dirs (list[str]): image directories
            bit_depth (int): data value range in n-bit
            norm_input (bool): normalize the input to [0, 1]
        """
        super(PSDataset, self).__init__()

        self.image_dirs = image_dirs
        self.bit_depth = bit_depth
        self.norm_input = norm_input
        self.image_ids = []
        self.image_prefix_names = []  # full-path filename prefix
        #print("image dirs:", image_dirs)

        for y in image_dirs:
            #print("y:", y)
            for x in os.listdir(y):
                if _is_pan_image(x):
                    self.image_ids.append(get_image_id(x))
                    self.image_prefix_names.append(os.path.join(y, get_image_id(x)))

    def __getitem__(self, index):
        # type: (int) -> dict[str, Union[torch.Tensor, str]]
        prefix_name = self.image_prefix_names[index]

        # input lr 去掉了.transpose(2, 0, 1)

        input_dict = dict(
            input_lr=load_image('{}_lr.tif'.format(prefix_name)),  # [4,32,32] LrMS
            input_pan=load_image('{}_pan.tif'.format(prefix_name))[np.newaxis, :],  # [1,128,128] PAN
            # input_lr_u=load_image('{}_lr_u.tif'.format(prefix_name)) # [4,128,128] LRMS 上采样4倍
        )
        #print("input lr u:", input_dict['input_lr_u'].shape)
        # print("input lr", input_dict['input_lr'].shape)
        # print("input pan", input_dict['input_pan'].shape)
        if os.path.exists('{}_mul.tif'.format(prefix_name)) and len(self.image_dirs) == 1:
           
            input_dict['target'] = load_image('{}_mul.tif'.format(prefix_name)) # [4,128,128] HrMS gt
            # 去掉了 .transpose(2, 0, 1) 
        #print("input target", input_dict['target'].shape)

        # [1,64,64] Gaussian Degraded PAN
        input_dict['input_pan_l'] = cv2.pyrDown(cv2.pyrDown(input_dict['input_pan'][0]))[np.newaxis, :]

        for key in input_dict:  # numpy2torch
            input_dict[key] = torch.from_numpy(input_dict[key]).float()

        if self.norm_input:
            input_dict = data_normalize(input_dict, self.bit_depth)

        input_dict['image_id'] = self.image_ids[index]

        #print("input dict:", input_dict)
        #print("input dict:", input_dict)
        return input_dict

    def __len__(self):
        #print("len:", len(self.image_ids))
        return len(self.image_ids)
