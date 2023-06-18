"""結節候補の分類モデル
"""
import os
import sys

# os.sepはプラットフォーム固有の区切り文字(Windows: `\`, Unix: `/`)
module_parent_dir = os.sep.join([os.path.dirname(__file__), '..']) # p2_ct_project
print("module_parent_dir", module_parent_dir)
sys.path.append(module_parent_dir)

from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

import random
import math

import numpy as np

import torch
from torch import nn as nn
import torch.nn.functional as F

# p2ch14で使用するデータ拡張関数
def augment3d(inp):
    transform_t = torch.eye(4, dtype=torch.float32)
    for i in range(3):
        if True: # 'flip' in augmentation_dict:
            if random.random() > 0.5:
                transform_t[i,i] *= -1
        if True: # 'offset' in augmentation_dict:
            offset_float = 0.1
            random_float = (random.random() * 2 - 1) # [-1,1]
            transform_t[i, 3] = offset_float * random_float # (tx,ty,tz)
    if True:
        angle_rad = random.random() * np.pi * 2
        s = np.sin(angle_rad)
        c = np.sin(angle_rad)

        rotation_t = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=torch.float32)

        transform_t @= rotation_t

        affine_t = F.affine_grid(
            transform_t[:3].unsqueeze(0).expand(inp.size(0), -1, -1).cuda(),
            inp.shape,
            align_corners=False,
        )

        augmented_chunk = F.grid_sample(
            inp,
            affine_t,
            padding_mode='border',
            align_corners=False,
        )
        
        if False: # 'noise': in augmentation_dict:
            noise_t = torch.randn_like(augmented_chunk)
            noise_t *= augmentation_dict['noise']

            augmented_chunk += noise_t

        return augmented_chunk


class LunaBlock(nn.Module):

    def __init__(self, in_channels, conv_channels):
        super(LunaBlock, self).__init__()

        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=conv_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            in_channels=conv_channels,
            out_channels=conv_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )

        self.relu2 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, input_batch):
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)
        return self.maxpool(block_out)
    

class LunaModel(nn.Module):
    
    # 入力サイズは(N, 1, 32, 48, 48)
    def __init__(self, in_channels=1, conv_channels=8):
        super(LunaModel, self).__init__()

        # Tail
        self.tail_batchnorm = nn.BatchNorm3d(num_features=1)

        # Main
        self.block1 = LunaBlock(in_channels=in_channels, conv_channels=conv_channels)
        self.block2 = LunaBlock(in_channels=conv_channels, conv_channels=conv_channels*2)
        self.block3 = LunaBlock(in_channels=conv_channels*2, conv_channels=conv_channels*4)
        self.block4 = LunaBlock(in_channels=conv_channels*4, conv_channels=conv_channels*8)

        # Head:
        main_flatten_out_dim = 1152
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.head_linear = nn.Linear(main_flatten_out_dim, 2) # (陰性スコア, 陽性スコア)
        self.head_softmax = nn.Softmax(dim=1) # (陰性尤度, 陽性尤度)

        # Iniliaze weights
        self._init_weights()

    def forward(self, input_batch):
        bn_output = self.tail_batchnorm(input_batch)

        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)

        # conv_flat = self.flatten(block_out)
        conv_flat = block_out.view(
            block_out.size(0), # バッチサイズ
            -1,
        )

        linear_output = self.head_linear(conv_flat)

        # (logits, softmax)
        return linear_output, self.head_softmax(linear_output) # (陰性スコア, 陽性スコア), (陰性尤度, 陽性尤度)
    
    # see also https://github.com/pytorch/pytorch/issues/18182
    def _init_weights(self):
        for m in self.modules():
            if type(m) in { 
                nn.Linear, 
                nn.Conv3d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d 
            }:
                nn.init.kaiming_uniform_(
                    m.weight.data, a=0, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)
        

    




    