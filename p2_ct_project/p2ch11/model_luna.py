"""結節候補の分類モデル
"""

import math

import torch.nn as nn

import os
import sys

# os.sepはプラットフォーム固有の区切り文字(Windows: `\`, Unix: `/`)
module_parent_dir = os.sep.join([os.path.dirname(__file__), '..']) # p2_ct_project
print("module_parent_dir", module_parent_dir)
sys.path.append(module_parent_dir)

from util.logconf import logging



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

        # Head
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
        

    




    