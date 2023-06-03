"""SegmentationTrainingApp
"""
import os
import sys

# os.sepはプラットフォーム固有の区切り文字(Windows: `\`, Unix: `/`)
module_parent_dir = os.sep.join([os.path.dirname(__file__), '..']) # p2_ct_project
print("module_parent_dir", module_parent_dir)
sys.path.append(module_parent_dir)

import argparse
import datetime
import hashlib
import shutil
import socket


import numpy as np
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim

from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from p2ch13.dataset_luna_seg import Luna2dSegmentationDataset, TrainingLuna2dSegmentationDataset, getCt
from util.logconf import logging
from p2ch13.model_unet import UNetWrapper, SegmentationAugmentation

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# Used for computeClassificationLoss and logMetrics to index into metrics_t/metrics_a
# METRICS_LABEL_NDX = 0
METRICS_LOSS_NDX = 1
# METRICS_FN_LOSS_NDX = 2
# METRICS_ALL_LOSS_NDX = 3

# METRICS_PTP_NDX = 4
# METRICS_PFN_NDX = 5
# METRICS_MFP_NDX = 6
METRICS_TP_NDX = 7
METRICS_FN_NDX = 8
METRICS_FP_NDX = 9

METRICS_SIZE = 10


class SegmentationTrainingApp:

    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
                            help='Batch size to use for training',
                            default=16,
                            type=int,
                            )
        parser.add_argument('--num-workers',
                            help='Number of worker processes for background data loading',
                            default=8,
                            type=int,
                            )
        parser.add_argument('--epochs',
                            help='Number of epochs to train for',
                            default=1,
                            type=int,
                            )
        
        # データ拡張
        parser.add_argument('--augmented',
                            help='Augment the training data.',
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--augment-flip',
                            help='Augment the training data by randomly flipping the data left-right, up-down, and front-back.',
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--augment-offset',
                            help='Augment the training data by randomly offsetting the data slightly along the X and Y axes.',
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--augment-scale',
                            help='Augment the training data by randomly increasing or decreasing the size of the candidate.',
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--augment-rotate',
                            help="Augment the training data by randomly rotating the data around the head-foot axis.",
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--augment-noise',
                            help="Augment the training data by randomly adding noise to the data.",
                            action='store_true',
                            default=False,
                            )

        # その他
        parser.add_argument('--tb-prefix',
                            default='p2ch13',
                            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
                            )

        parser.add_argument('comment',
                            help="Comment suffix for Tensorboard run.",
                            nargs='?',
                            default='none',
                            )
        
        # データセットディレクトリの指定
        parser.add_argument('--datasetdir',
            help="Luna raw dataset directory",
            default='E:/Luna16',
        )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.totalTrainingSamples_count = 0
        self.trn_writer = None
        self.val_writer = None

        

        # データ拡張 (パラメータは取り組み対象の経験則)
        self.augmentation_dict = {}
        if self.cli_args.augmented or self.cli_args.augment_flip:
            self.augmentation_dict['flip'] = True
        if self.cli_args.augmented or self.cli_args.augment_offset:
            self.augmentation_dict['offset'] = 0.03
        if self.cli_args.augmented or self.cli_args.augment_scale:
            self.augmentation_dict['scale'] = 0.2
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            self.augmentation_dict['rotate'] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            self.augmentation_dict['noise'] = 25.0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

    def initModel(self):
        segmentation_model = UNetWrapper(
            in_channels=7,
            n_classes=1,
            depth=3,
            wf=4,
            padding=True,
            batch_norm=True,
            up_mode='upconv',
        )

        augmentation_model = SegmentationAugmentation(**self.augmentation_dict)

        if self.use_cuda:
            log.info('Using CUDA; {} devices.'.format(
                torch.cuda.device_count()
            ))
            if torch.cuda.device_count() > 1:
                segmentation_model = nn.DataParallel(segmentation_model)
                augmentation_model = nn.DataParallel(augmentation_model)
            segmentation_model = segmentation_model.to(self.deivce)
            augmentation_model = augmentation_model.to(self.device)

        return segmentation_model, augmentation_model
    
    def initOptimizer(self):
        return Adam(self.segmentation_model.parameters())
    
    def initTrainDl(self):
        train_ds = TrainingLuna2dSegmentationDataset(
            raw_datasetdir=self.cli_args.datasetdir,
            val_stride=10,
            isValSet_bool=False,
            contextSlices_count=3,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return train_dl
    
    def initValDl(self):
        val_ds = Luna2dSegmentationDataset(
            raw_datasetdir=self.cli_args.datasetdir,
            val_stride=10,
            isValSet_bool=True,
            contextSlices_count=3,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return val_dl
    
    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.cli_args.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=log_dir + "-trn_cls-" + self.cli_args.comment
            )

            self.val_writer = SummaryWriter(
                log_dir=log_dir + "-val_cls-" + self.cli_args.comment
            )

    




            
