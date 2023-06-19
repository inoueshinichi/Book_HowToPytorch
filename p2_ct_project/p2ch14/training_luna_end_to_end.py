"""結節と非結節を分類するモデルの学習と検証"""

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

import numpy as np
from matplotlib import pyplot

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from util.logconf import logging
# from p2ch12.mod_dataset_luna import LunaDataset # p2ch12のLunaDatasetクラスを使用する
# from p2ch11.model_luna import LunaModel # モデルはp2ch11のまま
import p2ch11
import p2ch14
from p2ch14.dataset_luna_end_to_end import LunaDataset, MalignantLunaDataset

from util.util import enumerateWithEstimate
from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# Used for computeBatchLoss and logMetrics to index into metrics_t/metrics_a
METRICS_LABEL_NDX=0
METRICS_PRED_NDX=1
METRICS_PRED_P_NDX=2
METRICS_LOSS_NDX=3
METRICS_SIZE = 4


class ClassificationTrainingApp:

    def __init__(self,
                 sys_argv=None,
                 ):
        
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
                            help='Batch size to use for training.',
                            default=24,
                            type=int,
                            )
        parser.add_argument('--num-workers',
                            help='Number of worker processes for background data loading.',
                            default=8,
                            type=int,
                            )
        parser.add_argument('--epochs',
                            help='Number of epochs to train for',
                            default=1,
                            type=int,
                            )
        parser.add_argument('--dataset',
                            help='What to dataset to feed the model.',
                            action='store',
                            default='LunaDataset',
                            )
        parser.add_argument('--model',
                            help='What to model class name to use.',
                            action='store',
                            default='LunaModel',
                            )
        parser.add_argument('--malignant',
                            help='Train the model to classify nodules as benign or malignant.',
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--finetune',
                            help='Start finetuning from this model.',
                            default='',
                            )
        parser.add_argument('--finetune-depth',
                            help='Number of blocks (counted from the headd) to include in finetuning.',
                            type=int,
                            default=1,
                            )
        parser.add_argument('--tb-prefix',
                            default='p2ch14',
                            help='Data prefix to use for Tensorboard run. Defaults to chapter.',
                            )
        parser.add_argument('comment',
                            help='Comment suffix for Tensorboard run.',
                            nargs='?',
                            default='dlwpt',
                            )
        
        parser.add_argument('--dataset',
                            help='Raw dataset for Luna directory.',
                            default='E:/Luna16',
                            )
        
        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.augmentation_dict = {}
        if True:
            self.augmentation_dict['flip'] = True
            self.augmentation_dict['offset'] = 0.1
            self.augmentation_dict['scale'] = 0.2
            self.augmentation_dict['rotate'] = True
            self.augmentation_dict['noise'] = 25.0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

    
    def initModel(self):
        model_cls = getattr(p2ch11.model_luna, self.cli_args.model)
        model = model_cls()

        if self.cli_args.finetune:
            d = torch.load(self.cli_args.finetune, map_location='cpu')
            model_blocks = [
                n for n, subm in model.named_children()
                if len(list(subm.parameters())) > 0
            ]

            finetune_blocks = model_blocks[-self.cli_args.finetune_depth:]
            log.info(f"finetuning from {self.cli_args.finetune}, blocks {' '.join(finetune_blocks)}")

            model.load_state_dict(
                {
                    k: v for k, v in d['model_state'].items()
                    if k.split('.')[0] not in model_blocks[-1]
                }, 
                strict=False,
            )

            for n, p in model.named_parameters():
                if n.split('.')[0] not in finetune_blocks:
                    p.requires_grad_(False) # 勾配計算をさせない (凍結)
            
        if self.use_cuda:
            log.info('Using CUDA; {} device.'.format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        
        return model
    
    def initOptimizer(self):
        lr = 0.003 if self.cli_args.finetune else 0.001
        return SGD(self.model.parameters(), lr=lr, weight_decay=1e-4)
    
    def initTrainDl(self):
        ds_cls = getattr(p2ch14.dataset_luna_end_to_end, self.cli_args.dataset)

        train_ds = ds_cls(
            val_stride=10,
            isValSet_bool=False,
            ratio_int=1,
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
        ds_cls = getattr(p2ch14.dataset_luna_end_to_end, self.cli_args.dataset)

        val_ds = ds_cls(
            val_stride=10,
            isValSet_bool=True,
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
                log_dir=log_dir + '-trn_cls-' + self.cli_args.comment
            )
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val_cls-' + self.cli_args.comment
            )

    def main(self):

        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        best_score = 0.0
        validation_cadence = 5 if not self.cli_args.finetune else 1

        for epoch_ndx in range(1, self.cli_args.epochs + 1):

            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1),
            ))

            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            if epoch_ndx == 1 or epoch_ndx % validation_cadence == 0:
                valMetrics_t = self.doValidation(epoch_ndx, val_dl)
                score = self.logMetrics(epoch_ndx, 'val', valMetrics_t)
                best_score = max(score, best_score)

                # TODO: this 'cls' will need to change for the malignant classifier
                self.saveModel('cls', epoch_ndx, score == best_score)

            
        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_wirter.close()

    
    def doTraining(self,
                   epoch_ndx,
                   train_dl,
                   ):
        self.model.train()

        train_dl.dataset.shuffleSamples()

        trnMetrics_g = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
        )

        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )

        for batch_ndx, batch_tup in batch_iter:

            self.optimizer.zero_grad()

            loss_var = self.computeBatchLoss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trnMetrics_g,
                augment=True,
            )

            loss_var.backward()
            self.optimizer.step()

        self.totalTrainingSamples_count += len(train_dl.dataset)
        



