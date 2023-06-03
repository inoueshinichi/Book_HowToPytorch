"""LunaSegPrepCacheApp
"""

import os
import sys

# os.sepはプラットフォーム固有の区切り文字(Windows: `\`, Unix: `/`)
module_parent_dir = os.sep.join([os.path.dirname(__file__), '..']) # p2_ct_project
print("module_parent_dir", module_parent_dir)
sys.path.append(module_parent_dir)

import argparse

import numpy as np

import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD
from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from p2ch13.dataset_luna_seg import PrepcacheLunaDataset, getCtSampleSize # p2ch13のデータせっと
from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)


class LunaSegPrepCacheApp:

    @classmethod
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
                            help='Batch size to use for training',
                            default=1024,
                            type=int,
                            )
        parser.add_argument('--num-workers',
                            help='Number of worker processes for background data loading',
                            default=8,
                            type=int,
                            )
        # データセットディレクトリの指定
        parser.add_argument('--datasetdir',
            help="Luna raw dataset directory",
            default='E:/Luna16',
        )

        self.cli_args = parser.parse_args(sys_argv)

    def main(self):
        log.info('Starting {}, {}'.format(type(self).__name__, self.cli_args))

        self.prep_dl = DataLoader(
            PrepcacheLunaDataset(
                raw_datasetdir=self.cli_args.datasetdir,
                # sortby_str='series_uid',
            ),
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers,
        )

        batch_iter = enumerateWithEstimate(
            self.prep_dl,
            "Stuffing cache",
            start_ndx=self.prep_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            pass

if __name__ == "__main__":
    LunaSegPrepCacheApp().main()