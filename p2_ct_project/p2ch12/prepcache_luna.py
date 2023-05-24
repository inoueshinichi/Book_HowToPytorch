import argparse
import sys

import os

# os.sepはプラットフォーム固有の区切り文字(Windows: `\`, Unix: `/`)
module_parent_dir = os.sep.join([os.path.dirname(__file__), '..']) # p2_ct_project
print("module_parent_dir", module_parent_dir)
sys.path.append(module_parent_dir)

import numpy as np

import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD
from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from p2ch12.mod_dataset_luna import LunaDataset # p2ch12のLunaDatasetを使用
from util.logconf import logging
from p2ch11.model_luna import LunaModel # 引き続きp2ch11のモデルを使用

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)


class LunaPrepCacheApp:
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
        # 生データセットディレクトリの指定
        parser.add_argument('--raw_datasetdir',
            help="Luna raw dataset directory",
            default='E:/Luna16'
        )
        # キャッシュデータセットディレクトリの指定
        parser.add_argument("--cache_datasetdir",
            help="Luna cache dataset directory",
            default="F:/Luna16"
        )

        self.cli_args = parser.parse_args(sys_argv)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        self.prep_dl = DataLoader(
            LunaDataset(
                raw_datasetdir=self.cli_args.raw_datasetdir,
                cache_datasetdir=self.cli_args.cache_datasetdir,
                sortby_str='series_uid',
            ),
            batch_size=self.cli_args.batch_size,
            num_workers=self.cli_args.num_workers,
        )

        batch_iter = enumerateWithEstimate(
            self.prep_dl,
            "Stuffing cache",
            start_ndx=self.prep_dl.num_workers,
        )
        for _ in batch_iter:
            pass


if __name__ == '__main__':
    LunaPrepCacheApp().main()
