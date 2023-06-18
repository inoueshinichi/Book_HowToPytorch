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
from p2ch11.model_luna import LunaModel # モデルはp2ch11のまま
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


