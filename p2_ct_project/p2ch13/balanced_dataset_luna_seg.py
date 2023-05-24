"""クラスバランス調整されたセマセグ用データセット
"""

import os
import sys


# os.sepはプラットフォーム固有の区切り文字(Windows: `\`, Unix: `/`)
module_parent_dir = os.sep.join([os.path.dirname(__file__), '..']) # p2_ct_project
print("module_parent_dir", module_parent_dir)
sys.path.append(module_parent_dir)

import copy
import csv
import functools
import glob
import math
import random

from collections import namedtuple

import SimpleITK as sitk
import numpy as np

import torch.nn.functional as F

from util.disk import getCache
from util.util import XyzTuple, xyz2irc
from util.logconf import logging

raw_cache = getCache("F:/Luna16", "p2ch13_raw")


# セグメンテーション用マスク
# 生密度, 密度, 身体, 空気, 生結節候補, 結節候補, 肺, 良性結節, 悪性結節
MaskTuple = namedtuple(
    "MaskTuple",
    "raw_dense_mask, \
    dense_mask, \
    body_mask, \
    air_mask, \
    raw_candidate_mask, \
    candidate_mask, \
    lung_mask, \
    neg_mask, \
    pos_mask"
)

# 結節候補の状態
# 結節フラグ, アノテーションフラグ, 悪性腫瘍フラグ,  直径, 識別子, 結節候補の中心座標
CandidateInfoTuple = namedtuple(
    "CandidteInfoTuple",
    "isNodule_bool, \
    hasAnnotation_bool, \
    isMal_bool, \
    diameter_mm, \
    series_uid, \
    center_xyz"
)

# インメモリでキャッシングを行う標準的なライブラリ
# getCandidateInfoList関数の結果をメモリキャッシュに保存する
@functools.lru_cache(1)
def getCandidateInfoList(
    requireOnDisk_bool=True, 
    raw_datasetdir : str = "", 
    cache_datasetdir : str = "",
    ):
    # We construct a set with all series_uids that are present on disk.
    # This will let us use the data, even if we haven't downloaded all of
    # the subsets yet.

    # キャッシングした生データ(3D)
    regex_mhd_path = cache_datasetdir + "/subset*/*.mhd"
    mhd_list = glob.glob(regex_mhd_path)
    presentOnDisk_set = { os.path.split(p)[-1][:-4] for p in mhd_list } # filenames

    # 重複を排除した悪性腫瘍アノテーション(annotations_with_malignancy.csv)
    # から情報を取得
    candidateInfo_list = []
    anno_path = raw_datasetdir + "/annotations_with_malignancy.csv"
    with open(anno_path, "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue

            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])

            # new
            isMal_bool = {"False": False, "True": True}[row[5]]

            candidateInfo_list.append(
                CandidateInfoTuple(
                    True, # 結節フラグ
                    True, # アノテーションフラグ
                    isMal_bool, # 悪性腫瘍フラグ
                    annotationDiameter_mm, # 直径
                    series_uid, # 識別子
                    annotationCenter_xyz, # 中心
                )
            )

    # 結節でないデータを取得する
    cand_path = raw_datasetdir + "/candidates.csv"
    with open(cand_path, "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue

            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

            if not isNodule_bool:
                candidateInfo_list.append(
                    CandidateInfoTuple(
                        False,# 結節フラグ
                        False,# アノテーションフラグ
                        False,# 悪性腫瘍フラグ
                        0.0,# 直径
                        series_uid,# 識別子
                        candidateCenter_xyz,# 中心
                    )
                )

    candidateInfo_list.sort(reverse=True)
    # これにより, 最も大きいサイズのものから始まる実際の結節サンプルに続いて, 
    # (結節のサイズの情報を持たない)結節でないサンプルが続くことになる.
    return candidateInfo_list


@functools.lru_cache(1)
def getCandidateInfoDict(
    requireOnDisk_bool=True, 
    raw_datasetdir : str = "", 
    cache_datasetdir : str = "",
    ):
    candidateInfo_list = getCandidateInfoList(
        requireOnDisk_bool, raw_datasetdir, cache_datasetdir)
    
    candidateInfo_dict = {}

    for candidateInfo_tup in candidateInfo_list:
        candidateInfo_dict.setdefault(candidateInfo_tup.series_uid, []).append(
            candidateInfo_tup
        )

    return candidateInfo_dict


