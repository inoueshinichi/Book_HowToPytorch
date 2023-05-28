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
from torch.utils.data import Dataset

from util.disk import getCache
from util.util import XyzTuple, xyz2irc
from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# diskcacheのタグ
tag_version = "unversioned"

raw_cache = getCache("F:/Luna16", "p2ch13_raw")


# セグメンテーション用マスク
# 生密度, 密度, 身体, 空気, 生結節候補, 結節候補, 肺, 良性結節, 悪性結節
MaskTuple = namedtuple(
    "MaskTuple",
    ["raw_dense_mask",
    "dense_mask",
    "body_mask",
    "air_mask",
    "raw_candidate_mask",
    "candidate_mask",
    "lung_mask",
    "neg_mask",
    "pos_mas"]
)

# 結節候補の状態
# 結節フラグ, アノテーションフラグ, 悪性腫瘍フラグ,  直径, 識別子, 結節候補の中心座標
CandidateInfoTuple = namedtuple(
    "CandidteInfoTuple",
    ["isNodule_bool",
    "hasAnnotation_bool",
    "isMal_bool",
    "diameter_mm",
    "series_uid",
    "center_xyz"]
)

# インメモリでキャッシングを行う標準的なライブラリ
# getCandidateInfoList関数の結果をメモリキャッシュに保存する
@functools.lru_cache(1)
def getCandidateInfoList(
    requireOnDisk_bool=True, 
    raw_datasetdir : str = "", 
    ):
    # We construct a set with all series_uids that are present on disk.
    # This will let us use the data, even if we haven't downloaded all of
    # the subsets yet.

    # 生データ(3D)
    regex_mhd_path = raw_datasetdir + "/subset*/*.mhd"
    mhd_list = glob.glob(regex_mhd_path)
    presentOnDisk_set = { os.path.split(p)[-1][:-4] for p in mhd_list } # filenames

    # 重複を排除した結節アノテーション(annotations_with_malignancy.csv)
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
            isMal_bool = {"False": False, "True": True}[row[5]] # 悪性腫瘍フラグ

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

            # 結節でないデータ
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

# インメモリキャッシュ
@functools.lru_cache(1)
def getCandidateInfoDict(
    requireOnDisk_bool=True, 
    raw_datasetdir : str = "", 
    ):
    candidateInfo_list = getCandidateInfoList(
                            requireOnDisk_bool, 
                            raw_datasetdir
                            )
    
    candidateInfo_dict = {}

    for candidateInfo_tup in candidateInfo_list:
        candidateInfo_dict.setdefault(candidateInfo_tup.series_uid, []).append(
            candidateInfo_tup
        )

    return candidateInfo_dict


class Ct:

    def __init__(self, 
                 series_uid, 
                 raw_datasetdir : str):
        regex_mhd_path = raw_datasetdir + "/subset*/{}.mhd".format(series_uid)
        mhd_path = glob.glob(regex_mhd_path)[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        self.hu_a = np.array(sitk.getArrayFromImage(ct_mhd), dtype=np.float32)

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) begin -1000 and 1 g/cc (water) begin 0.

        self.series_uid = series_uid

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3,3)

        '''ここからがセマセグ用に追加'''
        candidateInfo_list = getCandidateInfoDict(raw_datasetdir = raw_datasetdir)[self.series_uid]

        self.positiveInfo_list = [candidate_tup for candidate_tup in candidateInfo_list if candidate_tup.isNodule_bool]
        self.positive_mask = self.buildAnnotationMask(self.positiveInfo_list)
        self.positive_indexes = (self.positive_mask.sum(axis=(1,2))).nonzero()[0].tolist()

    def buildAnnotationMask(self, 
                            positiveInfo_list,
                            threshold_hu=-700):
        
        boundingBox_a = np.zeros_like(self.hu_a, dtype=np.bool) # (I,H,W)

        for candidateInfo_tup in positiveInfo_list:
            center_irc = xyz2irc(candidateInfo_tup.center_xyz, 
                                 self.origin_xyz, 
                                 self.vxSize_xyz, 
                                 self.direction_a)
            
            ci = int(center_irc.index)
            cr = int(center_irc.row)
            cc = int(center_irc.col)

            # Vocelの範囲内でindex半径を取得(このやり方賢い)
            index_radius = 2
            try:
                while (self.hu_a[ci + index_radius, cr, cc] > threshold_hu
                       and self.hu_a[ci - index_radius, cr, cc] > threshold_hu):
                    index_radius += 1
            except IndexError:
                index_radius -= 1 # Vocelの境界内に収める

            # Vocelの範囲内でrow半径を取得
            row_radius = 2
            try:
                while (self.hu_a[ci, cr + row_radius, cc] > threshold_hu
                       and self.hu_a[ci, cr - row_radius, cc] > threshold_hu):
                    row_radius += 1
            except IndexError:
                row_radius -= 1 # Vocelの境界内に収める

            # Vocelの範囲内でcol半径を取得
            col_radius = 2
            try:
                while (self.hu_a[ci, cr, cc + col_radius] > threshold_hu
                       and self.hu_a[ci, cr, cc - col_radius] > threshold_hu):
                    col_radius += 1
            except IndexError:
                col_radius -= 1 # Vocelの境界内に収める

            # assert index_radius > 0, repr([candidateInfo_tup.center_xyz, center_irc, self.hu_a[ci, cr, cc]])
            # assert row_radius > 0
            # assert col_radius > 0

            boundingBox_a[
                ci - index_radius : ci + index_radius + 1,
                cr - row_radius : cr + row_radius + 1,
                cc - col_radius : cc + col_radius + 1,
            ] = True

        mask_a = boundingBox_a & (self.hu_a > threshold_hu)

        return mask_a
    
    def getRawCandidate(self, center_xyz, width_irc):

        center_irc = xyz2irc(center_xyz, self.origin_xyz, self.vxSize_xyz, self.direction_a)

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis]/2)) # 四捨五入
            end_ndx = int(start_ndx + width_irc[axis])

            assert center_val >=0 and center_val < self.hu_a.shape[axis], repr(
                [
                    self.series_uid,
                    center_xyz,
                    self.origin_xyz,
                    self.vxSize_xyz,
                    center_irc,
                    axis,
                ]
            )

            # Vocel境界の補正
            if start_ndx < 0:
                log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                    self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width_irc[axis])
            
            if end_ndx > self.hu_a.shape[axis]:
                log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                    self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]
        pos_chunk = self.positive_mask[tuple(slice_list)]

        return ct_chunk, pos_chunk, center_irc
    

@functools.lru_cache(1, typed=True)
def getCt(series_uid, raw_datasetdir : str):
    return Ct(series_uid=series_uid, raw_datasetdir=raw_datasetdir)


# ディスクキャッシュ
@raw_cache.memoize(typed=True, tag=tag_version)
def getCtRawCandidate(        
        raw_datasetdir : str, 
        series_uid, 
        center_xyz, 
        width_irc,
        ):
    
    ct = getCt(
            series_uid=series_uid, 
            raw_datasetdir=raw_datasetdir, 
            )
    
    ct_chunk, pos_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    ct_chunk.clip(-1000, 1000, ct_chunk)

    return ct_chunk, pos_chunk, center_irc # center_ircが追加された


# ディスクキャッシュ
@raw_cache.memoize(typed=True, tag=tag_version)
def getCtSampleSize(
    raw_datasetdir : str,
    series_uid,
    ):

    ct = Ct(series_uid, raw_datasetdir)
    return int(ct.hu_a.shape[0]), ct._positive_indexes # [0, N], [pi1, pi2, pi3, ...]



class Luna2dSegmentationDataset(Dataset):

    def __init__(
            self,
            val_stride=0,
            isValSet_bool=None,
            series_uid=None,
            contextSlices_count=3,
            fullCt_bool=False,
    ):
        self.contextSlices_count = contextSlices_count
        self.fullCt_bool = fullCt_bool

        # 続きはここから. '23/5/28
            



