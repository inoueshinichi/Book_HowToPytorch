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

import torch
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
    "pos_mask"]
)

# 結節候補の状態
# 結節フラグ, アノテーションフラグ, 悪性腫瘍フラグ,  直径, 識別子, 結節候補の中心座標
CandidateInfoTuple = namedtuple(
    "CandidateInfoTuple",
    ["isNodule_bool",
    "hasAnnotation_bool",
    "isMal_bool",
    "diameter_mm",
    "series_uid",
    "center_xyz"]
)

# cached in-memory
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
    return candidateInfo_list # (結節(悪性腫瘍), 結節(良性), 結節でない)の3つのラベルがごちゃ混ぜ

# cached in-memory
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

    return candidateInfo_dict # series_uidをkeyにしてcandidateInfo_listを整理


class Ct:

    def __init__(self, 
                 series_uid, 
                 raw_datasetdir : str):
        regex_mhd_path = raw_datasetdir + "/subset*/{}.mhd".format(series_uid)
        mhd_path = glob.glob(regex_mhd_path)[0]

        ct_mhd = sitk.ReadImage(mhd_path) # vocel
        self.hu_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) begin -1000 and 1 g/cc (water) begin 0.

        self.series_uid = series_uid

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3,3)

        '''ここからがセマセグ用に追加'''
        # 結節候補データセット
        # cached in-memory
        candidateInfo_list = getCandidateInfoDict(raw_datasetdir = raw_datasetdir)[self.series_uid]

        # 結節データセット
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

        center_irc = xyz2irc(
            center_xyz, 
            self.origin_xyz, 
            self.vxSize_xyz, 
            self.direction_a
            )

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

        ct_chunk = self.hu_a[tuple(slice_list)] # 生データ(ct-vocel)のスライスチャンク
        pos_chunk = self.positive_mask[tuple(slice_list)] # 結節マスクのスライスチャンク

        return ct_chunk, pos_chunk, center_irc
    

# cached in-memory
@functools.lru_cache(1, typed=True)
def getCt(series_uid, raw_datasetdir : str):
    return Ct(series_uid=series_uid, raw_datasetdir=raw_datasetdir)


# cached disk
@raw_cache.memoize(typed=True, tag=tag_version)
def getCtRawCandidate(        
        raw_datasetdir : str, 
        series_uid, 
        center_xyz, 
        width_irc,
        ):
    
    # cached in-memory
    ct = getCt(
            series_uid=series_uid, 
            raw_datasetdir=raw_datasetdir, 
            )
    
    ct_chunk, pos_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    ct_chunk.clip(-1000, 1000, ct_chunk)

    return ct_chunk, pos_chunk, center_irc # center_ircが追加された


# cached disk
@raw_cache.memoize(typed=True, tag=tag_version)
def getCtSampleSize(
    raw_datasetdir : str,
    series_uid,
    ):

    ct = Ct(series_uid, raw_datasetdir)
    return int(ct.hu_a.shape[0]), ct.positive_indexes # [0, N], [pi1, pi2, pi3, ...]



class Luna2dSegmentationDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            raw_datasetdir : str,
            val_stride=0,
            isValSet_bool=None,
            series_uid=None,
            contextSlices_count=3,
            fullCt_bool=False,
    ):
        self.raw_datasetdir = raw_datasetdir
        self.contextSlices_count = contextSlices_count
        self.fullCt_bool = fullCt_bool

        # 続きはここから. '23/5/28
        if series_uid:
            self.series_list = [series_uid]
        else:
            # cached in-memoryy
            self.series_list = sorted(getCandidateInfoDict(raw_datasetdir=self.raw_datasetdir).keys()) # [uid1, uid2...]

        
        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.series_list = self.series_list[::val_stride]
            assert self.series_list
        elif val_stride > 0:
            del self.series_list[::val_stride] # 全体のデータセットからValidationだけを除外
            assert self.series_list

        
        self.sample_list = []
        for series_uid in self.series_list:
            # cached disk
            index_count, positive_indexes = getCtSampleSize(self.raw_datasetdir, series_uid) 

            if self.fullCt_bool:
                self.sample_list += [
                    (series_uid, slice_ndx) for slice_ndx in range(index_count)
                ]
            else:
                self.sample_list += [
                    (series_uid, slice_ndx) for slice_ndx in positive_indexes
                ]
        # finish to make self.sample_list
        
        # cached in-memory
        self.candidateInfo_list = getCandidateInfoList(raw_datasetdir=self.raw_datasetdir) # 結節候補データセット(csv)

        series_set = set(self.series_list) # 高速化
        self.candidateInfo_list = [
            cit for cit in self.candidateInfo_list if cit.series_uid in series_set
        ]
        
        # 結節データセット(csv)
        self.pos_list = [nt for nt in self.candidateInfo_list if nt.isNodule_bool]

        log.info(
            "{!r}: {} {} seires, {} slices, {} nodules".format(
                self,
                len(self.series_list),
                {None: "general", True: "validation", False: "training"}[isValSet_bool],
                len(self.sample_list),
                len(self.pos_list),
            )
        )

    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, ndx):
        series_uid, slice_ndx = self.sample_list[ndx % len(self.sample_list)]
        return self.getitem_fullSlice(series_uid, slice_ndx)
    
    def getitem_fullSlice(self, series_uid, slice_ndx):
        ct = getCt(series_uid, self.raw_datasetdir)
        ct_t = torch.zeros((self.contextSlices_count * 2 + 1,512,512))

        start_ndx = slice_ndx - self.contextSlices_count
        end_ndx = slice_ndx + self.contextSlices_count + 1
        for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
            context_ndx = max(context_ndx, 0)
            context_ndx = min(context_ndx, ct.hu_a.shape[0] - 1)
            ct_t[i] = torch.from_numpy(ct.hu_a[context_ndx].astype(np.float32))

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
        # The upper bound nukes any weird hotspots and clamps bone down
        ct_t.clamp_(-1000, 1000)

        pos_t = torch.from_numpy(ct.positive_mask[slice_ndx]).unsqueeze(0) # (I,H,W) -> (1,I,H,W)

        return ct_t, pos_t, ct.series_uid, slice_ndx
    

class TrainingLuna2dSegmentationDataset(Luna2dSegmentationDataset):

    def __init__(self, raw_datasetdir, *args, **kwargs):
        super().__init__(raw_datasetdir, *args, **kwargs)

        self.ratio_int = 2

    def __len__(self):
        return 300000 # 30万
    
    def shuffleSamples(self):
        random.shuffle(self.candidateInfo_list)
        random.shuffle(self.pos_list)
    
    def __getitem__(self, ndx):
        candidateInfo_tup = self.pos_list[ndx % len(self.pos_list)]
        return self.getitem_trainingCrop(candidateInfo_tup)
    
    def getitem_trainingCrop(self, candidateInfo_tup):
        # cacked disk
        ct_a, pos_a, center_irc = getCtRawCandidate(
            self.raw_datasetdir,
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            (7, 96, 96)
        )

        pos_a = pos_a[3:4] # 結節候補Indexチャンクの中心

        row_offset = random.randrange(0, 32)
        col_offset = random.randrange(0, 32)
        ct_t = torch.from_numpy(
            ct_a[:, row_offset : row_offset + 64, col_offset : col_offset + 64]
        ).to(torch.float32)
        pos_t = torch.from_numpy(
            pos_a[:, row_offset : row_offset + 64, col_offset : col_offset + 64]
        ).to(torch.long) # long

        slice_ndx = center_irc.index

        return ct_t, pos_t, candidateInfo_tup.series_uid, slice_ndx
    

class PrepcacheLunaDataset(torch.utils.data.Dataset):

    def __init__(self, raw_datasetdir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_datasetdir = raw_datasetdir

        # cached in-memory
        self.candidateInfo_list = getCandidateInfoList(raw_datasetdir=self.raw_datasetdir)
        self.pos_list = [nt for nt in self.candidateInfo_list if nt.isNodule_bool]

        self.seen_set = set()
        self.candidateInfo_list.sort(key=lambda x: x.series_uid)

    def __len__(self):
        return len(self.candidateInfo_list)
    
    def __getitem__(self, ndx):
        # candidate_t, pos_t, series_uid, center_t = super().__getitem__(ndx)

        candidateInfo_tup = self.candidateInfo_list[ndx]
        # cached disk
        getCtRawCandidate(
            self.raw_datasetdir,
            candidateInfo_tup.series_uid, 
            candidateInfo_tup.center_xyz, 
            (7, 96, 96)
        )

        series_uid = candidateInfo_tup.series_uid
        if series_uid not in self.seen_set:
            self.seen_set.add(series_uid)

            getCtSampleSize(self.raw_datasetdir,
                            series_uid)
            # ct = getCt(self.raw_datasetdir, series_uid)
            # for mask_ndx in ct.positive_indexes:
            #     build2dLungMask(series_uid, mask_ndx)

        return 0, 1  # candidate_t, pos_t, series_uid, center_t
    

class TvTrainingLuna2dSegmentationDataset(torch.utils.data.Dataset):

    def __init__(self,
                 isValSet_bool=False,
                 val_stride=10,
                 contextSlices_count=3,
                 ):
        assert contextSlices_count == 3
        data = torch.load("./imgs_and_masks.pt")
        suids = list(set(data['suids']))
        trn_mask_suids = torch.arange(len(suids)) % val_stride < (val_stride - 1)
        trn_suids = {s for i, s in zip(trn_mask_suids, suids) if i}
        trn_mask = torch.tensor([(s in trn_suids) for s in data['suids']])
        if not isValSet_bool:
            self.imgs = data['imgs'][trn_mask]
            self.masks = data['masks'][trn_mask]
            self.suids = [s for s, i in zip(data['suids'], trn_mask) if i] # if s == i
        else:
            self.imgs = data['imgs'][~trn_mask]
            self.masks = data['masks'][~trn_mask]
            self.suids = [s for s, i in zip(data['suids'], trn_mask) if not i] # if s != i
        
        # discard spurious hotspots and clamp bone
        self.imgs.clamp_(-1000, 1000)
        self.imgs /= 1000

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, i):
        oh, ow = torch.randint(0, 32, (2,))
        s1 = self.masks.size(1) // 2
        return (
            self.imgs[i, :, oh : oh + 64, ow : ow + 64],
            1,
            self.masks[i, s1 : s1 + 1, oh : oh + 64, ow : ow + 64].to(torch.float32),
            self.suids[i],
            9999,
        )