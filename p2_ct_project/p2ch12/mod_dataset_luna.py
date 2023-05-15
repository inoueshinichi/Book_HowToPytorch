"""データセット内でクラスバランスを調整する
    ・訓練セットの陽性サンプル数と陰性サンプル数が1対1になるようにする
    ・陰性サンプルのリストと陽性サンプルのリストを作成し, これらのリストから交互にサンプルを取り出すようにする.
    ・副次効果として, ミニバッチの1バッチあたりのデータ比率が1対1になり, イテレーションごとにNNモデルが適切に学習できる.
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

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

raw_cache = getCache("part2ch12_raw")

# 結節の状態(分類対象), 直径, 識別子, 結節候補の中心座標
CandidateInfoTuple = namedtuple(
    'CandidateInfoTuple',
    'isNodule_bool, diameter_mm, series_uid, center_xyz',
)


# Make CandidateInfoTuple with Helper 関数 for loading luna dataset
@functools.lru_cache(1) # インメモリでキャッシングを行う標準的なライブラリ : getCandidateInfoList関数の結果をメモリキャッシュに保存する
def getCandidateInfoList(requireOnDisk_bool=True, datasetdir: str = ""):
    # We construct a set with all series_uids that are present on disk.
    # This will let us use the data, even if we haven't downloaded all of
    # the subsets yet.

    # 生データ(3D)
    regex_mhd_path = datasetdir + "/subset*/*.mhd"
    mhd_list = glob.glob(regex_mhd_path)
    presentOnDisk_set = { os.path.split(p)[-1][:-4] for p in mhd_list } # filenames

    # アノテーション(annotations.csv)から直径の情報をマージする
    diameter_dict = {}
    anno_path = datasetdir + "/annotations.csv"
    with open(anno_path, 'r') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])

            diameter_dict.setdefault(series_uid, []).append(
                (annotationCenter_xyz, annotationDiameter_mm)
            )
    
    # CandidateデータをAnnoatationデータでフィルタリングする
    # presetOnDisk_setに存在する`series_uid`のみを扱う
    # AnnotationとCandidateで各中心座標(x,y,z)の距離がAnnotationDiameter_mmの1/4以下のもののみを扱う
    candidateInfo_list = []
    cand_path = datasetdir + "/candidates.csv"
    with open(cand_path, 'r') as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                # series_uidが存在しない場合, ディクス上にないサブセットデータなのでスキップ.
                continue

            isNodule_bool = bool(int(row[4])) # 0 : 結節ではない, 1 : 結節である
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

            candidateDiameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []):
                annotationCenter_xyz, annotationDiameter_mm = annotation_tup
                for i in range(3):
                    delta_mm = abs(candidateCenter_xyz[i] - annotationCenter_xyz[i])
                    if delta_mm > annotationDiameter_mm / 4:
                        # 直径を2で除算して半径を求めた上で,
                        # 半径を2で除算することで,
                        # 結節に対する2つのファイルの中心座標のズレが
                        # 結節の大きさに対して大きくないことを確認する.
                        # (これは, 距離をチェックしているのではなく, アノテーションのバウンディングボックスの正常性をチェックしている)
                        break
                else:
                    candidateDiameter_mm = annotationDiameter_mm
                    break

            candidateInfo_list.append(CandidateInfoTuple(
                isNodule_bool,
                candidateDiameter_mm,
                series_uid,
                candidateCenter_xyz,
            ))

    candidateInfo_list.sort(reverse=True)
    # これにより, 最も大きいサイズのものから始まる実際の結節サンプルに続いて, 
    # (結節のサイズの情報を持たない)結節でないサンプルが続くことになる.
    return candidateInfo_list


# CTインスタンス
class Ct:
    # series_uidでデータを指定
    def __init__(self, series_uid, datasetdir:str):
        regex_mhd_path = datasetdir + "/subset*/{}.mhd".format(series_uid)
        mhd_path = glob.glob(regex_mhd_path)[0]

        # sitk.ReadImageは, 与えられた*.mhdファイルに加えて, *.rawファイルも暗黙的に使用する.
        ct_mhd = sitk.ReadImage(mhd_path) # numpy
        ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32) # as np.float32 (D,H,W)

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        # The lower bound gets rid of negative density stuff used to indicate out-of-FOV
        # The upper bound nukes any weird hotspots and clamps bone down
        ct_a.clip(-1000, 1000, ct_a) # ボクセル値のクリップ CTの慣例に従う

        self.series_uid = series_uid
        self.hu_a = ct_a # ハンスフィールド単位

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def getRawCandidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vxSize_xyz,
            self.direction_a
        )

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], \
                repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])
            
            if start_ndx < 0:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width_irc[axis])
            
            if end_ndx > self.hu_a.shape[axis]:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.hu_a.shape, width_irc))
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]

        return ct_chunk, center_irc


@functools.lru_cache(1, typed=True)
def getCt(series_uid, datasetdir:str):
    return Ct(series_uid, datasetdir=datasetdir)


@raw_cache.memoize(typed=True)
def getCtRawCandidate(datasetdir:str, series_uid, center_xyz, width_irc):
    ct = getCt(series_uid, datasetdir)
    ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
    return ct_chunk, center_irc


'''データ拡張関数'''
def getCtAugmentedCandidate(
    datasetdir: str, 
    augmentation_dict, 
    series_uid, 
    center_xyz, 
    width_irc, 
    use_cache=True
):
    if use_cache:
        ct_chunk, center_irc = getCtRawCandidate(datasetdir, series_uid, center_xyz, width_irc)
    else:
        ct = getCt(series_uid)
        ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)

    ct_t = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)

    # アフィン変換
    transform_t = torch.eye(4)
    # ... <1>

    # x,y,z
    for i in range(3):
        if "flip" in augmentation_dict:
            if random.random() > 0.5:
                transform_t[i, i] *= -1

        if "offset" in augmentation_dict:
            offset_float = augmentation_dict["offset"]
            random_float = random.random() * 2 - 1
            transform_t[i, 3] = offset_float * random_float # (tx,ty,tz)

        if "scale" in augmentation_dict:
            scale_float = augmentation_dict["scale"]
            random_float = random.random() * 2 - 1
            transform_t[i, i] *= 1.0 + scale_float * random_float # 1.0 ~ 2.0

    if "rotate" in augmentation_dict:
        angle_rad = random.random() * math.pi * 2
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)

        # x軸とy軸は同じスケールだが, z軸だけスケールが異なる(ボクセルは立方体でない.直方体)
        # 回転はxy平面に限定するべし.
        rotation_t = torch.tensor(
            [
                [c, -s, 0, 0],
                [s, c, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        transform_t @= rotation_t # (4,4)

    affine_t = F.affine_grid(
        transform_t[:3].unsqueeze(0).to(torch.float32), # (N,2,3)
        ct_t.size(), # output image size
        align_corners=False,
    )

    augmented_chunk = F.grid_sample(
        ct_t,
        affine_t,
        padding_mode="border",
        align_corners=False,
    ).to("cpu")

    if "noise" in augmentation_dict:
        noise_t = torch.randn_like(augmented_chunk)
        noise_t *= augmentation_dict["noise"]

        augmented_chunk += noise_t

    return augmented_chunk[0], center_irc




# データバランス調整済みLunaDataset
import torch
import torch.cuda
from torch.utils.data import Dataset

class LunaDataset(Dataset):

    def __init__(self,
                 datasetdir:str,
                 val_stride=0,
                 isValSet_bool=None,
                 series_uid=None,
                 sortby_str="random",
                 ratio_int=0,
                 augmentation_dict=None,
                 candidateInfo_list=None,
                 ):
        self.datasetdir = datasetdir
        self.ratio_int = ratio_int
        self.augmentation_dict = augmentation_dict

        if candidateInfo_list:
            self.candidateInfo_list = copy.copy(candidateInfo_list)
            self.use_cache = False
        else:
            self.candidateInfo_list = copy.copy(getCandidateInfoList(datasetdir=self.datasetdir))
            self.use_cache = True


        if series_uid:
            self.candidateInfo_list = [
                x for x in self.candidateInfo_list if x.series_uid == series_uid
            ]
        
        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.candidateInfo_list = self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list
        elif val_stride > 0:
            del self.candidateInfo_list[::val_stride]
            assert self.candidateInfo_list

        if sortby_str == "random":
            random.shuffle(self.candidateInfo_list)
        elif sortby_str == "series_uid":
            self.candidateInfo_list.sort(key=lambda x: (x.series_uid, x.center_xyz))
        elif sortby_str == "label_and_size":
            pass
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))


        # 陰性サンプルのリスト
        self.negative_list = [ 
            nt for nt in self.candidateInfo_list if not nt.isNodule_bool 
        ]

        # 陽性サンプルのリスト
        self.positive_list = [
            nt for nt in self.candidateInfo_list if nt.isNodule_bool
        ]

        log.info(
            "{!r}: {} {} samples, {} neg, {} pos, {} ratio".format(
                self,
                len(self.candidateInfo_list),
                "validation" if isValSet_bool else "training",
                len(self.negative_list),
                len(self.positive_list),
                "{}:1".format(self.ratio_int) if self.ratio_int else "unbalanced",
            )
        )


    def shuffleSamples(self):
        # 各エポックの最初にこの関数を実行し, サンプルの順序をランダムに変更する
        if self.ratio_int:
            random.shuffle(self.negative_list)
            random.shuffle(self.positive_list)
    

    def __len__(self):
        if self.ratio_int:
            # 訓練セットのバランスを取るために陽性サンプルを何度も繰り返し使うので,
            # 実際のサンプル数は関係がなくなり, 「エポック全体」が本来の意味を成さなくなった.
            # All : 551,065個, Pos : 1,351個, Neg : 549,714個
            return 200000 # 必ず必要ではないが, ハードコードする. 単純に訓練開始から結果が出るまでを早くする.
        else:
            return len(self.candidateInfo_list)
    

    def __getitem__(self, ndx):

        # バッチ内のデータ比率を整える(ratio_int=1の場合, 陽:陰=1:1, ratio_int=2の場合, 陽:陰=1:2)
        # 比率内個数 self.ratio_int + 1, 1:2の場合 2+1 = 3
        '''
        ratio_int = 2
        DS Index 0 1 2 3 4 5 6 7 8 9 ...
        Label    + - - + - - + - - +
        PosIndex 0     1     2     3
        NegIndex   0 1   2 3   4 5
        
        '''
        if self.ratio_int:
            pos_ndx = ndx // (self.ratio_int + 1)

            if ndx % (self.ratio_int + 1): # 余りが0でない場合は陰性サンプル
                neg_ndx = ndx - 1 - pos_ndx
                neg_ndx %= len(self.negative_list) # オーバーフロー対策, 陰性サンプルリストを一周したら0からスタート
                candidateInfo_tup = self.negative_list[neg_ndx]
            else:
                pos_ndx %= len(self.positive_list) # オーバーフロー対策, 陽性サンプルリストを一周したら0からスタート
                candidateInfo_tup = self.positive_list[pos_ndx]

        else:
            # バッチ内データ比率を揃えないデフォルト状態
            # バランスを取らない場合(ratio_int=0), N番目のサンプルをシンプルに返す.
            candidateInfo_tup = self.candidateInfo_list[ndx]


        width_irc = (32, 48, 48)

        # @warning データ拡張を行う前にキャッシュを作成すること.
        # さもないと, データ拡張が1回しか行われないことになる.
        # 最初のデータ拡張の結果をキャッシュしてしまうため.
        candidate_a, center_irc = getCtRawCandidate(
            self.datasetdir,
            candidateInfo_tup.series_uid,
            candidateInfo_tup.center_xyz,
            width_irc,
        )

        candidate_t = torch.from_numpy(candidate_a)
        candidate_t = candidate_t.to(torch.float32)
        candidate_t = candidate_t.unsqueeze(0)

        if self.augmentation_dict:
            candidate_t, center_irc = getCtAugmentedCandidate(
                self.datasetdir,
                self.augmentation_dict,
                candidateInfo_tup.series_uid,
                candidateInfo_tup.center_xyz,
                width_irc,
                self.use_cache,
            )
        elif self.use_cache:
            candidate_a, center_irc = getCtRawCandidate(
                self.datasetdir,
                candidateInfo_tup.series_uid,
                candidateInfo_tup.center_xyz,
                width_irc,
            )
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)
        else:
            ct = getCt(candidateInfo_tup.series_uid)
            candidate_a, center_irc = ct.getRawCandidate(
                candidateInfo_tup.center_xyz,
                width_irc,
            )
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)

        pos_t = torch.tensor([
            not candidateInfo_tup.isNodule_bool,
            candidateInfo_tup.isNodule_bool
            ],
            dtype=torch.long
        )

        return (
            candidate_t, 
            pos_t, 
            candidateInfo_tup.series_uid, 
            torch.tensor(center_irc),
        )

