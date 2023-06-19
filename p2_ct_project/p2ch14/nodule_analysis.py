"""各CTに対して、セグメンテーション、グループ化、結節候補の分類、腫瘍の分類
"""

import os
import sys


# os.sepはプラットフォーム固有の区切り文字(Windows: `\`, Unix: `/`)
module_parent_dir = os.sep.join([os.path.dirname(__file__), '..']) # p2_ct_project
print("module_parent_dir", module_parent_dir)
sys.path.append(module_parent_dir)

import argparse
import glob


import numpy as np
import scipy.ndimage.measurements as measurements
import scipy.ndimage.morphology as morphology

import torch
import torch.nn as nn
import torch.optim

from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from util.logconf import logging
from util.util import xyz2irc, irc2xyz

from p2ch13.dataset_luna_seg import Luna2dSegmentationDataset
from p2ch14.dataset_luna_end_to_end import (
    LunaDataset,
    getCt,
    getCandidateInfoDict,
    getCandidateInfoList,
    CandidateInfoTuple,
)
from p2ch13.model_unet import UNetWrapper

import p2ch11
import p2ch14


log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)
logging.getLogger("p2ch13.dataset_luna_seg").setLevel(logging.WARNING)
logging.getLogger("p2ch14.dataset_luna_end_to_end").setLevel(logging.WARNING)


def print_confusion(label, confusions, do_mal):
    row_labels = ['Non-Nodules', 'Benign', 'Malignant']

    if do_mal:
        col_labels = ['', 'Complete Miss', 'Filtered Out', 'Pred. Benign', 'Pred. Malignant']
    else:
        col_labels = ['', 'Complete Miss', 'Miltered Out', 'Pred. Nodule']
        confusions[:, -2] += confusions[:, -1]
        confusions = confusions[:, :-1]
    cell_width = 16
    f = '{:>}' + str(cell_width) + '}'
    print(label)
    print(' | '.join([f.format(s) for s in col_labels]))
    for i, (l,r) in enumerate(zip(row_labels, confusions)):
        r = [l] + list(r)
        if i == 0:
            r[1] = ''
        print(' | '.join([f.format(i) for i in r]))


def match_and_score(detections, truth, threshold=0.5, threshold_mal=0.5):
    # Returns 3x4 confusion matrix for:
    # Rows: Truth: Non-Nodules, Benign, Malignant
    # Cols: Not Detected, Detected by Seg, Detected as Benign, Detected as Malignant
    # If one true nodule matches multiple detections, the "highest" detection is considered
    # If one detection matches several true nodule annotations, it counts for all of them
    true_nodules = [c for c in truth if c.isNodule_bool]
    truth_diams = np.array([c.diameter_mm for c in true_nodules])
    truth_xyz = np.array([c.center_xyz for c in true_nodules])


    
    # cls_tup = (prob_nodule, prob_mal, center_xyz, center_irc)
    # detections: classifications_list.append(cls_tup)
    detected_xyz = np.array([n[2] for n in detections]) # セマセグモデルによる推論結果のxyz座標
    # detection classes will contain
    # 1 -> detected by seg but filtered by cls
    # 2 -> detected as benign nodule (or nodule if no malignancy model is used)
    # 3 -> detected as malignant nodule (if applicable)
    detected_classes = np.array([1 if d[0] < threshold
                                 else (2 if d[1] < threshold
                                       else 3) for d in detections])
    
    confusion = np.zeros((3, 4), dtype=np.int32)

    if len(detected_xyz) == 0: # セマセグ器で見逃し(Not Detected)
        for tn in true_nodules:
            confusion[2 if tn.isMal_bool else 1, 0] += 1 # Benign or Malignant
    elif len(truth_xyz) == 0: # 過検出(Detected by seg, Detected as Benign, Detected as Malignant)
        for dc in detected_classes:
            confusion[0, dc] += 1
    else: 
        # 推論結果xyzとGTxyzの距離をGT直径で割った値
        normalized_dicts = np.linalg.norm(
            truth_xyz[:,None] - detected_xyz[None], ord=2, axis=-1) / truth_diams[:, None]
        
        # 0.7 未満をマッチングしたと見なす
        matches = (normalized_dicts < 0.7)

        unmatched_detections = np.ones(len(detections), dtype=np.bool8)
        matched_true_nodules = np.zeros(len(true_nodules), dtype=np.int32)
        for i_tn, i_detection in zip(*matches.nonzero()): # ここがわからない
            matched_true_nodules[i_tn] = max(matched_true_nodules[i_tn], detected_classes[i_detection]) # セマセグ器と結節検出器の予測確率の大きい方を採用
            unmatched_detections[i_detection] = False
        
        # 過検出
        for ud, dc in zip(unmatched_detections, detected_classes):
            if ud:
                confusion[0, dc] += 1

        # セマセグ器or分類器で検出した結果
        for tn, dc in zip(true_nodules, matched_true_nodules):
            confusion[2 if tn.isMal_bool else 1, dc] += 1

    return confusion
    

class NoduleAnalysisApp:

    def __init__(self, sys_argv=None):
        if sys_argv is None:
            log.debug(sys.argv)
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
                            help='Batch size to use for training.',
                            default=4,
                            type=int,
                            )
        parser.add_argument('--num-workers',
                            help='Number of worker processes for background data loading.',
                            default=4,
                            type=int,
                            )
        parser.add_argument('--run-validation',
                            help='Run over validation rather than a single CT.',
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--include-train',
                            help='Include data that was in the training set. (default: validation data only).',
                            action='store_true',
                            default=False,
                            )
        parser.add_argument('--segmentation-path',
                            help='Path to the saved segmentation model',
                            nargs='?',
                            default=None,
                            )
        parser.add_argument('--cls-model',
                            help='What to model class name to user for the classifier.',
                            action='store',
                            default='LunaModel',
                            )
        parser.add_argument('--malignancy-model',
                            help='What to model class name to user for the malignancy classifier',
                            action='store',
                            default='LunaModel',
                            # default='ModifiedLunaModel',
                            )
        parser.add_argument('--malignancy-path',
                            help='Path to the saved malignancy classification model',
                            nargs='?',
                            default=None,
                            )
        parser.add_argument('--tb-prefix',
                            default='p2ch14',
                            help='Data prefix to user for Tensorbard run. Defaults to chapter.',
                            )
        parser.add_argument('series_uid',
                            nargs='?',
                            default=None,
                            help='Series UID to use.',
                            )
        # rawdatasetdir
        parser.add_argument('--datasetdir',
                            help="Luna raw dataset directory",
                            default='E:/Luna16',
                            )
        
        
        self.cli_args = parser.parse_args(sys_argv)
        # self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        if not (bool(self.cli_args.series_uid) ^ self.cli_args.run_validation):
            raise Exception('One and only one of series_uid and --run-validation should be given.')
        
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')

        if not self.cli_args.segmentaiton_path:
            self.cli_args.segmentation_path = self.initModelPath('seg')

        if not self.cli_args.classification_path:
            self.cli_args.classification_path = self.initModelPath('cls')

        self.seg_model, self.cls_model, self.malignancy_model = self.initModels()


    def initModelPath(self, type_str):
        local_path = os.path.join(
            'model',
            'p2ch13',
            'data-unversioned',
            type_str + '_{}_{}_{}.state'.format('*', '*', 'best'),
        )

        file_list = glob.glob(local_path)
        if not file_list:
            pretrained_path = os.path.join(
                'model',
                'data',
                type_str + '_{}_{}_{}'.format('*', '*', '*')
            )
            file_list = glob.glob(pretrained_path)
        else:
            pretrained_path = None
        
        file_list.sort()

        try:
            return file_list[-1]
        except IndexError:
            log.debug(self.cli_args.segmentaiton_path)
            raise
            
    def initModels(self):
        # Segmentaiton
        log.debug(self.cli_args.segmentaiton_path)
        seg_dict = torch.load(self.cli_args.segmentation_path)

        seg_model = UNetWrapper(
            in_channels=7,
            n_classes=1,
            depth=3,
            wf=4,
            padding=True,
            batch_norm=True,
            up_mode='upconv',
        )

        seg_model.load_state_dict(seg_dict['model_state'])
        seg_model.eval()

        # Classification
        log.debug(self.cli_args.classification_path)
        cls_dict = torch.load(self.cli_args.classification_path)

        model_cls = getattr(p2ch11.model_luna, self.cli_args.cls_model) # モデル名を取得
        cls_model = model_cls() # モデルをインスタンス化
        cls_model.load_state_dict(cls_dict['model_state'])
        cls_model.eval()

        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                # nn.DataParallelでラッピング
                seg_model = nn.DataParallel(seg_model)
                cls_model = nn.DataParallel(cls_model)

            seg_model.to(self.device)
            cls_model.to(self.device)

        # Malignancy
        if self.cli_args.malignancy_path:
            model_cls = getattr(p2ch14.model_malignancy, self.cli_args.malignancy_model)
            malignancy_model = model_cls()
            malignancy_dict = torch.load(self.cli_args.malignancy_path)
            malignancy_model.load_state_dict(malignancy_dict['model_state'])
            malignancy_model.eval()
            if self.use_cuda:
                malignancy_model.to(self.device)
        else:
            malignancy_model = None
        
        return seg_model, cls_model, malignancy_model
    

    def initSegmentationDl(self, series_uid):
        seg_ds = Luna2dSegmentationDataset(
            raw_datasetdir=self.cli_args.datasetdir,
            contextSlices_count=3,
            series_uid=series_uid,
            fullCt_bool=True,
        )
        seg_dl = DataLoader(
            seg_ds,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return seg_dl
    
    def initClassificationDl(self, candidateInfo_list):
        cls_ds = LunaDataset(
            raw_datasetdir=self.cli_args.datasetdir,
            sortby_str='series_uid',
            candidateInfo_list=candidateInfo_list,
        )
        cls_dl = DataLoader(
            cls_ds,
            batch_size=self.cli_args.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return cls_dl
    

    def main(self):
        log.info('Starting {}, {}'.format(type(self).__name__, self.cli_args))

        # 検証データセット
        val_ds = LunaDataset(
            raw_datasetdir=self.cli_args.datasetdir,
            val_stride=10,
            isValSet_bool=True,
        )
        
        # 検証データseries_uid
        val_set = set(
            candidateInfo_tup.series_uid
            for candidateInfo_tup in val_ds.candidateInfo_list
        )

        # 結節データseries_uid
        positive_set = set(
            candidateInfo_tup.series_uid
            for candidateInfo_tup in getCandidateInfoList(raw_datasetdir=self.cli_args.datasetdir)
            if candidateInfo_tup.isNodule_bool
        )

        # 全体データseries_uid
        if self.cli_args.series_uid:
            series_set = set(self.cli_args.series_uid.split(','))
        else:
            series_set = set(
                candidateInfo_tup.series_uid
                for candidateInfo_tup in getCandidateInfoList(raw_datasetdir=self.cli_args.datasetdir)
            )
        
        # 訓練データseries_uid
        if self.cli_args.include_train:
            train_list = sorted(series_set - val_set)
        else:
            train_list = []

        # 検証データseries_uid
        val_list = sorted(series_set & val_set) # これよくわからnn


        candidateInfo_dict = getCandidateInfoDict(raw_datasetdir=self.cli_args.datasetdir)

        series_iter = enumerateWithEstimate(
            val_list + train_list,
            'Series',
        )

        all_confusion = np.zeros((3,4), dtype=np.int32)

        # series_uidについてループ
        for _, series_uid in series_iter:

            ct = getCt(series_uid, raw_datasetdir=self.cli_args.datasetdir)

            mask_a = self.segmentCt(ct, series_uid) # セマセグ器を実行

            # 推論したセマセグアノテーション(結節候補)をグループ化
            candidateInfo_list = self.groupSegmentationOutput(
                series_uid,
                ct,
                mask_a,
            )

            # 結節候補分類器を実行
            classifications_list = self.classifyCandidates(
                ct,
                candidateInfo_list,
            )

            if not self.cli_args.run_validation:
                print(f'found nodule candidates in {series_uid}:')
                for prob, prob_mal, center_xyz, center_irc in classifications_list:
                    if prob > 0.5:
                        s = f'nodule prob {prob:.3f}, '
                        if self.malignancy_model:
                            s += f'malignancy prob {prob_mal:.3f}, '
                        s += f'center xyz {center_xyz}'
                        print(s)
            
            if series_uid in candidateInfo_dict:
                one_confusion = match_and_score(
                    classifications_list,
                    candidateInfo_dict[series_uid],
                )
                all_confusion += one_confusion
                print_confusion(
                    series_uid,
                    one_confusion,
                    self.malignancy_model is not None
                )

        print_confusion(
            'Total',
            all_confusion,
            self.malignancy_model is not None
        )


    def classifyCandidates(self,
                           ct,
                           candidateInfo_list,
                           ):
        
        # 結節候補に関するリストに基づいたデータローダーを作成
        cls_dl = self.initClassificationDl(candidateInfo_list)
        classifications_list = []
        for batch_nd, batch_tup in enumerate(cls_dl):
            input_t, _, _, series_list, center_list = batch_tup

            input_g = input_t.to(self.device)
            with torch.no_grad():
                # 結節分類器の実行
                _, probability_nodule_g = self.cls_model(input_g)

                # 悪性腫瘍モデルがあれば, それも実行
                if self.malignancy_model is not None:
                    _, probability_mal_g = self.malignancy_model(input_g)
                else:
                    probability_mal_g = torch.zeros_like(probability_nodule_g)
            
            zip_iter = zip(center_list,
                           probability_nodule_g[:,1].tolist(),
                           probability_mal_g[:,1].tolist(),
                           )
            
            for center_irc, prob_nodule, prob_mal in zip_iter:
                center_xyz = irc2xyz(
                    center_irc,
                    direction_a=ct.direction_a,
                    origin_xyz=ct.origin_xyz,
                    vxSize_xyz=ct.vxSize_xyz,
                )
                cls_tup = (prob_nodule, prob_mal, center_xyz, center_irc)
                classifications_list.append(cls_tup)
        return classifications_list
    

    def segmentCt(self,
                  ct,
                  series_uid,
                  ):
        
        # 勾配計算を必要としないのでグラフは作らない
        with torch.no_grad():
            output_a = np.zeros_like(ct.hu_a, dtype=np.float32)

            # CTをバッチ毎にループさせるデータローダ
            seg_dl = self.initSegmentationDl(series_uid)


            # [ct_t, pos_t, series_uid, slice_ndx]のバッチ
            # input_t : N個のc_t_t
            for input_t, _, _, slice_ndx_list in seg_dl:
                input_g = input_t.to(self.device) # to GPU

                # semantic segmentaiton
                prediction_g = self.seg_model(input_g)

                for i, slice_ndx in enumerate(slice_ndx_list):
                    output_a[slice_ndx] = prediction_g[i].cpu().numpy()

            # 確率値の閾値処理. その後, erosion(収縮)処理
            mask_a = output_a > 0.5
            mask_a = morphology.binary_erosion(mask_a, iterations=1)

            return mask_a
        
    
    def groupSegmentationOutput(self, 
                                series_uid, 
                                ct,
                                clean_a,
                                ):
        # 3Dラベリング
        candidateLabel_a, candidate_count = measurements.label(clean_a)

        # 各結節候補の塊の中心インデックス(I,R,C)
        centerIrc_list = measurements.center_of_mass(
            ct.hu_a.clip(-1000, 1000) + 1001,
            labels=candidateLabel_a,
            index=np.arange(1, candidate_count + 1),
        )

        # 結節候補を示すタプルを構築し, 検出リストに追加
        candidateInfo_list = []
        for i, center_irc in enumerate(centerIrc_list):
            center_xyz = irc2xyz(
                center_irc,
                ct.origin_xyz,
                ct.vxSize_xyz,
                ct.direction_a,
            )
            assert np.all(np.isfinite(center_irc)), repr(['irc', center_irc, i, candidate_count])
            assert np.all(np.isfinite(center_xyz)), repr(['xyz', center_xyz])
            candidateInfo_tup = CandidateInfoTuple(False, False, False, 0.0,series_uid, center_xyz)

            candidateInfo_list.append(candidateInfo_tup)


        return candidateInfo_list
    
    def logResult(self,
                  mode_str,
                  filtered_list,
                  series2iagnosis_dict,
                  positive_set,
                  ):
        count_dict = {
            'tp':0, # 真陽性
            'tn':0, # 偽陽性
            'fp':0, # 真陰性
            'fn':0, # 偽陰性
            }
    




        
        










    





