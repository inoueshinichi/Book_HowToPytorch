"""LunaTrainingApp
"""

import os
import sys

# os.sepはプラットフォーム固有の区切り文字(Windows: `\`, Unix: `/`)
module_parent_dir = os.sep.join([os.path.dirname(__file__), '..']) # p2_ct_project
print("module_parent_dir", module_parent_dir)
sys.path.append(module_parent_dir)

import argparse
import datetime

import numpy as np

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from util.logconf import logging
from p2ch10.dataset_luna import LunaDataset
from p2ch11.model_luna import LunaModel

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

# Used for computeBatchLoss and logMetrics to index into metrics_t/metrics_a
METRICS_LABEL_NDX=0
METRICS_PRED_NDX=1 # 陽性尤度
METRICS_LOSS_NDX=1
METRICS_SIZE=3

class LunaTrainingApp:

    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,
        )
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=32,
            type=int,    
        )
        parser.add_argument('--epochs',
            help='Number of epochs to train for',
            default=1,
            type=int,
        )
        parser.add_argument('--tb-prefix',
            help='Data prefix to use for Tensorboard run. Defaults to chapter',
            default='p2ch11',    
        )
        parser.add_argument('comment',
            help='Comment suffix for Tensorboard run',
            nargs='?',
            default='dwlpt',
        )
        # データセットディレクトリの指定
        parser.add_argument('--datasetdir',
            help="Luna raw dataset directory",
            default='E:/Luna16'
        )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

    def initModel(self):
        model = LunaModel()
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model
    
    def initOptimizer(self):
        return SGD(self.model.parameters(), lr=0.001, momentum=0.99)
        # return Adam(self.model_parameters())
    
    def initTrainDl(self):
        datasetdir = self.cli_args.datasetdir # データセットディレクトリ
        train_ds = LunaDataset(raw_datasetdir=datasetdir, 
                               val_stride=0, 
                               isValSet_bool=False)

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
        
        train_dl = DataLoader(
            dataset=train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
        )

        return train_dl
    
    def initValDl(self):
        datasetdir = self.cli_args.datasetdir # データセットディレクトリ
        val_ds = LunaDataset(raw_datasetdir=datasetdir,
                             val_stride=10, 
                             isValSet_bool=True)

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            dataset=val_ds,
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

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        for epoch_ndx in range(1, self.cli_args.epochs + 1):

            log.info("Epoch {} of {}, train: {} / validation: {} batches of size {}*{}".format(
                epoch_ndx,
                self.cli_args.epochs,
                len(train_dl),
                len(val_dl),
                self.cli_args.batch_size,
                (torch.cuda.device_count() if self.use_cuda else 1)
            ))

            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            valMetrics_t = self.doValidation(epoch_ndx, val_dl)
            self.logMetrics(epoch_ndx, 'val', valMetrics_t)

        if hasattr(self, 'trn_writer'):
            self.trn_writer.close()
            self.val_writer.close()

    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()

        # 評価指標用にからの配列を初期化
        trnMetrics_g = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
        )

        # 終了時間予測付きのバッチループ作成
        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            # 勾配テンソルを0に初期化
            self.optimizer.zero_grad()

            loss_var = self.computeBatchLoss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                trnMetrics_g
            )

            # モデルの重みを更新
            loss_var.backward()
            self.optimizer.step()

            # # This is for adding the model graph to TensorBoard.
            # if epoch_ndx == 1 and batch_ndx == 0:
            #     with torch.no_grad():
            #         model = LunaModel()
            #         self.trn_writer.add_graph(model, batch_tup[0], verbose=True)
            #         self.trn_writer.close()

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trnMetrics_g.to('cpu')
    
    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            batch_iter = enumerateWithEstimate(
                val_dl,
                "E{} Validation ".format(epoch_ndx),
                start_ndx=val_dl.num_workers,
            )
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(
                    batch_ndx,
                    batch_tup, 
                    val_dl.batch_size,
                    valMetrics_g
                )

        return valMetrics_g.to('cpu')

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t, _series_list, _center_list = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g, probability_g = self.model(input_g)

        # reduction='none'でサンプル毎の損失を計算
        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_g = loss_func(
            logits_g, 
            label_g[:, 1],
        )

        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0) # ミニバッチ数(端数あり)

        # 勾配を必要とする指標がないのでデタッチして, 計算グラフから切り離す.
        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_g[:,1].detach()
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = probability_g[:,1].detach() # 陽性尤度
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g.detach()

        return loss_g.mean() # サンプル毎の損失を1バッチ分に平均化
    

    def logMetrics(self, 
                   epoch_ndx, 
                   mode_str,
                   metrics_t, 
                   classificationThreshold=0.5,
        ):

        self.initTensorboardWriters()
        log.info("E{} {}".format(epoch_ndx, type(self).__name__))

        negLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshold
        negPred_mask = metrics_t[METRICS_PRED_NDX] <= classificationThreshold

        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask

        # Label count
        neg_count = int(negLabel_mask.sum())
        pos_count = int(posLabel_mask.sum())

        # Pred accuracy
        trueNeg_count = neg_correct = int((negLabel_mask & negPred_mask).sum())
        truePos_count  = pos_correct = int((posLabel_mask & posPred_mask).sum())
        falsePos_count = neg_count - neg_correct
        falseNeg_count = pos_count - pos_correct

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/neg'] = metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean()
        metrics_dict['loss/pos'] = metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean()

        metrics_dict['correct/all'] = (pos_correct + neg_correct) / np.float32(metrics_t.shape[1]) * 100
        metrics_dict['correct/neg'] = neg_correct / np.float32(neg_count) * 100
        metrics_dict['correct/pos'] = pos_correct / np.float32(pos_count) * 100

        # Metrics
        precision = metrics_dict['pr/precision'] = truePos_count / np.float32(truePos_count + falsePos_count) # TP / (TP + FP)
        recall = metrics_dict['pr/recall'] = truePos_count / np.float32(truePos_count + falseNeg_count) # TP / (TP + FN)
        f1_score = metrics_dict['pr/f1_score'] = 2 * (precision * recall) / (precision + recall)



        # all
        log.info(
            (  "E{} {:8} {loss/all:.4f} loss, "
             + "{correct/all:-5.1f}% correct, "
             + "{pr/precision:.4f} precision, "
             + "{pr/recall:.4f} recall, "
             + "{pr/f1_score:.4f} f1 score"
            ).format(
                epoch_ndx, 
                mode_str, 
                **metrics_dict,
            )
        ) 

        # neg
        log.info(
            ("E{} {:8} {loss/neg:.4f} loss, "
                 + "{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})"
            ).format(
                epoch_ndx,
                mode_str + '_neg',
                neg_correct=neg_correct,
                neg_count=neg_count,
                **metrics_dict,
            )
        )

        # pos
        log.info(
            ("E{} {:8} {loss/pos:.4f} loss, "
                + "{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})"
            ).format(
                epoch_ndx,
                mode_str + '_pos',
                pos_correct=pos_correct,
                pos_count=pos_count,
                **metrics_dict,
            )
        )

        writer = getattr(self, mode_str + '_writer')

        # Scalar
        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.totalTrainingSamples_count)

        # PR_Curve
        writer.add_pr_curve(
            'pr',
            metrics_t[METRICS_LABEL_NDX],
            metrics_t[METRICS_PRED_NDX],
            self.totalTrainingSamples_count
        )

        # Histogram
        bins = [x / 50.0 for x in range(51)]
        negHist_mask = negLabel_mask & (metrics_t[METRICS_PRED_NDX] > 0.01) # 陰性サンプル : 結節でない
        posHist_mask = posLabel_mask & (metrics_t[METRICS_PRED_NDX] < 0.99) # 陽性サンプル : 結節である

        if negHist_mask.any():
            sum_pos_prob_ls_001= np.sum(metrics_t[METRICS_PRED_NDX] > 0.01)
            print("sum(metrics_[METRICS_PRED_NDX] > 0.01) : 陽性尤度が>0.01な推論サンプル数", sum_pos_prob_ls_001)
            print("negHist_mask\n", negHist_mask)
            print("negHist_mask.shape", negHist_mask.shape)
            print("metrics_t[METRICS_PRED_NDX, negHist_mask]\n", metrics_t[METRICS_PRED_NDX, negHist_mask])
            print("metrics_t[METRICS_PRED_NDX, negHist_mask].shape", metrics_t[METRICS_PRED_NDX, negHist_mask].shape)

            writer.add_histogram(
                'is_neg',
                metrics_t[METRICS_PRED_NDX, negHist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )

        if posHist_mask.any():
            sum_pos_prob_gt_099 = np.sum(metrics_t[METRICS_PRED_NDX] < 0.99)
            print("sum(metrics_[METRICS_PRED_NDX] < 0.99) : 陽性尤度が<0.99な推論サンプル数", sum_pos_prob_gt_099)
            print("posHist_mask\n", posHist_mask)
            print("posHist_mask.shape", posHist_mask.shape)
            print("metrics_t[METRICS_PRED_NDX, posHist_mask]\n", metrics_t[METRICS_PRED_NDX, posHist_mask])
            print("metrics_t[METRICS_PRED_NDX, posHist_mask].shape", metrics_t[METRICS_PRED_NDX, posHist_mask].shape)

            writer.add_histogram(
                'is_pos',
                metrics_t[METRICS_PRED_NDX, posHist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )

        # score = 1 \
        #     + metrics_dict['pr/f1_score'] \
        #     - metrics_dict['loss/mal'] * 0.01 \
        #     - metrics_dict['loss/all'] * 0.0001
        #
        # return score

    # def logModelMetrics(self, model):
    #     writer = getattr(self, 'trn_writer')
    #
    #     model = getattr(model, 'module', model)
    #
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             min_data = float(param.data.min())
    #             max_data = float(param.data.max())
    #             max_extent = max(abs(min_data), abs(max_data))
    #
    #             # bins = [x/50*max_extent for x in range(-50, 51)]
    #
    #             try:
    #                 writer.add_histogram(
    #                     name.rsplit('.', 1)[-1] + '/' + name,
    #                     param.data.cpu().numpy(),
    #                     # metrics_a[METRICS_PRED_NDX, negHist_mask],
    #                     self.totalTrainingSamples_count,
    #                     # bins=bins,
    #                 )
    #             except Exception as e:
    #                 log.error([min_data, max_data])
    #                 raise  

if __name__ == "__main__":
    LunaTrainingApp().main()

