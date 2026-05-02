import os.path as osp
import numpy as np
import cv2
import os
import json
import torchvision
from .base_dataset import BaseDataset
from mfplnet.utils.pl_metric import LaneEval
from .registry import DATASETS
import logging
import random

SPLIT_FILES = {
    'trainval':
    ['label_data_0114.json', 'label_data_0115.json', 'label_data_0116.json', 'label_data_0117.json',
     'label_data_0118.json', 'label_data_0119.json', 'label_data_0120.json', 'label_data_0121.json'],
    'train': ['label_data_0114.json', 'label_data_0115.json', 'label_data_0116.json', 'label_data_0117.json'],
    'val': ['label_data_0118.json', 'label_data_0119.json', 'label_data_0120.json', 'label_data_0121.json'],
    'test': ['label_data_0122.json'],
}


@DATASETS.register_module
class Pldm01(BaseDataset):
    def __init__(self, data_root, split, processes=None, pipeline=None, cfg=None):
        super().__init__(data_root, split, processes, cfg)
        self.anno_files = SPLIT_FILES[split]
        self.load_annotations()
        self.h_samples = list(range(0, 540, 1))
        self.pipeline = pipeline

    def load_annotations(self):
        self.logger.info('Loading pldm_01 annotations...')
        self.data_infos = []
        max_lanes = 0
        for anno_file in self.anno_files:
            anno_file = osp.join(self.data_root, anno_file)
            anno_file = anno_file.replace('\\', '/')
            with open(anno_file, 'r') as anno_obj:
                lines = anno_obj.readlines()
            for line in lines:
                data = json.loads(line)
                y_samples = data['h_samples']
                gt_lanes = data['lanes']
                mask_path = data['raw_file'].replace('clips',
                                                     'seg_label')[:-3] + 'png'
                lanes = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0]
                         for lane in gt_lanes]
                lanes = [lane for lane in lanes if len(lane) > 0]
                max_lanes = max(max_lanes, len(lanes))
                self.data_infos.append({
                    'img_path':
                    osp.join(self.data_root, data['raw_file']),
                    'img_name':
                    data['raw_file'],
                    'mask_path':
                    osp.join(self.data_root, mask_path),
                    'lanes':
                    lanes,
                })

        if self.training:
            random.shuffle(self.data_infos)
        self.max_lanes = max_lanes

    def pred2lanes(self, pred):
        ys = np.array(self.h_samples) / self.cfg.ori_img_h
        lanes = []
        for lane in pred:
            xs = lane(ys)
            invalid_mask = xs < 0
            lane = (xs * self.cfg.ori_img_w).astype(int)
            lane[invalid_mask] = -2
            lanes.append(lane.tolist())

        return lanes

    def pred2tusimpleformat(self, idx, pred, runtime):
        runtime *= 1000.
        img_name = self.data_infos[idx]['img_name']
        lanes = self.pred2lanes(pred)
        output = {'raw_file': img_name, 'lanes': lanes, 'run_time': runtime}
        return json.dumps(output)

    def save_tusimple_predictions(self, predictions, filename, runtimes=None):
        if runtimes is None:
            runtimes = np.ones(len(predictions)) * 1.e-3
        lines = []
        for idx, (prediction, runtime) in enumerate(zip(predictions,
                                                        runtimes)):
            line = self.pred2tusimpleformat(idx, prediction, runtime)
            lines.append(line)
        with open(filename, 'w') as output_file:
            output_file.write('\n'.join(lines))

    def evaluate(self, predictions, output_basedir, runtimes=None):
        pred_filename = os.path.join(output_basedir,
                                     'pldm_01_predictions.json')
        self.save_tusimple_predictions(predictions, pred_filename, runtimes)
        result, acc, mean_iou = LaneEval.bench_one_submit(pred_filename,
                                                self.cfg.test_json_file)
        self.logger.info(result)
        self.logger.info(f"Mean Iou: {mean_iou:.4f}")
        return {
            'Accuracy': acc,
            'Mean Iou': mean_iou
        }