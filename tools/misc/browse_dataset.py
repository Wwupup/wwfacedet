# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os
from pathlib import Path

import mmcv
import numpy as np
from mmcv import Config, DictAction

from mmdet.core.utils import mask2ndarray
from mmdet.core.visualization import imshow_det_bboxes
from mmdet.datasets.builder import build_dataset
from mmdet.utils import update_data_root


class BBoxDistribution():

    def __init__(self) -> None:
        self.data = {}

    def insert(self, wh):
        wh.clip(0)
        xs = np.sqrt(wh[:, 0] * wh[:, 1]).astype(np.int32)
        for x in xs:
            x = int(x)
            if x in self.data:
                self.data[x] += 1
            else:
                self.data[x] = 1

    def save(self, outfilename):
        with open(outfilename, 'w') as f:
            json.dump(self.data, f)
        print(f'\nSave to: {outfilename}')

    def read(self, infilename):
        print(f'Read from: {infilename}\n')
        with open(infilename, 'r') as f:
            self.data = json.load(f)

    def export(self, outfilename):
        pass


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['DefaultFormatBundle', 'Normalize', 'Collect'],
        help='skip some useless pipeline')
    parser.add_argument(
        '--output-dir',
        default='./work_dirs/browse_dataset/',
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=float,
        default=0,
        help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument('--tag', default='', type=str)
    parser.add_argument('--epoches', default=1, type=int)
    args = parser.parse_args()
    return args


def retrieve_data_cfg(config_path, skip_type, cfg_options):

    def skip_pipeline_steps(config):
        config['pipeline'] = [
            x for x in config.pipeline if x['type'] not in skip_type
        ]

    cfg = Config.fromfile(config_path)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    train_data_cfg = cfg.data.train
    while 'dataset' in train_data_cfg and train_data_cfg[
            'type'] != 'MultiImageMixDataset':
        train_data_cfg = train_data_cfg['dataset']

    skip_pipeline_steps(train_data_cfg)

    return cfg


def main():
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type, args.cfg_options)

    dataset = build_dataset(cfg.data.train)
    bd = BBoxDistribution()
    for epoch in range(args.epoches):
        print(f'\nEpoch {epoch}:')
        progress_bar = mmcv.ProgressBar(len(dataset))
        for item in dataset:
            filename = os.path.join(args.output_dir, 'images',
                                    Path(item['filename']).name
                                    ) if args.output_dir is not None else None

            gt_bboxes = item['gt_bboxes']
            gt_labels = item['gt_labels']
            gt_masks = item.get('gt_masks', None)
            if gt_masks is not None:
                gt_masks = mask2ndarray(gt_masks)

            gt_kps = item.get('gt_keypointss', None)
            kps_ignore = True
            if kps_ignore:
                kps = gt_kps[..., :-1]
                kps_flag = np.mean(
                    gt_kps[..., 2], axis=1, keepdims=True).squeeze(1) > 0
                gt_kps = kps[kps_flag].reshape(-1, 2)
            else:
                # kps = kps[..., :-1].reshape(num_gt, -1)
                assert 'This dataset has kps ignore flag!'

            if not args.not_show:
                imshow_det_bboxes(
                    item['img'],
                    gt_bboxes,
                    gt_labels,
                    gt_masks,
                    gt_kps,
                    class_names=dataset.CLASSES,
                    show=not args.not_show,
                    wait_time=args.show_interval,
                    out_file=filename,
                    bbox_color=dataset.PALETTE,
                    text_color=(200, 200, 200),
                    mask_color=dataset.PALETTE,
                    kps_color=dataset.PALETTE)

            wh = gt_bboxes[:, -2:] - gt_bboxes[:, :2]
            bd.insert(wh)
            progress_bar.update()

    bd.save(os.path.join(args.output_dir, f'bd_{args.tag}.json'))


if __name__ == '__main__':
    main()
