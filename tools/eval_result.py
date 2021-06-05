import argparse
import mmcv
import numpy as np

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--result-pkl-path', help='result pkl file path')
    parser.add_argument('--gt-pkl-path', help='gt pkl file path')
    parser.add_argument('--iou-thr', type=float, default=0.5, help='threshold of iou')
    parser.add_argument('--score-thr', type=float, default=0.3, help='threshold of iou')
    args = parser.parse_args()
    return args

def _load_pkl(result_pkl_path, gt_pkl_path):
    result_pkl = mmcv.load(result_pkl_path)
    gt_pkl = mmcv.load(gt_pkl_path)
    return result_pkl, gt_pkl

def _eval_sample(det, gt, score_thr, iou_thr):
    num_classes = len(det)
    matrix = np.zeros([num_classes, num_classes])
    for i in range(num_classes):
        ids = np.where(det[i][..., -1] > score_thr)
        det[i] = det[i][ids[0]]
        for j in range(num_classes):
            ious = bbox_overlaps(det[i], gt[j])
            if ious.shape[1] > 0:
                ious = ious.max(1)
                matrix[i][j] += np.count_nonzero(ious > iou_thr)
    return matrix

def main():
    args = parse_args()
    result_pkl_path = args.result_pkl_path
    gt_pkl_path = args.gt_pkl_path
    score_thr = args.score_thr
    iou_thr = args.iou_thr
    result_pkl, gt_pkl = _load_pkl(result_pkl_path, gt_pkl_path)
    full_matrix = None
    for i in range(len(result_pkl)):
        matrix = _eval_sample(result_pkl[i], gt_pkl[i], score_thr, iou_thr)
        if i == 0:
            full_matrix = matrix
        else:
            full_matrix += matrix
    print(full_matrix.trace() / full_matrix.sum())

if __name__ == '__main__':
    main()