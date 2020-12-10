from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import os.path as osp
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet inference a model')
    parser.add_argument(
        '--image-path', help='path of images to be tested')
    parser.add_argument(
        '--result-path', help='path of results to be saved')
    parser.add_argument(
        '--config-file', help='config file to be used')
    parser.add_argument(
        '--ckpt-path', help='checkpoint file to be used')
    parser.add_argument(
        '--score-thr', default=0.3, help='threshold score to show the result')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    config_file = args.config_file
    checkpoint_file = args.ckpt_path
    result_path = args.result_path
    score_thr = args.score_thr
    mmcv.mkdir_or_exist(result_path)
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    with open(args.image_path) as f:
        names = f.readlines()
    for name in names:
        name = name[:-1]
        img = osp.join('data/karyotype/PNGImages', name + '.png')
        file_name = name.split('.')[0]
        result = inference_detector(model, img)
        if hasattr(model, 'module'):
            model = model.module
        img = model.show_result(img, result, score_thr=score_thr, show=False)
        mmcv.imwrite(img, osp.join(result_path, file_name + '.png'))
        print(f'{file_name} finished.')



if __name__ == '__main__':
    main()


