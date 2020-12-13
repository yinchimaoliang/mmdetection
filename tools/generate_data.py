import argparse
import mmcv
import glob
import numpy as np
import os.path as osp
import os
import xml.etree.ElementTree as ET

def parse_args():
    parser = argparse.ArgumentParser(description='Generate data')
    parser.add_argument(
        '--target-dir', default='/home1/yinhaoli/data/butterfly', help='the dir to save the generated data')
    parser.add_argument(
        '--train-ratio', default=0.7, type=int, help='ratio of the train number')

    args = parser.parse_args()
    return args

def _count_data_info(target_dir):
    annos = glob.glob(osp.join(target_dir, 'Annotations', '*.xml'))
    class_names = []
    bgrs = []
    for anno in annos:
        tree = ET.parse(anno)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            class_names.append(name)

    np.random.shuffle(annos)
    for anno in annos:
        tree = ET.parse(anno)
        root = tree.getroot()
        filename = root.find('filename').text
        img = mmcv.imread(osp.join(target_dir, 'JPEGImages', filename))
        img = mmcv.imresize(img, (img.shape[1] // 100, img.shape[0] // 100))
        bgrs.append(img.reshape(-1, 3))
        print(f'{anno} finished')
    bgrs = np.concatenate(bgrs, axis=0)
    mean = np.mean(bgrs, axis=0)
    std = np.std(bgrs, axis=0)
    print(mean, std)
    return set(class_names)


def _generate_division(target_path, train_ratio):
    mmcv.mkdir_or_exist(osp.join(target_path, 'ImageSets', 'Main'))
    names = os.listdir(osp.join(target_path, 'Annotations'))
    np.random.shuffle(names)
    train_names = names[:int(len(names) * train_ratio)]
    val_names = names[int(len(names) * train_ratio):]
    with open(osp.join(target_path, 'ImageSets', 'Main', 'trainval.txt'), 'w') as f:
        for name in names:
            f.write(name.split('.')[0] + '\n')
    with open(osp.join(target_path, 'ImageSets', 'Main', 'train.txt'), 'w') as f:
        for train_name in train_names:
            f.write(train_name.split('.')[0] + '\n')
    with open(osp.join(target_path, 'ImageSets', 'Main', 'val.txt'), 'w') as f:
        for val_name in val_names:
            f.write(val_name.split('.')[0] + '\n')


def main():
    args = parse_args()
    target_dir = args.target_dir
    train_ratio = args.train_ratio
    # _generate_division(target_dir, train_ratio)
    names = _count_data_info(target_dir)
    print(names)
if __name__ == '__main__':
    main()
