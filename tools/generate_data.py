import argparse
import mmcv
import glob
import numpy as np
import os.path as osp
import os
import shutil
import xml.etree.ElementTree as ET
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString

def parse_args():
    parser = argparse.ArgumentParser(description='Generate data')
    parser.add_argument('--source-dir', default='/home1/yinhaoli/data/cell/complete-1-26', help='the dir of the source data')
    parser.add_argument(
        '--target-dir', default='/home1/yinhaoli/data/cell/complete-1-26', help='the dir to save the generated data')
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
    for i in range(100):
        anno = annos[i]
        tree = ET.parse(anno)
        root = tree.getroot()
        filename = root.find('filename').text
        img = mmcv.imread(osp.join(target_dir, 'JPEGImages', filename))
        img = mmcv.imresize(img, (img.shape[1] // 10, img.shape[0] // 10))
        bgrs.append(img.reshape(-1, 3))
        print(i)
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





def _make_xml(target_path, obj_data, image_name):

    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'cell'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = image_name

    node_path = SubElement(node_root, 'path')
    node_path.text = osp.join(target_path, 'JPEGImages', image_name)

    node_object_num = SubElement(node_root, 'object_num')
    node_object_num.text = str(len(obj_data))

    img = mmcv.imread(osp.join(target_path, 'JPEGImages', image_name))
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(img.shape[1])

    node_height = SubElement(node_size, 'height')
    node_height.text = str(img.shape[0])

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    for obj_instance in obj_data:
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = obj_instance[-1]
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'

        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(obj_instance[0])
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(obj_instance[1])
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(int(obj_instance[0]) + int(obj_instance[2]))
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(int(obj_instance[1]) + int(obj_instance[3]))


    xml = tostring(node_root, pretty_print = True)
    dom = parseString(xml)
    #print xml 打印查看结果
    return dom

def _copy_data(source_dir, target_dir):
    root_folders = os.listdir(source_dir)
    mmcv.mkdir_or_exist(osp.join(target_dir, 'JPEGImages'))
    mmcv.mkdir_or_exist(osp.join(target_dir, 'Annotations'))
    for root_folder in root_folders:
        names = [osp.split(path)[-1].split('.')[0] for path in glob.glob(osp.join(source_dir, root_folder, '*.xml'))]
        for name in names:
            try:
                shutil.copyfile(osp.join(source_dir, root_folder, name+'.jpg'), osp.join(target_dir, 'JPEGImages', name+'.jpg'))
                shutil.copyfile(osp.join(source_dir, root_folder, name+'.xml'), osp.join(target_dir, 'Annotations', name+'.xml'))
            except:
                print(name)

    # annotations = []
    # for root_folder in root_folders:
    #     data_folders = glob.glob(osp.join(root_folder, '*'))
    #     for data_folder in data_folders:
    #         images = glob.glob(osp.join(data_folder, '*.jpg'))
    #         with open(osp.join(data_folder, 'type.txt')) as f:
    #             annotations += f.readlines()
    #         for image in images:
    #             filename = osp.split(image)[-1]
    #             shutil.copyfile(image, osp.join(target_dir, 'JPEGImages', filename))
    # return annotations

def _get_class_names(annotation_path):
    xml_names = os.listdir(annotation_path)
    class_names = []
    for xml_name in xml_names:
        tree = ET.parse(osp.join(annotation_path, xml_name))
        objects = tree.findall('object')
        for object in objects:
            class_names.append(object.find('name').text)

    print(set(class_names))

def _generate_ann(target_dir, annotations):
    annotations_dict = dict()
    mmcv.mkdir_or_exist(osp.join(target_dir, 'Annotations'))
    for annotation in annotations:
        name, x, y, height, width, class_name = annotation.split(',')
        if name not in annotations_dict.keys():
            annotations_dict[name] = [[x, y, height, width, class_name[:-1]]]
        else:
            annotations_dict[name].append([x, y, height, width, class_name[:-1]])

    for i, filename in enumerate(annotations_dict.keys()):
        if filename == '010008610797F9F5026D8B245A023F72AFBA0F76C702.jpg' or not osp.exists(osp.join(target_dir, 'JPEGImages', filename)) or filename == '010009AA7D8185BC63EA5FFBAED2D916A8DE1DC27602.jpg' or filename == '0100098862143E8B22ED715C1305DF63124F68439402.jpg':
            continue
        dom = _make_xml(target_dir, annotations_dict[filename], filename)
        with open(osp.join(target_dir, 'Annotations', filename.split('.')[0] + '.xml'), 'wb') as f:
            f.write(dom.toprettyxml(encoding='utf-8'))
        print(f'{filename} finished')

def main():
    args = parse_args()
    source_dir = args.source_dir
    target_dir = args.target_dir
    train_ratio = args.train_ratio
    # _copy_data(source_dir, target_dir)
    _get_class_names(osp.join(args.source_dir, 'Annotations'))
    # annotations = _copy_data(source_dir, target_dir)
    # _generate_ann(target_dir, annotations)
    # _generate_division(target_dir, train_ratio)
    # names = _count_data_info(target_dir)
    # print(names)
if __name__ == '__main__':
    main()
