import argparse
import mmcv
import glob
import shutil
import os.path as osp
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString

def parse_args():
    parser = argparse.ArgumentParser(description='Generate data')
    parser.add_argument('--source-dir', default='/home1/lixiao/celldata', help='the dir of the source data')
    parser.add_argument(
        '--target-dir', default='/home1/yinhaoli/data/cell', help='the dir to save the generated data')

    args = parser.parse_args()
    return args

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
    root_folders = glob.glob(osp.join(source_dir, '项目*'))
    mmcv.mkdir_or_exist(osp.join(target_dir, 'JPEGImages'))
    annotations = []
    for root_folder in root_folders:
        data_folders = glob.glob(osp.join(root_folder, '*'))
        for data_folder in data_folders:
            images = glob.glob(osp.join(data_folder, '*.jpg'))
            with open(osp.join(data_folder, 'type.txt')) as f:
                annotations += f.readlines()
            for image in images:
                filename = osp.split(image)[-1]
                # shutil.copyfile(image, osp.join(target_dir, 'JPEGImages', filename))
    return annotations

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
        with open(osp.join(target_dir, 'Annotations', filename.split('.')[0] + 'xml'), 'wb') as f:
            f.write(dom.toprettyxml(encoding='utf-8'))
        print(f'{filename} finished')

def main():
    args = parse_args()
    source_dir = args.source_dir
    target_dir = args.target_dir
    annotations = _copy_data(source_dir, target_dir)
    _generate_ann(target_dir, annotations)

if __name__ == '__main__':
    main()
