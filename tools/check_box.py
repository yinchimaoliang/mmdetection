import xml.etree.ElementTree as ET
import os

import cv2

xml_root = "./data/nuclei/Annotations"
new_xml_root = "./data/nuclei/new_Annotations"
image_root = "./data/nuclei/PNGImages"

xml_name_list = sorted(os.listdir(xml_root))


def print_all_classes():
    all_name_list = []
    for xml_name in xml_name_list:
        print(f"{xml_name}")
        xml_path = os.path.join(xml_root, xml_name)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            name = obj.find("name").text
            all_name_list.append(name)
        print(all_name_list)


def check_hw():
    tranposed_name_lists = []
    for xml_name in xml_name_list:
        xml_path = os.path.join(xml_root, xml_name)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        image_path = os.path.join(image_root, xml_name[:-4] + ".jpg")
        img = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
        h, w, _ = img.shape
        if height != h or width != w:
            print(width, w, height, h)
            print(f"{xml_name}'s h, w is tranposed.")
            tranposed_name_lists.append(xml_name)
    print(tranposed_name_lists)


def check_bbox():
    if not os.path.exists(new_xml_root):
        os.makedirs(new_xml_root)

    for xml_name in xml_name_list:
        xml_path = os.path.join(xml_root, xml_name)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            bnd_box = obj.find("bndbox")
            bbox = [
                int(float(bnd_box.find("xmin").text)),
                int(float(bnd_box.find("ymin").text)),
                int(float(bnd_box.find("xmax").text)),
                int(float(bnd_box.find("ymax").text)),
            ]
            image_path = os.path.join(image_root, xml_name[:-4] + ".png")
            img = cv2.imread(image_path, flags=cv2.IMREAD_COLOR)
            h, w, _ = img.shape

            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                print("bbox[0] >= bbox[2] or bbox[1] >= bbox[3]", bbox, xml_name)
                # bboxes = np.array([bbox])
                # mmcv.imshow_det_bboxes(img, bboxes, labels=np.array(["h"]))
                # bbox_min_ge_max_name_lists.append(xml_name)
                root.remove(obj)
            elif bbox[3] > h or bbox[2] > w:
                bnd_box.find("xmax").text = str(min(w, bbox[2]))
                bnd_box.find("ymax").text = str(min(h, bbox[3]))
                print("bbox[3] > h or bbox[2] > w", bbox, h, w, xml_name)
                # bboxes = np.array([bbox])
                # mmcv.imshow_det_bboxes(img, bboxes, labels=np.array(["h"]))
                # bbox_max_border_name_lists.append(xml_name)
        tree.write(os.path.join(new_xml_root, xml_name))

check_bbox()