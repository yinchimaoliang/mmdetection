from mmdet.apis import init_detector, inference_detector
import mmcv
import json
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
        '--threshold', help='threshold', type=int, default=0)
    args = parser.parse_args()

    return args

index = {
    "\u5df4\u9ece\u7fe0\u51e4\u8776": "0",
    "\u67d1\u6a58\u51e4\u8776": "1",
    "\u7389\u5e26\u51e4\u8776": "2",
    "\u78a7\u51e4\u8776": "3",
    "\u7ea2\u57fa\u7f8e\u51e4\u8776": "4",
    "\u84dd\u51e4\u8776": "5",
    "\u91d1\u88f3\u51e4\u8776": "6",
    "\u9752\u51e4\u8776": "7",
    "\u6734\u5599\u8776": "8",
    "\u5bc6\u7eb9\u98d2\u5f04\u8776": "9",
    "\u5c0f\u9ec4\u6591\u5f04\u8776": "10",
    "\u65e0\u6591\u73c2\u5f04\u8776": "11",
    "\u76f4\u7eb9\u7a3b\u5f04\u8776": "12",
    "\u82b1\u5f04\u8776": "13",
    "\u9690\u7eb9\u8c37\u5f04\u8776": "14",
    "\u7ee2\u6591\u8776": "15",
    "\u864e\u6591\u8776": "16",
    "\u4eae\u7070\u8776": "17",
    "\u5496\u7070\u8776": "18",
    "\u5927\u7d2b\u7409\u7483\u7070\u8776": "19",
    "\u5a40\u7070\u8776": "20",
    "\u66f2\u7eb9\u7d2b\u7070\u8776": "21",
    "\u6ce2\u592a\u7384\u7070\u8776": "22",
    "\u7384\u7070\u8776": "23",
    "\u7ea2\u7070\u8776": "24",
    "\u7ebf\u7070\u8776": "25",
    "\u7ef4\u7eb3\u65af\u773c\u7070\u8776": "26",
    "\u8273\u7070\u8776": "27",
    "\u84dd\u7070\u8776": "28",
    "\u9752\u6d77\u7ea2\u73e0\u7070\u8776": "29",
    "\u53e4\u5317\u62df\u9152\u773c\u8776": "30",
    "\u963f\u82ac\u773c\u8776": "31",
    "\u62df\u7a3b\u7709\u773c\u8776": "32",
    "\u7267\u5973\u73cd\u773c\u8776": "33",
    "\u767d\u773c\u8776": "34",
    "\u83e9\u8428\u9152\u773c\u8776": "35",
    "\u897f\u95e8\u73cd\u773c\u8776": "36",
    "\u8fb9\u7eb9\u9edb\u773c\u8776": "37",
    "\u4e91\u7c89\u8776": "38",
    "\u4f8f\u7c89\u8776": "39",
    "\u5927\u536b\u7c89\u8776": "40",
    "\u5927\u7fc5\u7ee2\u7c89\u8776": "41",
    "\u5bbd\u8fb9\u9ec4\u7c89\u8776": "42",
    "\u5c71\u8c46\u7c89\u8776": "43",
    "\u6a59\u9ec4\u8c46\u7c89\u8776": "44",
    "\u7a81\u89d2\u5c0f\u7c89\u8776": "45",
    "\u7bad\u7eb9\u4e91\u7c89\u8776": "46",
    "\u7bad\u7eb9\u7ee2\u7c89\u8776": "47",
    "\u7ea2\u895f\u7c89\u8776": "48",
    "\u7ee2\u7c89\u8776": "49",
    "\u83dc\u7c89\u8776": "50",
    "\u9549\u9ec4\u8fc1\u7c89\u8776": "51",
    "\u9ece\u660e\u8c46\u7c89\u8776": "52",
    "\u4f9d\u5e15\u7ee2\u8776": "53",
    "\u56db\u5ddd\u7ee2\u8776": "54",
    "\u73cd\u73e0\u7ee2\u8776": "55",
    "\u86c7\u76ee\u8910\u86ac\u8776": "56",
    "\u4e2d\u73af\u86f1\u8776": "57",
    "\u4e91\u8c79\u86f1\u8776": "58",
    "\u4f0a\u8bfa\u5c0f\u8c79\u86f1\u8776": "59",
    "\u5c0f\u7ea2\u86f1\u8776": "60",
    "\u626c\u7709\u7ebf\u86f1\u8776": "61",
    "\u6590\u8c79\u86f1\u8776": "62",
    "\u66f2\u6591\u73e0\u86f1\u8776": "63",
    "\u67f1\u83f2\u86f1\u8776": "64",
    "\u67f3\u7d2b\u95ea\u86f1\u8776": "65",
    "\u707f\u798f\u86f1\u8776": "66",
    "\u7384\u73e0\u5e26\u86f1\u8776": "67",
    "\u73cd\u86f1\u8776": "68",
    "\u7409\u7483\u86f1\u8776": "69",
    "\u767d\u94a9\u86f1\u8776": "70",
    "\u79c0\u86f1\u8776": "71",
    "\u7ee2\u86f1\u8776": "72",
    "\u7eff\u8c79\u86f1\u8776": "73",
    "\u7f51\u86f1\u8776": "74",
    "\u7f8e\u773c\u86f1\u8776": "75",
    "\u7fe0\u84dd\u773c\u86f1\u8776": "76",
    "\u8001\u8c79\u86f1\u8776": "77",
    "\u8368\u9ebb\u86f1\u8776": "78",
    "\u866c\u7709\u5e26\u86f1\u8776": "79",
    "\u87fe\u798f\u86f1\u8776": "80",
    "\u94a9\u7fc5\u773c\u86f1\u8776": "81",
    "\u94f6\u6591\u8c79\u86f1\u8776": "82",
    "\u94f6\u8c79\u86f1\u8776": "83",
    "\u94fe\u73af\u86f1\u8776": "84",
    "\u9526\u745f\u86f1\u8776": "85",
    "\u9ec4\u73af\u86f1\u8776": "86",
    "\u9ec4\u94a9\u86f1\u8776": "87",
    "\u9ed1\u7f51\u86f1\u8776": "88",
    "\u5c16\u7fc5\u7fe0\u86f1\u8776": "89",
    "\u7d20\u5f04\u8776": "90",
    "\u7fe0\u8896\u952f\u773c\u8776": "91",
    "\u84dd\u70b9\u7d2b\u6591\u8776": "92",
    "\u96c5\u5f04\u8776": "93"
    }

order = [
    '蟾福蛱蝶', '素弄蝶', '线灰蝶', '无斑珂弄蝶', '老豹蛱蝶', '突角小粉蝶', '尖翅翠蛱蝶', '柑橘凤蝶', '箭纹绢粉蝶', '碧凤蝶', '亮灰蝶', '银斑豹蛱蝶',
         '宽边黄粉蝶',
         '灿福蛱蝶', '蛇目褐蚬蝶', '巴黎翠凤蝶', '黄钩蛱蝶', '翠袖锯眼蝶', '红基美凤蝶', '虬眉带蛱蝶', '黄环蛱蝶', '翠蓝眼蛱蝶', '隐纹谷弄蝶', '蓝点紫斑蝶',
         '大紫琉璃灰蝶',
         '古北拟酒眼蝶', '绿豹蛱蝶', '西门珍眼蝶', '伊诺小豹蛱蝶', '网蛱蝶', '阿芬眼蝶', '波太玄灰蝶', '红灰蝶', '雅弄蝶', '花弄蝶', '美眼蛱蝶', '银豹蛱蝶',
         '牧女珍眼蝶', '柳紫闪蛱蝶', '婀灰蝶', '扬眉线蛱蝶', '绢斑蝶', '箭纹云粉蝶', '中环蛱蝶', '青海红珠灰蝶', '大卫粉蝶', '蓝凤蝶', '曲斑珠蛱蝶', '金裳凤蝶',
         '边纹黛眼蝶', '链环蛱蝶', '荨麻蛱蝶', '黎明豆粉蝶', '秀蛱蝶', '艳灰蝶', '依帕绢蝶', '白眼蝶', '白钩蛱蝶', '青凤蝶', '云粉蝶', '珍蛱蝶', '直纹稻弄蝶',
         '拟稻眉眼蝶', '蓝灰蝶', '大翅绢粉蝶', '玄珠带蛱蝶', '珍珠绢蝶', '曲纹紫灰蝶', '琉璃蛱蝶', '山豆粉蝶', '云豹蛱蝶', '菜粉蝶', '小红蛱蝶', '橙黄豆粉蝶',
         '绢蛱蝶',
         '朴喙蝶', '玄灰蝶', '虎斑蝶', '玉带凤蝶', '钩翅眼蛱蝶', '侏粉蝶', '小黄斑弄蝶', '咖灰蝶', '柱菲蛱蝶', '密纹飒弄蝶', '镉黄迁粉蝶', '斐豹蛱蝶', '四川绢蝶',
         '黑网蛱蝶', '维纳斯眼灰蝶', '绢粉蝶', '红襟粉蝶', '锦瑟蛱蝶', '菩萨酒眼蝶'
         ]

types = 94 #一共有94个类

def main():
    args = parse_args()
    jsbase = {}
    for type in range(types):
        jsbase[str(type)] = []

    config_file = args.config_file
    checkpoint_file = args.ckpt_path
    result_path = args.result_path
    mmcv.mkdir_or_exist(result_path)
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    img_path = args.image_path
    names = mmcv.scandir(img_path)
    for name in names:
        img = osp.join(img_path, name)
        file_name = name.split('.')[0]
        result = inference_detector(model, img)
        for type in range(types):
            row, col = result[type].shape
            if row != 0:
                if float(result[type][0, 4]) > args.threshold:
                    jsbase[index[order[type]]].append(
                        [file_name, float(result[type][0, 4]), float(result[type][0, 0]),
                         float(result[type][0, 1]), float(result[type][0, 2]), float(result[type][0, 3])])
                # 当这一行存在预测结果时将其记录起来
        print(f'{file_name} finished')

    with open(osp.join(args.result_path, 'submission.json'),'w') as f:
        f.write(json.dumps(jsbase))


if __name__ == '__main__':
    main()


