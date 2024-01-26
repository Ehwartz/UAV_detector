import os
import json
from tqdm import tqdm
import argparse
import cv2
import PIL.Image as Image

classes1 = ['cookies', 'mixed_congee', 'chocolate', 'melon_seeds',
            'milk', 'water', 'cola', 'coffee', 'ad_milk',
            'knife', 'underwear_detergent', 'book', 'floral_water', 'toothpaste',
            'folder', 'water_glass', 'food_grade_detergent', 'slippers', 'pen']

classes2 = ['biscuits', '', '', '',
            '', '', '', '', 'AD_calcium_milk',
            'fruit_knife', 'laundry_detergent', '', 'toilet_water', '',
            '', '', 'dish_soap', '', '']


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2]-box[0]
    h = box[3]-box[1]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


if __name__ == '__main__':
    path = './COCO/'
    files = os.listdir(path + 'coco_labels')

    for f in files:
        J = json.load(open(path + 'coco_labels/' + f))['labels']
        img_name = f.split('.')[-2]
        img_size = Image.open(path + 'Images/'+img_name+'.jpg').size
        f_txt = open(path + 'labels/' + img_name + '.txt', 'w')

        for elems in J:
            # print(img_size)
            box = []
            box.append(elems['x1'])
            box.append(elems['y1'])
            box.append(elems['x2'])
            box.append(elems['y2'])
            bbox = convert(img_size, box)
            cls = 0
            if elems['name'] in classes1:
                cls = classes1.index(elems['name'])
            if elems['name'] in classes2:
                cls = classes2.index(elems['name'])
            if elems['name'] not in classes1 and elems['name'] not in classes2:
                continue

            f_txt.write('%s %s %s %s %s\n' % (cls, bbox[0], bbox[1], bbox[2], bbox[3]))
        f_txt.close()

    print('End')
