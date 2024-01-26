import torch
import numpy as np
# from det.models.common import DetectMultiBackend
from det.det import imgTransform, predict, load_images, create_det_model
from det.utils.plots import Annotator, colors
from classify.classify import cls_predict, create_cls_model, img_transform

# from classify.classify import create_cls_model
import os
import sys
from pathlib import Path
import cv2
from PIL import Image


def load_data(path: str):
    """
    :param path: Path to the folder of all images that need processing
    :return: A list of images of BGR array
    """

    img_paths = load_images(path)
    imgs = list()
    for i, img_path in enumerate(img_paths):
        print('Loading image: [ ', img_path, ' ]')
        imgs.append(cv2.imread(img_path))

    return imgs


def clip_images(det, img0):
    """
    AyRphtz
    :param det: Detection results of detection model
    :param img0: Original image
    :return:
    """

    clipped_imgs = list()
    for n in range(det.shape[0]):
        clipped = img0[int(det[n][1]):int(det[n][3]), int(det[n][0]):int(det[n][2])]
        clipped_imgs.append(clipped)
    return clipped_imgs


def array2Image(arrays: list):
    """

    :param arrays: List of cv2 BGR arrays
    :return: List of PIL Images
    """
    images = list()
    for arr in arrays:
        images.append(Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)))
    return images


def annotate(model, img0, det, clss, annotate_normal=True):
    """

    :param model: Detection model
    :param img0: Original image
    :param det: Detection results of detection model
    :param clss: Classification results of classification model
    :param annotate_normal: Annotate all the detected objects in det
    """
    line_thickness = 3
    names = model.names
    annotator = Annotator(img0, line_width=line_thickness, example=str(names))
    i = len(det) - 1
    for *xyxy, conf, cls in reversed(det):
        if int(clss[i]) == 0 and (not annotate_normal):
            i -= 1
            continue
        c = int(clss[i])  # integer class
        label = f'class {clss[i]}  {conf:.2f}'
        annotator.box_label(xyxy, label, color=colors(c, True))
        i -= 1


def process_folder(folder_path: str, output_path: str, det_model, cls_model, annotate_normal=False):
    """

    :param folder_path: Path to the folder of all images that need processing
    :param output_path: Path to the folder to save all results
    :param det_model: Detection model
    :param cls_model: Classification model
    :param annotate_normal: Annotate all the detected objects in det
    """
    data = load_data(folder_path)

    print('\nLoad %d images' % len(data))
    print('Processing ... ...')
    for n in range(len(data)):
        print('>>>\n\t---Detecting Image %d ...:' % n)
        det = predict(det_model, data[n])
        print('\tNum of target objects: %d' % len(det))
        if len(det):

            clipped = clip_images(det, data[n])
            images = array2Image(clipped)
            img0 = data[n].copy()
            clipped_imgs = img_transform(images, 224)
            print('\t--- Classifying ...')
            clss = cls_predict(cls_model, clipped_imgs)
            print('\t->> Classification of Image %d: \n\t\t' % n, clss)
            print('\t--- Annotating ...')
            annotate(model=det_model, img0=img0, det=det, clss=clss, annotate_normal=annotate_normal)
            print('\t Save result as: [ ' + output_path + '/output%d.jpg' % n, ' ]')
            cv2.imwrite(output_path + '/output%d.jpg' % n, img0)
        else:
            print('\t No target objects')
            img0 = data[n].copy()
            print('\t Save result as: [ ' + output_path + '/output%d.jpg' % n, ' ]')
            cv2.imwrite(output_path + '/output%d.jpg' % n, img0)
        # img0 = cv2.resize(img0, (800, 800))
        # cv2.imshow('img0', img0)
        # cv2.waitKey(0)


if __name__ == '__main__':
    pass
    det_model = create_det_model(weight_path='./det/best.pt')
    cls_model = create_cls_model(num_cls=5, weight_path='./classify/weights/model-9.pth')
    process_folder('./test_images', './output_images', det_model, cls_model, True)
