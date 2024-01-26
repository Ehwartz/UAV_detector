import torch
import numpy as np
from .models.common import DetectMultiBackend
# import torchvision
# from PIL import Image
from .utils.augmentations import letterbox
from .utils.general import non_max_suppression, scale_coords
# from .utils.plots import Annotator, colors
import cv2
import os

# import sys
# from pathlib import Path

imgsz = (640, 640)  # inference size (height, width)
conf_thres = 0.3  # confidence threshold
iou_thres = 0.4  # NMS IOU threshold
max_det = 1000
agnostic_nms = False
line_thickness = 3


def create_det_model(weight_path: str, img_size=imgsz):
    """

    :param weight_path: Path to weights file for detection model
    :param img_size: Size of input image to resize
    :return: Detection model
    """
    print('Creating model for detection: ... ...')

    model = DetectMultiBackend(weights=weight_path)
    stride, names, pt = model.stride, model.names, model.pt
    bs = 1
    model.warmup(imgsz=(1 if pt else bs, 3, *img_size))
    model.warmup()
    print('Model for detection created and warmed up')
    return model


def imgTransform(model, img0):
    """

    :param model: Provide the data type of Tensor
    :param img0: Input image, cv2 array
    :return: Tensor as input for detection model
    """
    img = img0.copy()
    img = letterbox(img)[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    img = img.half() if model.fp16 else img.float()
    img /= 255
    img = img[None]
    return img


def predict(model, img):
    """

    :param model: Detection model
    :param img: input image for detection
    :return: List of detection results
    """
    # print('\t Predicting image... ...')
    # print(img.shape)
    img0 = img.copy()
    img = imgTransform(model, img)
    pred = model(img)
    pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=max_det)

    det = pred[0]
    if len(det):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

    return det


def load_images(path: str):
    """

    :param path: Path to folder of images
    :return: List of image paths
    """
    imgs = os.listdir(path)
    for i, img in enumerate(imgs):
        imgs[i] = path + '/' + img
    return imgs


def clip(model, imgs: list, dst_path):
    """
    Useless function

    :param model:
    :param imgs:
    :param dst_path:
    """
    for i, img_path in enumerate(imgs):
        folder_name = imgs[i][0:-4].split('/')[-1]
        dst_folder = dst_path + '/' + folder_name
        os.makedirs(dst_folder)
        img = cv2.imread(img_path)
        img0 = img.copy()
        det = predict(model, img)
        for n in range(det.shape[0]):
            clipped = img0[int(det[n][1]):int(det[n][3]), int(det[n][0]):int(det[n][2])]
            print('Saving: ', dst_folder + '/%d' % n + '.jpg')
            cv2.imwrite(dst_folder + '/%d' % n + '.jpg', clipped)


if __name__ == '__main__':
    pass
