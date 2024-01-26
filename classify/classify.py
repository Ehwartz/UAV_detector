# import os
# import json

import torch
# from PIL import Image
from torchvision import transforms
# import matplotlib.pyplot as plt

from .model import swin_tiny_patch4_window7_224 as create_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_cls_model(num_cls: int, weight_path: str):
    """

    :param num_cls: Number of class
    :param weight_path: Path to weights file for classification model
    :return: Classification model
    """
    print('Creating model for classification: ... ...')
    model = create_model(num_classes=num_cls)  # .to(device)

    model.load_state_dict(torch.load(weight_path, map_location=device))
    print('Model for classification created')
    return model


def img_transform(images, img_size=224):
    """

    :param images:
    :param img_size:
    :return: Integrated Tensor
    """
    channel = 3
    n = len(images)

    transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # integrate all clipped images into a whole Tensor
    imgs = torch.empty(n, channel, img_size, img_size)
    for i in range(n):
        img = images[i].copy()
        img = transform(img)
        imgs[i, :, :, :] = img
    return imgs


def cls_predict(model, imgs):
    """

    :param model: Classification model
    :param imgs: Clipped and transformed images' Tensors
    :return: Classification results
    """

    model.eval()
    with torch.no_grad():
        output = model(imgs)
        pred = torch.softmax(output, dim=1)

    return torch.argmax(pred, dim=1)


if __name__ == '__main__':
    imgs = img_transform('./imagenet/1', 224)
    print(imgs.shape)
    model = create_model(num_classes=5).to(device)
    # load model weights
    model_weight_path = "./weights/model-9.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        output = model(imgs)
        pred = torch.softmax(output, dim=1)

    pass
    print(pred)
    print(torch.argmax(pred, dim=1))
