# import torch
# import numpy as np
# from det.models.common import DetectMultiBackend
from det.det import imgTransform, predict, load_images,  create_det_model

# from classify.classify import cls_predict
from utils import process_folder
from classify.classify import create_cls_model
import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

det_model = create_det_model(weight_path='./det/best.pt')
cls_model = create_cls_model(num_cls=5, weight_path='./classify/weights/model-9.pth')


if __name__ == '__main__':
    det_model = create_det_model(weight_path='./det/best.pt')
    cls_model = create_cls_model(num_cls=5, weight_path='./classify/weights/model-9.pth')
    process_folder('./test_images', './output_images', det_model, cls_model, True)