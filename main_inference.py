import argparse

parser = argparse.ArgumentParser(description='Model inference.')
parser.add_argument('--model-name', help='Name of model to train.', choices=['attnet', 'cenet', 'deeplabv3plus', 'doubleunet', 'mnet', 'mobilenet_unet', 'resnet_unet', 'resunet', 'unet', 'unetpp', 'sam'])
parser.add_argument('--image-path', help='Path to fundus image.', required=True)
parser.add_argument('--mask-path', help='Path to file location where the mask will be saved.', required=True)
parser.add_argument('--path-model', help='Path to the saved model.', required=True)
parser.add_argument('--img-size', type=int, help='Size to which the images should be reshaped (one number, i.e. 256 or 512).', required=True)
parser.add_argument('--binary', type=bool, help='Whether the segmentation masks are binary (True) or multi-class (False).', default=False)

args = parser.parse_args()

from utils.metrics import *
import numpy as np
import os
from PIL import Image
from utils.data_utils import *
import pandas as pd
import os.path as osp
from tqdm import tqdm

from models.attnet import AttNet
from models.cenet import CENet
from models.deeplabv3plus import DeepLabV3Plus
from models.doubleunet import DoubleUnet
from models.mnet import MNet
from models.mobilenet_unet import MobileNetUnet
from models.resnet_unet import ResNetUnet
from models.resunet import ResUnet
from models.unet import Unet
from models.unetpp import UnetPlusPlus
from models.sam import SAM

from utils.data_utils import *

img_size = (args.img_size, args.img_size)
n_classes = 2 if args.binary else 3
model = {
    'attnet': AttNet,
    'cenet': CENet,
    'deeplabv3plus': DeepLabV3Plus,
    'doubleunet': DoubleUnet,
    'mnet': MNet,
    'mobilenet_unet': MobileNetUnet,
    'resnet_unet': ResNetUnet,
    'resunet': ResUnet,
    'unet': Unet,
    'unetpp': UnetPlusPlus,
    'sam': SAM
}[args.model_name]((img_size[0],img_size[1],3), n_classes)

torch_models = ['cenet', 'sam']
polar_models = [] # ['mnet']

img = process_img(args.image_path, img_size, binary=args.binary==1, polar=(args.model_name in polar_models), channelsFirst=(args.model_name in torch_models))

model.load(args.path_model)
pred = model.predict(np.expand_dims(img, axis=0))[0]
if args.model_name in torch_models:
    pred = np.moveaxis(pred, 0, -1)
im_pred = np.argmax(pred, axis=2) if pred.shape[-1] > 1 else pred.reshape(pred.shape[:-1])
im = Image.fromarray((im_pred * 255).astype(np.uint8))
im.save(args.mask_path)