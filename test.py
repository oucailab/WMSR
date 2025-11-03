import math
import argparse, yaml
import utils
import os
from tqdm import tqdm
import sys
import time
import importlib
import glob
from torchvision.transforms import Compose
import torch.nn.parallel
import cv2
import argparse
import glob
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
 
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
parser = argparse.ArgumentParser(description='WMSR')
parser.add_argument('--config', type=str, default='./configs/wmsr_x4.yml', help = 'pre-config file for training')
parser.add_argument('--resume', type=str, default=None, help = 'resume training or not')
import random
from datetime import datetime



args = parser.parse_args()
if args.config:
   opt = vars(args)
   yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
   opt.update(yaml_args)
## set visibel gpu   
gpu_ids_str = str(args.gpu_ids).replace('[','').replace(']','')
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_ids_str)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from datas.utils import create_datasets

device = torch.device('cuda')
torch.set_num_threads(args.threads)
model = utils.import_module('models.{}_network'.format(args.model, args.model)).create_model(args)
model = nn.DataParallel(model).to(device)
ckpt = torch.load('./X2/model_x4_10.pt')#'./model_x2_30.pt'#
model.load_state_dict(ckpt['model_state_dict'])
torch.set_grad_enabled(False)
model = model.eval()


def Pixel_rule2(input1):
    lr = input1
    lr = np.array(lr)
    lr = torch.from_numpy(lr)
    lr = lr.permute(2, 1, 0)
    lr = lr.float() 
    lr = lr[np.newaxis,:,:,:]
    lr = lr.to(device)
    with torch.no_grad():
        sr = model(lr)
    srr = sr.detach().cpu().numpy()
    srr = srr[0,:,:,:]
    srr = srr.transpose(1,2,0)

   # srrr = srr[:,:,::-1]

    return srr

   # cv2.imwrite('./b/{}_10pt.jpg'.format(i), srrr)


dir="./X4"
files = os.listdir(dir)
for f in files:
    if os.path.splitext(f)[1] == ".png":
        basename=os.path.splitext(f)[0]
        print(basename)
        img = cv2.imread("./X4/{}".format(f))
        img = cv2.flip(img, 1)
        rows, cols, channel = img.shape[ :3]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
        img = cv2.warpAffine(img, M, (cols, rows))
        result = Pixel_rule2(img)
        cv2.imwrite('./b/x4-{}'.format(f), result)
       # result.save('./b/{}.png'.format(basename))  
