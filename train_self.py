# self-semi training like MAE

import glob
import numpy as np
from ctypes.wintypes import RGB
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms


def show_img(path = r"D:\studyzc\QNRF\UCF-QNRF-1024x1024-mod16\train\img\1.jpg"):
    img = Image.open(path).convert('RGB')
    img = transforms.ToTensor()(img)
    print(img.shape)
    

if __name__=='__main__':
    l = glob.glob(r"D:\studyzc\jhu_crowd_v2.0\train\gt\*")
    d = {}
    for id, i in enumerate(l):
        
        d[id] = np.array(np.loadtxt(i, dtype=int))
        print(d[0])
        print(d[0].shape)
        print(type(d[0]))
        break