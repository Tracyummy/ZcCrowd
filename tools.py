import re
import numpy as np
import torch
import torch.nn as nn
import glob
import cv2
import PIL.Image as Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import scipy.io as sio
import pandas as pd
import os


def read_img2tensor(path = u""):
    img = cv2.imread(path)
    img = transforms.ToTensor()(img)
    return img

def pad_img(img):
    if img.shape==3:
        img = img.unsqueeze(0)
    img = F.pad(img, (128,128,128,128), 'reflect')
    img = img.squeeze(0)
    return img

def show_img(img):
    img = transforms.ToPILImage()(img)
    img.show("img")
    return img

def read_density():
    den = sio.loadmat(os.path.join('.mat'))
    den = den['map']
    den = pd.read_csv(os.path.join('.csv'), sep=',',header=None).values
    return den

def read_image_and_gt(index):
    img = Image.open()
    if img.mode == 'L':
        img = img.convert('RGB')
    den = pd.read_csv("", sep=',',header=None).values
    den = den.astype(np.float32, copy=False)
    den = Image.fromarray(den)
    return img, den

def gaussian_kernel(size, sigma):
    x, y = np.mgrid[-size:size+1, -size:size+1]
    kernel = np.exp(-0.5*(x*x+y*y)/(sigma*sigma))
    kernel /= kernel.sum()
    return kernel


class SSIM_Loss(torch.nn.modules.loss._Loss):
    def __init__(self, in_channels, size=11, sigma=1.5, size_average=True):
        super(SSIM_Loss, self).__init__(size_average)
        #assert in_channels == 1, 'Only support single-channel input'
        self.in_channels = in_channels
        self.size = int(size)
        self.sigma = sigma
        self.size_average = size_average

        kernel = gaussian_kernel(self.size, self.sigma)
        self.kernel_size = kernel.shape
        weight = np.tile(kernel, (in_channels, 1, 1, 1))
        self.weight = nn.Parameter(torch.from_numpy(weight).float(), requires_grad=False)

    def forward(self, input, target, mask=None):
        # _assert_no_grad(target)
        mean1 = F.conv2d(input, self.weight, padding=self.size, groups=self.in_channels)
        mean2 = F.conv2d(target, self.weight, padding=self.size, groups=self.in_channels)
        mean1_sq = mean1*mean1
        mean2_sq = mean2*mean2
        mean_12 = mean1*mean2

        sigma1_sq = F.conv2d(input*input, self.weight, padding=self.size, groups=self.in_channels) - mean1_sq
        sigma2_sq = F.conv2d(target*target, self.weight, padding=self.size, groups=self.in_channels) - mean2_sq
        sigma_12 = F.conv2d(input*target, self.weight, padding=self.size, groups=self.in_channels) - mean_12
    
        C1 = 0.01**2
        C2 = 0.03**2

        ssim = ((2*mean_12+C1)*(2*sigma_12+C2)) / ((mean1_sq+mean2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
        if self.size_average:
            out = 1 - ssim.mean()
        else:
            out = 1 - ssim.view(ssim.size(0), -1).mean(1)
        return out



if __name__=='__main__':
    p = r"D:\studyzc\NWPU-Dataset\NWPU-Crowd\images_part1\0001.jpg"
    img = read_img2tensor(p)
    img = pad_img(img)
    show_img(img)