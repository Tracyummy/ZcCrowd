import glob
import torchvision.transforms as transforms
from PIL import Image
import csv
import numpy as np

transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),
                            ])

class Jhu_Density_Dataset():
    def __init__(self, train_mode = True, cfg_data = None) -> None:
        self.train_img_list = glob.glob(r"D:\studyzc\jhu_crowd_v2.0\train\images\*")  #  把路径都存在这个列表里
        self.test_img_list = glob.glob(r"D:\studyzc\jhu_crowd_v2.0\test\images\*")

        self.train_label = self.train_list.replace("images","gt").replace("jpg","txt")
        self.train_points = {}
        for id, i in enumerate(self.train_label):
            self.train_points[id] = np.array(np.loadtxt(i, dtype=int))

        self.test_label = self.test_list.replace("images","gt").replace("jpg","txt")
        self.test_points = {}
        for id, i in enumerate(self.test_label):
            self.test_points[id] = np.array(np.loadtxt(i, dtype=int))

        self.cfg_data = cfg_data
        self.train_mode = True
        self.transform = transform


    def __getitem__(self, index):
        img_path = self.train_img_list[index]
        img = Image.open(img_path).convert('RGB')
        img =  self.transform(img)

        gt_path = img_path.replace("img","gt").replace("jpg","txt")
        points = np.array(np.loadtxt(gt_path, dtype=int))[:, :4]

        if self.train_mode:
            if self.cfg_data.crop:
                img, points = self.crop_img(img, points)
            if self.horizon_flip:
                img, points = self.hflip(img, points)
            return img, len(points)

        else:
            return img, len(points)

    def __len__(self):
        return len(self.train_list)


if __name__=='__main__':
    l = glob.glob(r"D:\studyzc\QNRF\UCF-QNRF-1024x1024-mod16\train\img\*")
    print(len(l))