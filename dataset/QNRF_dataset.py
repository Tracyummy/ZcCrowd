import glob
import torchvision.transforms as transforms
from PIL import Image
import csv

transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),
                            ])

class QNRF_Density_Dataset():
    def __init__(self, train_mode = True, cfg_data) -> None:
        self.train_img_list = glob.glob(r"D:\studyzc\QNRF\UCF-QNRF-1024x1024-mod16\train\img\*")  #  把路径都存在这个列表里
        self.test_img_list = glob.glob(r"D:\studyzc\QNRF\UCF-QNRF-1024x1024-mod16\test\img\*")
        self.train_label = [x.replace("img","den").replace("jpg","csv")  for x  in  self.train_img_list]
        self.test_label  = [x.replace("img","den").replace("jpg","csv")  for x  in  self.test_img_list]
        self.cfg_data = cfg_data
        self.train_mode = True
        self.transform = transform


    def __getitem__(self, index):
        img_path = self.train_img_list[index]
        gt_path = img_path.replace("img","den").replace("jpg","csv")
        img = Image.open(img_path).convert('RGB')
        gt = 0
        img =  self.transform(img)
        with open(gt_path, "r") as f:
            for row in csv.reader(f):
                gt += 1
        if self.train_mode:
            if self.cfg_data.crop:
                img = transforms
            return img, gt
        else:
            return img, gt

    def __len__(self):
        return len(self.train_list)


if __name__=='__main__':
    l = glob.glob(r"D:\studyzc\QNRF\UCF-QNRF-1024x1024-mod16\train\img\*")
    print(len(l))