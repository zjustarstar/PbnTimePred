from torch.utils.data import Dataset
import cv2
import json
import torch
from PIL import Image
import numpy as np
import os
import random


from torchvision import transforms
transform_train_1 = transforms.Compose([
            transforms.Normalize([79.09], [24.38]),
        ])

transform_train_2 = transforms.Compose([
            transforms.Normalize([834.76], [351.01]),
        ])

transform_test_1 = transforms.Compose([
            transforms.Normalize([79.09], [24.38]),
        ])

transform_test_2 = transforms.Compose([
            transforms.Normalize([834.76], [351.01]),
        ])

transform_depth_image = transforms.Compose([
            transforms.Resize([512,512]),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # transforms.Normalize([0.5], [0.5]),
        ])

transform_vit = transforms.Compose([
            transforms.Resize([384,384]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # transforms.Normalize([0.5], [0.5]),
        ])

class PbnImgDataSet(Dataset):
    """
    data_path:包含所有的图片的路径和标签
    """
    # 初始化
    def __init__(self, dataset_path="./file/train.json",
                 img_root_path="D://LexinData//pbn_timepred//",
                 model='vit'):
        self.img_name = []
        self.time = []
        self.model = model
        self.img_rootpath = img_root_path

        with open(dataset_path, "r") as r:
            data = json.load(r)

        for i in range(1):
            root = os.path.join(img_root_path, "copy"+str(i))
            for key, value in data.items():
                fullname = os.path.join(root, key)
                self.img_name.append(fullname)
                # json中的内容为[sehaoshu[i], sekuaishu[i], plan[indexes_1[i] + ".png"], time[i]]
                self.time.append(value[3])

        self.transforms = 0
        if model == 'resnet':
            self.transforms = transform_depth_image
        elif model == 'vit':
            self.transforms = transform_vit

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):
        img_name = self.img_name[index]
        label = self.time[index]
        # img_path = os.path.join(self.img_rootpath, img_name)
        default_size = 512
        if self.model == 'vit':
            default_size = 384
        try:
            im = Image.open(img_name)
            if im.mode == 'RGBA':
                im = np.array(im)
                rgb_info = im[:, :, :3]
                a_info = im[:, :, 3]
                rgb_info[a_info == 0] = [255, 255, 255]
                im = Image.fromarray(rgb_info)

            if self.transforms:
                try:
                    img = self.transforms(im)
                except Exception as e:
                    print("Cannot transform image: {}".format(img_name))
        except IOError:
            # print(f"fail to open image {img_name}")
            return torch.zeros((3,default_size,default_size)), 0

        return img, label


class PbnImgDataSet_Test(Dataset):
    """
    data_path:包含所有的图片的路径和标签
    """
    # 初始化
    def __init__(self, dataset_path="./file/train.json",
                 img_root_path="D://LexinData//pbn_timepred//copy0",
                 model='vit'):
        self.img_name = []
        self.time = []
        self.img_rootpath = img_root_path
        self.model = model

        with open(dataset_path, "r") as r:
            data = json.load(r)

        for key, value in data.items():
            self.img_name.append(key)
            # json中的内容为[sehaoshu[i], sekuaishu[i], plan[indexes_1[i] + ".png"], time[i]]
            self.time.append(value[3])

        self.transforms = 0
        if model == 'resnet':
            self.transforms = transform_depth_image
        elif model == 'vit':
            self.transforms = transform_vit

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):
        img_name = self.img_name[index]
        label = self.time[index]
        img_path = os.path.join(self.img_rootpath, img_name)

        default_size = 512
        if self.model == 'vit':
            default_size = 384

        try:
            im = Image.open(img_path)
            if im.mode == 'RGBA':
                im = np.array(im)
                rgb_info = im[:, :, :3]
                a_info = im[:, :, 3]
                rgb_info[a_info == 0] = [255, 255, 255]
                im = Image.fromarray(rgb_info)
        except IOError:
            print(f'fail to open {img_path}')
            return torch.zeros((3,default_size,default_size)), 0

        if self.transforms:
            try:
                img = self.transforms(im)
            except Exception as e:
                print("Cannot transform image: {}".format(img_name))

        return img, label
