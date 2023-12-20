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

transform_depth_image = transforms.Compose([
            transforms.Resize([512,512]),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.Normalize([0.5], [0.5]),
        ])


class TimePredictionDataSet_Stru(Dataset):
    """
    data_path:包含所有的图片的路径和标签
    """
    # 初始化
    def __init__(self, dataset_path="./file/train_stru.json"):
        self.img_name = []
        self.num_color = []
        self.num_blocks = []
        self.time = []
        self.hint = []
        self.blk_per_color = []
        self.area_per_color = []
        with open(dataset_path, "r") as r:
            data = json.load(r)

        # 依次读取数据
        for key, value in data.items():
            self.img_name.append(key)
            self.num_color.append(value[0])
            self.num_blocks.append(value[1])
            self.blk_per_color.append(value[2])
            self.area_per_color.append(value[3])
            self.hint.append(value[4])
            self.time.append(value[5])

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):
        num_color = self.num_color[index]
        num_blocks = self.num_blocks[index]
        blk_per_color = self.blk_per_color[index]
        area_per_color = self.area_per_color[index]
        hint = self.hint[index]
        time = self.time[index]

        input_color = (torch.from_numpy(np.array(num_color))-79.092)/24.377 # 归一化，减去均值，除以标准差
        input_blocks = (torch.from_numpy(np.array(num_blocks))-834.756)/351.01 # 归一化，减去均值，除以标准差
        input_blk_per_color = torch.from_numpy(np.array(blk_per_color))
        input_area_per_color = torch.from_numpy(np.array(blk_per_color))
        input_hint = (torch.from_numpy(np.array(num_blocks))-0.573)/0.442
        input_time = torch.from_numpy(np.array(time))

        # print("名称:", img_name, "sehao:", input_sehao, "sekuai:", input_sekuai, "time:", input_time)
        # raise

        # torch.zeros(2, 2)用来占位，不用管
        return torch.zeros(2, 2), input_color, input_blocks, input_blk_per_color, input_area_per_color, input_hint, input_time