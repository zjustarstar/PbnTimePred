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
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.Normalize([0.5], [0.5]),
        ])


class TimePredictionDataSet(Dataset):
    """
    data_path:包含所有的图片的路径和标签
    """
    # 初始化
    def __init__(self, dataset_path="./file/train_stru.json"):
        self.img_name = []
        self.num_sehao = []
        self.num_sekuai = []
        self.time = []
        self.num_area = []
        self.pos_distribution = []
        self.sekuai_distribution = []
        with open(dataset_path, "r") as r:
            data = json.load(r)

        # 依次读取数据
        for key, value in data.items():
            self.img_name.append(key)
            self.num_sehao.append(value[0])
            self.num_sekuai.append(value[1])
            self.num_area.append(value[2])
            self.pos_distribution.append(value[3])
            self.sekuai_distribution.append(value[4])
            self.time.append(value[5])

        self.transforms = transform_depth_image

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):
        num_sehao = self.num_sehao[index]
        num_sekuai = self.num_sekuai[index]
        num_area = self.num_area[index]
        pos_distribution = self.pos_distribution[index]
        sekuai_distribution = self.sekuai_distribution[index]
        time = self.time[index]

        input_sehao = (torch.from_numpy(np.array(num_sehao))-79.09247191011237)/24.377676506738464 # 归一化，减去均值，除以标准差
        input_sekuai = (torch.from_numpy(np.array(num_sekuai))-834.7568539325842)/351.00993773055467 # 归一化，减去均值，除以标准差
        input_num_area = torch.from_numpy(np.array(num_area))
        input_pos_distribution = torch.from_numpy(np.array(pos_distribution))
        input_sekuai_distribution = torch.from_numpy(np.array(sekuai_distribution))
        input_time = torch.from_numpy(np.array(time))

        # print("名称:", img_name, "sehao:", input_sehao, "sekuai:", input_sekuai, "time:", input_time)
        # raise

        # torch.zeros(2, 2)用来占位，不用管
        return torch.zeros(2, 2), input_sehao, input_sekuai, input_num_area, input_pos_distribution, input_sekuai_distribution, input_time



class TimePredictionDataSetTrain(Dataset):
    """
    data_path:包含所有的图片的路径和标签
    """
    # 初始化
    def __init__(self, dataset_path="/home/cgim/wushukai/code/LeXin/LexinTimePrediction/Data/train.json"):
        self.img_name = []
        self.num_sehao = []
        self.num_sekuai = []
        self.time = []
        self.num_area = []
        self.pos_distribution = []
        self.sekuai_distribution = []
        with open(dataset_path, "r") as r:
            data = json.load(r)
        for key, value in data.items():
            self.img_name.append(key)
            self.num_sehao.append(value[0])
            self.num_sekuai.append(value[1])
            self.num_area.append(value[2])
            self.pos_distribution.append(value[3])
            self.sekuai_distribution.append(value[4])
            self.time.append(value[5])

        self.transforms = transform_depth_image
        # self.transforms_2 = transform_depth_image_2

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):

        # img_name = self.img_name[index]
        # print(img_name)
        # img = cv2.imread(os.path.join("/home/cgim/wushukai/code/LeXin/LexinTimePrediction/TimePrediction/suoluetu",img_name))
        # img = Image.fromarray(img)

        # img = Image.open(os.path.join("/home/cgim/wushukai/code/LeXin/LexinTimePrediction/TimePrediction/souluetu-single-channel", img_name))

        # img = cv2.imread(os.path.join("/home/cgim/wushukai/code/LeXin/LexinTimePrediction/TimePrediction/souluetu-single-channel", img_name), 0)
        # print(img)
        # print(len(img.split()))
        # print(type(img))
        # img = Image.fromarray(img)

        num_sehao = self.num_sehao[index]
        num_sekuai = self.num_sekuai[index]
        time = self.time[index]
        num_area = self.num_area[index]
        pos_distribution = self.pos_distribution[index]
        sekuai_distribution = self.sekuai_distribution[index]
        input_sehao = (torch.from_numpy(np.array(num_sehao))-79.09247191011237)/24.377676506738464
        input_sekuai = (torch.from_numpy(np.array(num_sekuai))-834.7568539325842)/351.00993773055467
        input_num_area = torch.from_numpy(np.array(num_area))
        input_pos_distribution = torch.from_numpy(np.array(pos_distribution))
        input_sekuai_distribution = torch.from_numpy(np.array(sekuai_distribution))
        input_time = torch.from_numpy(np.array(time))

        # print("名称:", img_name, "sehao:", input_sehao, "sekuai:", input_sekuai, "time:", input_time)
        # raise

        # if self.transforms:
        #     try:
        #         img = self.transforms(img)
        #     except:
        #         print("Cannot transform image: {}".format(img_name))

        # img = torch.unsqueeze(img, 1)


        return torch.zeros(2, 2), input_sehao, input_sekuai, input_num_area, input_pos_distribution, input_sekuai_distribution, input_time


class TimePredictionDataSetTest(Dataset):
    """
    data_path:包含所有的图片的路径和标签
    """
    # 初始化
    def __init__(self, dataset_path="/home/cgim/wushukai/code/LeXin/LexinTimePrediction/Data/test.json"):
        self.img_name = []
        self.num_sehao = []
        self.num_sekuai = []
        self.num_area = []
        self.pos_distribution = []
        self.sekuai_distribution = []
        self.time = []
        with open(dataset_path, "r") as r:
            data = json.load(r)
        for key, value in data.items():
            self.img_name.append(key)
            self.num_sehao.append(value[0])
            self.num_sekuai.append(value[1])
            self.num_area.append(value[2])
            self.pos_distribution.append(value[3])
            self.sekuai_distribution.append(value[4])
            self.time.append(value[5])

        self.transforms = transform_depth_image

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):

        # img_name = self.img_name[index]
        # print(img_name)
        # img = cv2.imread(os.path.join("/home/cgim/wushukai/code/LeXin/LexinTimePrediction/TimePrediction/suoluetu",img_name))
        # img = Image.fromarray(img)

        # img = Image.open(os.path.join("/home/cgim/wushukai/code/LeXin/LexinTimePrediction/TimePrediction/souluetu-single-channel", img_name))

        # img = cv2.imread(os.path.join("/home/cgim/wushukai/code/LeXin/LexinTimePrediction/TimePrediction/souluetu-single-channel", img_name), 0)
        # print(img)
        # print(len(img.split()))
        # print(type(img))
        # img = Image.fromarray(img)

        num_sehao = self.num_sehao[index]
        num_sekuai = self.num_sekuai[index]
        num_area = self.num_area[index]
        time = self.time[index]
        pos_distribution = self.pos_distribution[index]
        sekuai_distribution = self.sekuai_distribution[index]
        input_sehao = (torch.from_numpy(np.array(num_sehao))-79.09247191011237)/24.377676506738464
        input_sekuai = (torch.from_numpy(np.array(num_sekuai))-834.7568539325842)/351.00993773055467
        input_num_area = torch.from_numpy(np.array(num_area))
        input_pos_distribution = torch.from_numpy(np.array(pos_distribution))
        input_sekuai_distribution = torch.from_numpy(np.array(sekuai_distribution))
        input_time = torch.from_numpy(np.array(time))

        # print("名称:", img_name, "sehao:", input_sehao, "sekuai:", input_sekuai, "time:", input_time)
        # raise

        # if self.transforms:
        #     try:
        #         img = self.transforms(img)
        #     except:
        #         print("Cannot transform image: {}".format(img_name))

        # img = torch.unsqueeze(img, 1)


        return torch.zeros(2, 2), input_sehao, input_sekuai, input_num_area, input_pos_distribution, input_sekuai_distribution, input_time
