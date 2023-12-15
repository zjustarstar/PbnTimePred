# 存储每张图中每个块的面积

import json
import pandas
import random
import os
import numpy as np
import cv2

def get_color_range(img):
    """
    获取 region_img 中的色彩范围
    :param img:
    :return:
    """
    img[np.all(img - [255, 255, 255] == 0, axis=2)] = [0, 0, 0]
    img = np.array(img, dtype=int)
    min_color = (np.min(img[:, :, 0] + img[:, :, 1] * 256))
    max_color = (np.max(img[:, :, 0] + img[:, :, 1] * 256))

    unique_colors = img[:, :, 0] + img[:, :, 1] * 256
    unique_colors = np.unique(unique_colors)
    unique_colors = np.delete(unique_colors, np.where(unique_colors==0))

    return min_color, max_color, unique_colors


img_path = "/home/cgim/wushukai/code/LeXin/LexinTimePrediction/TimePrediction/suoluetu-final"
data = {}


count = 0
max_area_value = 0
for file in os.listdir(img_path):
    origin_region_img = cv2.imread(os.path.join(img_path, file))
    min_color, max_color, unique_colors = get_color_range(origin_region_img)
    count += 1
    print(count, file)

    # num_patch = 0
    area_list = []

    for color_type in unique_colors:

        # num_patch += 1
        # print(num_patch)

        b_type = color_type % 256
        g_type = int(color_type / 256)

        # mask表示当前分割区域
        region_img = np.copy(origin_region_img)
        mask = cv2.inRange(region_img, np.array([b_type, g_type, 0]), np.array([b_type, g_type, 0])) # 通过对比原图和上下界值，返回一个二值化的掩码图像

        area = np.count_nonzero(mask)
        area_list.append(area)

    area_list.sort()
    print(len(area_list), sum(area_list), area_list)
    data[file] = area_list

    if area_list[-1] > max_area_value:
        max_area_value = area_list[-1]


print("最大面积:", max_area_value)

whole_json_path ="/home/cgim/wushukai/code/LeXin/LexinTimePrediction/Data/data-1/area.json"
with open(whole_json_path, "w") as f:
    json.dump(data, f)
    print("ok")
