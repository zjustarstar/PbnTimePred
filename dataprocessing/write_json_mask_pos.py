# 存储每张图每个块的中心坐标
import json
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
pos_data = {}


count = 0
for file in os.listdir(img_path):
    origin_region_img = cv2.imread(os.path.join(img_path, file))
    min_color, max_color, unique_colors = get_color_range(origin_region_img)
    count += 1
    print(count, file)

    # num_patch = 0
    pos_list = []

    for color_type in unique_colors:

        # num_patch += 1
        # print(num_patch)

        b_type = color_type % 256
        g_type = int(color_type / 256)

        # mask表示当前分割区域
        region_img = np.copy(origin_region_img)
        mask = cv2.inRange(region_img, np.array([b_type, g_type, 0]), np.array([b_type, g_type, 0])) # 通过对比原图和上下界值，返回一个二值化的掩码图像

        # 寻找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # print(len(contours))
        # if len(contours) > 1:
        #     print(len(contours))

        # 遍历每个轮廓
        for contour in contours:
            # 计算轮廓的中心坐标
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

            pos_list.append((cX, cY))

    pos_list.sort()
    print(len(pos_list), pos_list)
    pos_data[file] = pos_list


whole_json_path ="/home/cgim/wushukai/code/LeXin/LexinTimePrediction/Data/data-2/mask_pos.json"
with open(whole_json_path, "w") as f:
    json.dump(pos_data, f)
    print("ok")
