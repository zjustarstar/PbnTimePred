# 存储色块 色号 面积统计 坐标分布　色号对应的色块数目　特征
# 会按照4比１的比例随机生成train.json文件和test.json文件，这个文件在train.py的第18-20行分别指明

import json
import pandas
import random
from PIL import  Image
import numpy as np
import os

# 最多120种颜色?
MAX_COLORS = 120

def get_color_based16(clr_str):
    r = int(clr_str[0:2], 16)
    g = int(clr_str[2:4], 16)
    b = int(clr_str[4:6], 16)

    return (r,g,b)


def get_stat_info(data):
    ave_val = np.mean(np.array(data)) # 79.09247191011237
    max_val = np.max(np.array(data))
    std_val = np.std(np.array(data))  # 24.377676506738464
    return ave_val, max_val, std_val


def get_block_info(img_root, id, blocks_list, total_blocks_num, colors_list):
    img_path = os.path.join(img_root, id)
    if not os.path.exists(img_path):
        print(f'{img_path} do not exist!')
        return

    im = Image.open(img_path)
    if im.mode == 'RGBA':
        im = np.array(im)
        rgb_info = im[:, :, :3]
        a_info = im[:, :, 3]
        rgb_info[a_info == 0] = [255, 255, 255]
        im = Image.fromarray(rgb_info)

    im_array = np.array(im)

    # 每种颜色的区块个数，以及每种颜色的面积
    blks_per_color = []
    area_per_color = []
    s = im_array.shape[0] * im_array.shape[1]
    for ind, blocks in enumerate(blocks_list):
        blk = blocks.split(',')
        rgb_list = []
        for b in blk:
            bb = int(b) % 256
            gg = int(b) // 256
            rr = 0
            rgb_list.append([rr, gg, bb])

        # 查找img_array中数值是pixels中任意一个的像素mask，以及总数
        pixels = np.array(rgb_list)
        mask = np.zeros(im_array.shape[:2], dtype=bool)
        for pv in pixels:
            mask = np.any([mask, np.all(im_array == pv, axis=-1)], axis=0)

        area = np.sum(mask == True)

        blks_per_color.append(len(blk)/total_blocks_num)
        area_per_color.append(area/s)

    # 不够数量的，直接补充到最大数量
    margin = MAX_COLORS - len(blks_per_color)
    flag = True
    if margin > 0:
        t = [0] * margin
        blks_per_color = blks_per_color + t
        area_per_color = area_per_color + t
    else:
        flag = False

    return flag, blks_per_color, area_per_color


def get_all_data_info(cvs_file_path, whole_json_path, img_root_path):
    book_1 = pandas.read_csv(cvs_file_path, encoding="gb2312")
    ids = book_1['id'].values.tolist()
    plan_1 = book_1['plan'].values.tolist()
    total_color_num = book_1['色号'].values.tolist()   # 最多不超过120
    total_block_num = book_1['色块'].values.tolist()
    time = book_1['时长'].values.tolist()
    hint = book_1['hint'].values.tolist()

    # 求色号和色块数等的均值和标准差
    ave_clr_num, max_clr_num, std_clr_num = get_stat_info(total_color_num)
    print(f'ave_color_num={ave_clr_num}, max={max_clr_num}, std={std_clr_num}')
    ave_blk_num, max_blk_num, std_blk_num = get_stat_info(total_block_num)
    print(f'ave_block_num={ave_blk_num}, max={max_blk_num}, std={std_blk_num}')
    ave_hint, max_hint, std_hint = get_stat_info(hint)
    print(f'ave_hit={ave_hint}, max={max_hint}, std={std_hint}')

    # ----处理色号对应的色块数量特征 -------------
    plan = {}

    for i in range(len(ids) - 1):
        img_name = ids[i]
        content = str(plan_1[i])
        teams = content.split('|')

        block_indexs = []
        block_colors = []
        for item in teams:
            sharp_index = item.find('#')
            blk_index = item[:sharp_index]
            clr = item[sharp_index+1:len(item)]
            block_indexs.append(blk_index)
            block_colors.append(get_color_based16(clr))

        block_dict = {}
        block_dict['blocks_index'] = block_indexs
        block_dict['blocks_color'] = block_colors

        if len(block_indexs) == 0:
            continue

        plan[img_name + ".png"] = block_dict

    # 读取色号色块特征和时长
    data = {}
    for i in range(len(ids)):
        print(f'processing {i}/{len(ids)}...')
        id = ids[i]
        key = str(id) + ".png"
        if key not in plan:
            continue
        block_dict = plan[key]
        blk_index = block_dict['blocks_index']
        blk_color = block_dict['blocks_color']

        # 获取每个色块的区块个数占比，以及面积占比;
        # 如果色块数超过最大值，这个样本不考虑;
        flag, blks_per_color, area_per_color = get_block_info(img_root_path, key, blk_index, total_block_num[i], blk_color)
        if not flag:
            print(f'file id = {id}, color num={total_color_num}, exceed 120....drop it')
            continue

        data[ids[i] + ".png"] = [total_color_num[i], total_block_num[i], blks_per_color, area_per_color, hint[i], time[i]]

    # 将所有数据写入whole_original.json文件
    with open(whole_json_path, "w") as f:
        json.dump(data, f)

    return data


# 将数据分别写入train.json和test.json文件
def create_train_test_json(data, ratio=0.8):
    train_json_path = "../file/train_stru.json"
    test_json_path = "../file/test_stru.json"
    train_data = {}
    test_data = {}

    with open(train_json_path, "w") as f_train:
        with open(test_json_path, "w") as f_test:
            for key, value in data.items():
                random_num = random.random()
                if random_num < ratio:
                    train_data[key] = value
                else:
                    test_data[key] = value
                    # train_data[data[i]]
            json.dump(train_data, f_train)
            json.dump(test_data, f_test)

    print("训练集大小:", len(train_data))
    print("测试集大小:", len(test_data))
# -------------------- 将数据分别写入train.json和test.json文件 ---------------------------------


if __name__ == '__main__':
    # 将所有信息统计后，输出到whole_json_path文件中
    whole_json_path = "../file/whole_original_stru.json"

    # 输入的cvs文件
    # cvs_file_path = "..//file//test_data_8900_with_plan_whole.csv"
    cvs_file_path = "..//file//test1.csv"

    img_root_path = "F://data//乐信//PBN//PBN_TimePred//suoluetu"
    data = get_all_data_info(cvs_file_path, whole_json_path, img_root_path)
    # 80%的训练数据
    create_train_test_json(data, 0.8)
