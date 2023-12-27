# 存储色块 色号 面积统计 坐标分布　色号对应的色块数目　特征
# 会按照4比１的比例随机生成train.json文件和test.json文件，这个文件在train.py的第18-20行分别指明

import json
import math

import pandas
import random
from multiprocessing import Pool
from functools import  partial
from PIL import  Image
import numpy as np
import os

# 最多140种颜色?
MAX_COLORS = 140

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


def get_block_index(mask, im_array, block_size):
    N = block_size
    blocksize_x, blocksize_y = im_array.shape[1] / N, im_array.shape[0] / N

    # 平均x,y坐标;
    indices = np.where(mask)
    meanx = np.mean(indices[1])
    meany = np.mean(indices[0])

    # 确定坐标所在的区块
    if not np.isnan(meanx):
        blkx = min(int(meanx // blocksize_x), N-1)
        blky = min(int(meany // blocksize_y), N-1)
        blk_ind = int(blky * N + blkx)
    else:
        blk_ind = -1

    return blk_ind


def get_block_info(img_root, id, blocks_list, total_blocks_num, colors_list):
    img_path = os.path.join(img_root, id)
    if not os.path.exists(img_path):
        print(f'{img_path} do not exist!')
        return False, 0, 0, 0, 0
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
        return False, 0, 0, 0, 0

    im_array = np.array(im)

    # 每种颜色的区块个数，以及每种颜色的面积
    blks_per_color = []
    area_per_color = []
    s = im_array.shape[0] * im_array.shape[1]
    small_area_nums = [0] * 100   #[40,80,120,160,200,240,..]
    block_counts = [0] * 100     #区域的平均分布，分为10*10区域
    for ind, blocks in enumerate(blocks_list):
        blk = blocks.split(',')
        rgb_list = []
        # 每一种颜色的面积
        total_area = 0
        for b in blk:
            bb = int(b) % 256
            gg = int(b) // 256
            rr = 0
            rgb_list.append([rr, gg, bb])
            mask = np.all(im_array == (rr, gg, bb), axis=-1)
            area = np.sum(mask == True)
            # 小区域的个数
            if area//40 < 100:
                small_area_nums[area//40] += 1
            # 总区域大小
            total_area += area

            # 当前blcok所在的区块
            ind = get_block_index(mask, im_array, 10)
            if ind>=0:
                block_counts[ind] += 1

        blks_per_color.append(len(blk)/total_blocks_num)
        area_per_color.append(total_area/s)

    # 不够数量的，直接补充到最大数量
    margin = MAX_COLORS - len(blks_per_color)
    flag = True
    if margin > 0:
        t = [0] * margin
        blks_per_color = blks_per_color + t
        area_per_color = area_per_color + t
    elif margin < 0:
        flag = False

    # 归一化
    block_counts = [num / total_blocks_num for num in block_counts]

    return flag, blks_per_color, area_per_color, small_area_nums, block_counts


def main_thread(i, total_num, total_color_num, time, hint, img_root_path, ids, total_block_num, plan):
    # i, total_num, total_color_num, time, hint, img_root_path, ids, total_block_num, plan = args
    print(f'processing {i}/{total_num}...')
    id = ids[i]
    key = str(id) + ".png"
    if key not in plan:
        return {}

    block_dict = plan[key]
    blk_index = block_dict['blocks_index']
    blk_color = block_dict['blocks_color']

    # 获取每个色块的区块个数占比，以及面积占比;
    # 如果色块数超过最大值，这个样本不考虑;
    flag, blks_per_color, area_per_color, small_area_nums, block_counts = get_block_info(img_root_path, key, blk_index, total_block_num[i], blk_color)
    if not flag:
        print(f'file id = {id}, color num={total_color_num[i]}, exceed {MAX_COLORS}....drop it')
        return {}
    print(f'small areas:{small_area_nums[0:10]}, block_dist:{block_counts[0:5]}, time={time[i]}, total_block_num={total_block_num[i]}')

    data = {}
    data[ids[i] + ".png"] = [total_color_num[i], total_block_num[i], blks_per_color, area_per_color, small_area_nums, block_counts, hint[i], time[i]]
    return data


def post_func(dict_data:dict, new_item):
    dict_data.update(new_item)


def get_all_data_info(cvs_file_path, whole_json_path, img_root_path):
    book_1 = pandas.read_csv(cvs_file_path, encoding="gb2312")
    ids = book_1['id'].values.tolist()
    plan_1 = book_1['plan'].values.tolist()
    total_color_num = book_1['色号'].values.tolist()
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
    ave_time, max_time, std_time = get_stat_info(time)
    print(f'ave_hit={ave_time}, max={max_time}, std={std_time}')

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

    # 创建一个进程池
    # 要处理的数据
    total = len(ids)-1
    print(f'total samples={total}')
    data_size = list(range(0, total))

    data = {}
    # 使用map进行并行处理
    with Pool(processes=8) as pool:
        # res = pool.map(main_thread,
        #          [(item, total, total_color_num, time, hint, img_root_path, ids, total_block_num, plan, data) for item in data_size])
        # data.update(res)
        results = []
        for item in data_size:
            param = (item, total, total_color_num, time, hint, img_root_path, ids, total_block_num, plan,)
            callback_func = partial(post_func, data)
            result = pool.apply_async(main_thread, param)
            results.append(result)

        for res in results:
            # 如果返回不为空
            if res.get():
                data.update(res.get())

        # # 关闭进程池
        # pool.close()
        # pool.join()

    # # 读取色号色块特征和时长
    # for i in range(len(ids)-8000):
    #     print(f'processing {i}/{len(ids)-8000}...')
    #     id = ids[i]
    #     key = str(id) + ".png"
    #     if key not in plan:
    #         continue
    #     block_dict = plan[key]
    #     blk_index = block_dict['blocks_index']
    #     blk_color = block_dict['blocks_color']
    #
    #     # 获取每个色块的区块个数占比，以及面积占比;
    #     # 如果色块数超过最大值，这个样本不考虑;
    #     flag, blks_per_color, area_per_color, small_area_nums = get_block_info(img_root_path, key, blk_index, total_block_num[i], blk_color)
    #     if not flag:
    #         print(f'file id = {id}, color num={total_color_num[i]}, exceed {MAX_COLORS}....drop it')
    #         continue
    #     print(f'small areas:{small_area_nums}, time={time[i]}')
    #     data[ids[i] + ".png"] = [total_color_num[i], total_block_num[i], blks_per_color, area_per_color, small_area_nums, hint[i], time[i]]

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
    # 输入的cvs文件
    cvs_file_path = "..//file//test_data_8900_with_plan_whole.csv"
    # cvs_file_path = "..//file//test1.csv"

    # 将所有信息统计后，输出到whole_json_path文件中
    whole_json_path = "../file/whole_original_stru.json"

    img_root_path = "D://myproject//suoluetu"
    # img_root_path = "F:\data\乐信\PBN\PBN_TimePred\suoluetu"
    data = get_all_data_info(cvs_file_path, whole_json_path, img_root_path)
    # 80%的训练数据
    create_train_test_json(data, 0.85)
