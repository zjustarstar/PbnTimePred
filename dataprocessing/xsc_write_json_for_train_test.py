# 存储色块 色号 面积统计 坐标分布　色号对应的色块数目　特征
# 会按照4比１的比例随机生成train.json文件和test.json文件，这个文件在train.py的第18-20行分别指明

import json
import pandas
import random
import os

# cvs_file_path = "F://MyProject//乐信合作//各类数据//202307pbn时长数据//test_data_8900_with_plan_whole.csv"
cvs_file_path = "..//file//test1.csv"

def get_color_based16(clr_str):
    r = int(clr_str[0:2], 16)
    g = int(clr_str[2:4], 16)
    b = int(clr_str[4:6], 16)

    return (r,g,b)


def get_all_data_info(cvs_file_path, whole_json_path):
    book_1 = pandas.read_csv(cvs_file_path, encoding="gb2312")
    indexes_1 = book_1['id'].values.tolist()
    plan_1 = book_1['plan'].values.tolist()
    sehaoshu = book_1['色号'].values.tolist()
    sekuaishu = book_1['色块'].values.tolist()
    time = book_1['时长'].values.tolist()

    # ----处理色号对应的色块数量特征 -------------
    plan = {}

    for i in range(len(indexes_1) - 1):
        img_name = indexes_1[i]
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

    # --------------------读取色号色块特征和时长 -------------------
    data = {}
    for i in range(len(indexes_1)):
        fl = indexes_1[i]
        if str(fl) + ".png" not in plan:
            continue
        data[indexes_1[i] + ".png"] = [sehaoshu[i], sekuaishu[i], plan[indexes_1[i] + ".png"], time[i]]

    # 求均值和标准差
    import numpy as np
    ave_sehaoshu = np.mean(np.array(sehaoshu)) # 79.09247191011237
    max_sehaoshu = np.max(np.array(sehaoshu))
    print(f'ave_sehaoshu={ave_sehaoshu}, max_sehaoshu={max_sehaoshu}')
    std_sehaoshu = np.std(np.array(sehaoshu))  # 24.377676506738464
    print(std_sehaoshu)

    ave_sekuaishu = np.mean(np.array(sekuaishu))
    print(ave_sekuaishu)  # 834.7568539325842
    std_sehaoshu = np.std(np.array(sekuaishu))  # 351.00993773055467
    print(std_sehaoshu)
    # --------------------读取色号色块特征和时长 -------------------

    # -------------------- 将所有数据写入whole_original.json文件 ---------------------------------
    with open(whole_json_path, "w") as f:
        json.dump(data, f)

    return data
    # -------------------- 将所有数据写入whole_original.json文件 ---------------------------------

# 将数据分别写入train.json和test.json文件
def create_train_test_json(data, ratio=0.8):
    train_json_path = "../file/train.json"
    test_json_path = "../file/test.json"
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

    # print(train_data)
    # print(test_data)
    print("训练集大小:", len(train_data))
    print("测试集大小:", len(test_data))
# -------------------- 将数据分别写入train.json和test.json文件 ---------------------------------

whole_json_path = "../file/whole_original.json"
data = get_all_data_info(cvs_file_path, whole_json_path)
# 90%的训练数据
create_train_test_json(data, 0.8)
