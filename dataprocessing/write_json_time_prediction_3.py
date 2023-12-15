# 存储色块 色号 面积统计 坐标分布　色号对应的色块数目　特征
# 会按照4比１的比例随机生成train.json文件和test.json文件，这个文件在train.py的第18-20行分别指明

import json
import pandas
import random
import os


file_path = "/home/cgim/wushukai/code/LeXin/LexinTimePrediction/TimePrediction/6k-test_data_filled_2_ori_with_plan.csv"
book_1 = pandas.read_csv(file_path, encoding="gb2312")
indexes_1 = book_1['id'].values.tolist()
values_1 = book_1['plan'].values.tolist()


max_color_num = 0
plan = {}

for i in range(len(indexes_1)-1):
    img_name = indexes_1[i]
    item = str(values_1[i])
    item = item.replace("|", ",").replace("#", ",#")
    item = item.split(",")

    temp_item = []
    color_dict = {}
    for i in item:
        if i.startswith("#"):
            count_area_num = len(temp_item)
            color_dict[i] = count_area_num
            temp_item = []
        else:
            temp_item.append(i)

    # sorted_color_dict = color_dict
    sorted_color_dict = dict(sorted((color_dict.items())))
    sorted_color_list = list(sorted_color_dict.values())

    # if len(sorted_color_list) > max_color_num:
    #     max_color_num = len(sorted_color_list)

    sorted_color_list = [int(x) for x in sorted_color_list]


    while len(sorted_color_list) < 800:
        sorted_color_list.append(0)

    # print(sorted_color_list)
    if sum(sorted_color_list) == 0:
        continue

    plan[img_name+".png"] = [x/sum(sorted_color_list) for x in sorted_color_list]



whole_json_path_1 = "/home/cgim/wushukai/code/LeXin/LexinTimePrediction/Data/data-1/area_prob.json"
with open(whole_json_path_1, "r") as f:
    area_data = json.load(f)

whole_json_path_2 = "/home/cgim/wushukai/code/LeXin/LexinTimePrediction/Data/data-2/pos_distribution_64_patch.json"
with open(whole_json_path_2, "r") as f:
    pos_data = json.load(f)



book_1 = pandas.read_csv('/home/cgim/wushukai/code/LeXin/LexinTimePrediction/TimePrediction/data_ori.csv', encoding="utf-8")
indexes_1 = book_1['id'].values.tolist()
sehaoshu_1 = book_1['色号数'].values.tolist()
sekuaishu_1 = book_1['色块数'].values.tolist()
time_1 = book_1['时长'].values.tolist()

data = {}

for i in range(len(indexes_1)):
    if indexes_1[i]+".png" not in area_data or indexes_1[i]+".png" not in pos_data or indexes_1[i]+".png" not in plan:
        continue
    data[indexes_1[i]+".png"] = [sehaoshu_1[i], sekuaishu_1[i], area_data[indexes_1[i]+".png"], pos_data[indexes_1[i]+".png"], plan[indexes_1[i]+".png"], time_1[i]]


book_2 = pandas.read_csv('/home/cgim/wushukai/code/LeXin/LexinTimePrediction/TimePrediction/6k.csv', encoding="utf-8")
indexes_2 = book_2['id'].values.tolist()
sehaoshu_2 = book_2['色号数'].values.tolist()
sekuaishu_2 = book_2['色块数'].values.tolist()
time_2 = book_2['时长'].values.tolist()

for i in range(len(indexes_2)):
    if indexes_2[i]+".png" not in area_data or indexes_2[i]+".png" not in pos_data or indexes_2[i]+".png" not in plan:
        continue
    data[indexes_2[i]+".png"] = [sehaoshu_2[i], sekuaishu_2[i], area_data[indexes_2[i]+".png"], pos_data[indexes_2[i]+".png"], plan[indexes_2[i]+".png"], time_2[i]]

sekuaishu = sekuaishu_1 + sekuaishu_2
sehaoshu = sehaoshu_1 + sehaoshu_2
time = time_1 + time_2

# 相关性分析
# import matplotlib.pyplot as plt
# plt.scatter(sehaoshu, time, s=1)
# plt.title("Analysis")
# plt.xlabel("Sehao")
# plt.ylabel("Time")
# plt.savefig('../relavenceanalysis/sehao.png')
# plt.show()
# 相关性分析

# 求均值和方差
import numpy as np
ave_sehaoshu = np.mean(np.array(sehaoshu))
print(ave_sehaoshu)  # 79.09247191011237
std_sehaoshu = np.std(np.array(sehaoshu))  # 24.377676506738464
print(std_sehaoshu)

ave_sekuaishu = np.mean(np.array(sekuaishu))
print(ave_sekuaishu)  # 834.7568539325842
std_sehaoshu = np.std(np.array(sekuaishu))  # 351.00993773055467
print(std_sehaoshu)
# 求均值和方差


# 根据缩略图过滤=
suoluetu_path = "/home/cgim/wushukai/code/LeXin/LexinTimePrediction/TimePrediction/suoluetu-final"
suoluetu_list = os.listdir(suoluetu_path)
temp_data = data.copy()
for key, value in data.items():
    if key not in suoluetu_list:
        temp_data.pop(key)
data = temp_data


whole_json_path ="/home/cgim/wushukai/code/LeXin/LexinTimePrediction/Data/data-3/whole_original.json"
with open(whole_json_path, "w") as f:
    json.dump(data, f)


train_json_path = "/home/cgim/wushukai/code/LeXin/LexinTimePrediction/Data/data-3/train.json"
test_json_path = "/home/cgim/wushukai/code/LeXin/LexinTimePrediction/Data/data-3/test.json"
train_data = {}
test_data = {}

with open(train_json_path, "w") as f_train:
    with open(test_json_path, "w") as f_test:
        for key, value in data.items():
            random_num = random.random()
            if random_num < 0.8:
                train_data[key] = value
            else:
                test_data[key] = value
                # train_data[data[i]]
        json.dump(train_data, f_train)
        json.dump(test_data, f_test)

print(train_data)
print(test_data)
print("训练集大小:", len(train_data))
print("测试集大小:", len(test_data))