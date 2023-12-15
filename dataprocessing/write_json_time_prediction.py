# 存储色号色块特征到json
# 会按照4比１的比例随机生成train.json文件和test.json文件，这个文件在train.py的第18-20行分别指明

import json
import pandas
import random
import os


book_1 = pandas.read_csv('/home/cgim/wushukai/code/LeXin/LexinTimePrediction/TimePrediction/data_ori.csv', encoding="utf-8")
indexes_1 = book_1['id'].values.tolist()
sehaoshu_1 = book_1['色号数'].values.tolist()
sekuaishu_1 = book_1['色块数'].values.tolist()
time_1 = book_1['时长'].values.tolist()

data = {}

for i in range(len(indexes_1)):
    data[indexes_1[i]+".png"] = [sehaoshu_1[i], sekuaishu_1[i], time_1[i]]


book_2 = pandas.read_csv('/home/cgim/wushukai/code/LeXin/LexinTimePrediction/TimePrediction/6k.csv', encoding="utf-8")
indexes_2 = book_2['id'].values.tolist()
sehaoshu_2 = book_2['色号数'].values.tolist()
sekuaishu_2 = book_2['色块数'].values.tolist()
time_2 = book_2['时长'].values.tolist()

for i in range(len(indexes_2)):
    data[indexes_2[i]+".png"] = [sehaoshu_2[i], sekuaishu_2[i], time_2[i]]

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

whole_json_path ="/home/cgim/wushukai/code/LeXin/LexinTimePrediction/Data/whole_original.json"
with open(whole_json_path, "w") as f:
    json.dump(data, f)

# 根据缩略图过滤=
suoluetu_path = "/home/cgim/wushukai/code/LeXin/LexinTimePrediction/TimePrediction/souluetu-single-channel"
suoluetu_list = os.listdir(suoluetu_path)
temp_data = data.copy()
for key, value in data.items():
    if key not in suoluetu_list:
        temp_data.pop(key)
data = temp_data


train_json_path = "/home/cgim/wushukai/code/LeXin/LexinTimePrediction/Data/train.json"
test_json_path = "/home/cgim/wushukai/code/LeXin/LexinTimePrediction/Data/test.json"
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