# 从excel中获取色号对应的色块数量，存入json文件

import json
import pandas
import random
import os


# color_list = [48,1,129,2,181,197,103,327,107,'#1C52AC',3,403,4,390,312,520,473,346,332,366,'#1F5DA0',162,195,5,'#B5CFC8',6,'#7399BA']
# color_list = str(color_list)

# # 初始化一个字典用于存放颜色和对应区域编号的列表
# color_dict = {}
# current_color = None
#
# # 遍历颜色列表
# for item in color_list:
#     # 如果是颜色项，更新current_color
#     if isinstance(item, str) and item.startswith('#'):
#         current_color = item[1:]
#         color_dict[current_color] = []
#     else:
#         # 如果current_color不为None，将当前项添加到对应颜色的列表中
#         if current_color is not None:
#             color_dict[current_color].append(item)
#
# # 对颜色字典按照颜色进行升序排序
# sorted_color_dict = {k: sorted(v) for k, v in sorted(color_dict.items())}
#
# # 打印结果
# print(sorted_color_dict)
# raise



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
    # print(img_name+".png",sum(plan[img_name+".png"]), len(sorted_color_list)-sorted_color_list.count(0),sum(sorted_color_list))




# print(list(plan.items())[-1])
# raise
