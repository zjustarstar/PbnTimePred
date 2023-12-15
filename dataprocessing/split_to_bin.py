# 统计每张图中面积的分布，形成面积分布向量，存成area_prob.json文件
# 首先读取area.json文件，area.json文件里面存放了每张图的色块面积
import json

whole_json_path ="/home/cgim/wushukai/code/LeXin/LexinTimePrediction/Data/data-1/area.json"
with open(whole_json_path, "r") as f:
    data = json.load(f)

print(data)

min_value, max_value = 100000, 0
for key,value in data.items():
    min_num = min(value)
    max_num = max(value)
    if min_num < min_value:
        min_value = min_num
    if max_num > max_value:
        max_value = max_num

print(min_value, max_value)



# 将数值映射到字典和数组中
def count_values(data):

    length = len(data)

    result_dict = {'0-499': 0, '500-999': 0, '1000-1499': 0, '1500-1999': 0, '2000-2499': 0, '2500-2999': 0,
                   '3000-3499': 0, '3500-3999': 0, '4000-4499': 0, '4500-4999': 0, '5000-5499': 0, '5500-5999': 0,
                   '6000-6499': 0, '6500-6999': 0, '7000-7499': 0, '7500-7999': 0, '8000-8499': 0, '8500-8999': 0,
                   '9000-9499': 0, '9500-9999': 0, '10000 and above': 0}

    result_list = []

    for value in data:
        if value < 10000:
            range_key = f'{(value // 500) * 500}-{((value // 500) * 500) + 499}'
            result_dict[range_key] += 1
        else:
            result_dict['10000 and above'] += 1

    for key, value in result_dict.items():
        result_list.append(value/length)

    print(sum(result_list))

    return result_dict, result_list


res_count_area_num_dict = {}

for key, value in data.items():
    _, result_list = count_values(value)
    res_count_area_num_dict[key] = result_list

json_path ="/home/cgim/wushukai/code/LeXin/LexinTimePrediction/Data/data-1/area_prob.json"
with open(json_path, "w") as f:
    json.dump(res_count_area_num_dict, f)
    print(res_count_area_num_dict)
    print("ok")





