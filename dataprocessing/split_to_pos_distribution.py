# 将一张图片等分为Ｎ个块，在每个块内部统计坐标出现的次数，转换为坐标分布向量
# 可以更改45行Ｎ的大小

# 读取json文件
import json
import os
import cv2

whole_json_path ="/home/cgim/wushukai/code/LeXin/LexinTimePrediction/Data/data-2/mask_pos.json"
with open(whole_json_path, "r") as f:
    pos_data = json.load(f)

# print(pos_data)

# 将数值映射到list中
def count_coordinates_in_blocks(coordinates, A, B, N):  # A是宽，B是高
    # 初始化统计列表
    block_counts = [0] * N

    # 计算每个块的大小
    block_size_x = A / (N ** 0.5)
    block_size_y = B / (N ** 0.5)

    # 遍历每个坐标
    for x, y in coordinates:
        # 确定坐标所在的块
        block_x = min(int(x // block_size_x), int(N ** 0.5) - 1)
        block_y = min(int(y // block_size_y), int(N ** 0.5) - 1)

        # 将对应块的计数加一
        block_index = int(block_y * (N ** 0.5) + block_x)
        block_counts[block_index] += 1

    return block_counts


count = 0
res_dict = {}
for key, value in pos_data.items():
    count += 1
    print(count, key)
    img_path = os.path.join("/home/cgim/wushukai/code/LeXin/LexinTimePrediction/TimePrediction/suoluetu-final", key)
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    pos_count = count_coordinates_in_blocks(value, width, height, 25)
    # res_dict[key] = pos_count
    res_dict[key] = [num / sum(pos_count) for num in pos_count]
    # print(sum(res_dict[key]))


json_path = "/home/cgim/wushukai/code/LeXin/LexinTimePrediction/Data/data-2/pos_distribution_25_patch.json"
with open(json_path, "w") as f:
    json.dump(res_dict, f)
    print(res_dict)
    print("ok")
