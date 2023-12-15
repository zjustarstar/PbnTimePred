from PIL import Image
import os
import json
import random

data_folder = "/home/cgim/wushukai/code/LeXin/LexinTimePrediction/TimePrediction/suoluetu"
yichangtupian = []


for img_name in os.listdir(data_folder):
    try:
        img = Image.open(
        os.path.join("/TimePrediction/suoluetu", img_name))
    except:
        print("Cannot transform image: {}".format(img_name))
        yichangtupian.append(img_name)
        continue


with open("/home/cgim/wushukai/code/LeXin/LexinTimePrediction/TimePrediction-Lexin-1/data/whole_original.json", "r") as f:
    whole_data = json.load(f)

for x in yichangtupian:
    whole_data.pop(x)

with open("/home/cgim/wushukai/code/LeXin/LexinTimePrediction/TimePrediction-Lexin-1/data/whole.json", "w") as fw:
    json.dump(whole_data, fw)



train_json_path = "/home/cgim/wushukai/code/LeXin/LexinTimePrediction/TimePrediction-Lexin-1/data/train.json"
test_json_path = "/home/cgim/wushukai/code/LeXin/LexinTimePrediction/TimePrediction-Lexin-1/data/test.json"
train_data = {}
test_data = {}

with open(train_json_path, "w") as f_train:
    with open(test_json_path, "w") as f_test:
        for key, value in whole_data.items():
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
















# with open("/home/cgim/wushukai/code/LeXin/LexinTimePrediction/TimePrediction-Lexin-1/data/train.json", "r") as f:
#     train_data = json.load(f)
#
# for x in yichangtupian:
#     train_data.pop(x)
#
# with open("/home/cgim/wushukai/code/LeXin/LexinTimePrediction/TimePrediction-Lexin-1/data/train-1.json", "w") as fw:
#     json.dump(train_data, fw)
#
#
# with open("/home/cgim/wushukai/code/LeXin/LexinTimePrediction/TimePrediction-Lexin-1/data/test.json", "r") as f:
#     test_data = json.load(f)
#
# for x in yichangtupian:
#     test_data.pop(x)
#
# with open("/home/cgim/wushukai/code/LeXin/LexinTimePrediction/TimePrediction-Lexin-1/data/test-1.json", "w") as fw:
#     json.dump(test_data, fw)
