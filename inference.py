import torch

import os
import cv2
import numpy as np

from networks import ScoreModel

from torchvision import transforms


model_path = "/home/cgim/wushukai/code/LeXin/ImageScore/checkpoints/user_score_color/resnet18_-2/best_checkpoint.pth"
test_img_path = "/home/cgim/wushukai/dataset/LeXin/ImageScore/dataset/color_dataset/testset"




# Model & Optimizer Definition
model = ScoreModel().cuda()
model.eval()

ckp = torch.load(model_path)
model.load_state_dict(ckp)
model = model.eval()
print("predict......")


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

max_value = -1
min_value = 100

count = 0
for file in os.listdir(test_img_path):
    img_name = os.path.join(test_img_path, file)
    img = cv2.imread(img_name)

    img = transform_test(img).cuda()
    img = torch.unsqueeze(img, 0)  # 给最高位添加一个维度，也就是batchsize的大小

    res = model(img)
    res = res.detach().cpu().numpy()[0][0]
    print(img_name, res)


    temp = float(img_name.split("_")[4])
    # print(temp)
    chazhi_res = np.abs(res-temp)
    print(chazhi_res)
    # print(chazhi_res)
    # raise

    if chazhi_res > max_value:
        max_value = chazhi_res
    if chazhi_res < min_value:
        min_value = chazhi_res
print(max_value, min_value)



