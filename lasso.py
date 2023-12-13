from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
from joblib import dump, load
import numpy as np
import json


# whole_json_path ="/home/cgim/wushukai/code/LeXin/LexinTimePrediction/Data/data-1/whole_original.json"
# with open(whole_json_path, "r") as f:
#     data_dict = json.load(f)
#
# X = []
# y = []
# for key, value in data_dict.items():
#     X.append([value[0], value[1]] + value[2])
#     y += [value[-1]]


whole_json_path ="/home/cgim/wushukai/code/LeXin/LexinTimePrediction/Data/data-3/whole_original.json"
with open(whole_json_path, "r") as f:
    data_dict = json.load(f)

X = []
y = []
for key, value in data_dict.items():
    X.append([value[0], value[1]] + value[2] + value[3] + value[4])
    y += [value[-1]]


# 假设你的字典为data_dict
# data_dict的每个项是一个list，前23项是特征，最后一项是目标值

# 1. 将数据准备成特征矩阵X和目标向量y
X = np.array(X)
y = np.array(y)

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 初始化Lasso回归模型
alpha = 0.1  # 超参数，可以根据需要调整
lasso_model = Lasso(alpha=alpha)

# 4. 训练模型
lasso_model.fit(X_train, y_train)

# 5. 在测试集上进行预测
y_pred = lasso_model.predict(X_test)

# 6. 计算误差
mae = mean_absolute_error(y_test, y_pred)
error_60 = sum(abs(y_test - y_pred) <= 60) / len(y_test) * 100
error_120 = sum(abs(y_test - y_pred) <= 120) / len(y_test) * 100
error_180 = sum(abs(y_test - y_pred) <= 180) / len(y_test) * 100

# 输出误差
print(f'Mean Absolute Error: {mae}')
print(f'Error within 60: {error_60}%')
print(f'Error within 120: {error_120}%')
print(f'Error within 180: {error_180}')


