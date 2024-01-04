from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error
import numpy as np
import json
import util


# 读取x和y,返回nparray
def load_data(json_path):
    with open(json_path, "r") as f:
        data_dict = json.load(f)

    X = []
    y = []
    fl = []
    for key, value in data_dict.items():
        fl.append(key)
        X.append([value[0], value[1]] + value[2] + value[3] + value[4] + value[5] )
        y += [value[-1]]

    return fl, np.array(X), np.array(y)


if __name__ == '__main__':
    # 假设你的字典为data_dict
    # data_dict的每个项是一个list，前23项是特征，最后一项是目标值
    # whole_json_path = "./file/whole_original_stru.json"
    # filename, X, y = load_data(whole_json_path)

    # 1. 将数据准备成特征矩阵X和目标向量y
    train_json_path = "./file/train_stru.json"
    test_json_path = "./file/test_stru.json"
    _, X_train, y_train = load_data(train_json_path)
    filename, X_test, y_test = load_data(test_json_path)
    print(f'train samples={X_train.shape[0]}, test samples={X_test.shape[0]}')

    # 2. 划分训练集和测试集
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    util.draw_lines(y_pred, y_test)
    util.save_to_cvs(filename, y_pred, y_test,"lasso_pred.csv")

    # 输出误差
    print(f'true:{y_test[0:30]}')
    print(f'pred:{y_pred[0:30]}')

    print(f'Mean Absolute Error: {mae}')
    print(f'Error within 60: {error_60}%')
    print(f'Error within 120: {error_120}%')
    print(f'Error within 180: {error_180}')


