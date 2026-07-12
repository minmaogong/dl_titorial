import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch

def get_digit_data():
    # 1. 加载数据集
    dataset = pd.read_csv('../data/train.csv')

    # 2. 划分数据集
    X = dataset.drop('label', axis=1) # DataFrame 类型
    y = dataset['label'] # Series 类型

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # train_test_split 会保持输入类型，所以返回的x_train和x_test仍然是DataFrame，y_train和y_test仍然是Series
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # 3. 特征工程（归一化）
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train) # 输入是DataFrame，输出是numpy.ndarray （一个二维数组）
    x_test = scaler.transform(x_test) # 输入是DataFrame，输出是numpy.ndarray （一个二维数组）

    # 4. 统一转换为Tensor torch.tensor() 基于内容创建张量，可以是列表或者ndarray，如果是pandas的DataFrame或者Series类型，需要先转换为numpy.ndarray，再转换为Tensor
    x_train = torch.tensor(x_train).float()
    x_test = torch.tensor(x_test).float()
    y_train = torch.tensor(y_train.values) # y_train 是Series类型，通常先转为 numpy.ndarray（如 .values 或 .to_numpy()）再转 Tensor，更稳妥
    y_test = torch.tensor(y_test.values) # y_test 是Series类型，通常先转为 numpy.ndarray（如 .values 或 .to_numpy()）再转 Tensor，更稳妥

    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = get_digit_data()
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    print(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)


