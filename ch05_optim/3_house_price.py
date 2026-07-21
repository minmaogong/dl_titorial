import torch
from torch import nn, optim
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader # 数据集和数据加载器
from matplotlib import pyplot as plt # 绘图

from common.load_data import get_house_data

# 1. 获取数据
x_train, x_test, y_train, y_test = get_house_data()
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)
# print(x_train)

# 2. 构建数据集
train_dataset = TensorDataset(torch.tensor(x_train).float(), torch.tensor(y_train).float())
val_dataset = TensorDataset(torch.tensor(x_test).float(), torch.tensor(y_test).float())
# print(len(train_dataset), len(val_dataset))

# 3. 构建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32)

