import torch
from torch import nn, optim
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader # 数据集和数据集加载器
from matplotlib import pyplot as plt # 绘图

from common.load_data import get_house_data

# 1. 获取数据
x_train, x_test, y_train, y_test = get_house_data()
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# print(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)
# print(x_train)

# 2. 构建数据集
train_dataset = TensorDataset(torch.tensor(x_train).float(), torch.tensor(y_train).float())
val_dataset = TensorDataset(torch.tensor(x_test).float(), torch.tensor(y_test).float())

# 3. 定义模型
features_num = x_train.shape[1]
model = nn.Sequential(
    nn.Linear(features_num, 128), # 全连接层
    nn.BatchNorm1d(128), # 正则化-批量标准化
    nn.ReLU(), # 激活函数-修正线性单元（引入非线性）
    nn.Dropout(p=0.2), # 正则化-Dropout层，随机失活，防止过拟合
    nn.Linear(128, 1) # 输出层
) # 回归任务，输出层后不需要激活函数，直接输出预测值

# 4. 初始化参数
def init_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight) # Kaiming正态分布初始化

model.apply(init_weights) # 将初始化函数添加到模型中，模型会自动调用该函数对每一层进行初始化

# 5. 定义设备
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 6. 定义超参数
lr = 0.01 # 学习率 如果使用动量发momentum，学习率不要太大，太大容易冲过头，震荡
epochs = 200 # 整个训练集训练200轮
batch_size = 64 # 每次用64个样本计算一次梯度并更新模型参数

# 7. 构建数据集加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# 8. 优化器
optimizer = optim.Adam(model.parameters(), lr=lr)

# 9. 损失函数
def log_rmse(y_pred, y_target):
    mse = nn.MSELoss()
    y_pred = torch.clamp(y_pred, 1, float("inf")) # 将预测值限制在1到正无穷之间，避免对数运算出现负数或零
    loss = torch.sqrt(mse(torch.log(y_pred), torch.log(y_target)))
    return loss
