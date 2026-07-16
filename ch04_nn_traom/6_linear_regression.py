import torch
import matplotlib.pyplot as plt
from torch import nn, optim # 模型、损失函数和优化器
from torch.utils.data import TensorDataset, DataLoader # 数据集和数据加载器

# 1. 准备数据, N * 1
x = torch.randn(100, 1) # 100条数据，每条数据1个特征
# 定义预设的模型参数：权重和偏置
w = torch.tensor(2.5)
b = torch.tensor(5.2)
# 噪声
noise = torch.randn(100, 1) * 0.1
# 基于直线方程构建目标值y
y = w * x + b + noise

# 2. 构建数据集和数据加载器

# 3， 定义模型（Linear）

# 4. 定义超参数

# 5. 定义损失函数和优化器

# 6. 训练模型

# 画图
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# 训练损失随epoch的变化曲线

# 散点图和拟合直线
ax[1].scatter(x, y)

plt.show()
