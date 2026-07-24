import torch
from torch import nn, optim
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from common.load_data import get_fashion_data

# 1. 加载并构建数据集
x_train, x_test, y_train, y_test = get_fashion_data()
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# 显示一张图片及其标签 x_train.shape = torch.Size([60000, 1, 28, 28])
img = x_train[12345, 0, :, :] # 取第12345张图片，0表示通道数，28表示高度，28表示宽度
plt.imshow(img, cmap='gray') # 显示灰度图像
plt.title(f'Label: {y_train[12345]}') # 显示标签
plt.show()

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

