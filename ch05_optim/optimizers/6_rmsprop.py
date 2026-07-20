import torch
from torch import optim
import numpy as np
from matplotlib import pyplot as plt
from torch.optim import RMSprop # Root Mean Square Propagation 均方根传播

# 定义目标函数（模拟损失函数），参数是二维向量
def f(x):
    # return 0.05 * x[0]**2 + x[1]**2
    return (x**2).dot(torch.tensor([0.05, 1.0]))

# 梯度下降函数
def gradient_descent(x, optimizer, num_iters):
    # 用列表保存更新轨迹
    x0_list = []
    x1_list = []

    # 循环迭代
    for iter in range(num_iters):
        # x0_list.append(x[0].detach().numpy().copy())
        # x1_list.append(x[1].detach().numpy().copy())
        x0_list.append(x[0].item()) # item() 张量转换为标量
        x1_list.append(x[1].item()) # item() 张量转换为标量

        # 1. 计算损失（带入f）
        y = f(x)
        # 2. 反向传播，计算梯度
        y.backward()
        # 3. 更新参数
        optimizer.step()
        # 4， 梯度清零
        optimizer.zero_grad()

    return x0_list, x1_list

if __name__ == '__main__':
    # 1. 初始化参数
    x = torch.tensor([-7.0, 2.0], requires_grad=True)

    # 2. 定义超参数
    lr = 0.1
    num_iters = 500

    # SGD
    # 3. 定义优化器
    x_sgd = x.clone().detach().requires_grad_(True)
    optimizer = optim.SGD([x_sgd], lr=lr)

    # 4. 梯度下降法寻找最小值
    x0_list, x1_list = gradient_descent(x_sgd, optimizer, num_iters)

    # 画图
    plt.plot(x0_list, x1_list, color='red')

    # RMSprop 均方根传播
    # 3. 定义优化器
    x_adagrad = x.clone().detach().requires_grad_(True)
    optimizer = RMSprop([x_adagrad], lr=lr, alpha=0.99)

    # 4. 梯度下降法寻找最小值
    x0_list, x1_list = gradient_descent(x_adagrad, optimizer, num_iters)

    # 画图
    plt.plot(x0_list, x1_list, color='blue')

    # 绘制等高线
    x0_grid, x1_grid = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-2, 2, 100))
    y_grid = 0.05 * x0_grid**2 + x1_grid**2
    plt.contour(x0_grid, x1_grid, y_grid, levels=30, colors='gray')
    plt.legend(["SGD", "RMSprop"])
    plt.show()
