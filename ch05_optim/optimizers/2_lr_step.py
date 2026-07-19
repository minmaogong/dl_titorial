import torch
from torch import optim
import numpy as np
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import StepLR

# 定义目标函数（模拟损失函数），参数是二维向量
def f(x):
    # return 0.05 * x[0]**2 + x[1]**2
    return (x**2).dot(torch.tensor([0.05, 1.0]))

# 梯度下降函数
def gradient_descent(x, optimizer, lr_scheduler, num_iters):
    # 用列表保存更新轨迹
    x0_list = []
    x1_list = []
    lr_list = []

    # 循环迭代
    for iter in range(num_iters):
        # x0_list.append(x[0].detach().numpy().copy())
        # x1_list.append(x[1].detach().numpy().copy())
        x0_list.append(x[0].item()) # item() 张量转换为标量
        x1_list.append(x[1].item()) # item() 张量转换为标量
        lr_list.append(optimizer.param_groups[0]['lr'])

        # 1. 计算损失（带入f）
        y = f(x)
        # 2. 反向传播，计算梯度
        y.backward()
        # 3. 更新参数
        optimizer.step()
        # 4， 梯度清零
        optimizer.zero_grad()

        # 5. 更新学习率
        lr_scheduler.step()

    return x0_list, x1_list, lr_list

if __name__ == '__main__':
    # 1. 初始化参数
    x = torch.tensor([-7.0, 2.0], requires_grad=True)

    # 2. 定义超参数
    lr = 0.9
    num_iters = 500

    # SGD
    # 3. 定义优化器
    optimizer = optim.SGD([x], lr=lr)

    # 4. 定义学习率调度器
    lr_scheduler = StepLR(optimizer, step_size=20, gamma=0.7) # lr_scheduler.step() 每20次迭代后，学习率乘以0.1

    # 5. 梯度下降法寻找最小值
    x0_list, x1_list, lr_list = gradient_descent(x, optimizer, lr_scheduler=lr_scheduler, num_iters=num_iters)

    # 画图
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(x0_list, x1_list, color='red')
    ax[0].set_title('Gradient Descent')

    # 绘制等高线
    x0_grid, x1_grid = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-2, 2, 100))
    y_grid = 0.05 * x0_grid**2 + x1_grid**2
    ax[0].contour(x0_grid, x1_grid, y_grid, levels=30, colors='gray')

    ax[1].plot(lr_list, color='blue')
    ax[1].set_title('Learning Rate')

    plt.show()
