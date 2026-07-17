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
noise = torch.randn(100, 1) * 0.2
# 基于直线方程构建目标值y
y = w * x + b + noise

# 2. 定义超参数
lr = 0.1 # 学习率，学习率越大，模型收敛越快，但可能会错过最优解；学习率越小，模型收敛越慢，但更容易找到最优解
epochs = 10 # 将整个数据集反复学习10次
batch_size = 10 # 每次用10个样本计算一次梯度并更新模型参数

# 3. 构建数据集和数据加载器
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # 构造数据加载器对象，batch_size为本次训练的样本数，shuffle为是否打乱数据

# 4. 定义模型（Linear）
model = nn.Linear(1, 1)

# 5. 定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr) # 随机梯度下降

# 6. 训练模型
loss_list = []
for epoch in range(epochs): # 循环训练epochs次
    # 统计每个轮次的训练损失
    total_loss = 0
    iter_num = 0
    for x_batch, y_batch in dataloader: # 循环遍历数据加载器中的每个批次
        # 1. 前向传播 得到输出值
        y_pred = model.forward(x_batch)

        # 2. 计算损失
        loss = loss_fn(y_pred, y_batch)

        # 3. 反向传播，计算梯度
        loss.backward()

        # 4. 更新参数
        optimizer.step()

        # 5. 梯度清零
        optimizer.zero_grad()

        # 累加本轮损失
        total_loss += loss.item() # item()方法从张量中获取数值
        iter_num += 1

    # 将本轮训练平均损失，添加到列表中
    loss_list.append(total_loss / iter_num)

print("模型的权重：", model.weight)
print("模型的偏置：", model.bias)

# 画图
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# 训练损失随epoch的变化曲线
ax[0].plot(loss_list)
ax[0].set_title('Train_loss')
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('loss')

# 散点图和拟合直线
ax[1].scatter(x, y)
# 预测
y_pred = model(x).detach().numpy() # detach()：从计算图中分离出来，返回一个新的张量，不会计算梯度 numpy()：将张量转换为numpy数组
ax[1].plot(x, y_pred, color='red')

plt.show()
