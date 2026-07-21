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
lr = 0.1 # 学习率 如果使用动量发momentum，学习率不要太大，太大容易冲过头，震荡
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
    y_pred = y_pred.squeeze() # y_pred 是二维张量，形状为(batch_size, 1)，y_target 是一维张量，形状为(batch_size,)，需要将y_pred压缩为一维张量，squeeze()作用是移除张量中所有大小为1的维度
    y_pred = torch.clamp(y_pred, 1, float("inf")) # 将预测值限制在1到正无穷之间，避免对数运算出现负数或零
    loss = torch.sqrt(mse(torch.log(y_pred), torch.log(y_target)))
    return loss

# 10. 模型训练和验证
train_loss_list = []
val_loss_list = []
for epoch in range(epochs):
    # 训练
    model.train()
    train_total_loss = 0
    for (inputs, targets) in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        # 前向传播，得出输出值
        outputs = model(inputs)
        # 计算损失
        loss = log_rmse(outputs, targets) # 一批数据的平均损失
        # 反向传播，计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()
        # 梯度清零
        optimizer.zero_grad()
        # 累加损失
        train_total_loss += loss.item() * inputs.shape[0] # loss.item()是平均损失，乘以样本数得到总损失
    # 本轮训练完毕，计算平均损失
    train_loss = train_total_loss / len(train_dataset)
    train_loss_list.append(train_loss)

    # 验证
    model.eval()
    val_total_loss = 0
    with torch.no_grad(): # 在验证阶段不需要计算梯度，节省内存和计算资源
        for (inputs, targets) in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # 前向传播，得出输出值
            outputs = model(inputs)
            # 计算损失
            loss = log_rmse(outputs, targets) # 一批数据的平均损失
            # 累加损失
            val_total_loss += loss.item() * inputs.shape[0]  # loss.item()是平均损失，乘以样本数得到总损失
    # 本轮验证完毕，计算平均损失
    val_loss = val_total_loss / len(val_dataset)
    val_loss_list.append(val_loss)

    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# 画图
plt.plot(train_loss_list, 'r-', label="Train Loss", linewidth=2)
plt.plot(val_loss_list, 'b--', label="Val Loss", linewidth=2)
plt.legend(loc='best') # legend() 显示图例，loc='best' 自动选择最佳位置
plt.show()


