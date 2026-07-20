import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from common.load_data import get_digit_data

# 1. 记载数据
x_train, x_val, y_train, y_val = get_digit_data()

# 2. 定义模型
model = nn.Sequential(
    nn.Linear(784, 50), # 全连接层
    nn.ReLU(), # 激活函数-线性修正单元（引入非线性）
    nn.Linear(50, 100), # 全连接层
    nn.ReLU(), # 激活函数-线性修正单元 （引入非线性）
    nn.Linear(100, 10) # 输出层 输出0-9 十个分类
)

# # 对模型参数赋初始值
# def init_params(layer):
#     if isinstance(layer, nn.Linear):
#         nn.init.zeros_(layer.weight)
#         # nn.init.zeros_(layer.bias)
#
# model.apply(init_params) # 将初始化函数添加到模型中，模型会自动调用该函数对每一层进行初始化

# 3. 定义超参数
lr = 0.1 # 学习率 学习率越大，模型收敛越快，但容易震荡，学习率越小，模型收敛越慢，但更稳定
epochs = 20 # 将整个训练集反复训练10轮
batch_size = 64 # 每次用64个样本计算一次梯度并更新模型参数

# 4. 构建数据集和数据加载器
train_dataset = TensorDataset(x_train, y_train) # 训练集
val_dataset = TensorDataset(x_val, y_val) # 验证集
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size) # 验证集不需要打乱数据

# 5. 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss() # 多分类交叉熵损失函数
optimizer = optim.SGD(model.parameters(), lr=lr) # 随机梯度下降优化器

# 6. 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 7. 训练模型
for epoch in range(epochs):

    # 7.1 训练
    model.train() # 设置模型为训练模式，启用dropout和batchnorm

    train_loss_total = 0
    train_acc_num = 0
    for inputs,  targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        # 1. 前向传播，得出输出值
        outputs = model.forward(inputs)

        # 2. 计算损失
        loss = loss_fn(outputs, targets)

        # 3. 反向传播，计算梯度
        loss.backward()

        # 4. 更新参数
        optimizer.step()

        # 5. 梯度清零
        optimizer.zero_grad()

        # 累加损失
        train_loss_total += loss.item() * inputs.shape[0] # loss.item()是平均损失，乘以样本数得到总损失
        # 累计预测准确的个数
        y_pred_class = torch.argmax(outputs, -1)# torch.argmax(dim=-1) 返回最后一个维度的最大值索引号，即每条数据预测分数最大的分类索引号
        train_acc_num += y_pred_class.eq(targets).sum().item() # 计算预测分类标签与真实分类标签相等的数量

    # 本轮训练完毕，计算训练集平均损失和准确率
    this_train_loss = train_loss_total / len(train_dataset) # 平均损失 = 总损失 / 训练集总样本数
    this_train_acc = train_acc_num / len(train_dataset) # 准确率 = 预测正确的样本数 / 训练集总样本数

    # 7.2 验证
    model.eval() # 设置模型为评估模式，禁用dropout和batchnorm

    val_loss_total = 0
    val_acc_num = 0

    with torch.no_grad(): # 模型验证，禁用梯度计算，节省内存和计算资源
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 1. 前向传播，得出输出值
            outputs = model.forward(inputs)

            # 2. 计算损失
            loss = loss_fn(outputs, targets)

            # 累加损失
            val_loss_total += loss.item() * inputs.shape[0] # loss.item()是平均损失，乘以样本数得到总损失
            # 累计预测准确的个数
            y_pred_class = torch.argmax(outputs, dim=-1) # torch.argmax(dim=-1) 返回最后一个维度的最大值索引号，即每条数据预测分数最大的分类索引号
            val_acc_num += y_pred_class.eq(targets).sum().item() # 计算预测分类标签与真实分类标签相等的数量

    # 本轮验证完毕，计算验证集平均损失和准确率
    this_val_loss = val_loss_total / len(val_dataset)
    this_val_acc = val_acc_num / len(val_dataset)

    print(f"Epoch: {epoch + 1}, Train Loss: {this_train_loss:.4f}, Train Acc: {this_train_acc:.4f}, Val Loss: {this_val_loss:.4f}, Val Acc: {this_val_acc:.4f}")


