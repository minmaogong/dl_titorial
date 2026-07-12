import torch
import torch.nn as nn
from common.load_data import get_digit_data

# 1. 加载数据
x_train, x_test, y_train, y_test = get_digit_data()

# 2. 定义模型
model = nn.Sequential(
    nn.Linear(784, 50), # 全连接层
    nn.ReLU(), # 激活函数-修正线性单元
    nn.Linear(50, 100),
    nn.ReLU(),
    nn.Linear(100, 10), # 输出层 输出0-9 十个分类
    # nn.Softmax(dim=-1)
)

# 3. 加载训练好的模型参数
state_dict = torch.load('../data/nn_example.pt')
model.load_state_dict(state_dict)

# 4. 预测
y_pred = model(x_test)
print(y_pred.shape) # torch.Size([12600, 10]) 12600条数据，每条数据10个分类的预测分数

# 5. 将预测输出值最大索引号，转换为预测分类标签
y_pred_class = torch.argmax(y_pred, dim=-1) # torch.argmax(dim=-1) 返回最后一个维度的最大值索引号，即每条数据预测分数最大的分类索引号
print(y_pred_class.shape) # torch.Size([12600]) 12600条数据，每条数据一个预测分类标签

# 6. 评估：计算准确率
# acc_cnt = torch.sum(y_pred_class == y_test).item() # 计算预测分类标签与真实分类标签相等的数量
acc_cnt = y_pred_class.eq(y_test).sum().item() # 计算预测分类标签与真实分类标签相等的数量
acc = acc_cnt / len(y_test)
print("准确率：", acc)