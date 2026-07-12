import torch
import torch.nn as nn
from torchsummary import summary

# 1. 输入数据
X = torch.randn(10, 3) # 生成一个10行3列的随机张量，表示10条数据，每条数据3个特征
x = torch.tensor([1, 2.0, 3])

# 2. 定义模型
model = nn.Sequential(
    nn.Linear(3, 4),
    nn.Tanh(), # 激活函数-双曲正切
    nn.Linear(4, 4),
    nn.ReLU(), # 激活函数-修正线性单元
    nn.Linear(4, 2),
    nn.Softmax(dim=-1) # 激活函数-Softmax dim=-1 表示最后一个维度，Softmax将一个任意的实数向量转换为一个概率分布，确保输出值的总和为1
)

# 前向传播
y = model(x)
print(y)

summary(model, input_size=(3, ), batch_size=10, device='cpu')

print("state_dict: ", model.state_dict())

# 模型的保存
# torch.save(model.state_dict(), "model.pt") # 保存模型参数到文件

# 加载模型
state_dict = torch.load("model.pt") # 加载模型参数

model.load_state_dict(state_dict) # 将加载的参数赋值给模型

# 用模型进行测试
y_pred = model(x)
print(y_pred)