import torch
import torch.nn as nn
from torchsummary import summary

"""
    自定义神经网络模型类
"""
class MyModel(nn.Module): # 继承nn.Module类
    def __init__(self):
        super(MyModel, self).__init__() # 调用父类的构造函数
        # 定义三个全连接层（线性层）
        self.fc1 = nn.Linear(in_features=3, out_features=4)
        self.fc2 = nn.Linear(in_features=4, out_features=4)
        self.out = nn.Linear(in_features=4, out_features=2)

    # 前向传播
    def forward(self, x): # 定义前向传播函数
        # 逐层计算
        # # 第一层
        # a1 = self.fc1(x)
        # z1 = torch.tanh(a1) # 激活函数-tanh 双曲正切函数
        # # 第二层
        # a2 = self.fc2(z1) # 第一层的全连接层的输出作为输入
        # z2 = torch.relu(a2) # 激活函数-ReLU 修正线性单元
        # # 输出层
        # a3 = self.out(z2) # 第二层的全连接层的输出作为输出层的输入
        # y = torch.softmax(a3, dim=-1) # 激活函数-Softmax

        # 第一层
        x = self.fc1(x)
        x = torch.tanh(x)  # 激活函数-tanh 双曲正切函数
        # 第二层
        x = self.fc2(x)  # 第一层的全连接层的输出作为输入
        x = torch.relu(x)  # 激活函数-ReLU 修正线性单元
        # 输出层
        x = self.out(x)  # 第二层的全连接层的输出作为输出层的输入
        y = torch.softmax(x, dim=-1)  # 激活函数-Softmax，将一个任意的实数向量转换为一个概率分布，确保输出值的总和为1

        return y


if __name__ == '__main__':
    # 1. 输入数据
    x = torch.randn(3)
    X = torch.randn(10, 3) # 生成一个10行3列的随机张量，表示10个样本，每个样本有3个特征
    # 2. 创建模型
    model = MyModel()
    # 3. 前向传播
    y = model(x)
    print(y)
    y = model(X)
    print(y)

    # 查看参数
    # 1. 直接查找每个全连接层的权重和偏置
    print(model.fc1.weight)
    print(model.fc1.bias)

    # 2. 遍历模型的参数
    print("遍历模型参数：")
    for name, param in model.named_parameters():
        print(name, param)

    # 3，查看状态字典
    print("查看状态字典：")
    dict = model.state_dict()
    print(dict)

    # 查看模型结构和参数数量
    summary(model, input_size=(3, ), device='cpu')
