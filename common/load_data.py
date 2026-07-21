import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
from sklearn.model_selection import train_test_split # 划分训练集和测试集
from sklearn.preprocessing import StandardScaler, OneHotEncoder # 标准化缩放器、独热编码器
from sklearn.compose import ColumnTransformer # 列转换器
from sklearn.pipeline import Pipeline # 管道
from sklearn.impute import SimpleImputer # 处理缺失值

def get_digit_data():
    # 1. 加载数据集
    dataset = pd.read_csv('../data/train.csv')

    # 2. 划分数据集
    X = dataset.drop('label', axis=1) # DataFrame 类型
    y = dataset['label'] # Series 类型

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # train_test_split 会保持输入类型，所以返回的x_train和x_test仍然是DataFrame，y_train和y_test仍然是Series
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # 3. 特征工程（归一化）
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train) # 输入是DataFrame，输出是numpy.ndarray （一个二维数组）
    x_test = scaler.transform(x_test) # 输入是DataFrame，输出是numpy.ndarray （一个二维数组）

    # 4. 统一转换为Tensor torch.tensor() 基于内容创建张量，可以是列表或者ndarray，如果是pandas的DataFrame或者Series类型，需要先转换为numpy.ndarray，再转换为Tensor
    x_train = torch.tensor(x_train).float()
    x_test = torch.tensor(x_test).float()
    y_train = torch.tensor(y_train.values) # y_train 是Series类型，通常先转为 numpy.ndarray（如 .values 或 .to_numpy()）再转 Tensor，更稳妥
    y_test = torch.tensor(y_test.values) # y_test 是Series类型，通常先转为 numpy.ndarray（如 .values 或 .to_numpy()）再转 Tensor，更稳妥

    return x_train, x_test, y_train, y_test

def get_house_data():
    # 1. 读取数据
    data = pd.read_csv('../data/house_prices.csv')
    # 2. 数据预处理（特征选择）
    data.drop(['Id'], axis=1, inplace=True) # 删除Id列
    # 3. 划分特征和目标值
    X = data.drop('SalePrice', axis=1)
    y = data['SalePrice']
    # 4. 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. 特征工程（特征转换）
    # 5.1 特征分为两组：数值型、类别型
    numerical_features = X.select_dtypes(exclude=['object']).columns # 数值型特征
    categorical_features = X.select_dtypes(include=['object']).columns # 类别型特征
    # 5.2 定义两组特征的转换操作（Pipeline）
    num_pipeline = Pipeline(steps = [
        ('imputer', SimpleImputer(strategy='mean')), # 缺失值处理，数值型特征用均值填充
        ('scaler', StandardScaler()) # 标准化
    ])
    cat_pipeline = Pipeline(steps = [
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), # 缺失值处理，类别型特征用'missing'填充
        ('onehot', OneHotEncoder(handle_unknown='ignore')) # 独热编码
    ])
    # 5.3 定义列转换器
    ct = ColumnTransformer(transformers=[
        ('num', num_pipeline, numerical_features), # 数值型特征的转换器
        ('cat', cat_pipeline, categorical_features) # 类别型特征的转换器
    ])
    # 5.4 执行特征转换
    x_train = ct.fit_transform(x_train) # 转换后数据类型从pd.Series变为numpy.ndarray
    x_test = ct.transform(x_test) # 转换后数据类型从pd.Series变为numpy.ndarray

    # x_train 和 x_test 通过toarray()方法解决Sparse Matrix稀疏矩阵问题，转换为numpy.ndarray类型。稀疏矩阵是由于独热编码后产生的，OneHotEncoder默认返回稀疏矩阵，toarray()方法将其转换为密集矩阵
    # y_train 和 y_test 仍然是 pd.Series 类型，转换为 numpy.ndarray 类型
    return x_train.toarray(), x_test.toarray(), y_train.values, y_test.values


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = get_digit_data()
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    print(x_train.dtype, x_test.dtype, y_train.dtype, y_test.dtype)


