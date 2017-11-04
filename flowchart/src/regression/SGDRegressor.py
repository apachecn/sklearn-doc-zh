#!/usr/bin/env python
# encoding: utf-8
'''
Created on 2017-10-26
Update  on 2017-10-26
@author: 片刻
sklearn 中文文档 更新地址：https://github.com/apachecn/scikit-learn-doc-zh
'''

# import gc
import time
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

start_time = time.time()

# n_samples, n_features = 10, 5
# np.random.seed(0)
# # 生成一个 10*1 的矩阵
# y = np.random.randn(n_samples)
# # 生成一个 10*5 的矩阵
# X = np.random.randn(n_samples, n_features)

X, y = np.arange(50).reshape((10, 5)), range(10)
print X, '\n', y

'''
train_test_split: 用于拆分数据集
X, y
    表示要拆分的数据集
test_size
    表示测试数据集占全集的比例
train_size
    (默认是：1-test_size)，设置训练数据集占全集的比例，例如：随机森林会
shuffle [很关键的一步，是否要让数据随机化]
    boolean，可选（default = True）, 是否在拆分之前洗牌，如果shuffle = False，则stratify必须为None。
stratify
    数组式或无（默认为无）
    如果不是None，则数据以分层方式分割，使用它作为类标签。

拆分训练数据与测试数据， 80%做训练 20%做测试
'''
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.6, test_size=0.2)
print '拆分数据：\n', x_train, '\t', x_test, '\n', y_train, '\t', y_test


'''
SGDRegressor: 随机梯度下降，解决回归问题
loss
    loss="squared_loss": 普通最小二乘法。（默认值）
    loss="huber": Huber 损失稳健回归，"huber" 修改 "squared_loss"，更少地集中在通过从ε平方到直线损失之间切换，使异常值正确。
    loss="epsilon_insensitive": 线性支持向量回归，"epsilon_insensitive" 忽略小于 epsilon 的错误，并且是线性的，这是SVR中使用的损失函数。
    loss="squared_epsilon_insensitive":  "squared_epsilon_insensitive" 是相同的，但是通过一个容忍的ε被平均损失。
penalty
    'none', 'l2', 'l1', or 'elasticnet'
    使用的penalty（也称为正规化术语）。 默认为“l2”，它是线性SVM模型的标准正则符。 'l1'和'elasticnet' 可能会使模型（特征选择）带来稀疏性，而'l2'无法实现。
alpha
    alpha代表向目标移动的步长
    将正则化项扩大的常数。默认为0.0001当设置为“最优”时，也用于计算learning_rate。
max_iter
    最大迭代次数
'''

clf = linear_model.SGDRegressor()
print clf.fit(X, y)
print clf.predict(x_test)

stop_time = time.time()
print("example run in %.2fs" % (stop_time - start_time))
