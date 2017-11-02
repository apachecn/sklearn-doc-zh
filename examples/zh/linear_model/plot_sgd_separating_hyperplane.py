#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================
SGD: 最大间隔的分离超平面
=========================================


使用利用 SGD 训练的线性支持向量机分类器，绘制在两类可分离数据集中分离超平面的最大余量。
"""
print(__doc__)

# 导入 numpy, matplotlib, linear_model 等需要用到的模块
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.datasets.samples_generator import make_blobs

# 我们首先创建 50 个可分离的点
# sklearn.datasets.make_blobs(n_samples=100, n_features=2,centers=3, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=None)[source] 用法如下:
# 作用：会根据用户指定的特征数量、中心点数量、范围等来生成几类数据，这些数据可用于测试聚类算法的效果。具体参见：http://blog.csdn.net/kevinelstri/article/details/52622960
# 参数含义：
# n_samples --- 待生成的样本的总数
# n_features --- 每个样本的特征数
# centers --- 类别数
# cluster_std --- 每个类别的方差，例如我们希望生成2类数据，其中一类比另一类具有更大的方差，我们可以将 cluster_std 设置为 [1.0, 3.0]
X, Y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)

# 拟合模型
# SGDClassifier() 随机梯度下降的 logistic 回归
# 具体的参见：http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
# 参数： loss --- 损失函数, alpha --- 将正则化项扩大的常数, 默认为 0.0001，当设置为'optimal', 也可以用于计算 learning_rate, max_iter --- 训练数据的最大迭代次数
# 注意： SGDClassifier() 其中的参数 max_iter，这个参数是最新的，之前的参数是 n_iter，请大家看好了sklearn 的版本再使用
clf = SGDClassifier(loss="hinge", alpha=0.01, n_iter=200, fit_intercept=True)
clf.fit(X, Y)

# 绘制线，点和距离平面最近的向量
# numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None) 在指定的间隔内返回均匀间隔的数字
# 具体参见：http://blog.csdn.net/you_are_my_dream/article/details/53493752
xx = np.linspace(-1, 5, 10)
yy = np.linspace(-1, 5, 10)

# np.meshgrid() 参见：http://blog.csdn.net/grey_csdn/article/details/69663432
X1, X2 = np.meshgrid(xx, yy)
# numpy.empty(shape, dtype=float, order='C') 返回给定形状和类型的数组，而不初始化其中的条目。
# 详情请参阅：https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.empty.html
Z = np.empty(X1.shape)
# np.ndenumerate() 多维索引迭代器，返回一个产生数组坐标和值的迭代器。
for (i, j), val in np.ndenumerate(X1):
    x1 = val
    x2 = X2[i, j]
    # clf.decision_function() 计算支持向量到分割超平面的函数距离
    p = clf.decision_function([[x1, x2]])
    Z[i, j] = p[0]
levels = [-1.0, 0.0, 1.0]
# 画线的种类
linestyles = ['dashed', 'solid', 'dashed']
colors = 'k'
plt.contour(X1, X2, Z, levels, colors=colors, linestyles=linestyles)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired,
            edgecolor='black', s=20)

plt.axis('tight')
plt.show()
