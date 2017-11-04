#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
================================
最近邻分类
================================

最近邻分类的使用方法示例.
它将绘制出每个类别的决策边界.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 15 # 最近邻数量

# 导入一些用来玩的数据
iris = datasets.load_iris()

# 我们只要前两个特征.
# 我们可以使用一个二维的数据集以更方便的操作它
X = iris.data[:, :2] # 训练数据
y = iris.target # 目标变量

h = .02  #  mesh 中的 step size（步长）

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform', 'distance']:
    # 我们创建一个近邻分类器的实例, 并且拟合数据.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y) # 拟合数据

    # 绘制决策边界.
    # 为此，我们为 mesh [x_min, x_max]x[y_min, y_max] 中的每个点都分配了一个颜色.
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) # 预测数据

    # 将结果放入彩色图中
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # 绘制训练数据
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()
