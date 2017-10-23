# -*- coding:UTF-8 -*-

"""
================================
SVM 练习
================================

使用不同的 SVM kernels 的教程练习。

这个教程应用于 :ref:`supervised_learning_tut` 章节的 :ref:`stat_learn_tut_index` 的 :ref:`using_kernels_tut` 这一部分。
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm

# 加载 iris 数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

X = X[y != 0, :2]
y = y[y != 0]

n_sample = len(X)

np.random.seed(0)
# np.random.permutation() 和 np.random.shuffle() 功能类似，不同之处在于: 1、如果传给 permutation() 一个矩阵，它会返回一个洗牌后的矩阵副本,而 shuffle() 只是对一个矩阵进行洗牌，无返回值。如果向 permutation()传入的是一个整数，它会返回一个洗牌后的arange。
# 参考官网链接: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.permutation.html
order = np.random.permutation(n_sample)
# 这行代码的意思是 X 中的数据将按照生成的打乱的 order 顺序，order中的顺序是原本 X 中的数据的下标
X = X[order]
# y 同样按照上面的 X 的套路，并且转化为 float 类型
y = y[order].astype(np.float)

X_train = X[:int(.9 * n_sample)]
y_train = y[:int(.9 * n_sample)]
X_test = X[int(.9 * n_sample):]
y_test = y[int(.9 * n_sample):]

# 拟合模型，我们这里选用 SVM 的三种核函数来拟合，分别为 linear， rbf， poly
for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):
    clf = svm.SVC(kernel=kernel, gamma=10)
    clf.fit(X_train, y_train)

    plt.figure(fig_num)
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Paired,
                edgecolor='k', s=20)

    # 圈出 test 数据
    plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none',
                zorder=10, edgecolor='k')

    plt.axis('tight')
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # 将结果放入彩色图
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                linestyles=['--', '-', '--'], levels=[-.5, 0, .5])

    plt.title(kernel)
plt.show()
