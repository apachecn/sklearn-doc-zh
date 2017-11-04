# -*- coding:UTF-8 -*-

"""
=================================================
SVM: 分离不平衡类的超平面
=================================================

使用不平衡类的 SVC 找到最优分离超平面。

我们首先找到具有平滑 SVC 的分离平面，然后绘制（虚线）分离超平面，并对不平衡类进行自动校正。

.. currentmodule:: sklearn.linear_model

.. note::

    这个例子也可以使用 ``SGDClassifier(loss="hinge")`` 替换 ``SVC(kernel="linear")`` 。设置 :class:`SGDClassifier` 的 ``loss`` 参数等于 ``hinge`` 将产生一个具有线性内核的 SVC 的 behavior .

    例如使用的不是 ``SVC``::

        clf = SGDClassifier(n_iter=100, alpha=0.01)

"""
print(__doc__)

# 导入 numpy, matplotlib, sklearn 等模块
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# 我们创建了 40 个分离的点
rng = np.random.RandomState(0)
n_samples_1 = 1000
n_samples_2 = 100
# np.r_ 按 row 来组合 array, 如 a = np.array([1,2,3]) b = np.array([4,5,6]) np.r_[a,b] 得到的结果是 array([1,2,3,4,5,6])
# 扩展 np.c_ 是按照 column 来组合 array 的，如 np.c_[a,b] 得到的结果是 array([[1, 4], [2, 5], [3, 6]])
X = np.r_[1.5 * rng.randn(n_samples_1, 2),
          0.5 * rng.randn(n_samples_2, 2) + [2, 2]]
y = [0] * (n_samples_1) + [1] * (n_samples_2)

# 拟合模型并获取分类超平面
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, y)

# 拟合模型并使用加权类来得到分类超平面
wclf = svm.SVC(kernel='linear', class_weight={1: 10})
wclf.fit(X, y)

# 绘制分类超平面和样本
# 在这里做了一下小修改，添加了一个参数 label=" " 否则运行的时候会报错：UserWarning: No labelled objects found. Use label='...' kwarg on individual plots. warnings.warn("No labelled objects found. "
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k', label=" ")
plt.legend()

# 绘制两个分类器的决策函数
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# 创建网格来评估模型
# np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None) 在指定的间隔内返回均匀间隔的数字. 
# start --- 序列的起始点, stop --- 序列的终点, num --- 生成的样本数, endpoint --- 如果为true，则一定包括 stop,为 flase, 则一定不包括 stop
# 详情参见： https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
# meshgrid() 函数用两个坐标轴上的点在平面上画格。具体参阅：https://docs.scipy.org/doc/numpy/reference/generated/numpy.meshgrid.html
YY, XX = np.meshgrid(yy, xx)
# np.vstack() 在垂直方向上合并数组
# numpy.ravel() 与 numpy.flatten() 功能一致，将多维数组降为一维, 不同的地方在于 flatten() 返回一份拷贝，对拷贝做修改不会影响原始矩阵，而 ravel() 返回视图，对视图做修改会影响原始矩阵
# 详情参见： https://docs.scipy.org/doc/numpy/reference/generated/numpy.ravel.html
xy = np.vstack([XX.ravel(), YY.ravel()]).T

# 获得分离超平面
# clf.decision_function() 计算样本点到分割超平面的函数距离
# reshape() 创建一个改变了尺寸的新数组，原数组的shape保持不变, 参考：https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html 
Z = clf.decision_function(xy).reshape(XX.shape)

# 绘制决策边界和间距
# contour() 用来绘制矩阵数据的等高线，详情参见：http://blog.csdn.net/kai165416/article/details/73743201
a = ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])

# 得到加权类的分离超平面
Z = wclf.decision_function(xy).reshape(XX.shape)

# 绘制加权类的决策边界和边距
b = ax.contour(XX, YY, Z, colors='r', levels=[0], alpha=0.5, linestyles=['-'])

# plt.legend() 用来显示图例，详情请看：http://blog.csdn.net/you_are_my_dream/article/details/53440964
plt.legend([a.collections[0], b.collections[0]], ["non weighted", "weighted"],
           loc="upper right")

# 添加这个语句，将上面我们画出来的图显示出来
plt.show()