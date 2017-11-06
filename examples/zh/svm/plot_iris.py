# -*- coding:UTF-8 -*-

"""
===========================================================
在鸢尾花卉数据集上绘制不同的 SVM 分类器
===========================================================

在鸢尾花卉数据集的 2D 投影上的不同线性 SVM 分类器的比较。我们只考虑这个数据集的前 2 个特征: 

- 萼片长度
- 萼片宽度

此示例显示如何绘制具有不同 kernel 的四个 SVM 分类器的决策表面。

线性模型 ``LinearSVC()`` 和 ``SVC(kernel='linear')`` 产生稍微不同的决策边界。这可能是以下差异的结果:

- ``LinearSVC`` 可以最大限度地减少 squared hinge loss 而 ``SVC`` 最大限度地减少 regular hinge loss.

- ``LinearSVC`` 使用 One-vs-All (也被称作 One-vs-Rest) multiclass reduction ，而 ``SVC`` 则使用 One-vs-One multiclass reduction 。

两个线性模型具有线性决策边界（相交超平面），而非线性内核模型（多项式或 高斯 RBF）具有更灵活的非线性决策边界，其形状取决于内核的种类及其参数。

.. NOTE:: 在绘制玩具 2D 数据集分类器的决策函数的时候可以帮助您直观了解其各自的表现力，请注意，这些直觉并不总是推广到更加接近于现实的高维度的问题。

"""
print(__doc__)

# 加载 numpy, matplotlib, sklearn 等模块
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def make_meshgrid(x, y, h=.02):
    """创建一个要绘制的点的网格

    参数
    ----------
    x: 基于 x 轴的网格数据
    y: 基于 y 轴的网格数据
    h: meshgrid 的步长参数, 是可选的

    返回
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    # meshgrid() 函数用两个坐标轴上的点在平面上画格。具体参阅：https://docs.scipy.org/doc/numpy/reference/generated/numpy.meshgrid.html
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """绘制分类器的决策边界.

    参数
    ----------
    ax: matplotlib 轴对象
    clf: 一个分类器
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: params 的字典传递给 contourf, 可选
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # contourf() 函数 具体用法参见：http://www.labri.fr/perso/nrougier/teaching/matplotlib/   和 http://matplotlib.org/examples/pylab_examples/contourf_demo.html
    out = ax.contourf(xx, yy, Z, **params)
    return out


# 加载一些需要玩的数据
iris = datasets.load_iris()
# 只取前两个特征数据，我们可以通过 2 维数据集来避免这种情况
X = iris.data[:, :2]
y = iris.target

# 我们创建了一个 SVM 的实例并填充了数据。我们不扩展我们的数据，因为我们想绘制支持向量。
C = 1.0  # SVM 正则化参数
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
models = (clf.fit(X, y) for clf in models)

# 绘图区域的标题
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

# 设置 2x2 的网格进行绘制.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()
