# -*- coding:UTF-8 -*-

"""
===================================================================
多输出决策树回归
===================================================================

用决策树说明多输出回归的一个例子。

:ref:`decision trees <tree>` 用于同时预测给定单个底层特征的圆的 noisy x 和 y 观测值。因此，它学习近似圆的局部线性回归。

我们可以看到，如果树的最大深度 (由 `max_depth` 参数控制)设置得太高，则决策树会学习过多的训练数据的细节，并且会从噪声中学习，即它们过拟合了。
"""
# print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# 创建一个随机数据集
rng = np.random.RandomState(1)
# rand(m, n) 以参数列表的形式指定参数，而非元组,也就是给定了随机数的 shape, 内部指定的区间是 [0, 1); 其中的 m 代表生成随机序列的行数; n 代表生成随机序列的列数
# numpy.sort(a, axis=-1, kind='quicksort', order=None) 其中 a 是 需要排序的 array, axis=0 表示按列排序
X = np.sort(200 * rng.rand(100, 1) - 100, axis=0)
# numpy.ravel() 与 numpy.flatten() 所实现的功能均为: 将多维数组降维到一维数组, 但是有一些区别. ravel() 返回视图，会影响原始矩阵; flatten() 返回拷贝，对拷贝修改不会影响原始矩阵。
# 具体参见: http://blog.csdn.net/lanchunhui/article/details/50354978
# np.pi 就是 π , np.sin() 对应 sin 函数; np.cos() 对应 cos 函数, 需要了解的知识点: [sin(x)]^2 + [cos(x)]^2 = 1，这实际上是一个圆的图形
y = np.array([np.pi * np.sin(X).ravel(), np.pi * np.cos(X).ravel()]).T
# y[::5, :] 的意思是从第一行开始，每 5 行取全部的列，也就是每隔 4 行取一行，如第 1 行，第 6 行， 第 11 行...以此类推，100行会提取出来 20 行
y[::5, :] += (0.5 - rng.rand(20, 2))

# 拟合回归模型，设置三个最大深度，分别拟合
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_3 = DecisionTreeRegressor(max_depth=8)
regr_1.fit(X, y)
regr_2.fit(X, y)
regr_3.fit(X, y)

# 进行预测
# arange() 函数 与 range() 函数作用是一样的，只不过 arange() 返回的是 array, 而 range() 返回的是 list, 其中的三个参数，分别代表，起始点，终止点，还有步长。
# np.newaxis 的作用是为 numpy.ndarray(多维数组)增加一个轴，例如：b = np.array([1, 2, 3, 4, 5, 6])，shape 为 [6,];然后我们使用 b[np.newaxis] 生成的是新的 b 为 array([[1, 2, 3, 4, 5, 6]]) ,对应的 shape 为[1L, 6L]
X_test = np.arange(-100.0, 100.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)
y_3 = regr_3.predict(X_test)

# Plot the results
plt.figure()
s = 50
s = 25
plt.scatter(y[:, 0], y[:, 1], c="navy", s=s,
            edgecolor="black", label="data")
plt.scatter(y_1[:, 0], y_1[:, 1], c="cornflowerblue", s=s,
            edgecolor="black", label="max_depth=2")
plt.scatter(y_2[:, 0], y_2[:, 1], c="red", s=s,
            edgecolor="black", label="max_depth=5")
plt.scatter(y_3[:, 0], y_3[:, 1], c="orange", s=s,
            edgecolor="black", label="max_depth=8")
plt.xlim([-6, 6])
plt.ylim([-6, 6])
plt.xlabel("target 1")
plt.ylabel("target 2")
plt.title("Multi-output Decision Tree Regression")
plt.legend(loc="best")
plt.show()
