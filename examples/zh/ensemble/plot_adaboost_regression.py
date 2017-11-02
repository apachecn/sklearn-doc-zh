#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
======================================
使用 AdaBoost 的决策树回归
======================================

在具有少量高斯噪声的 1D 正弦数据集上使用 AdaBoost.R2 [1]_ 算法来提升决策树。
将 299 个提升（300 个决策树）与单个决策树回归比较。随着 boosts 的数量增加，回归器可以更多的细节。

.. [1] H. Drucker, "Improving Regressors using Boosting Techniques", 1997.

"""
print(__doc__)

# Author: Noel Dawe <noel.dawe@gmail.com> Joy yx <chinachenyyx@gmail.com>
#
# License: BSD 3 clause

# 导入一些必要的模块
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

# 创建我们所需的数据集
rng = np.random.RandomState(1)
X = np.linspace(0, 6, 100)[:, np.newaxis]
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])

# 拟合回归模型
regr_1 = DecisionTreeRegressor(max_depth=4)

regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300, random_state=rng)

regr_1.fit(X, y)
regr_2.fit(X, y)

# 预测数据
y_1 = regr_1.predict(X)
y_2 = regr_2.predict(X)

# 绘制出我们拟合和预测之后的结果
plt.figure()
plt.scatter(X, y, c="k", label="training samples")
plt.plot(X, y_1, c="g", label="n_estimators=1", linewidth=2)
plt.plot(X, y_2, c="r", label="n_estimators=300", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Boosted Decision Tree Regression")
plt.legend()
plt.show()
