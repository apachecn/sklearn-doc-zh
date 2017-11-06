"""
===========================================================
绘制凸度系数作为正则化函数
===========================================================

显示共线性对估计器系数的影响。

.. currentmodule:: sklearn.linear_model

:class:`Ridge` 回归是本例中使用的估计量。
每个颜色代表系数向量的不同特征，并且这被显示为正则化参数的函数。

本例还显示了将 Ridge 回归应用于高度病态矩阵的有用性。
对于这种矩阵，目标变量的轻微变化可能导致计算权重的巨大差异。
在这种情况下，设置一定的正则化（alpha）以减少这种变化（噪声）是有用的。

当 α 非常大时，正则化效应支配平方丢失函数，系数趋于零。
在路径的最后，由于 α 趋于零，解决方案趋向于普通的最小二乘法，所以系数呈现大的振荡。
在实践中，有必要调整 alpha，以保持两者之间的平衡。
"""

# Author: Fabian Pedregosa -- <fabian.pedregosa@inria.fr>
# License: BSD 3 clause

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# X is the 10x10 Hilbert matrix
X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)

# #############################################################################
# Compute paths

n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)

coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)

# #############################################################################
# Display results

ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()
