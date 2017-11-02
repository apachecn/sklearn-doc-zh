"""
===================================================================
支持向量回归（SVR），使用 linear and non-linear 内核
===================================================================

使用 linear, polynomial 和 RBF 内核的 1D 回归的玩具示例。

"""
print(__doc__)

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# #############################################################################
# Generate sample data
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

# #############################################################################
# Add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(8))

# #############################################################################
# Fit regression model
'''
SVR
kernel：string，optional（default ='rbf'）
    指定算法中要使用的内核类型。它必须是 ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ 或可调用的之一。如果没有给出，将使用'rbf'。如果给出了一个可调用函数，它将用于预计算内核矩阵。
C：float，optional（default = 1.0）
    错误项的惩罚参数C。
gamma：float，可选（default ='auto'）
    'rbf'，'poly'和'sigmoid'的内核系数。如果gamma是'auto'，那么将使用1 / n_features。
degree：int，可选（默认= 3）
    多项式核函数的程度（'poly'）。被所有其他内核忽略。
'''
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)

# #############################################################################
# Look at the results
lw = 2
plt.scatter(X, y, color='darkorange', label='data')
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
