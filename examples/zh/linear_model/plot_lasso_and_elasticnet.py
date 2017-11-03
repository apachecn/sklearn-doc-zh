"""
========================================
Lasso and Elastic Net for Sparse Signals
========================================

估计手动产生的稀疏信号 Lasso 和 Elastic-Net 回归模型与附加噪声相关。 估计的系数与真实数据进行比较。

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# #############################################################################
# Generate some sparse data to play with
# 生成一些稀疏数据来使用
np.random.seed(42)

n_samples, n_features = 50, 200
X = np.random.randn(n_samples, n_features)
coef = 3 * np.random.randn(n_features)
inds = np.arange(n_features)
np.random.shuffle(inds)
coef[inds[10:]] = 0  # sparsify coef
y = np.dot(X, coef)

# add noise
y += 0.01 * np.random.normal(size=n_samples)

# Split data in train set and test set
n_samples = X.shape[0]
X_train, y_train = X[:n_samples // 2], y[:n_samples // 2]
X_test, y_test = X[n_samples // 2:], y[n_samples // 2:]

# #############################################################################
# Lasso
from sklearn.linear_model import Lasso

# '''
# Lasso：是估计稀疏系数的线性模型
# alpha
#     参数控制估计系数的稀疏度。
#     将L1项倍增的常数，默认为1.0。
#     alpha=0 相当于通过 LinearRegression 对象求解的普通最小二乘法。
#     出于数值原因，不建议使用 alpha=0 与 Lasso 对象。给定这个，你应该使用 LinearRegression 对象。
#     其中 C 是通过 alpha=1/C 或者 alpha=1/(n_samples*C) 得到的。
# '''
# reg = linear_model.Lasso(alpha=0.1)
# print reg.fit([[0, 0], [1, 1]], [0, 1])
# print reg.predict([[1, 1]])
alpha = 0.1
lasso = Lasso(alpha=alpha)

y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred_lasso)
print(lasso)
print("r^2 on test data : %f" % r2_score_lasso)

# #############################################################################
# ElasticNet
from sklearn.linear_model import ElasticNet

enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)
print(enet)
print("r^2 on test data : %f" % r2_score_enet)

plt.plot(enet.coef_, color='lightgreen', linewidth=2,
         label='Elastic net coefficients')
plt.plot(lasso.coef_, color='gold', linewidth=2,
         label='Lasso coefficients')
plt.plot(coef, '--', color='navy', label='original coefficients')
plt.legend(loc='best')
plt.title("Lasso R^2: %f, Elastic Net R^2: %f"
          % (r2_score_lasso, r2_score_enet))
plt.show()
