"""
======================================================
在构建估计器之前估算缺失值
======================================================

此示例显示，丢弃缺少的值可以比丢弃包含任何缺失值的样本更好的结果。
估算并不总是改进预测，因此请通过交叉验证进行检查。
有时删除行或使用标记值更有效。

缺省值可以用平均值，中值或最常用的值替换为 ``策略`` 超参数。
中位数是具有高幅度变量的数据的更强大的估计，可以主导结果（也称为 '长尾'）。

脚本输出 ::

  Score with the entire dataset = 0.56
  Score without the samples containing missing values = 0.48
  Score after imputation of the missing values = 0.55

在这种情况下，估算有助于分类器接近原始分数。

"""
import numpy as np

from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score

rng = np.random.RandomState(0)

dataset = load_boston()
X_full, y_full = dataset.data, dataset.target
n_samples = X_full.shape[0]
n_features = X_full.shape[1]

# 估计整个数据集上的分数，没有缺少值
'''
RandomForestRegressor
n_estimators : integer, optional (default=10)
  森林里的树木数量。
random_state : int, RandomState实例或无，可选 (default=None)
  如果int，random_state 是随机数生成器使用的种子; 
  如果 RandomState 的实例，random_state 是随机数生成器; 
  如果没有，随机数生成器所使用的 RandomState 实例 np.random。
'''
estimator = RandomForestRegressor(random_state=0, n_estimators=100)
score = cross_val_score(estimator, X_full, y_full).mean()
print("Score with the entire dataset = %.2f" % score)

# 在75％的行中添加缺失值
missing_rate = 0.75
n_missing_samples = int(np.floor(n_samples * missing_rate))
missing_samples = np.hstack((np.zeros(n_samples - n_missing_samples,
                                      dtype=np.bool),
                             np.ones(n_missing_samples,
                                     dtype=np.bool)))
rng.shuffle(missing_samples)
missing_features = rng.randint(0, n_features, n_missing_samples)

# 估计没有包含缺失值的行的分数
X_filtered = X_full[~missing_samples, :]
y_filtered = y_full[~missing_samples]
estimator = RandomForestRegressor(random_state=0, n_estimators=100)
score = cross_val_score(estimator, X_filtered, y_filtered).mean()
print("Score without the samples containing missing values = %.2f" % score)

# 估计缺失值估算后的分数
X_missing = X_full.copy()
X_missing[np.where(missing_samples)[0], missing_features] = 0
y_missing = y_full.copy()
estimator = Pipeline([("imputer", Imputer(missing_values=0,
                                          strategy="mean",
                                          axis=0)),
                      ("forest", RandomForestRegressor(random_state=0,
                                                       n_estimators=100))])
score = cross_val_score(estimator, X_missing, y_missing).mean()
print("Score after imputation of the missing values = %.2f" % score)
