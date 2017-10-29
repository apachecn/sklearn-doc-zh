# Sklearn 快速入门教程

![](images/ml_map.png)

## Regression 回归

> 考虑下面 == 如果样本的数据量 >= 100k

### [SGD Refressor（随机梯度下降）](http://sklearn.apachecn.org/cn/0.19.0/modules/sgd.html)

> 考虑下面 == 如果样本的数据量 < 100k 并且 特征少并重要

### [Lasso](http://sklearn.apachecn.org/cn/0.19.0/modules/linear_model.html) 或 [ElasticNet](http://sklearn.apachecn.org/cn/0.19.0/modules/linear_model.html#elastic-net)

> 考虑下面 == 如果样本的数据量 < 100k 并且 特征少并不重要

### [ridge-regression](http://sklearn.apachecn.org/cn/0.19.0/modules/linear_model.html#ridge-regression) 或 [SVR(kernel='linear')](http://sklearn.apachecn.org/cn/0.19.0/modules/svm.html#regression)

> 考虑下面 == 如果无效

### [EnsembleRegressors](http://sklearn.apachecn.org/cn/0.19.0/modules/ensemble.html) 或 [SVR(kernel='rbf')](http://sklearn.apachecn.org/cn/0.19.0/modules/svm.html#regression)

