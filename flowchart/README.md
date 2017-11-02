# Sklearn 快速入门教程

![](images/ml_map.png)

## Regression 回归

> 考虑下面 == 如果样本的数据量 >= 100k

### [SGD Refressor（随机梯度下降）](http://sklearn.apachecn.org/cn/0.19.0/modules/sgd.html)

> 考虑下面 == 如果样本的数据量 < 100k 并且 少数特征是重要

### [Lasso](http://sklearn.apachecn.org/cn/0.19.0/modules/linear_model.html) 或 [ElasticNet](http://sklearn.apachecn.org/cn/0.19.0/modules/linear_model.html#elastic-net)

[Lasso 和 ElasticNet 项目案例](http://sklearn.apachecn.org/cn/0.19.0/auto_examples/linear_model/plot_lasso_and_elasticnet.html#sphx-glr-auto-examples-linear-model-plot-lasso-and-elasticnet-py)

> 考虑下面 == 如果样本的数据量 < 100k 并且 少数特征是不重要

### [ridge-regression](http://sklearn.apachecn.org/cn/0.19.0/modules/linear_model.html#ridge-regression) 或 [SVR(kernel='linear')](http://sklearn.apachecn.org/cn/0.19.0/modules/svm.html#regression)

[ridge-regression 项目案例](http://sklearn.apachecn.org/cn/0.19.0/auto_examples/linear_model/plot_ridge_path.html#sphx-glr-auto-examples-linear-model-plot-ridge-path-py)

[SVR(kernel='linear') 项目案例](http://sklearn.apachecn.org/cn/0.19.0/auto_examples/svm/plot_svm_regression.html#sphx-glr-auto-examples-svm-plot-svm-regression-py)

> 考虑下面 == 如果无效

### [EnsembleRegressors](http://sklearn.apachecn.org/cn/0.19.0/modules/ensemble.html) 或 [SVR(kernel='rbf')](http://sklearn.apachecn.org/cn/0.19.0/modules/svm.html#regression)

[EnsembleRegressors-RandomForestRegressor 项目案例](http://scikit-learn.org/stable/auto_examples/plot_missing_values.html#sphx-glr-auto-examples-plot-missing-values-py)

[EnsembleRegressors-AdaBoostRegressor 项目案例](http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_regression.html#sphx-glr-auto-examples-ensemble-plot-adaboost-regression-py)

[SVR(kernel='rbf') 项目案例](http://sklearn.apachecn.org/cn/0.19.0/auto_examples/svm/plot_svm_regression.html#sphx-glr-auto-examples-svm-plot-svm-regression-py)

## Dimensionality Reduction 降维

### [Randomized PCA](http://sklearn.apachecn.org/cn/0.19.0/modules/decomposition.html#principal-component-analysis-pca)

[Randomized PCA 项目案例](http://sklearn.apachecn.org/cn/0.19.0/auto_examples/decomposition/plot_pca_vs_lda.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-lda-py)

> 考虑下面 == 无效 并且 如果样本的数据量 < 10k

### [Isomap](http://sklearn.apachecn.org/cn/0.19.0/modules/manifold.html#isomap) 或 [Spectral Embedding](http://sklearn.apachecn.org/cn/0.19.0/modules/manifold.html#spectral-embedding)

> 考虑下面 == 无效 并且 如果样本的数据量 >= 10k

### [Kernel Approximation](http://sklearn.apachecn.org/cn/0.19.0/modules/kernel_approximation.html)

> 考虑下面 == 无效 并且 无效

### [LLE](http://sklearn.apachecn.org/cn/0.19.0/modules/manifold.html#locally-linear-embedding)
