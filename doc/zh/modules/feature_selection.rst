.. currentmodule:: sklearn.feature_selection

.. _feature_selection:

=================
特征选择 
=================

 
在:mod:`sklearn.feature_selection` 模块中的类可以用来对样本集进行特征选择（feature selection）和降维（dimensionality reduction ），这将会提高估计的准确度或者增加他们在高维数据上的性能。


.. _variance_threshold:

移除低方差特征
===================================

:class:`VarianceThreshold` 是特征选择的一个简单基本方法，它会移除所有那些方差不满足一些阈值的特征。默认情况下，它将会移除所有的零方差特征，比如，特征在所有的样本上的值都是一样的（即方差为0）。

例如，假设我们有一个特征是布尔值的数据集，我们想要移除那些在整个数据集中特征值为0或者为1的比例超过80%的特征。布尔特征是伯努利（ Bernoulli ）随机变量，变量的方差为

.. math:: \mathrm{Var}[X] = p(1 - p)

因此，我们可以使用阈值 ``.8 * (1 - .8)``进行选择::

  >>> from sklearn.feature_selection import VarianceThreshold
  >>> X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
  >>> sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
  >>> sel.fit_transform(X)
  array([[0, 1],
         [1, 0],
         [0, 0],
         [1, 1],
         [1, 0],
         [1, 1]])

正如预期一样，VarianceThreshold 移除了第一列，
它的值为0的概率为 :math:`p = 5/6 > .8` .

.. _univariate_feature_selection:

单变量特征选择
============================

单变量的特征选择是通过基于单变量的统计测试来选择最好的特征。它可以当做是评估器的预处理步骤。Scikit-learn将特征选择的内容作为实现了transform方法的对象：

 * :class:`SelectKBest`移除那些除了评分最高的K个特征之外的所有特征

 * :class:`SelectPercentile` 移除除了用户指定的最高得分百分比之外的所有特征

 * using common univariate statistical tests for each feature:
   false positive rate :class:`SelectFpr`, false discovery rate
   :class:`SelectFdr`, or family wise error :class:`SelectFwe`.

 * :class:`GenericUnivariateSelect` 允许使用可配置方法来进行单变量特征选择。它允许超参数搜索评估器来选择最好的单变量特征。
例如下面的实例，我们可以使用 :math:`\chi^2` 检验样本集来选择最好的两个特征：

  >>> from sklearn.datasets import load_iris
  >>> from sklearn.feature_selection import SelectKBest
  >>> from sklearn.feature_selection import chi2
  >>> iris = load_iris()
  >>> X, y = iris.data, iris.target
  >>> X.shape
  (150, 4)
  >>> X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
  >>> X_new.shape
  (150, 2)

这些对象将得分函数作为输入，返回单变量的得分和p值 (或者仅仅是 :class:`SelectKBest` 和
:class:`SelectPercentile` 的分数):

 * 对于回归: :func:`f_regression`, :func:`mutual_info_regression`

 * 对于分类: :func:`chi2`, :func:`f_classif`, :func:`mutual_info_classif`

这些基于F-test的方法计算两个随机变量之间的线性相关程度。另一方面，mutual information 
methods能够计算任何种类的统计相关性，但是是非参数的，需要更多的样本来进行准确的估计。

.. topic:: 稀疏数据的特征选择

  如果你使用的是稀疏的数据 (用稀疏矩阵来表示数据),
   :func:`chi2`, :func:`mutual_info_regression`, :func:`mutual_info_classif`
   处理数据时不会使它变密集。

.. warning::

    不要使用一个回归得分函数来处理分类问题，你会得到无用的结果。

.. topic:: Examples:

    * :ref:`sphx_glr_auto_examples_feature_selection_plot_feature_selection.py`

    * :ref:`sphx_glr_auto_examples_feature_selection_plot_f_test_vs_mi.py`

.. _rfe:

递归特征消除
=============================

给定一个外部的估计器，将特征设置一定的权重 （比如，线性模型的相关系数），  recursive feature elimination (:class:`RFE`)
通过考虑越来越小的特征集合来递归的选择特征。 首先，训练器在初始的特征集合上面训练并且每一个特征的重要程度是通过一个 ``coef_`` 属性
或者 ``feature_importances_`` 属性. 然后，从当前的特征集合中移除最不重要的特征。在特征集合上不断的重复递归这个步骤，知道达到所需要的特征数量为止。
:class:`RFECV` 在一个交叉验证的循环中执行RFE 来找到最优的特征数量

.. topic:: Examples:

    * :ref:`sphx_glr_auto_examples_feature_selection_plot_rfe_digits.py`: A recursive feature elimination example
      showing the relevance of pixels in a digit classification task.

    * :ref:`sphx_glr_auto_examples_feature_selection_plot_rfe_with_cross_validation.py`: A recursive feature
      elimination example with automatic tuning of the number of features
      selected with cross-validation.

.. _select_from_model:

使用 SelectFromModel 选取特征
=======================================

:class:`SelectFromModel` 是一个meta-transformer ，它可以用来处理任何带有 ``coef_`` 或者 ``feature_importances_`` 属性的训练之后的训练器。
如果相关的``coef_`` or ``featureimportances`` 属性值低于预先设置的阈值，这些特征将会被认为不重要并且移除掉。除了指定数值上的阈值之外，还可以使用启发式的方法用字符串参数来找到一个合适的阈值。可以使用的启发式方法有mean、median以及使用浮点数乘以这些（例如，0.1*mean）。

有关如何使用的例子，可以参阅下面的例子。

.. topic:: Examples

    * :ref:`sphx_glr_auto_examples_feature_selection_plot_select_from_model_boston.py`: Selecting the two
      most important features from the Boston dataset without knowing the
      threshold beforehand.

.. _l1_feature_selection:

基于 L1 的特征选取
--------------------------

.. currentmodule:: sklearn

:ref:`Linear models <linear_model>` 使用L1正则化的线性模型会得到稀疏解：他们的许多系数为0。 
当目标是降低使用另一个分类器的数据集的纬度，
他们可以与 :class:`feature_selection.SelectFromModel`
一起使用来选择非零系数。特别的，用于此目的的稀疏估计量是用于回归的
:class:`linear_model.Lasso` , 以及 :class:`linear_model.LogisticRegression` 和
分类器:class:`svm.LinearSVC`
::

  >>> from sklearn.svm import LinearSVC
  >>> from sklearn.datasets import load_iris
  >>> from sklearn.feature_selection import SelectFromModel
  >>> iris = load_iris()
  >>> X, y = iris.data, iris.target
  >>> X.shape
  (150, 4)
  >>> lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
  >>> model = SelectFromModel(lsvc, prefit=True)
  >>> X_new = model.transform(X)
  >>> X_new.shape
  (150, 3)

在svm和逻辑回归中，参数C是用来控制稀疏性的：小的C会导致少的特征被选择。使用Lasso,alpha的值越大，
越少的特征会被选择。

.. topic:: 示例:

    * :ref:`sphx_glr_auto_examples_text_document_classification_20newsgroups.py`: Comparison
      of different algorithms for document classification including L1-based
      feature selection.

.. _compressive_sensing:

.. topic:: **L1-recovery and compressive sensing**

   For a good choice of alpha, the :ref:`lasso` can fully recover the
   exact set of non-zero variables using only few observations, provided
   certain specific conditions are met. In particular, the number of
   samples should be "sufficiently large", or L1 models will perform at
   random, where "sufficiently large" depends on the number of non-zero
   coefficients, the logarithm of the number of features, the amount of
   noise, the smallest absolute value of non-zero coefficients, and the
   structure of the design matrix X. In addition, the design matrix must
   display certain specific properties, such as not being too correlated.

   There is no general rule to select an alpha parameter for recovery of
   non-zero coefficients. It can by set by cross-validation
   (:class:`LassoCV` or :class:`LassoLarsCV`), though this may lead to
   under-penalized models: including a small number of non-relevant
   variables is not detrimental to prediction score. BIC
   (:class:`LassoLarsIC`) tends, on the opposite, to set high values of
   alpha.

   **Reference** Richard G. Baraniuk "Compressive Sensing", IEEE Signal
   Processing Magazine [120] July 2007
   http://dsp.rice.edu/sites/dsp.rice.edu/files/cs/baraniukCSlecture07.pdf


基于 Tree（树）的特征选取
----------------------------

基于树的estimators (查阅 :mod:`sklearn.tree` 模块和树的森林 在 :mod:`sklearn.ensemble` 
模块) 可以用来计算特征的重要性，然后可以消除不相关的特征
(when coupled with the :class:`sklearn.feature_selection.SelectFromModel`
meta-transformer)::

  >>> from sklearn.ensemble import ExtraTreesClassifier
  >>> from sklearn.datasets import load_iris
  >>> from sklearn.feature_selection import SelectFromModel
  >>> iris = load_iris()
  >>> X, y = iris.data, iris.target
  >>> X.shape
  (150, 4)
  >>> clf = ExtraTreesClassifier()
  >>> clf = clf.fit(X, y)
  >>> clf.feature_importances_  # doctest: +SKIP
  array([ 0.04...,  0.05...,  0.4...,  0.4...])
  >>> model = SelectFromModel(clf, prefit=True)
  >>> X_new = model.transform(X)
  >>> X_new.shape               # doctest: +SKIP
  (150, 2)

.. topic:: Examples:

    * :ref:`sphx_glr_auto_examples_ensemble_plot_forest_importances.py`: example on
      synthetic data showing the recovery of the actually meaningful
      features.

    * :ref:`sphx_glr_auto_examples_ensemble_plot_forest_importances_faces.py`: example
      on face recognition data.

特征选取作为 pipeline（管道）的一部分
=======================================

特征选择通常在实际的学习之前用来做预处理。在scikit-learn中推荐的方式是使用
::class:`sklearn.pipeline.Pipeline`::

  clf = Pipeline([
    ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
    ('classification', RandomForestClassifier())
  ])
  clf.fit(X, y)

在这个小节中，我们利用 :class:`sklearn.svm.LinearSVC`
和 :class:`sklearn.feature_selection.SelectFromModel`
来评估特征的重要性并且选择出相关的特征。
然后，在转化后的输出中使用一个  :class:`sklearn.ensemble.RandomForestClassifier` 分类器,
比如只使用相关的特征。你可以使用其他特征选择的方法和提供评估特征重要性的分类器执行相似的操作。
请查阅 :class:`sklearn.pipeline.Pipeline` 更多
 的实例。
