.. _ensemble:

================
Ensemble methods
================

.. currentmodule:: sklearn.ensemble

The goal of **ensemble methods** is to combine the predictions of several
base estimators built with a given learning algorithm in order to improve
generalizability / robustness over a single estimator.

Two families of ensemble methods are usually distinguished:

- In **averaging methods**, the driving principle is to build several
  estimators independently and then to average their predictions. On average,
  the combined estimator is usually better than any of the single base
  estimator because its variance is reduced.

  **Examples:** :ref:`Bagging methods <bagging>`, :ref:`Forests of randomized trees <forest>`, ...

- By contrast, in **boosting methods**, base estimators are built sequentially
  and one tries to reduce the bias of the combined estimator. The motivation is
  to combine several weak models to produce a powerful ensemble.

  **Examples:** :ref:`AdaBoost <adaboost>`, :ref:`Gradient Tree Boosting <gradient_boosting>`, ...


.. _bagging:

Bagging meta-estimator
======================

In ensemble algorithms, bagging methods form a class of algorithms which build
several instances of a black-box estimator on random subsets of the original
training set and then aggregate their individual predictions to form a final
prediction. These methods are used as a way to reduce the variance of a base
estimator (e.g., a decision tree), by introducing randomization into its
construction procedure and then making an ensemble out of it. In many cases,
bagging methods constitute a very simple way to improve with respect to a
single model, without making it necessary to adapt the underlying base
algorithm. As they provide a way to reduce overfitting, bagging methods work
best with strong and complex models (e.g., fully developed decision trees), in
contrast with boosting methods which usually work best with weak models (e.g.,
shallow decision trees).

Bagging methods come in many flavours but mostly differ from each other by the
way they draw random subsets of the training set:

  * When random subsets of the dataset are drawn as random subsets of the
    samples, then this algorithm is known as Pasting [B1999]_.

  * When samples are drawn with replacement, then the method is known as
    Bagging [B1996]_.

  * When random subsets of the dataset are drawn as random subsets of
    the features, then the method is known as Random Subspaces [H1998]_.

  * Finally, when base estimators are built on subsets of both samples and
    features, then the method is known as Random Patches [LG2012]_.

In scikit-learn, bagging methods are offered as a unified
:class:`BaggingClassifier` meta-estimator  (resp. :class:`BaggingRegressor`),
taking as input a user-specified base estimator along with parameters
specifying the strategy to draw random subsets. In particular, ``max_samples``
and ``max_features`` control the size of the subsets (in terms of samples and
features), while ``bootstrap`` and ``bootstrap_features`` control whether
samples and features are drawn with or without replacement. When using a subset
of the available samples the generalization accuracy can be estimated with the
out-of-bag samples by setting ``oob_score=True``. As an example, the
snippet below illustrates how to instantiate a bagging ensemble of
:class:`KNeighborsClassifier` base estimators, each built on random subsets of
50% of the samples and 50% of the features.

    >>> from sklearn.ensemble import BaggingClassifier
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> bagging = BaggingClassifier(KNeighborsClassifier(),
    ...                             max_samples=0.5, max_features=0.5)

.. topic:: Examples:

 * :ref:`sphx_glr_auto_examples_ensemble_plot_bias_variance.py`

.. topic:: References

  .. [B1999] L. Breiman, "Pasting small votes for classification in large
         databases and on-line", Machine Learning, 36(1), 85-103, 1999.

  .. [B1996] L. Breiman, "Bagging predictors", Machine Learning, 24(2),
         123-140, 1996.

  .. [H1998] T. Ho, "The random subspace method for constructing decision
         forests", Pattern Analysis and Machine Intelligence, 20(8), 832-844,
         1998.

  .. [LG2012] G. Louppe and P. Geurts, "Ensembles on Random Patches",
         Machine Learning and Knowledge Discovery in Databases, 346-361, 2012.

.. _forest:

Forests of randomized trees
===========================

The :mod:`sklearn.ensemble` module includes two averaging algorithms based
on randomized :ref:`decision trees <tree>`: the RandomForest algorithm
and the Extra-Trees method. Both algorithms are perturb-and-combine
techniques [B1998]_ specifically designed for trees. This means a diverse
set of classifiers is created by introducing randomness in the classifier
construction.  The prediction of the ensemble is given as the averaged
prediction of the individual classifiers.

As other classifiers, forest classifiers have to be fitted with two
arrays: a sparse or dense array X of size ``[n_samples, n_features]`` holding the
training samples, and an array Y of size ``[n_samples]`` holding the
target values (class labels) for the training samples::

    >>> from sklearn.ensemble import RandomForestClassifier
    >>> X = [[0, 0], [1, 1]]
    >>> Y = [0, 1]
    >>> clf = RandomForestClassifier(n_estimators=10)
    >>> clf = clf.fit(X, Y)

Like :ref:`decision trees <tree>`, forests of trees also extend
to :ref:`multi-output problems <tree_multioutput>`  (if Y is an array of size
``[n_samples, n_outputs]``).

Random Forests
--------------

In random forests (see :class:`RandomForestClassifier` and
:class:`RandomForestRegressor` classes), each tree in the ensemble is
built from a sample drawn with replacement (i.e., a bootstrap sample)
from the training set. In addition, when splitting a node during the
construction of the tree, the split that is chosen is no longer the
best split among all features. Instead, the split that is picked is the
best split among a random subset of the features. As a result of this
randomness, the bias of the forest usually slightly increases (with
respect to the bias of a single non-random tree) but, due to averaging,
its variance also decreases, usually more than compensating for the
increase in bias, hence yielding an overall better model.

In contrast to the original publication [B2001]_, the scikit-learn
implementation combines classifiers by averaging their probabilistic
prediction, instead of letting each classifier vote for a single class.

Extremely Randomized Trees
--------------------------

In extremely randomized trees (see :class:`ExtraTreesClassifier`
and :class:`ExtraTreesRegressor` classes), randomness goes one step
further in the way splits are computed. As in random forests, a random
subset of candidate features is used, but instead of looking for the
most discriminative thresholds, thresholds are drawn at random for each
candidate feature and the best of these randomly-generated thresholds is
picked as the splitting rule. This usually allows to reduce the variance
of the model a bit more, at the expense of a slightly greater increase
in bias::

    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.datasets import make_blobs
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.ensemble import ExtraTreesClassifier
    >>> from sklearn.tree import DecisionTreeClassifier

    >>> X, y = make_blobs(n_samples=10000, n_features=10, centers=100,
    ...     random_state=0)

    >>> clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,
    ...     random_state=0)
    >>> scores = cross_val_score(clf, X, y)
    >>> scores.mean()                             # doctest: +ELLIPSIS
    0.97...

    >>> clf = RandomForestClassifier(n_estimators=10, max_depth=None,
    ...     min_samples_split=2, random_state=0)
    >>> scores = cross_val_score(clf, X, y)
    >>> scores.mean()                             # doctest: +ELLIPSIS
    0.999...

    >>> clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,
    ...     min_samples_split=2, random_state=0)
    >>> scores = cross_val_score(clf, X, y)
    >>> scores.mean() > 0.999
    True

.. figure:: ../auto_examples/ensemble/images/sphx_glr_plot_forest_iris_001.png
    :target: ../auto_examples/ensemble/plot_forest_iris.html
    :align: center
    :scale: 75%

Parameters
----------

The main parameters to adjust when using these methods is ``n_estimators``
and ``max_features``. The former is the number of trees in the forest. The
larger the better, but also the longer it will take to compute. In
addition, note that results will stop getting significantly better
beyond a critical number of trees. The latter is the size of the random
subsets of features to consider when splitting a node. The lower the
greater the reduction of variance, but also the greater the increase in
bias. Empirical good default values are ``max_features=n_features``
for regression problems, and ``max_features=sqrt(n_features)`` for
classification tasks (where ``n_features`` is the number of features
in the data). Good results are often achieved when setting ``max_depth=None``
in combination with ``min_samples_split=1`` (i.e., when fully developing the
trees). Bear in mind though that these values are usually not optimal, and
might result in models that consume a lot of RAM. The best parameter values
should always be cross-validated. In addition, note that in random forests,
bootstrap samples are used by default (``bootstrap=True``)
while the default strategy for extra-trees is to use the whole dataset
(``bootstrap=False``).
When using bootstrap sampling the generalization accuracy can be estimated
on the left out or out-of-bag samples. This can be enabled by
setting ``oob_score=True``.

.. note::

    The size of the model with the default parameters is :math:`O( M * N * log (N) )`,
    where :math:`M` is the number of trees and :math:`N` is the number of samples.
    In order to reduce the size of the model, you can change these parameters:
    ``min_samples_split``, ``min_samples_leaf``, ``max_leaf_nodes`` and ``max_depth``.

Parallelization
---------------

Finally, this module also features the parallel construction of the trees
and the parallel computation of the predictions through the ``n_jobs``
parameter. If ``n_jobs=k`` then computations are partitioned into
``k`` jobs, and run on ``k`` cores of the machine. If ``n_jobs=-1``
then all cores available on the machine are used. Note that because of
inter-process communication overhead, the speedup might not be linear
(i.e., using ``k`` jobs will unfortunately not be ``k`` times as
fast). Significant speedup can still be achieved though when building
a large number of trees, or when building a single tree requires a fair
amount of time (e.g., on large datasets).

.. topic:: Examples:

 * :ref:`sphx_glr_auto_examples_ensemble_plot_forest_iris.py`
 * :ref:`sphx_glr_auto_examples_ensemble_plot_forest_importances_faces.py`
 * :ref:`sphx_glr_auto_examples_plot_multioutput_face_completion.py`

.. topic:: References

 .. [B2001] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.

 .. [B1998] L. Breiman, "Arcing Classifiers", Annals of Statistics 1998.

 * P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized
   trees", Machine Learning, 63(1), 3-42, 2006.

.. _random_forest_feature_importance:

Feature importance evaluation
-----------------------------

The relative rank (i.e. depth) of a feature used as a decision node in a
tree can be used to assess the relative importance of that feature with
respect to the predictability of the target variable. Features used at
the top of the tree contribute to the final prediction decision of a 
larger fraction of the input samples. The **expected fraction of the 
samples** they contribute to can thus be used as an estimate of the
**relative importance of the features**.

By **averaging** those expected activity rates over several randomized
trees one can **reduce the variance** of such an estimate and use it
for feature selection.

The following example shows a color-coded representation of the relative
importances of each individual pixel for a face recognition task using
a :class:`ExtraTreesClassifier` model.

.. figure:: ../auto_examples/ensemble/images/sphx_glr_plot_forest_importances_faces_001.png
   :target: ../auto_examples/ensemble/plot_forest_importances_faces.html
   :align: center
   :scale: 75

In practice those estimates are stored as an attribute named
``feature_importances_`` on the fitted model. This is an array with shape
``(n_features,)`` whose values are positive and sum to 1.0. The higher
the value, the more important is the contribution of the matching feature
to the prediction function.

.. topic:: Examples:

 * :ref:`sphx_glr_auto_examples_ensemble_plot_forest_importances_faces.py`
 * :ref:`sphx_glr_auto_examples_ensemble_plot_forest_importances.py`

.. _random_trees_embedding:

Totally Random Trees Embedding
------------------------------

:class:`RandomTreesEmbedding` implements an unsupervised transformation of the
data.  Using a forest of completely random trees, :class:`RandomTreesEmbedding`
encodes the data by the indices of the leaves a data point ends up in.  This
index is then encoded in a one-of-K manner, leading to a high dimensional,
sparse binary coding.
This coding can be computed very efficiently and can then be used as a basis
for other learning tasks.
The size and sparsity of the code can be influenced by choosing the number of
trees and the maximum depth per tree. For each tree in the ensemble, the coding
contains one entry of one. The size of the coding is at most ``n_estimators * 2
** max_depth``, the maximum number of leaves in the forest.

As neighboring data points are more likely to lie within the same leaf of a tree,
the transformation performs an implicit, non-parametric density estimation.

.. topic:: Examples:

 * :ref:`sphx_glr_auto_examples_ensemble_plot_random_forest_embedding.py`

 * :ref:`sphx_glr_auto_examples_manifold_plot_lle_digits.py` compares non-linear
   dimensionality reduction techniques on handwritten digits.

 * :ref:`sphx_glr_auto_examples_ensemble_plot_feature_transformation.py` compares
   supervised and unsupervised tree based feature transformations.

.. seealso::

   :ref:`manifold` techniques can also be useful to derive non-linear
   representations of feature space, also these approaches focus also on
   dimensionality reduction.


.. _adaboost:

AdaBoost
========

模型 :mod:`sklearn.ensemble` 包含最流行的提升算法AdaBoost, 这个算法是由 Freund and Schapire 在1995提出来的 [FS1995]_.

AdaBoost的核心思想是用训练数据来反复修正数据来学习一系列的弱学习器(例如:一个弱学习器模型仅仅比随机猜测好一点,
比如一个简单的决策树),由这些弱学习器学到的结果然后通过加权投票(或加权求和)的方式组合起来,
从而得到我们最终的预测结果.在每一次提升迭代中修改的数据由应用于每一个训练样本所生成的弱学习器
的权重 :math:`w_1`, :math:`w_2`, ..., :math:`w_N` 组成.初始化时,将所有弱学习器的权重都
设置为 :math:`w_i = 1/N` ,因此第一次迭代仅仅是通过原始数据训练出一个弱学习器.在接下来的
连续迭代中,样本的权重逐个的被修改,学习算法也因此要重新应用这些已经修改的权重.在给定的步骤,
那些在之前预测出错误结果的弱学习器的权重将会被加强,而那些在之前预测出正确结果的弱学习器的权
重将会被减弱.随着迭代次数的增加,那些难以预测的样本的影响将会越来越大,每一个随后的弱学习器都将
会更加关注那些在序列中之前错误的例子 [HTF]_.

.. figure:: ../auto_examples/ensemble/images/sphx_glr_plot_adaboost_hastie_10_2_001.png
   :target: ../auto_examples/ensemble/plot_adaboost_hastie_10_2.html
   :align: center
   :scale: 75

AdaBoost既可以用在分类问题也可以用在回归问题中:

  - For multi-class classification, :class:`AdaBoostClassifier` implements
    AdaBoost-SAMME and AdaBoost-SAMME.R [ZZRH2009]_.

  - For regression, :class:`AdaBoostRegressor` implements AdaBoost.R2 [D1997]_.

Usage
-----

下面的例子展示了如何用100个弱学习器来拟合一个AdaBoost分类器::

    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.ensemble import AdaBoostClassifier

    >>> iris = load_iris()
    >>> clf = AdaBoostClassifier(n_estimators=100)
    >>> scores = cross_val_score(clf, iris.data, iris.target)
    >>> scores.mean()                             # doctest: +ELLIPSIS
    0.9...

弱学习器的数量是由参数 ``n_estimators`` 来控制. ``learning_rate`` 参数用来控制每个弱学习器对
最终的结果的贡献程度.弱学习器默认为决策树桩.不同的弱学习器可以通过参数 ``base_estimator``
来指定.获取一个好的预测结果主要调整的参数是 ``n_estimators`` 和 ``base_estimator`` 的复杂度
(例如:对于弱学习器为决策树的情况,树的深度 ``max_depth`` 或叶节点的最小样本数 ``min_samples_leaf``
等都是控制树的复杂度的参数)

.. topic:: Examples:

 * :ref:`sphx_glr_auto_examples_ensemble_plot_adaboost_hastie_10_2.py` compares the
   classification error of a decision stump, decision tree, and a boosted
   decision stump using AdaBoost-SAMME and AdaBoost-SAMME.R.

 * :ref:`sphx_glr_auto_examples_ensemble_plot_adaboost_multiclass.py` shows the performance
   of AdaBoost-SAMME and AdaBoost-SAMME.R on a multi-class problem.

 * :ref:`sphx_glr_auto_examples_ensemble_plot_adaboost_twoclass.py` shows the decision boundary
   and decision function values for a non-linearly separable two-class problem
   using AdaBoost-SAMME.

 * :ref:`sphx_glr_auto_examples_ensemble_plot_adaboost_regression.py` demonstrates regression
   with the AdaBoost.R2 algorithm.

.. topic:: References

 .. [FS1995] Y. Freund, and R. Schapire, "A Decision-Theoretic Generalization of
             On-Line Learning and an Application to Boosting", 1997.

 .. [ZZRH2009] J. Zhu, H. Zou, S. Rosset, T. Hastie. "Multi-class AdaBoost",
               2009.

 .. [D1997] H. Drucker. "Improving Regressors using Boosting Techniques", 1997.

 .. [HTF] T. Hastie, R. Tibshirani and J. Friedman, "Elements of
              Statistical Learning Ed. 2", Springer, 2009.


.. _gradient_boosting:

梯度树提升(Gradient Tree Boosting)
======================

`Gradient Tree Boosting <https://en.wikipedia.org/wiki/Gradient_boosting>`_
或梯度提升回归树(GBRT)是提升算法推广到任意可微的损失函数的泛化.GBRT是一个准确高效的现有程序,
它既能用于分类问题也可以用于回归问题.梯度树提升模型被应用到各种领域包括网页搜索排名和生态领域.

GBRT的优点:

  + 对混合型数据的自然处理(异构特性)

  + 强大的预测能力

  + 在输出空间中对异常点的鲁棒性(通过鲁棒损失函数实现)

GBRT的缺点:

  + 可扩展性差.由于提升算法的有序性(也就是说下一步的结果依赖于上一步),因此很难做并行.

模型 :mod:`sklearn.ensemble` 通过梯度提升树提供了分类和回归的方法.

分类(Classification)
---------------

:class:`GradientBoostingClassifier` 既支持二分类又支持多分类问题.
下面的例子展示了如何用100个弱学习器来拟合一个梯度提升分类器::
    >>> from sklearn.datasets import make_hastie_10_2
    >>> from sklearn.ensemble import GradientBoostingClassifier

    >>> X, y = make_hastie_10_2(random_state=0)
    >>> X_train, X_test = X[:2000], X[2000:]
    >>> y_train, y_test = y[:2000], y[2000:]

    >>> clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    ...     max_depth=1, random_state=0).fit(X_train, y_train)
    >>> clf.score(X_test, y_test)                 # doctest: +ELLIPSIS
    0.913...

弱学习器(例如:回归树)的数量由参数 ``n_estimators`` 来控制;每个树的大小可以被控制通过参数 ``max_depth``
设置树的深度通或者通过参数 ``max_leaf_nodes`` 设置叶节点数目. ``learning_rate``
是一个在(0,1]之间的超参数,这个参数通过shrinkage(缩减步长)来控制过拟合.

.. note::

   超过两类的分类问题需要在每一次迭代时诱导 ``n_classes`` 个回归树.因此,所有的诱导树数量等
   于 ``n_classes * n_estimators`` .对于拥有大量类别的数据集我们强烈推荐使用
   :class:`RandomForestClassifier` 来代替 :class:`GradientBoostingClassifier` .

回归(Regression)
----------

对于回归问题 :class:`GradientBoostingRegressor` 支持 :ref:`different loss functions <gradient_boosting_loss>` ,
这些损失函数可以通过参数 ``loss`` 来指定;对于回归问题默认的损失函数是最小二乘损失函数( ``'ls'`` ).

::

    >>> import numpy as np
    >>> from sklearn.metrics import mean_squared_error
    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.ensemble import GradientBoostingRegressor

    >>> X, y = make_friedman1(n_samples=1200, random_state=0, noise=1.0)
    >>> X_train, X_test = X[:200], X[200:]
    >>> y_train, y_test = y[:200], y[200:]
    >>> est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
    ...     max_depth=1, random_state=0, loss='ls').fit(X_train, y_train)
    >>> mean_squared_error(y_test, est.predict(X_test))    # doctest: +ELLIPSIS
    5.00...

下图展示了应用GradientBoostingRegressor算法,设置其损失函数为最小二乘损失,基本学习器个数为500来处理
:func:`sklearn.datasets.load_boston` 数据集的结果.左图表示每一次迭代的训练误差和测试误差.每一次迭
代的训练误差保存在提升树模型的 :attr:`~GradientBoostingRegressor.train_score_` 属性中,每一次迭代的测试误差能够通过
:meth:`~GradientBoostingRegressor.staged_predict` 方法获取,该方法返回一个生成器,用来产生每一
步的预测结果.像下面这种方式画图,可以通过提前停止的方法来决定最优的树的数量.右图表示每个特征的重要性,它
可以通过 ``feature_importances_`` 属性来获取.

.. figure:: ../auto_examples/ensemble/images/sphx_glr_plot_gradient_boosting_regression_001.png
   :target: ../auto_examples/ensemble/plot_gradient_boosting_regression.html
   :align: center
   :scale: 75

.. topic:: Examples:

 * :ref:`sphx_glr_auto_examples_ensemble_plot_gradient_boosting_regression.py`
 * :ref:`sphx_glr_auto_examples_ensemble_plot_gradient_boosting_oob.py`

.. _gradient_boosting_warm_start:

拟合额外的弱学习器(Fitting additional weak-learners)
--------------------------------

 :class:`GradientBoostingRegressor` 和 :class:`GradientBoostingClassifier`都支持设置参数
``warm_start=True``,这样设置允许我们在已经拟合的模型上面添加更多的estimators.

::

  >>> _ = est.set_params(n_estimators=200, warm_start=True)  # set warm_start and new nr of trees
  >>> _ = est.fit(X_train, y_train) # fit additional 100 trees to est
  >>> mean_squared_error(y_test, est.predict(X_test))    # doctest: +ELLIPSIS
  3.84...

.. _gradient_boosting_tree_size:

控制树的大小(Controlling the tree size)
-------------------------

回归树基本学习器的大小定义了变量影响的级别,这个变量影响可以被梯度提升模型捕获到.通常一棵树的深度
``h`` 能捕获秩为  ``h`` 的影响.这里有两种控制单棵回归树大小的方法.

如果你指定 ``max_depth=h`` 然后深度为 ``h`` 的完全二叉树将会生成.这棵树将会有 ``2**h`` 个
叶节点 和 ``2**h - 1`` 个切分节点.

另外,你能通过参数 ``max_leaf_nodes`` 指定叶节点的数量来控制树的大小.在这种情况下,树将会
使用最优搜索来生成,这种搜索方式是通过选取纯度提升最大的节点来最先被扩大.一棵树的 ``max_leaf_nodes=k``
拥有 ``k - 1`` 切分节点,因此可以模拟高达 ``max_leaf_nodes - 1`` 的相互影响.

我们发现 ``max_leaf_nodes=k`` 给出与 ``max_depth=k-1`` 相当的结果,但是其训练速度更快,同时
也会损失一点训练误差作为代价.参数 ``max_leaf_nodes`` 对应于文章 [F2001]_ 中的梯度提升章节的变量 ``J``
同时与R语言的gbm包的参数 ``interaction.depth`` 相关,两者间的关系是 ``max_leaf_nodes == interaction.depth + 1``.

数学公式(Mathematical formulation)
-------------------------

GBRT可以认为是下面形式的加法模型:

  .. math::

    F(x) = \sum_{m=1}^{M} \gamma_m h_m(x)

其中 :math:`h_m(x)` 是基本函数,在提升算法中它通常被称作 *weak learners* .梯度树提升利用固定大小
的 :ref:`decision trees <tree>` 作为弱学习器,决策树本身拥有的一些特性使它能够在提升中变得有价值.
这种特性即:能够处理混合类型的数据和构建复杂的功能.

与其他提升算法类似,GBRT构建加法模型也是使用到了前向分步算法:

  .. math::

    F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)

在每一步给定当前模型 :math:`F_{m-1}` 和它的拟合函数 :math:`F_{m-1}(x_i)` 的前提下,
:math:`h_m(x)` 的选择是根据最小化损失函数 :math:`L` 得到的.

  .. math::

    F_m(x) = F_{m-1}(x) + \arg\min_{h} \sum_{i=1}^{n} L(y_i,
    F_{m-1}(x_i) - h(x))

初始化模型 :math:`F_{0}` 是一个具体的问题,对于最小二乘回归一个通常的选择是目标值的平均数.

.. note:: 初始化模型也能够通过参数 ``init`` 来指定.传过去的参数必须实现 ``fit`` 和 ``predict`` 方法.

梯度提升尝试以数值的方式通过最速下降法解决这个最小化问题.最速下降的方向是当前模型 :math:`F_{m-1}` 的
损失函数的梯度的负方向,对于任意的可微损失函数,这个梯度都是可以计算得到的:

  .. math::

    F_m(x) = F_{m-1}(x) + \gamma_m \sum_{i=1}^{n} \nabla_F L(y_i,
    F_{m-1}(x_i))

其中步长:math:`\gamma_m` 通过线性查找得到来选择:

  .. math::

    \gamma_m = \arg\min_{\gamma} \sum_{i=1}^{n} L(y_i, F_{m-1}(x_i)
    - \gamma \frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)})

这个算法对于分类和回归问题的唯一不同点在于对于具体损失函数的使用.

.. _gradient_boosting_loss:

损失函数(Loss Functions)
...............

以下就是目前支持的损失函数,这些损失函数可以通过参数``loss``指定:

  * Regression

    * Least squares (``'ls'``): 在回归问题中经常被用作损失函数是因为其优越的计算性能,
      初始化模型通过目标值的均值来给出.
    * Least absolute deviation (``'lad'``): 一个鲁棒性的损失函数,模型的初始化通过目
      标值的中位数给出.
    * Huber (``'huber'``): 另一个鲁棒性的损失函数,它是由最小二乘和最小绝对偏差结合得到.
      其利用 ``alpha`` 来控制模型对于异常点的敏感度(详细介绍请参考 [F2001]_).
    * Quantile (``'quantile'``): 分位数损失函数.用 ``0 < alpha < 1`` 来指定分位数这个损
      失函数可以用来产生预测间隔.(详见 :ref:`sphx_glr_auto_examples_ensemble_plot_gradient_boosting_quantile.py` ).

  * Classification

    * Binomial deviance (``'deviance'``): 对于二分类问题(提供概率估计)即负的二项log似然
      损失函数.模型以log的比值比来初始化.
    * Multinomial deviance (``'deviance'``): 对于多分类问题的负的多项log似然损失函数具有
       ``n_classes`` 个互斥的类.其提供概率估计,初始模型以每个类的先验概率来给出.在每一次迭代中
       ``n_classes`` 回归树被构建,这使得GBRT对于大量类的数据集而言十分低效.
    * Exponential loss (``'exponential'``): 与 :class:`AdaBoostClassifier` 相同的损失
      函数.相比如 ``'deviance'`` 对于有错误类别的样本的鲁棒性很差,只能用在二分类问题中.

正则化(Regularization)
----------------

.. _gradient_boosting_shrinkage:

缩减(Shrinkage)
..........

[F2001]_ 提出了一种简单的正则化策略,通过一个因素 :math:`\nu`来缩小每个弱学习器对于最终结果的贡献:

.. math::

    F_m(x) = F_{m-1}(x) + \nu \gamma_m h_m(x)

参数 :math:`\nu` 也叫作 **learning rate** 因为它缩小了梯度下降的步长,它可以通过参数 ``learning_rate`` 来设置.

在拟合一定数量的弱学习器时,参数 ``learning_rate`` 和参数 ``n_estimators`` 之间有很强的制约关系.
较小的``learning_rate`` 需要大量的弱学习器才能保证训练误差的不变.经验表明较小值的 ``learning_rate``
将会得到更好的测试误差. [HTF2009]_ 推荐把 ``learning_rate`` 设置为一个较小的常数
(例如: ``learning_rate <= 0.1`` )同时通过提前停止策略来选择合适的 ``n_estimators`` .
有关 ``learning_rate`` 和 ``n_estimators`` 更详细的讨论可以参考 [R2007]_.

子采样(Subsampling)
............

[F1999]_ 提出了随机梯度提升,这种方法将梯度提升和bootstrap averaging(bagging)结合.在每次迭代
中,基本分类器通过抽取所有训练数据中一小部分的 ``subsample`` 来训练得到.子样本采用无放回的方式采
样. ``subsample`` 参数的值一般设置为0.5.

下图表明了缩减和子采样对于模型拟合好坏的影响.我们可以明显看到缩减比无缩减拥有更好的表现.而子采
样和缩减的结合能更进一步的增加模型的准确率.相反,使用子采样而不使用缩减的结果十分糟糕.

.. figure:: ../auto_examples/ensemble/images/sphx_glr_plot_gradient_boosting_regularization_001.png
   :target: ../auto_examples/ensemble/plot_gradient_boosting_regularization.html
   :align: center
   :scale: 75

另一个减少方差的策略是特征子采样,这种方法类似于 :class:`RandomForestClassifier` 中的随机分割.
子采样的特征数可以通过参数 ``max_features`` 来控制.

.. note:: 采用一个较小的 ``max_features`` 值能大大缩减模型的训练时间.

随机梯度提升允许通过计算那些不在自助采样之内的样本偏差的改进来计算测试偏差的包外估计.这个改进保存在属性
:attr:`~GradientBoostingRegressor.oob_improvement_` 中. ``oob_improvement_[i]`` 如果将
第i步添加到当前预测中,则对OOB样本的损失进行改进.包外估计可以使用在模型选择中,例如决定最优迭代次
数.OOB估计通常都很悲观,因此我们推荐使用交叉验证来代替它,只有当交叉验证需要的时间太长时才考虑用OOB.

.. topic:: Examples:

 * :ref:`sphx_glr_auto_examples_ensemble_plot_gradient_boosting_regularization.py`
 * :ref:`sphx_glr_auto_examples_ensemble_plot_gradient_boosting_oob.py`
 * :ref:`sphx_glr_auto_examples_ensemble_plot_ensemble_oob.py`

解释性(Interpretation)
--------------

单棵决策树可通过简单的可视化树结构来解释.然而对于梯度提升模型来说,一般拥有几百棵树,因此它们不可能简单的通
过可视化单独的每一棵树来解释.幸运的是,有很多关于总结和解释梯度提升模型的技术被提出.

特征重要性(Feature importance)
..................

通常情况下每个特征对于预测膜表的贡献不是相同的.在很多情形下大多数特征实际上是无关的.当解释一个模型时,第一
个问题通常是:这些重要的特征是什么?它们是如何有助于预测目标的响应?

单个决策树本质上是通过选择最佳切分点来进行特征选择.这个信息可以用来检测每个特征的重要性.一个基本的观点:
一个特征被用作切分点的次数越多,则该特征就越重要.这个重要的概念可以通过简单的平均每棵树的特征的重要性被扩
展到决策树的组合中.(详见 :ref:`random_forest_feature_importance` ).

一个梯度提升模型中特征的重要性分数可以通过属性 ``feature_importances_`` 获取::

    >>> from sklearn.datasets import make_hastie_10_2
    >>> from sklearn.ensemble import GradientBoostingClassifier

    >>> X, y = make_hastie_10_2(random_state=0)
    >>> clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    ...     max_depth=1, random_state=0).fit(X, y)
    >>> clf.feature_importances_  # doctest: +ELLIPSIS
    array([ 0.11,  0.1 ,  0.11,  ...

.. topic:: Examples:

 * :ref:`sphx_glr_auto_examples_ensemble_plot_gradient_boosting_regression.py`

.. currentmodule:: sklearn.ensemble.partial_dependence

.. _partial_dependence:

部分依赖(Partial dependence)
..................

部分依赖图(PDP)展示了目标响应和一系列目标特征的依赖关系,同时边缘化了其他所有特征值(候选特征).
直觉上,我们可以解释为作为目标特征函数 [2]_ 的预期目标响应 [1]_ .

由于人类感知能力的限制,目标特征的设置必须小一点(通常是1到2),因此目标特征通常在最重要的特征中选择.

下表展示了加尼福尼亚房价数据集的四个单向和一个双向的部分依赖图:

.. figure:: ../auto_examples/ensemble/images/sphx_glr_plot_partial_dependence_001.png
   :target: ../auto_examples/ensemble/plot_partial_dependence.html
   :align: center
   :scale: 70

单向PDPs告诉我们目标响应和目标特征的相互影响(例如:线性或者非线性).上表中的左上图展示了一个地区中等收入对
中等房价的影响.我们可以清楚的看到两者之间是线性相关的.

两个目标特征的PDPs展示了这两个特征之间的相互影响.例如:上图中两个变量的PDP展示了房价中位数与房子年龄和
每户平均入住人数之间的依赖关系.我们能清楚的看到这两个特征之间的影响:对于每户入住均值而言,当其值大于2时,
房价与房屋年龄几乎是相对独立的,而其值小于2的时,房价对年龄的依赖性就会很强.

模型 :mod:`partial_dependence` 提供了一个便捷的函数
:func:`~sklearn.ensemble.partial_dependence.plot_partial_dependence` 来产生单向或双向部分依
赖图.在下图的例子中我们展示如何创建一个部分依赖的网格图:特征值介于 ``0`` 和 ``1`` 的两个单向依赖PDPs和一
个在两个特征间的双向PDPs::

    >>> from sklearn.datasets import make_hastie_10_2
    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> from sklearn.ensemble.partial_dependence import plot_partial_dependence

    >>> X, y = make_hastie_10_2(random_state=0)
    >>> clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    ...     max_depth=1, random_state=0).fit(X, y)
    >>> features = [0, 1, (0, 1)]
    >>> fig, axs = plot_partial_dependence(clf, X, features) #doctest: +SKIP

对于多类别的模型,你需要通过参数 ``label`` 设置类别标签来创建PDP标签::

    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> mc_clf = GradientBoostingClassifier(n_estimators=10,
    ...     max_depth=1).fit(iris.data, iris.target)
    >>> features = [3, 2, (3, 2)]
    >>> fig, axs = plot_partial_dependence(mc_clf, X, features, label=0) #doctest: +SKIP

如果你需要部分依赖函数的原始值而不是图,你可以利用
:func:`~sklearn.ensemble.partial_dependence.partial_dependence` 函数::

    >>> from sklearn.ensemble.partial_dependence import partial_dependence

    >>> pdp, axes = partial_dependence(clf, [0], X=X)
    >>> pdp  # doctest: +ELLIPSIS
    array([[ 2.46643157,  2.46643157, ...
    >>> axes  # doctest: +ELLIPSIS
    [array([-1.62497054, -1.59201391, ...

这个函数要求参数 ``grid`` ,这个参数的指定应该评估部分依赖函数的的目标特征值或参数
``X`` ,这个参数对于训练数据自动产生 ``grid`` 来说是一个十分方便的模型.如果 ``X``
给定, 函数 ``axes`` 的值将被返回通过给定目标特征的轴.

对于 ``grid`` 中的每一个'目标'特征的值,部分依赖函数需要边缘化一棵树中所有候选特征的可能值的
预测.在决策树中,这个函数可以在不参考训练数据的情况下被高效的评估,对于每一网格点执行加权遍历:如果切
分点包含'目标'特征,遍历其相关的左分支或相关的右分支,否则就遍历两个分支每一个分支的加权是通过进
入该分支的训练样本的占比,最后,部分依赖通过所有访问的叶节点的权重的平均值给出.对于树组合的结果,
需要对每棵树的结果再次平均得到.

.. rubric:: Footnotes

.. [1] For classification with ``loss='deviance'``  the target
   response is logit(p).

.. [2] More precisely its the expectation of the target response after
   accounting for the initial model; partial dependence plots
   do not include the ``init`` model.

.. topic:: Examples:

 * :ref:`sphx_glr_auto_examples_ensemble_plot_partial_dependence.py`


.. topic:: References

 .. [F2001] J. Friedman, "Greedy Function Approximation: A Gradient Boosting Machine",
   The Annals of Statistics, Vol. 29, No. 5, 2001.

 .. [F1999] J. Friedman, "Stochastic Gradient Boosting", 1999

 .. [HTF2009] T. Hastie, R. Tibshirani and J. Friedman, "Elements of Statistical Learning Ed. 2", Springer, 2009.

 .. [R2007] G. Ridgeway, "Generalized Boosted Models: A guide to the gbm package", 2007


 .. _voting_classifier:

Voting Classifier
========================

The idea behind the :class:`VotingClassifier` is to combine
conceptually different machine learning classifiers and use a majority vote
or the average predicted probabilities (soft vote) to predict the class labels.
Such a classifier can be useful for a set of equally well performing model
in order to balance out their individual weaknesses.


Majority Class Labels (Majority/Hard Voting)
--------------------------------------------

In majority voting, the predicted class label for a particular sample is
the class label that represents the majority (mode) of the class labels
predicted by each individual classifier.

E.g., if the prediction for a given sample is

- classifier 1 -> class 1
- classifier 2 -> class 1
- classifier 3 -> class 2

the VotingClassifier (with ``voting='hard'``) would classify the sample
as "class 1" based on the majority class label.

In the cases of a tie, the `VotingClassifier` will select the class based
on the ascending sort order. E.g., in the following scenario

- classifier 1 -> class 2
- classifier 2 -> class 1

the class label 1 will be assigned to the sample.

Usage
.....

The following example shows how to fit the majority rule classifier::

   >>> from sklearn import datasets
   >>> from sklearn.model_selection import cross_val_score
   >>> from sklearn.linear_model import LogisticRegression
   >>> from sklearn.naive_bayes import GaussianNB
   >>> from sklearn.ensemble import RandomForestClassifier
   >>> from sklearn.ensemble import VotingClassifier

   >>> iris = datasets.load_iris()
   >>> X, y = iris.data[:, 1:3], iris.target

   >>> clf1 = LogisticRegression(random_state=1)
   >>> clf2 = RandomForestClassifier(random_state=1)
   >>> clf3 = GaussianNB()

   >>> eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')

   >>> for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
   ...     scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
   ...     print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
   Accuracy: 0.90 (+/- 0.05) [Logistic Regression]
   Accuracy: 0.93 (+/- 0.05) [Random Forest]
   Accuracy: 0.91 (+/- 0.04) [naive Bayes]
   Accuracy: 0.95 (+/- 0.05) [Ensemble]


Weighted Average Probabilities (Soft Voting)
--------------------------------------------

In contrast to majority voting (hard voting), soft voting
returns the class label as argmax of the sum of predicted probabilities.

Specific weights can be assigned to each classifier via the ``weights``
parameter. When weights are provided, the predicted class probabilities
for each classifier are collected, multiplied by the classifier weight,
and averaged. The final class label is then derived from the class label
with the highest average probability.

To illustrate this with a simple example, let's assume we have 3
classifiers and a 3-class classification problems where we assign
equal weights to all classifiers: w1=1, w2=1, w3=1.

The weighted average probabilities for a sample would then be
calculated as follows:

================  ==========    ==========      ==========
classifier        class 1       class 2         class 3
================  ==========    ==========      ==========
classifier 1	  w1 * 0.2      w1 * 0.5        w1 * 0.3
classifier 2	  w2 * 0.6      w2 * 0.3        w2 * 0.1
classifier 3      w3 * 0.3      w3 * 0.4        w3 * 0.3
weighted average  0.37	        0.4             0.23
================  ==========    ==========      ==========

Here, the predicted class label is 2, since it has the
highest average probability.

The following example illustrates how the decision regions may change
when a soft `VotingClassifier` is used based on an linear Support
Vector Machine, a Decision Tree, and a K-nearest neighbor classifier::

   >>> from sklearn import datasets
   >>> from sklearn.tree import DecisionTreeClassifier
   >>> from sklearn.neighbors import KNeighborsClassifier
   >>> from sklearn.svm import SVC
   >>> from itertools import product
   >>> from sklearn.ensemble import VotingClassifier

   >>> # Loading some example data
   >>> iris = datasets.load_iris()
   >>> X = iris.data[:, [0,2]]
   >>> y = iris.target

   >>> # Training classifiers
   >>> clf1 = DecisionTreeClassifier(max_depth=4)
   >>> clf2 = KNeighborsClassifier(n_neighbors=7)
   >>> clf3 = SVC(kernel='rbf', probability=True)
   >>> eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)], voting='soft', weights=[2,1,2])

   >>> clf1 = clf1.fit(X,y)
   >>> clf2 = clf2.fit(X,y)
   >>> clf3 = clf3.fit(X,y)
   >>> eclf = eclf.fit(X,y)

.. figure:: ../auto_examples/ensemble/images/sphx_glr_plot_voting_decision_regions_001.png
    :target: ../auto_examples/ensemble/plot_voting_decision_regions.html
    :align: center
    :scale: 75%

Using the `VotingClassifier` with `GridSearch`
----------------------------------------------

The `VotingClassifier` can also be used together with `GridSearch` in order
to tune the hyperparameters of the individual estimators::

   >>> from sklearn.model_selection import GridSearchCV
   >>> clf1 = LogisticRegression(random_state=1)
   >>> clf2 = RandomForestClassifier(random_state=1)
   >>> clf3 = GaussianNB()
   >>> eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')

   >>> params = {'lr__C': [1.0, 100.0], 'rf__n_estimators': [20, 200],}

   >>> grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)
   >>> grid = grid.fit(iris.data, iris.target)

Usage
.....

In order to predict the class labels based on the predicted
class-probabilities (scikit-learn estimators in the VotingClassifier
must support ``predict_proba`` method)::

   >>> eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')

Optionally, weights can be provided for the individual classifiers::

   >>> eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft', weights=[2,5,1])
