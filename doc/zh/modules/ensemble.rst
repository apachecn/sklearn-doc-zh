.. _ensemble:

================
集成方法
================

.. currentmodule:: sklearn.ensemble

``注意，在本文中bagging和boosting为了更好的保留原文意图，不进行翻译``
``estimator->估计器  base estimator->基估计器``

**集成方法** 的目标是把使用给定学习算法构建的几个基估计器的预测结果结合起来，从而获得比单个估计器更好的泛化能力/鲁棒性。

集成方法通常分为两种:

- **平均方法**，该方法的驱动原则是构建几个独立的估计器，然后平均化它们的预测结果。一般来说组合之后的估计器是会要比单个估计器要好的，因为它的方差减小了。

  **示例:** :ref:`Bagging 方法 <bagging>`, :ref:`随机森林 <forest>`, ...

- 相比之下，在 **boosting 方法** 中，基估计器是依次构建的并且每一个都尝试去减少组合估计器的偏差。主要目的是为了把多个弱模型相结合变得更加强大。

  **示例:** :ref:`AdaBoost <adaboost>`, :ref:`梯度提升树 <gradient_boosting>`, ...


.. _bagging:

Bagging meta-estimator（Bagging 元估计）
========================================

在集成算法中，bagging方法会在原始训练集的随机子集上构建几个黑盒估计器，然后把这几个估计器的预测结果结合起来形成最终的预测。
该方法通过在构建模型的过程中引入随机性，以此来减少基估计器的方差(例如，决策树)。
在多数情况下，bagging方法提供了一种非常简单的方式来对单一模型进行改进，同时无需适应底层算法。
因为bagging方法可以减小过拟合，所以很适合在强分类器和复杂模型上使用（例如，完全决策树，fully developed decision trees），相比之下boosting方法在弱模型上表现更好（例如，浅层决策树，shallow decision trees）。


bagging 方法有很多种，区别大多数在于抽取训练子集的方法：

  * 如果抽取的数据集是样本的的子集，我们叫做粘贴(Pasting) [B1999]_ 。

  * 如果样本抽取是放回的，我们称为Bagging [B1996]_ 。

  * 如果抽取的数据集的随机子集是特征的随机子集，我们叫做随机子空间(Random Subspaces) [H1998]_ 。

  * 最后，如果估计器构建在样本和特征的子集之上时，我们叫做随机补丁(Random Patches) [LG2012]_ 。



在sklearn中，bagging方法使用统一的 :class:`BaggingClassifier` 元估计器（或者 :class:`BaggingRegressor` ），输入的参数和策略由用户指定。``max_samples`` 和 ``max_features`` 控制着子集的大小， ``bootstrap`` 和 ``bootstrap_features`` 控制着样本和特征是放回抽样还是不放回抽样。
当使用样本子集时，通过设置 ``oob_score=True`` ，可以使用袋外(out-of-bag)样本来评估泛化精度。下面的代码片段说明了如何实构造一个 :class:`KNeighborsClassifier` 估计器的bagging集成，每一个基估计器都建立在50%的样本随机子集和特征随机子集上。

    >>> from sklearn.ensemble import BaggingClassifier
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> bagging = BaggingClassifier(KNeighborsClassifier(),
    ...                             max_samples=0.5, max_features=0.5)

.. topic:: 示例:

 * :ref:`sphx_glr_auto_examples_ensemble_plot_bias_variance.py`

.. topic:: 参考文献

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

由随机树组成的森林
===========================

:mod:`sklearn.ensemble` 模块包含两个基于 :ref:`随机决策树 <tree>` 的平均算法： RandomForest 算法和 Extra-Trees算法。
这两种算法都是专门为树而设计的扰动和组合技术（perturb-and-combine techniques） [B1998]_ 。
这意味着通过在分类器构造过程中引入随机性来创建一组不同的分类器。集成分类器的预测是单个分类器预测结果的平均值。 


与其他分类器一样，森林分类器必须拟合（fitted）两个数组：
保存训练样本的数组（可能稀疏或稠密）X，大小为 ``[n_samples, n_features]``。
保存训练样本目标值（类标签）的数组Y，大小为 ``[n_samples]``::

    >>> from sklearn.ensemble import RandomForestClassifier
    >>> X = [[0, 0], [1, 1]]
    >>> Y = [0, 1]
    >>> clf = RandomForestClassifier(n_estimators=10)
    >>> clf = clf.fit(X, Y)


同 :ref:`决策树 <tree>` 一样，随机森林算法（forests of trees）也能够通过扩展来解决 :ref:`多输出问题 <tree_multioutput>` (如果 Y 的大小是 ``[n_samples, n_outputs])``. 

随机森林
--------------

在随机森林中（参见 :class:`ExtraTreesClassifier` 和 :class:`ExtraTreesRegressor` 类），
集成模型中的每棵树构建时的样本都是由训练集经过替换得来的（例如，自助采样法-bootstrap sample，这里采用西瓜书中的译法）。
另外，在构建树的过程中进行结点分割时，选择的分割点不再是所有特征中最佳分割点，而是特征的随机子集中的最佳分割点。
由于这种随机性，森林的偏差通常会有略微的增大（相对于单个非随机树的偏差），但是由于平均，其方差也会减小，通常能够补偿偏差的增加，从而产生更好的模型。


与原始版本的实现 [B2001]_ 相反，scikit-learn的实现是把每个分类器的预测概率进行平均化，而不是让每个分类器对类别进行投票。 


极限随机树
--------------------------

在极限随机树中（参见 :class:`ExtraTreesClassifier` 和 :class:`ExtraTreesRegressor` 类)，
计算分割点方法中的随机性进一步增强。 
在随机森林中，使用的是候选特征的随机子集，而不是寻找最具有区分度的阈值，
这里的阈值是针对每个候选特征而随机生成的，并且会把这些随机生成的阈值中的最佳值作为分割规则。
这种做法通常能够更多地减少模型的方差，代价则是轻微地增大偏差：

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

参数
----------

使用这些方法时要调整的参数主要是 ``n_estimators`` 和 ``max_features``。
前者（n_estimators）是森林里的树木数量，越大越好，但是计算时间会增加。
此外要注意树的数量超过临界值之后算法的效果并不会很显著的变好。
后者（max_features）是分割节点时要考虑的特征的随机子集的大小。
方差减小得越多，偏差增大得越多。
根据经验回归问题最好使用默认值 max_features = n_features，
分类问题最好使用 ``max_features = sqrt（n_features``
（其中 ``n_features`` 是特征的个数）。
``max_depth = None 和 min_samples_split = 1`` 结合通常会有不错的效果。
请记住，这些（默认）值通常不是最佳的，同时还可能消耗大量的内存，最佳参数值应由交叉验证获得。
另外，请注意，在随机林中，默认使用自助采样法（``bootstrap = True``），
然而extra-trees的默认策略是使用整个数据集（``bootstrap = False``）。
当使用自助采样法方法抽样时，泛化精度是可以通过剩余的或者袋子外的样本来估算的，设置 ``oob_score = True`` 即可。 

.. topic:: 提示:

    默认参数下模型复杂度是：``O(M*N*log(N))``， 
    其中M是树的数目，``N`` 是样本数。 
    可以通过设置以下参数来降低模型复杂度：``min_samples_split``, ``min_samples_leaf``, ``max_leaf_nodes`` and ``max_depth``. 


并行化 
---------------

最后，这个模块还支持树的并行构建，可以通过 ``n_jobs`` 参数来规划并行计算。
如果 ``n_jobs = k``，则计算被划分为 ``k`` 个作业，并运行在机器的 ``k`` 个核上。 
如果 ``n_jobs = -1``，则可以使用机器的所有核。 
注意由于进程间通信具有开销，这里的提速并不是线性的（即，使用 ``k`` 个作业不会快k倍）。 
当然，在建立大量的树，或者构建单个树需要相当长的时间（例如，在大型数据集上）时，（通过并行化）仍然可以实现显著的加速。 

.. topic:: 示例:

 * :ref:`sphx_glr_auto_examples_ensemble_plot_forest_iris.py`
 * :ref:`sphx_glr_auto_examples_ensemble_plot_forest_importances_faces.py`
 * :ref:`sphx_glr_auto_examples_plot_multioutput_face_completion.py`

.. topic:: 参考文献

 .. [B2001] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.

 .. [B1998] L. Breiman, "Arcing Classifiers", Annals of Statistics 1998.

 * P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized
   trees", Machine Learning, 63(1), 3-42, 2006.

.. _random_forest_feature_importance:

特征重要性评估
-----------------------------

特征对目标变量预测的重要性可以通过（树中的决策节点的）特征使用的顺序（即深度）来进行评估。
决策树顶部使用的特征对最终预测结果的贡献度更大，因此，可以使用该特征对最后结果的贡献度来评估该 **特征相对重要性** 。 


通过 **平均** 多个随机树中的 **预期贡献率** （expected activity rates），可以减少这种估计的 **方差** ，并将其用于特征选择。 


下面的例子展示了在面部识别中用颜色编码表示每个像素的相对重要性，使用的模型是ExtraTreesClassifier。 

.. figure:: ../auto_examples/ensemble/images/sphx_glr_plot_forest_importances_faces_001.png
   :target: ../auto_examples/ensemble/plot_forest_importances_faces.html
   :align: center
   :scale: 75

实际上，在拟合模型时这些估计值存储在 ``feature_importances_`` 属性中。
这是一个大小为``(n_features,)``的数组，其值为正，并且总和为1.0。值越高，匹配特征对预测函数的贡献越大。 

.. topic:: Examples:

 * :ref:`sphx_glr_auto_examples_ensemble_plot_forest_importances_faces.py`
 * :ref:`sphx_glr_auto_examples_ensemble_plot_forest_importances.py`

.. _random_trees_embedding:

完全随机树嵌入
------------------------------

:class:`RandomTreesEmbedding` 实现了无监督的数据转换。
通过由完全随机树构成的森林，:class:`RandomTreesEmbedding` 使用数据尾部的叶子节点的索引对数据进行编码。
该索引以one-of-K方式编码，能够形成一个高维的稀疏二进制编码。 这种编码的方式非常高效，可以作为其他学习任务的基础。
编码的大小和稀疏度可以通过选择树的数量和每棵树的最大深度来影响。对于集成中的每棵树，编码包含一个实体。 
编码的大小最多为 ``n_estimators * 2 ** max_depth``，即森林中的叶子节点的最大数。 


由于相邻数据点更可能位于树的同一叶子中，此时该变换表现为隐式非参数密度估计。 


.. topic:: 示例:

 * :ref:`sphx_glr_auto_examples_ensemble_plot_random_forest_embedding.py`

 * :ref:`sphx_glr_auto_examples_manifold_plot_lle_digits.py` 比较手写体数字的非线性降维技术。

 * :ref:`sphx_glr_auto_examples_ensemble_plot_feature_transformation.py` 比较了基于特征变换的有监督和无监督的树.

.. seealso::

   :ref:`manifold` 也可以用于特征空间的非线性表示, 这些方法的关注点同样在降维.


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

用法
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

.. topic:: 参考

 .. [FS1995] Y. Freund, and R. Schapire, "A Decision-Theoretic Generalization of
             On-Line Learning and an Application to Boosting", 1997.

 .. [ZZRH2009] J. Zhu, H. Zou, S. Rosset, T. Hastie. "Multi-class AdaBoost",
               2009.

 .. [D1997] H. Drucker. "Improving Regressors using Boosting Techniques", 1997.

 .. [HTF] T. Hastie, R. Tibshirani and J. Friedman, "Elements of
              Statistical Learning Ed. 2", Springer, 2009.


.. _gradient_boosting:

梯度树提升（Gradient Tree Boosting）
====================================

`Gradient Tree Boosting <https://en.wikipedia.org/wiki/Gradient_boosting>`_
或梯度提升回归树(GBRT)是提升算法推广到任意可微的损失函数的泛化.GBRT是一个准确高效的现有程序,
它既能用于分类问题也可以用于回归问题.梯度树提升模型被应用到各种领域包括网页搜索排名和生态领域.

GBRT 的优点:

  + 对混合型数据的自然处理(异构特性)

  + 强大的预测能力

  + 在输出空间中对异常点的鲁棒性(通过鲁棒损失函数实现)

GBRT 的缺点:

  + 可扩展性差.由于提升算法的有序性(也就是说下一步的结果依赖于上一步),因此很难做并行.

模型 :mod:`sklearn.ensemble` 通过梯度提升树提供了分类和回归的方法.

分类
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

回归
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

拟合额外的弱学习器
--------------------------------

 :class:`GradientBoostingRegressor` 和 :class:`GradientBoostingClassifier`都支持设置参数
``warm_start=True``,这样设置允许我们在已经拟合的模型上面添加更多的estimators.

::

  >>> _ = est.set_params(n_estimators=200, warm_start=True)  # set warm_start and new nr of trees
  >>> _ = est.fit(X_train, y_train) # fit additional 100 trees to est
  >>> mean_squared_error(y_test, est.predict(X_test))    # doctest: +ELLIPSIS
  3.84...

.. _gradient_boosting_tree_size:

控制树的大小
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

数学公式
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

损失函数
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

正则化
----------------

.. _gradient_boosting_shrinkage:

缩减（Shrinkage）
.................

[F2001]_ 提出了一种简单的正则化策略,通过一个因素 :math:`\nu`来缩小每个弱学习器对于最终结果的贡献:

.. math::

    F_m(x) = F_{m-1}(x) + \nu \gamma_m h_m(x)

参数 :math:`\nu` 也叫作 **learning rate** 因为它缩小了梯度下降的步长,它可以通过参数 ``learning_rate`` 来设置.

在拟合一定数量的弱学习器时,参数 ``learning_rate`` 和参数 ``n_estimators`` 之间有很强的制约关系.
较小的``learning_rate`` 需要大量的弱学习器才能保证训练误差的不变.经验表明较小值的 ``learning_rate``
将会得到更好的测试误差. [HTF2009]_ 推荐把 ``learning_rate`` 设置为一个较小的常数
(例如: ``learning_rate <= 0.1`` )同时通过提前停止策略来选择合适的 ``n_estimators`` .
有关 ``learning_rate`` 和 ``n_estimators`` 更详细的讨论可以参考 [R2007]_.

子采样（Subsampling）
.....................

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

解释性
--------------

单棵决策树可通过简单的可视化树结构来解释.然而对于梯度提升模型来说,一般拥有几百棵树,因此它们不可能简单的通
过可视化单独的每一棵树来解释.幸运的是,有很多关于总结和解释梯度提升模型的技术被提出.

特征重要性
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

部分依赖
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

.. rubric:: 附录

.. [1] For classification with ``loss='deviance'``  the target
   response is logit(p).

.. [2] More precisely its the expectation of the target response after
   accounting for the initial model; partial dependence plots
   do not include the ``init`` model.

.. topic:: 示例:

 * :ref:`sphx_glr_auto_examples_ensemble_plot_partial_dependence.py`


.. topic:: 参考

 .. [F2001] J. Friedman, "Greedy Function Approximation: A Gradient Boosting Machine",
   The Annals of Statistics, Vol. 29, No. 5, 2001.

 .. [F1999] J. Friedman, "Stochastic Gradient Boosting", 1999

 .. [HTF2009] T. Hastie, R. Tibshirani and J. Friedman, "Elements of Statistical Learning Ed. 2", Springer, 2009.

 .. [R2007] G. Ridgeway, "Generalized Boosted Models: A guide to the gbm package", 2007


 .. _voting_classifier:

投票分类器
=============================

:class:`VotingClassifier` （投票分类器）的原理是结合了多个不同的机器学习分类器，并且采用 majority vote （多数表决）的方式或者 soft vote （平均预测概率）的方式来预测类标签。这样的分类器（指 Voting Classifier）可以用于一组 equally well performing model （同样出色的模型），以平衡它们各自的弱点。


多数类标签（也叫 Majority/Hard Voting）
--------------------------------------------

majority vote(采用多数投票)的时候，特定样本的预测类标签是每个分类器预测的类标签中占据多数的那个类标签。

举个例子, 如果给定一个样本进行预测

- 分类器 1 预测得到的结果是 类别 1
- 分类器 2 预测得到的结果是 类别 1
- 分类器 3 预测得到的结果是 类别 2

类别 1 占据多数，所以 VotingClassifier (投票分类器)使用 ``voting='hard'`` ，即 majority vote (多数表决)的方式，会得到该样本的预测结果是类别 1。

如果得到的票数最多的类标签不止一个，VotingClassifier(投票分类器)会按照类标签升序排序的结果，选择靠前的类标签。
举个例子，在下边的场景中:

- 分类器1 预测得到的结果是 类别2
- 分类器2 预测得到的结果是 类别1

这种情况下，该样本的预测结果会是类别1。

用法
.....

下边这个示例程序说明了如何去 fit (拟合)、去构建一个采用 majority vote (多数表决)方法的分类器::

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


加权平均概率（也叫 Soft Voting）
--------------------------------------------

与多数表决 (majority voting / hard voting) 的方法相反，采用加权平均概率的方法得到的是预测概率值总和最大的那一个类标签。

可以通过参数 ``weights`` 来给每个分类器分配一个特定的权重。
当提供参数 ``weights`` 时，每个分类器的预测类概率需要乘以分类器权重并平均化。
最后得到的类标签采用拥有最高平均概率的类标签。

用一个简单的例子来说明上述这个方法。假设现在有一个分类问题，可供选择的类标签有 3 个，我们有 3 个分类器，在这里我们给这三个分类器分配相同的权重：w1=1, w2=1, w3=1.

如下所示，针对一个特定的样本输入，来计算加权平均概率：

================  ==========    ==========      ==========
  分类器            类别 1         类别 2          类别 3
================  ==========    ==========      ==========
  分类器 1          w1 * 0.2      w1 * 0.5        w1 * 0.3
  分类器 2          w2 * 0.6      w2 * 0.3        w2 * 0.1
  分类器 3          w3 * 0.3      w3 * 0.4        w3 * 0.3
 加权平均的结果       0.37	   0.4             0.23
================  ==========    ==========      ==========

这里可以看出，预测的类标签是类别 2，因为它有 highest average probability (最大的平均概率)。

下边的示例程序说明了当加权平均概率(也叫 soft voting )是基于 linear SupportVector Machine (线性支持向量机)、Decision Tree(决策树)、K-nearest neighbor(K近邻)这三种分类器的时候，decision regions (决策域) 可能会变化:

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

Using the `VotingClassifier` with `GridSearch`（网格搜索下的投票分类器）
------------------------------------------------------------------------

为了调整每个 estimators 的 hyperparameters ，`VotingClassifier` 可以和 `GridSearch` 一起使用::

   >>> from sklearn.model_selection import GridSearchCV
   >>> clf1 = LogisticRegression(random_state=1)
   >>> clf2 = RandomForestClassifier(random_state=1)
   >>> clf3 = GaussianNB()
   >>> eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')

   >>> params = {'lr__C': [1.0, 100.0], 'rf__n_estimators': [20, 200],}

   >>> grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)
   >>> grid = grid.fit(iris.data, iris.target)

用法
.....

根据预测的类概率来预测类标签(在 VotingClassifier 中的 scikit-learn estimators 必须支持 ``predict_proba`` 函数方法)::

   >>> eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')

可选地，也可以为单个分类器提供权重::

   >>> eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft', weights=[2,5,1])
