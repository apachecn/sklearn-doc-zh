
sklearn.model_selection

.. _grid_search:

===========================================
调整估计器的超参数
===========================================

超参数，即不直接在估计器内学习的参数。在 scikit-learn 包中，它们作为估计器类中构造函数的参数进行传递。典型的例子有：用于支持向量分类器的 ``C`` 、``kernel`` 和 ``gamma`` ，用于Lasso的 ``alpha`` 等。

搜索超参数空间以便获得最好 `交叉验证 <http://sklearn.apachecn.org/doc/cn/0.19.0/modules/cross_validation.html#cross-validation>`_ 分数的方法是可能的而且是值得提倡的。

通过这种方式，构造估计器时被提供的任何参数或许都能被优化。具体来说，要获取到给定估计器的所有参数的名称和当前值，使用::

  estimator.get_params()

搜索包括:

- 估计器(回归器或分类器，例如 ``sklearn.svm.SVC()``)
- 参数空间
- 搜寻或采样候选的方法
- 交叉验证方案
- :ref:`计分函数 <gridsearch_scoring>`

有些模型支持专业化的、高效的参数搜索策略, :ref:`描述如下 <alternative_cv>` 。在 scikit-learn 包中提供了两种采样搜索候选的通用方法:对于给定的值, `GridSearchCV <http://sklearn.apachecn.org/doc/cn/0.19.0/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV>`_ 考虑了所有参数组合；而 `RandomizedSearchCV <http://sklearn.apachecn.org/doc/cn/0.19.0/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV>`_ 可以从具有指定分布的参数空间中抽取给定数量的候选。介绍完这些工具后，我们将详细介绍适用于这两种方法的 :ref:`最佳实践 <grid_search_tips>` 。

**注意**，通常这些参数的一小部分会对模型的预测或计算性能有很大的影响，而其他参数可以保留为其默认值。
建议阅读估计器类的相关文档，以更好地了解其预期行为，可能的话还可以阅读下引用的文献。


网格追踪法--穷尽的网格搜索
====================================

:class:`GridSearchCV` 提供的网格搜索从通过 ``param_grid`` 参数确定的网格参数值中全面生成候选。例如，下面的 ``param_grid``::

  param_grid = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
   ]

探索两个网格的详细解释：
一个具有线性内核并且C在[1,10,100,1000]中取值；
另一个具有RBF内核，C值的交叉乘积范围在[1,10，100,1000]，gamma在[0.001，0.0001]中取值。

:class:`GridSearchCV` 实例实现了常用估计器 API：当在数据集上“拟合”时，参数值的所有可能的组合都会被评估，从而计算出最佳的组合。

.. topic:: 示例:

    - 有关在数字数据集上的网格搜索计算示例，请参阅  `基于交叉验证的网格搜索参数估计 <http://sklearn.apachecn.org/doc/cn/0.19.0/auto_examples/model_selection/plot_grid_search_digits.html#sphx-glr-auto-examples-model-selection-plot-grid-search-digits-py>`_。

    - 有关来自文本文档特征提取器（n-gram计数向量化器和TF-IDF变换器）的网格搜索耦合参数与分类器（这里是使用具有弹性网格的SGD训练的线性SVM 或L2惩罚）使用 `pipeline.Pipeline` 示例,请参阅  `用于文本特征提取和评估的示例管道 <http://sklearn.apachecn.org/doc/cn/0.19.0/auto_examples/model_selection/grid_search_text_feature_extraction.html#sphx-glr-auto-examples-model-selection-grid-search-text-feature-extraction-py>`_。

    - 有关iris数据集的交叉验证循环中的网格搜索示例, 请参阅  `嵌套与非嵌套交叉验证 <http://sklearn.apachecn.org/doc/cn/0.19.0/auto_examples/model_selection/plot_nested_cross_validation_iris.html#sphx-glr-auto-examples-model-selection-plot-nested-cross-validation-iris-py>`_。

    - 有关用于同时评估多个指标的GridSearchCV示例，请参阅  `cross_val_score 与 GridSearchCV 多指标评价的实证研究 <http://sklearn.apachecn.org/doc/cn/0.19.0/auto_examples/model_selection/plot_multi_metric_evaluation.html#sphx-glr-auto-examples-model-selection-plot-multi-metric-evaluation-py>`_。

.. _randomized_parameter_search:

随机参数优化
=================================
尽管使用参数设置的网格法是目前最广泛使用的参数优化方法, 其他搜索方法也具有更有利的性能。 `RandomizedSearchCV <http://sklearn.apachecn.org/doc/cn/0.19.0/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV>`_ 实现了对参数的随机搜索, 其中每个设置都是从可能的参数值的分布中进行取样。
这对于穷举搜索有两个主要优势:

* 可以选择独立于参数个数和可能值的预算
* 添加不影响性能的参数不会降低效率


指定如何取样的参数是使用字典完成的, 非常类似于为 :class:`GridSearchCV` 指定参数。
此外, 通过 ``n_iter`` 参数指定计算预算, 即取样候选项数或取样迭代次数。
对于每个参数, 可以指定在可能值上的分布或离散选择的列表 (均匀取样)::

  {'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1),
    'kernel': ['rbf'], 'class_weight':['balanced', None]}

本示例使用 ``scipy.stats`` 模块, 它包含许多用于采样参数的有用分布, 如 ``expon``，``gamma``，``uniform`` 或者 ``randint``。
原则上, 任何函数都可以通过提供一个 ``rvs`` （随机变量样本）方法来采样一个值。
对 ``rvs`` 函数的调用应在连续调用中提供来自可能参数值的独立随机样本。

    .. warning::

        The distributions in ``scipy.stats`` prior to version scipy 0.16
        do not allow specifying a random state. Instead, they use the global
        numpy random state, that can be seeded via ``np.random.seed`` or set
        using ``np.random.set_state``. However, beginning scikit-learn 0.18,
        the `sklearn.model_selection <http://sklearn.apachecn.org/doc/cn/0.19.0/modules/classes.html#module-sklearn.model_selection>`_ module sets the random state provided
        by the user if scipy >= 0.16 is also available.

对于连续参数 (如上面提到的 ``C`` )，指定连续分布以充分利用随机化是很重要的。这样，有助于 ``n_iter`` 总是趋向于更精细的搜索。

.. topic:: 示例:

    * 随机搜索和网格搜索的使用和效率的比较： `有关随机搜索和网格搜索超参数估计的对比 <http://sklearn.apachecn.org/doc/cn/0.19.0/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py>`_

.. topic:: 引用:

    * Bergstra, J. and Bengio, Y.,
      Random search for hyper-parameter optimization,
      The Journal of Machine Learning Research (2012)

.. _grid_search_tips:

参数搜索技巧
=========================

.. _gridsearch_scoring:

指定目标度量
------------------------------

默认情况下, 参数搜索使用估计器的评分函数来评估（衡量）参数设置。
比如 `sklearn.metrics.accuracy_score <http://sklearn.apachecn.org/doc/cn/0.19.0/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score>`_ 用于分类和 `sklearn.metrics.r2_score <http://sklearn.apachecn.org/doc/cn/0.19.0/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score>`_ 用于回归。
对于一些应用, 其他评分函数将会更加适合 (例如在不平衡的分类, 精度评分往往是信息不足的)。
一个可选的评分功能可以通过评分参数指定给 :class:`GridSearchCV`， :class:`RandomizedSearchCV` 和许多下文将要描述的、专业化的交叉验证工具。
有关详细信息, 请参阅 `评分参数:定义模型评估规则 <http://sklearn.apachecn.org/doc/cn/0.19.0/modules/model_evaluation.html#scoring-parameter>`_。

.. _multimetric_grid_search:

为评估指定多个指标
------------------------------------------

``GridSearchCV`` 和 ``RandomizedSearchCV`` 允许为评分参数指定多个指标。

多指标评分可以被指定为一个预先定义分数名称字符串列表或者是一个得分手名字到得分手的函数或预先定义的记分员名字的映射字典。
有关详细信息, 请参阅 `多指标评估 <http://sklearn.apachecn.org/doc/cn/0.19.0/modules/model_evaluation.html#multimetric-scoring>`_。

在指定多个指标时,必须将 ``refit`` 参数设置为要在其中找到 ``best_params_``,并用于在整个数据集上构建 ``best_estimator_`` 的度量标准（字符串）。
如果搜索不应该 refit, 则设置 ``refit=False``。在使用多个度量值时,如果将 refit 保留为默认值,不会导致结果错误。

有关示例用法, 请参见 `cross_val_score 与 GridSearchCV 多指标评价的实证研究 <http://sklearn.apachecn.org/doc/cn/0.19.0/auto_examples/model_selection/plot_multi_metric_evaluation.html#sphx-glr-auto-examples-model-selection-plot-multi-metric-evaluation-py>`_。

复合估计和参数空间
-----------------------------------------

`管道：链式评估器 <http://sklearn.apachecn.org/doc/cn/0.19.0/modules/pipeline.html#pipeline>`_ 描述了如何使用这些工具搜索参数空间构建链式评估器。

模型选择：开发和评估
-------------------------------------------

通过评估各种参数设置，可以将模型选择视为使用标记数据训练网格参数的一种方法。

在评估结果模型时, 重要的是在网格搜索过程中未看到的 held-out 样本数据上执行以下操作: 
建议将数据拆分为开发集 (**development set**,供 ``GridSearchCV`` 实例使用)和评估集(**evaluation set**)来计算性能指标。

这可以通过使用效用函数 `train_test_split <http://sklearn.apachecn.org/doc/cn/0.19.0/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split>`_ 来完成。


并行机制
-----------------

:class:`GridSearchCV` 和 :class:`RandomizedSearchCV` 可以独立地评估每个参数设置。如果您的OS支持,通过使用关键字 ``n_jobs=-1`` 可以使计算并行运行。
有关详细信息, 请参见函数签名。



对故障的鲁棒性
---------------------

某些参数设置可能导致无法 ``fit`` 数据的一个或多个折叠。
默认情况下, 这将导致整个搜索失败, 即使某些参数设置可以完全计算。
设置 ``error_score=0`` (或`=np.NaN`) 将使程序对此类故障具有鲁棒性,发出警告并将该折叠的分数设置为0(或`NaN`), 但可以完成搜索。

.. _alternative_cv:

暴力参数搜索的替代方案
============================================

模型特定交叉验证
-------------------------------


某些模型可以与参数的单个值的估计值一样有效地适应某一参数范围内的数据。
此功能可用于执行更有效的交叉验证, 用于此参数的模型选择。

该策略最常用的参数是编码正则化矩阵强度的参数。在这种情况下, 我们称之为, 计算估计器的正则化路径(**regularization path**)。

以下是这些模型的列表:

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   linear_model.ElasticNetCV
   linear_model.LarsCV
   linear_model.LassoCV
   linear_model.LassoLarsCV
   linear_model.LogisticRegressionCV
   linear_model.MultiTaskElasticNetCV
   linear_model.MultiTaskLassoCV
   linear_model.OrthogonalMatchingPursuitCV
   linear_model.RidgeCV
   linear_model.RidgeClassifierCV


信息标准
---------------------

一些模型通过计算一个正则化路径 (代替使用交叉验证得出数个参数), 可以给出正则化参数最优估计的信息理论闭包公式。

以下是从 Akaike 信息标准 (AIC) 或贝叶斯信息标准 (可用于自动选择模型) 中受益的模型列表:

.. autosummary::
   :toctree: generated/
   :template: class.rst

   linear_model.LassoLarsIC


.. _out_of_bag:

出袋估计
--------------------

当使用基于装袋的集合方法时，即使用具有替换的采样产生新的训练集，部分训练集保持不用。 对于集合中的每个分类器，训练集的不同部分被忽略。

这个省略的部分可以用来估计泛化误差，而不必依靠单独的验证集。 此估计是"免费的"，因为不需要额外的数据，可以用于模型选择。

目前该方法已经实现的类以下几个:

.. autosummary::
   :toctree: generated/
   :template: class.rst

    ensemble.RandomForestClassifier
    ensemble.RandomForestRegressor
    ensemble.ExtraTreesClassifier
    ensemble.ExtraTreesRegressor
    ensemble.GradientBoostingClassifier
    ensemble.GradientBoostingRegressor
