
.. _multiclass:

====================================
多分类和多标签算法
====================================

.. currentmodule:: sklearn.multiclass

.. warning::
    All classifiers in scikit-learn do multiclass classification
    out-of-the-box. You don't need to use the :mod:`sklearn.multiclass` module
    unless you want to experiment with different multiclass strategies.

The :mod:`sklearn.multiclass` module implements *meta-estimators* to solve
``multiclass`` and ``multilabel`` classification problems
by decomposing such problems into binary classification problems. Multitarget
regression is also supported.

- **Multiclass classification** 意味着一个分类任务有多于两个类的类别。比如，分类一系列橘子、
苹果、或者梨的水果图片。多分类假设每一个样本都有一个并且仅仅一个标签：一个水果可能是苹果，也可能
是梨，但不能同时是两者。

- **Multilabel classification** 多标签分类 给每一个样本分配一系列标签。这可以被认为是预测不
相互排斥的数据点的属性，例如与文档类型相关的主题。一个文本可以属于许多类别，可以同时为政治、金融、
教育或者不属于任何这些类别。

- **Multioutput regression** 为每个样本分配一组目标值。这可以认为是为每一个样本预测多个属性，
比如说在一个确定的地点风的方向和大小。

- **Multioutput-multiclass classification** and **multi-task classification**
  意味着单个的训练器要解决多个联合的分类任务。这是只考虑二分类的 multi-label classification 
  任务的推广，  *输出的格式是一个二维数组或者一个稀疏矩阵。*

 每个输出变量的标签集合可以是各不相同的。比如说，一个水果样本可以将梨作为一个输出变量，这个输出变
 量在一个比如梨、苹果等的有限集合中取可能的值；输出蓝色或者黄色的第二个输出变量在一个有限的颜色集
 合绿色、红色、蓝色等取可能的值...

 这意味着任何处理 multi-output multiclass or multi-task classification 的分类器，在特殊的
 情况下支持 multi-label classification 任务。Multi-task classification 与具有不同模型公式
 的 multi-output classification 相似。详细情况请查阅相关的分类器的文档。

所有的 scikit-learn 分类器都能处理 multiclass classification 任务，
但是 :mod:`sklearn.multiclass` 提供的 meta-estimators 允许改变处理超多两类的方式，因为这会对分类器的性能产生影响
（无论是在泛化误差或者所需要的计算资源方面）

下面是按照 scikit-learn 策略分组的分类器的总结，如果你使用其中的一个，则不需要此类中的 meta-estimators，除非你想要定值多分类方式。

- **Inherently multiclass:**

  - :class:`sklearn.naive_bayes.BernoulliNB`
  - :class:`sklearn.tree.DecisionTreeClassifier`
  - :class:`sklearn.tree.ExtraTreeClassifier`
  - :class:`sklearn.ensemble.ExtraTreesClassifier`
  - :class:`sklearn.naive_bayes.GaussianNB`
  - :class:`sklearn.neighbors.KNeighborsClassifier`
  - :class:`sklearn.semi_supervised.LabelPropagation`
  - :class:`sklearn.semi_supervised.LabelSpreading`
  - :class:`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`
  - :class:`sklearn.svm.LinearSVC` (setting multi_class="crammer_singer")
  - :class:`sklearn.linear_model.LogisticRegression` (setting multi_class="multinomial")
  - :class:`sklearn.linear_model.LogisticRegressionCV` (setting multi_class="multinomial")
  - :class:`sklearn.neural_network.MLPClassifier`
  - :class:`sklearn.neighbors.NearestCentroid`
  - :class:`sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis`
  - :class:`sklearn.neighbors.RadiusNeighborsClassifier`
  - :class:`sklearn.ensemble.RandomForestClassifier`
  - :class:`sklearn.linear_model.RidgeClassifier`
  - :class:`sklearn.linear_model.RidgeClassifierCV`


- **Multiclass as One-Vs-One:**

  - :class:`sklearn.svm.NuSVC`
  - :class:`sklearn.svm.SVC`.
  - :class:`sklearn.gaussian_process.GaussianProcessClassifier` (setting multi_class = "one_vs_one")


- **Multiclass as One-Vs-All:**

  - :class:`sklearn.ensemble.GradientBoostingClassifier`
  - :class:`sklearn.gaussian_process.GaussianProcessClassifier` (setting multi_class = "one_vs_rest")
  - :class:`sklearn.svm.LinearSVC` (setting multi_class="ovr")
  - :class:`sklearn.linear_model.LogisticRegression` (setting multi_class="ovr")
  - :class:`sklearn.linear_model.LogisticRegressionCV` (setting multi_class="ovr")
  - :class:`sklearn.linear_model.SGDClassifier`
  - :class:`sklearn.linear_model.Perceptron`
  - :class:`sklearn.linear_model.PassiveAggressiveClassifier`


- **Support multilabel:**

  - :class:`sklearn.tree.DecisionTreeClassifier`
  - :class:`sklearn.tree.ExtraTreeClassifier`
  - :class:`sklearn.ensemble.ExtraTreesClassifier`
  - :class:`sklearn.neighbors.KNeighborsClassifier`
  - :class:`sklearn.neural_network.MLPClassifier`
  - :class:`sklearn.neighbors.RadiusNeighborsClassifier`
  - :class:`sklearn.ensemble.RandomForestClassifier`
  - :class:`sklearn.linear_model.RidgeClassifierCV`


- **Support multiclass-multioutput:**

  - :class:`sklearn.tree.DecisionTreeClassifier`
  - :class:`sklearn.tree.ExtraTreeClassifier`
  - :class:`sklearn.ensemble.ExtraTreesClassifier`
  - :class:`sklearn.neighbors.KNeighborsClassifier`
  - :class:`sklearn.neighbors.RadiusNeighborsClassifier`
  - :class:`sklearn.ensemble.RandomForestClassifier`


.. warning::

    At present, no metric in :mod:`sklearn.metrics`
    supports the multioutput-multiclass classification task.

多标签分类格式
================================

在 multilabel learning 中，二分类任务的合集表示为二进制数组：每一个样本是 shape 为 (n_samples, n_classes) 的二维数组中的一行二进制值，比如非0元素，1表示为对应标签的
子集。 一个数组 ``np.array([[1, 0, 0], [0, 1, 1], [0, 0, 0]])`` 表示第一个样本属于第 0 个标签，第二个样本属于第一个和第二个标签，第三个样本不属于任何标签。

Producing multilabel data as a list of sets of labels may be more intuitive.
The :class:`MultiLabelBinarizer <sklearn.preprocessing.MultiLabelBinarizer>`
transformer can be used to convert between a collection of collections of
labels and the indicator format.

  >>> from sklearn.preprocessing import MultiLabelBinarizer
  >>> y = [[2, 3, 4], [2], [0, 1, 3], [0, 1, 2, 3, 4], [0, 1, 2]]
  >>> MultiLabelBinarizer().fit_transform(y)
  array([[0, 0, 1, 1, 1],
         [0, 0, 1, 0, 0],
         [1, 1, 0, 1, 0],
         [1, 1, 1, 1, 1],
         [1, 1, 1, 0, 0]])

.. _ovr_classification:

One-Vs-The-Rest
===============

这个方法也被称为 **one-vs-all**, 在 :class:`OneVsRestClassifier` 模块中执行。 这个方法在于每一个类都将拟合出一个分类器。对于每一个分类器，该类将会和其他所有的类区别。除了她的计算效率之外 (只需要 `n_classes` 个分类器), 这种方法的优点是它具有可解释性。通过检查相关的分类器它可以获得知识。这是最常用的方法，也是一个公平的默认选择。

Multiclass learning
-------------------

下面是一个使用 OvR 的一个例子：

  >>> from sklearn import datasets
  >>> from sklearn.multiclass import OneVsRestClassifier
  >>> from sklearn.svm import LinearSVC
  >>> iris = datasets.load_iris()
  >>> X, y = iris.data, iris.target
  >>> OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)
  array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

多标签学习
-------------------

:class:`OneVsRestClassifier`  也支持 multilabel classification.
要使用该功能，给分类器提供一个指示矩阵，比如 [i,j] 表示样本 i 的标签为 j。


.. figure:: ../auto_examples/images/sphx_glr_plot_multilabel_001.png
    :target: ../auto_examples/plot_multilabel.html
    :align: center
    :scale: 75%


.. topic:: 示例:

    * :ref:`sphx_glr_auto_examples_plot_multilabel.py`

.. _ovo_classification:

One-Vs-One
==========

:class:`OneVsOneClassifier` 将会为每一对类别构造出一个分类器，在预测阶段，收到最多投票的类别将会被选择出来。在两个类具有同样的票数的时候， it
selects the class with the highest aggregate classification confidence by
summing over the pair-wise classification confidence levels computed by the
underlying binary classifiers.

因为这需要训练出 ``n_classes * (n_classes - 1) / 2`` 个分类器,
由于复杂度为 O(n_classes^2)，这个方法通常比 one-vs-the-rest 慢。然而，这个方法也有优点，比如说是在没有很好的缩放 ``n_samples`` 数据的核方法中。每个单独的学习问题只涉及一小部分数据，
而 one-vs-the-rest 将会使用 ``n_classes`` 个完整的数据。

多类别学习
-------------------

Below is an example of multiclass learning using OvO::

  >>> from sklearn import datasets
  >>> from sklearn.multiclass import OneVsOneClassifier
  >>> from sklearn.svm import LinearSVC
  >>> iris = datasets.load_iris()
  >>> X, y = iris.data, iris.target
  >>> OneVsOneClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)
  array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])


.. topic:: 参考文献:

    * "Pattern Recognition and Machine Learning. Springer",
      Christopher M. Bishop, page 183, (First Edition)

.. _ecoc:

误差校正输出代码
=============================

基于Output-code的方法不同于 one-vs-the-rest 和 one-vs-one。使用这些方法，每一个类将会被映射到欧几里得空间，每一个维度上的值为0或者为1。另一种解释它的方法是，每一个类被表示为二进制
码（一个 0 和 1 数组）。保存 location （位置）/ 每一个类的编码的矩阵被称为 code book。编码的大小是前面提到的欧几里得空间的纬度。直观上，每一个类应该使用一个唯一的编码，好的 code book 应该能够优化分类的精度。
在实现上，我们使用随机产生的 code book，正如在 [3]_ 提倡的方式，然而，更加详尽的方法会在未来加入进来。

在训练时，code book 每一位的二分类器将会被训练。在预测时，分类器将映射到类空间中选中的点的附近。
 
在 :class:`OutputCodeClassifier`, ``code_size`` 属性允许用户设置将会用到的分类器的数量。
它是类别总数的百分比。

在 0 或 1 之中的一个数字会比 one-vs-the-rest 使用更少的分类器。理论上 ``log2(n_classes) / n_classes`` 足以明确表示每个类。然而，在实际上，这也许会导致不太好的精确度，因为 ``log2(n_classes)`` 小于 n_classes.

比 1 大的数字比 one-vs-the-rest 需要更多的分类器数数量。在这种情况下，一些分类器理论上会纠正其他分类器的错误，因此命名为 "error-correcting" 。然而在实际上这通常不会发生，因为许多分类器的错误通常意义上来说是相关的。error-correcting output codes 和 bagging 有一个相似的作用效果。


多类别学习
-------------------

Below is an example of multiclass learning using Output-Codes::

  >>> from sklearn import datasets
  >>> from sklearn.multiclass import OutputCodeClassifier
  >>> from sklearn.svm import LinearSVC
  >>> iris = datasets.load_iris()
  >>> X, y = iris.data, iris.target
  >>> clf = OutputCodeClassifier(LinearSVC(random_state=0),
  ...                            code_size=2, random_state=0)
  >>> clf.fit(X, y).predict(X)
  array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1,
         1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

.. topic:: 参考文献:

    * "Solving multiclass learning problems via error-correcting output codes",
      Dietterich T., Bakiri G.,
      Journal of Artificial Intelligence Research 2,
      1995.

    .. [3] "The error coding method and PICTs",
        James G., Hastie T.,
        Journal of Computational and Graphical statistics 7,
        1998.

    * "The Elements of Statistical Learning",
      Hastie T., Tibshirani R., Friedman J., page 606 (second-edition)
      2008.

多输出回归
======================

Multioutput regression 支持 :class:`MultiOutputRegressor` 可以被添加到任何回归器中。这个策略包括对每个目标拟合一个回归。因为每一个目标可以被一个回归器精确的表示，通过检查其他回归器，它可以获取关于目标的知识。因为 :class:`MultiOutputRegressor` 对于每一个目标可以训练出一个回归器，所以它可能忽略属性之间的关系。

以下是 multioutput regression（多输出回归）的示例:

  >>> from sklearn.datasets import make_regression
  >>> from sklearn.multioutput import MultiOutputRegressor
  >>> from sklearn.ensemble import GradientBoostingRegressor
  >>> X, y = make_regression(n_samples=10, n_targets=3, random_state=1)
  >>> MultiOutputRegressor(GradientBoostingRegressor(random_state=0)).fit(X, y).predict(X)
  array([[-154.75474165, -147.03498585,  -50.03812219],
         [   7.12165031,    5.12914884,  -81.46081961],
         [-187.8948621 , -100.44373091,   13.88978285],
         [-141.62745778,   95.02891072, -191.48204257],
         [  97.03260883,  165.34867495,  139.52003279],
         [ 123.92529176,   21.25719016,   -7.84253   ],
         [-122.25193977,  -85.16443186, -107.12274212],
         [ -30.170388  ,  -94.80956739,   12.16979946],
         [ 140.72667194,  176.50941682,  -17.50447799],
         [ 149.37967282,  -81.15699552,   -5.72850319]])

多输出分类
==========================

Multioutput classification 支持能够被添加到任何分类器中的 :class:`MultiOutputClassifier`. 这种方法训练每一个目标一个分类器。这允许多目标变量分类器。这种类的目的是扩展能够评估一系列目标函数的评估器 (f1,f2,f3…,fn) ，这些函数在一个单独的预测矩阵上训练来预测一系列 (y1,y2,y3…,yn)。

Below is an example of multioutput classification:
    
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.multioutput import MultiOutputClassifier
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.utils import shuffle
    >>> import numpy as np
    >>> X, y1 = make_classification(n_samples=10, n_features=100, n_informative=30, n_classes=3, random_state=1)
    >>> y2 = shuffle(y1, random_state=1)
    >>> y3 = shuffle(y1, random_state=2)
    >>> Y = np.vstack((y1, y2, y3)).T
    >>> n_samples, n_features = X.shape # 10,100
    >>> n_outputs = Y.shape[1] # 3
    >>> n_classes = 3
    >>> forest = RandomForestClassifier(n_estimators=100, random_state=1)
    >>> multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
    >>> multi_target_forest.fit(X, Y).predict(X)
    array([[2, 2, 0],
           [1, 2, 1],
           [2, 1, 0],
           [0, 0, 2],
           [0, 2, 1],
           [0, 0, 2],
           [1, 1, 0],
           [1, 1, 1],
           [0, 0, 2],
           [2, 0, 0]])

链式分类器
================

Classifier chains (查看 :class:`ClassifierChain`) 是一种集合多个二分类器为一个单独的 multi-label 模型，能够发掘目标之间的相关性信息。

有 N 个类别的 multi-label 分类问题，将 N 个二分类器分配 0 到 N-1 之间的一个整数。这些整数定义了模型在 chain 中的顺序。 每一个分类器在可用的训练数据加上具有较低数字的模型的类的真正标签上训练。

当预测时，真正的标签将不可利用。每一个模型的预测将会传递个链上的下一个模型来作为特征使用。

很明显链的顺序是十分重要的。链上的第一个模型没有关于其他标签的有效的利用信息，而链上的最后一个模型将会具有所有其他的标签信息。在一般情况下，不知道链上模型最优的顺序，因此通常会使用许多随机的顺序，将他们的预测求平均。

.. topic:: 参考文献:

    Jesse Read, Bernhard Pfahringer, Geoff Holmes, Eibe Frank,
        "Classifier Chains for Multi-label Classification", 2009.
