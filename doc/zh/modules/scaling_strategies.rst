.. _scaling_strategies:

=================================================
大规模计算的策略: 更大量的数据
=================================================

对于一些应用程序的实例数量，它们需要被处理的特征（或两者）和/或速度对传统的方法而言非常具有挑战性。在这些情况下，scikit-learn 有许多你值得考虑的选项可以使你的系统规模化。

使用外核学习实例进行拓展
=================================================

外核（或者称作 “外部存储器”）学习是一种用于学习那些无法装进计算机主存储（RAM）的数据的技术。

这里描述了一种为了实现这一目的而设计的系统：

  1. 一种用流来传输实例的方式
  2. 一种从实例中提取特征的方法
  3. 增量式算法

流式实例
-------------------
基本上， 1. 可能是从硬盘、数据库、网络流等文件中产生实例的读取器。然而，关于如何实现的相关细节已经超出了本文档的讨论范围。

提取特征
-------------------
\2. 可以是 scikit-learn 支持的的不同 :ref: `特征提取 <feature_extraction>` 方法中的任何相关的方法。然而，当处理那些需要矢量化并且特征或值的集合你预先不知道的时候，就得明确注意了。一个好的例子是文本分类，其中在训练的期间你很可能会发现未知的项。从应用的角度上来看，如果在数据上进行多次通过是合理的，则可以使用有状态的向量化器。否则，可以通过使用无状态特征提取器来提高难度。目前，这样做的首选方法是使用所谓的 :ref:`哈希技巧<feature_hashing>`，在 :class:`sklearn.feature_extraction.FeatureHasher` 中，其中有分类变量的表示为 Python 列表或 :class:`sklearn.feature_extraction.text.HashingVectorizer` 文本文档。

增量学习
--------------------
最后，对于3. 我们在 scikit-learn 之中有许多选择。虽软不是所有的算法都能够增量学习（即不能一次性看到所有的实例），所有实 ``partial_fit`` 的 API 估计器都作为了候选。实际上，从小批量的实例（有时称为“在线学习”）逐渐学习的能力是外核学习的关键，因为它保证在任何给定的时间内只有少量的实例在主存储中，选择适合小批量的尺寸来平衡相关性和内存占用可能涉及一些调整 [1]_。

以下是针对不同任务的增量估算器列表：

  - Classification（分类）
      + :class:`sklearn.naive_bayes.MultinomialNB`
      + :class:`sklearn.naive_bayes.BernoulliNB`
      + :class:`sklearn.linear_model.Perceptron`
      + :class:`sklearn.linear_model.SGDClassifier`
      + :class:`sklearn.linear_model.PassiveAggressiveClassifier`
      + :class:`sklearn.neural_network.MLPClassifier`
  - Regression（回归）
      + :class:`sklearn.linear_model.SGDRegressor`
      + :class:`sklearn.linear_model.PassiveAggressiveRegressor`
      + :class:`sklearn.neural_network.MLPRegressor`
  - Clustering（聚类）
      + :class:`sklearn.cluster.MiniBatchKMeans`
      + :class:`sklearn.cluster.Birch`
  - Decomposition / feature Extraction（分解/特征提取）
      + :class:`sklearn.decomposition.MiniBatchDictionaryLearning`
      + :class:`sklearn.decomposition.IncrementalPCA`
      + :class:`sklearn.decomposition.LatentDirichletAllocation`
  - Preprocessing（预处理）
      + :class:`sklearn.preprocessing.StandardScaler`
      + :class:`sklearn.preprocessing.MinMaxScaler`
      + :class:`sklearn.preprocessing.MaxAbsScaler`

对于分类，有一点要注意的是，虽然无状态特征提取程序可能能够应对新的/不可见的属性，但增量学习者本身可能无法应对新的/不可见的目标类。在这种情况下，你必须使用 ``classes=`` 参数将所有可能的类传递给第一个 ``partial_fit`` 调用。

选择合适的算法时要考虑的另一个方面是，所有这些算法在每个示例中都不会对时间保持一致。比如说， ``Perceptron`` 仍然对错误标签的例子是敏感的，即使经过多次的例子，而 ``SGD*`` 和 ``PassiveAggressive*`` 族对这些鲁棒性更好。相反，在学习速率随着时间不断降低时，合适标记的例子在流中迟来了也变得越来越不重要了，并不会有显著的区别。

示例
----------
最后，我们有一个完整的 :ref:`sphx_glr_auto_examples_applications_plot_out_of_core_classification.py` 文本文档的核心分类的示例。旨在为想要构建核心学习系统的人们提供一个起点，并展示上述大多数概念。

此外，它还展现了不同算法性能随着处理例子的数量的演变。

.. |accuracy_over_time| image::  ../auto_examples/applications/images/sphx_glr_plot_out_of_core_classification_001.png
    :target: ../auto_examples/applications/plot_out_of_core_classification.html
    :scale: 80

.. centered:: |accuracy_over_time|

现在我们来看不同部分的计算时间，我们看到矢量化的过程比学习本身耗时还多。对于不同的算法，MultinomialNB 是耗时最多的，但通过增加其 mini-batches 的大小可以减轻开销。（练习：minibatch_size 在程序中更改为100和10000，并进行比较）。

.. |computation_time| image::  ../auto_examples/applications/images/sphx_glr_plot_out_of_core_classification_003.png
    :target: ../auto_examples/applications/plot_out_of_core_classification.html
    :scale: 80

.. centered:: |computation_time|


注释
-------

.. [1] 根据算法，mini-batch 大小可以影响结果。SGD*，PassiveAggressive* 和离散的 NaiveBayes 是真正在线的，不受 batch 大小的影响。相反，MiniBatchKMeans 收敛速度受 batch 大小影响。此外，其内存占用可能会随 batch 大小而显着变化。
