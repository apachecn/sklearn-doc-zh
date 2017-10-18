.. _outlier_detection:

======================
新奇和异常检测
======================

.. currentmodule:: sklearn

许多应用需要能够判断新观测是否属于与现有观测相同的分布（它是一个非线性的），或者应该被认为是不同的（这是一个异常值）。 通常，这种能力用于清理实际的数据集。 必须做出两个重要的区别：

:新奇检测:
  训练数据不被异常值污染，我们有兴趣检测新观察中的异常情况。

:异常值检测:
  训练数据包含异常值，我们需要适应训练数据的中心模式，忽略偏差观测值。

scikit-learn项目提供了一套可用于新颖性或异常值检测的机器学习工具。 该策略是通过数据从无监督的方式学习的对象来实现的::

    estimator.fit(X_train)

然后可以使用 `predict` 方法将新观察值作为内在值或异常值排序::

    estimator.predict(X_test)

正常被标记为1，而异常值被标记为-1。

新奇检测
=================

从 :math:`p` 个特征描述的相同分布考虑 :math:`n` 个观察值的数据集。 现在考虑，我们再添加一个观察数据集。 新的观察与其他观察是不同的，我们可以怀疑它是正常的吗？ （即是否来自相同的分配？）或者相反，是否与另一个相似，我们无法将其与原始观察结果区分开来？ 这是新奇检测工具和方法所解决的问题。

一般来说，它将要学习一个粗略且紧密的边界，界定初始观测分布的轮廓，绘制在嵌入的 :math:`p` 维空间中。 那么，如果进一步的观察在边界划分的子空间内，则它们被认为来自与初始观察相同的群体。 否则，如果他们在边界之外，我们可以说他们是异常的，对我们的评估有一定的信心。

One-Class SVM 由Schölkopf等人介绍。 为此目的并在 :ref:`svm` 模块的 :class:`svm.OneClassSVM` 对象中实现。
需要选择kernel和scalar参数来定义边界。 通常选择RBF内核，尽管没有确切的公式或算法来设置其带宽参数。 这是scikit-learn实现中的默认值。 :math:`\nu` 参数，也称为一级SVM的边距，对应于在边界之外找到新的但常规的观察的概率。

.. topic:: 参考文献:

    * `Estimating the support of a high-dimensional distribution
      <http://dl.acm.org/citation.cfm?id=1119749>`_ Schölkopf,
      Bernhard, et al. Neural computation 13.7 (2001): 1443-1471.

.. topic:: 例子:

   * 参见 :ref:`sphx_glr_auto_examples_svm_plot_oneclass.py` 用于通过 :class:`svm.OneClassSVM` 对象可视化围绕某些数据学习的边界。

.. figure:: ../auto_examples/svm/images/sphx_glr_plot_oneclass_001.png
   :target: ../auto_examples/svm/plot_oneclass.html
   :align: center
   :scale: 75%


异常检测
=================

异常值检测类似于新奇检测，其目的是将正常观察的核心与一些被称为“异常值”的污染物进行分离。 然而，在异常值检测的情况下，我们没有一个干净的数据集代表可用于训练任何工具的常规观察值的群体。


Fitting an elliptic envelope
------------------------------

执行异常值检测的一种常见方式是假设常规数据来自已知分布（例如，数据是高斯分布的）。 从这个假设来看，我们通常试图定义数据的“形状”，并且可以将离散观察值定义为足够远离拟合形状的观测值。

scikit-learn提供了一个对象
:class:`covariance.EllipticEnvelope` ，它适合于对数据的鲁棒协方差估计，从而将椭圆适配到中央数据点，忽略中央模式之外的点。

例如，假设异构数据是高斯分布的，它将以鲁棒的方式（即不受异常值的影响）来估计非线性位置和协方差。 从该估计得到的马氏距离距离用于得出偏离度量。 这个策略如下图所示。

.. figure:: ../auto_examples/covariance/images/sphx_glr_plot_mahalanobis_distances_001.png
   :target: ../auto_examples/covariance/plot_mahalanobis_distances.html
   :align: center
   :scale: 75%

.. topic:: 例子:

   * 参见 :ref:`sphx_glr_auto_examples_covariance_plot_mahalanobis_distances.py` 说明使用标准
     (:class:`covariance.EmpiricalCovariance`) 或稳健估计
     (:class:`covariance.MinCovDet`) 的位置和协方差来评估观察的偏离程度的差异。

.. topic:: 参考文献:

    * Rousseeuw, P.J., Van Driessen, K. "A fast algorithm for the minimum
      covariance determinant estimator" Technometrics 41(3), 212 (1999)

.. _isolation_forest:

Isolation Forest
----------------------------

在高维数据集中执行异常值检测的一种有效方法是使用随机森林。
 :class:`ensemble.IsolationForest` 通过随机选择特征然后随机选择所选特征的最大值和最小值之间的分割值来隔离观察值。

由于递归分区可以由树结构表示，因此隔离样本所需的分裂次数等同于从根节点到终止节点的路径长度。

在这样的随机树的森林中平均的这个路径长度是正态性和我们的决策功能的量度。

随机分区产生明显较短的异常路径。 因此，当一个随机树林共同为特定样本产生较短的路径长度时，它们很有可能是异常的。

这个策略如下图所示。

.. figure:: ../auto_examples/ensemble/images/sphx_glr_plot_isolation_forest_001.png
   :target: ../auto_examples/ensemble/plot_isolation_forest.html
   :align: center
   :scale: 75%

.. topic:: 例子:

   * 参见 :ref:`sphx_glr_auto_examples_ensemble_plot_isolation_forest.py`  说明IsolationForest的用法。

   * 参见 :ref:`sphx_glr_auto_examples_covariance_plot_outlier_detection.py` 将 :class:`ensemble.IsolationForest` 与
     :class:`neighbors.LocalOutlierFactor`,
     :class:`svm.OneClassSVM` (调整为执行类似异常值检测方法）和基于协方差的异常值检测与协方差的
     :class:`covariance.EllipticEnvelope`.

.. topic:: 参考文献:

    * Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. "Isolation forest."
      Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on.


局部异常因子
--------------------
对中等高维数据集执行异常值检测的另一种有效方法是使用局部离群因子（LOF）算法。

 :class:`neighbors.LocalOutlierFactor` （LOF）算法计算反映观测值异常程度的分数（称为局部离群因子）。 它测量给定数据点相对于其邻居的局部密度偏差。 这个想法是检测具有比其邻居明显更低密度的样品。

实际上，从k个最近的邻居获得局部密度。 观察的LOF得分等于他的k-最近邻居的平均局部密度与其本身密度的比值：正常情况预期具有与其邻居类似的局部密度，而异常数据 预计本地密度要小得多。

考虑的邻居数（k个别名参数n_neighbors）通常选择1）大于集群必须包含的对象的最小数量，以便其他对象可以是相对于该集群的本地异常值，并且2）小于最大值 靠近可能是本地异常值的对象的数量。 在实践中，这样的信息通常不可用，并且n_neighbors = 20似乎总体上很好地工作。 当异常值的比例高（即大于10％时，如下面的例子），n邻居应该更大（在下面的例子中，n_neighbors = 35）。

LOF算法的优点是考虑到数据集的局部和全局属性：即使在异常样本具有不同基础密度的数据集中，它也能够很好地执行。 问题不在于，样本是如何孤立的，而是与周边邻里有多孤立。

这个策略如下图所示。

.. figure:: ../auto_examples/neighbors/images/sphx_glr_plot_lof_001.png
   :target: ../auto_examples/neighbors/plot_lof.html
   :align: center
   :scale: 75%

.. topic:: 例子:

   * 参见 :ref:`sphx_glr_auto_examples_neighbors_plot_lof.py`，  :class:`neighbors.LocalOutlierFactor` 使用说明。

   * 参见 :ref:`sphx_glr_auto_examples_covariance_plot_outlier_detection.py`， 与其他异常检测方法进行比较。

.. topic:: 参考文献:

   *  Breunig, Kriegel, Ng, and Sander (2000)
      `LOF: identifying density-based local outliers.
      <http://www.dbs.ifi.lmu.de/Publikationen/Papers/LOF.pdf>`_
      Proc. ACM SIGMOD

One-class SVM 与 Elliptic Envelope 与 Isolation Forest 与 LOF
-------------------------------------------------------------------------

严格来说，One-class SVM 不是异常检测方法，而是一种新颖性检测方法：其训练集不应该被异常值污染，因为它可能适合它们。 也就是说，高维度的异常值检测或对内容数据的分布没有任何假设是非常具有挑战性的， One-class SVM 在这些情况下给出了有用的结果。

下面的例子说明了当数据越来越少的单峰时，
:class:`covariance.EllipticEnvelope` 如何降低。
:class:`svm.OneClassSVM` 在具有多种模式和 :class:`ensemble.IsolationForest` 和
:class:`neighbors.LocalOutlierFactor` 的数据在每种情况下表现良好。

.. |outlier1| image:: ../auto_examples/covariance/images/sphx_glr_plot_outlier_detection_001.png
   :target: ../auto_examples/covariance/plot_outlier_detection.html
   :scale: 50%

.. |outlier2| image:: ../auto_examples/covariance/images/sphx_glr_plot_outlier_detection_002.png
   :target: ../auto_examples/covariance/plot_outlier_detection.html
   :scale: 50%

.. |outlier3| image:: ../auto_examples/covariance/images/sphx_glr_plot_outlier_detection_003.png
   :target: ../auto_examples/covariance/plot_outlier_detection.html
   :scale: 50%

.. list-table:: **Comparing One-class SVM, Isolation Forest, LOF, and Elliptic Envelope**
   :widths: 40 60

   *
      - 对于以和well-centered的非线性模式，
        :class:`svm.OneClassSVM` 不能受益于inlier群体的旋转对称性。 此外，它适合训练集中存在的异常值。 相反，基于拟合协方差的
        :class:`covariance.EllipticEnvelope` 学习一个椭圆，这适合于inlier分布。
        :class:`ensemble.IsolationForest`
        和 :class:`neighbors.LocalOutlierFactor` 表现也好。
      - |outlier1| 

   *
      - 由于inlier分布变为双峰，所以
        :class:`covariance.EllipticEnvelope` 不适合内部值。 但是，我们可以看到
        :class:`ensemble.IsolationForest`,
        :class:`svm.OneClassSVM` 和 :class:`neighbors.LocalOutlierFactor`
        在检测这两种模式时遇到困难，而且 :class:`svm.OneClassSVM` 往往会过度复杂：因为它没有 内在模型，它解释了一些区域，偶尔有一些异常值聚集在一起，作为内在的。
      - |outlier2|

   *
      - 如果inlier分布非常高斯，则
        :class:`svm.OneClassSVM`
        :class:`ensemble.IsolationForest`
        和 :class:`neighbors.LocalOutlierFactor` 一样
        能够恢复合理的近似,
        而 :class:`covariance.EllipticEnvelope` 完全失败。
      - |outlier3|

.. topic:: 例子:

   * 参见 :ref:`sphx_glr_auto_examples_covariance_plot_outlier_detection.py` ，
     :class:`svm.OneClassSVM` （调整为执行异常检测方法）, 
     :class:`ensemble.IsolationForest`, :class:`neighbors.LocalOutlierFactor`
     和基于协方差的异常值检测协方差的 :class:`covariance.EllipticEnvelope`.
