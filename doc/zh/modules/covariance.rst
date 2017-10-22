.. _covariance:

===================================================
Covariance estimation
协方差估计
===================================================

.. currentmodule:: sklearn.covariance


Many statistical problems require at some point the estimation of a
population's covariance matrix, which can be seen as an estimation of
data set scatter plot shape. Most of the time, such an estimation has
to be done on a sample whose properties (size, structure, homogeneity)
has a large influence on the estimation's quality. The
`sklearn.covariance` package aims at providing tools affording
an accurate estimation of a population's covariance matrix under
various settings.
许多统计问题需要在某个时候估计总体的协方差矩阵，这可以被看作是数据集散布情况的估计。
大多数情况下，基于样本的估计（基于其属性，如尺寸，结构，均匀性），
对估计质量有很大影响。`sklearn.covariance` 方法的目的是
提供一个能在各种设置下准确估计总体协方差矩阵的工具。

We assume that the observations are independent and identically
distributed (i.i.d.).
我们假设观察是独立的，相同分布的 (i.i.d.)。


Empirical covariance
经验协方差 （Empirical covariance）
====================

The covariance matrix of a data set is known to be well approximated
with the classical *maximum likelihood estimator* (or "empirical
covariance"), provided the number of observations is large enough
compared to the number of features (the variables describing the
observations). More precisely, the Maximum Likelihood Estimator of a
sample is an unbiased estimator of the corresponding population
covariance matrix.
已知数据集的协方差矩阵与经典最大似然估计（或“经验协方差”）
很好地近似，条件是与特征数量（描述观测值的变量）相比，观测数量足够大。
更准确地说，样本的最大似然估计是相应的总体协方差矩阵的无偏估计。

The empirical covariance matrix of a sample can be computed using the
:func:`empirical_covariance` function of the package, or by fitting an
:class:`EmpiricalCovariance` object to the data sample with the
:meth:`EmpiricalCovariance.fit` method.  Be careful that depending
whether the data are centered or not, the result will be different, so
one may want to use the ``assume_centered`` parameter accurately. More precisely
if one uses ``assume_centered=False``, then the test set is supposed to have the
same mean vector as the training set. If not so, both should be centered by the
user, and ``assume_centered=True`` should be used.
样本的经验协方差矩阵可以使用 :func:`empirical_covariance` 包的函数计算 ，
或者通过 :class:`EmpiricalCovariance` 使用 :meth:`EmpiricalCovariance.fit`
方法将对象与数据样本拟合 。
要注意，取决于数据是否居中，结果会有所不同，所以可能需要准确使用参数 ``assume_centered``。
如果使用 ``assume_centered=False`` ，则结果更准确。且测试集应该具有与训练集相同的均值向量。
如果不是这样，两者都应该使用中心值， ``assume_centered=True`` 应该使用。

.. topic:: Examples例子:

   * See :ref:`sphx_glr_auto_examples_covariance_plot_covariance_estimation.py` for
     an example on how to fit an :class:`EmpiricalCovariance` object
     to data.


.. _shrunk_covariance:

Shrunk Covariance
收敛协方差（Shrunk Covariance）
=================

Basic shrinkage
基本收敛（Basic shrinkage）
---------------

Despite being an unbiased estimator of the covariance matrix, the
Maximum Likelihood Estimator is not a good estimator of the
eigenvalues of the covariance matrix, so the precision matrix obtained
from its inversion is not accurate. Sometimes, it even occurs that the
empirical covariance matrix cannot be inverted for numerical
reasons. To avoid such an inversion problem, a transformation of the
empirical covariance matrix has been introduced: the ``shrinkage``.
尽管是协方差矩阵的无偏估计，
最大似然估计不是协方差矩阵的特征值的一个很好的估计，
所以从反演得到的精度矩阵是不准确的。
有时，甚至出现数学原因，经验协方差矩阵不能反转。
为了避免这样的反演问题，引入了经验协方差矩阵的一种变换方式：``shrinkage`` 。

In the scikit-learn, this transformation (with a user-defined shrinkage
coefficient) can be directly applied to a pre-computed covariance with
the :func:`shrunk_covariance` method. Also, a shrunk estimator of the
covariance can be fitted to data with a :class:`ShrunkCovariance` object
and its :meth:`ShrunkCovariance.fit` method.  Again, depending whether
the data are centered or not, the result will be different, so one may
want to use the ``assume_centered`` parameter accurately.
在 scikit-learn 中，该变换（具有用户定义的收缩系数）
可以直接应用于使用 :func:`shrunk_covariance` 方法预先计算协方差。
此外，协方差的收缩估计可以用 :class:`ShrunkCovariance` 对象
及其 :meth:`ShrunkCovariance.fit` 方法拟合到数据中。
再次，根据数据是否居中，结果会不同，所以可能要准确使用参数 ``assume_centered`` 。

Mathematically, this shrinkage consists in reducing the ratio between the
smallest and the largest eigenvalue of the empirical covariance matrix.
It can be done by simply shifting every eigenvalue according to a given
offset, which is equivalent of finding the l2-penalized Maximum
Likelihood Estimator of the covariance matrix. In practice, shrinkage
boils down to a simple a convex transformation : :math:`\Sigma_{\rm
shrunk} = (1-\alpha)\hat{\Sigma} + \alpha\frac{{\rm
Tr}\hat{\Sigma}}{p}\rm Id`.
在数学上，这种收缩在于减少经验协方差矩阵的最小和最大特征值之间的比率。
可以通过简单地根据给定的偏移量移动每个特征值来完成，
这相当于找到协方差矩阵的l2惩罚的最大似然估计器（l2-penalized Maximum
Likelihood Estimator）。在实践中，收缩归结为简单的凸变换：
:math:`\Sigma_{\rm
shrunk} = (1-\alpha)\hat{\Sigma} + \alpha\frac{{\rm
Tr}\hat{\Sigma}}{p}\rm Id`.

Choosing the amount of shrinkage, :math:`\alpha` amounts to setting a
bias/variance trade-off, and is discussed below.
选择收缩量， :math:`\alpha` 相当于设置偏差/方差权衡，下面将讨论。

.. topic:: Examples例子:

   * See :ref:`sphx_glr_auto_examples_covariance_plot_covariance_estimation.py` for
     an example on how to fit a :class:`ShrunkCovariance` object
     to data.


Ledoit-Wolf shrinkage
Ledoit-Wolf收敛（Ledoit-Wolf shrinkage）
---------------------

In their 2004 paper [1]_, O. Ledoit and M. Wolf propose a formula so as
to compute the optimal shrinkage coefficient :math:`\alpha` that
minimizes the Mean Squared Error between the estimated and the real
covariance matrix.
在他们的2004年的文章 [1]_ 中， O.Ledoit 和 M.Wolf 提出了一个公式，
用来计算优化的收敛系数 :math:`\alpha` ，
它使得估计协方差和实际协方差矩阵之间的均方差进行最小化。

The Ledoit-Wolf estimator of the covariance matrix can be computed on
a sample with the :meth:`ledoit_wolf` function of the
`sklearn.covariance` package, or it can be otherwise obtained by
fitting a :class:`LedoitWolf` object to the same sample.
在 `sklearn.covariance` 包中，可以使用 :meth:`ledoit_wolf` 函数来计算样本的
基于 Ledoit-Wolf estimator 的协方差， 或者可以针对同样的样本
通过拟合 :class:`LedoitWolf` 对象来获得。

.. topic:: Examples例子:

   * See :ref:`sphx_glr_auto_examples_covariance_plot_covariance_estimation.py` for
     an example on how to fit a :class:`LedoitWolf` object to data and
     for visualizing the performances of the Ledoit-Wolf estimator in
     terms of likelihood.
     See :ref:`sphx_glr_auto_examples_covariance_plot_covariance_estimation.py`
     关于如何将 :class:`LedoitWolf` 对象与数据拟合
     并将 Ledoit-Wolf 估计器的性能进行可视化的示例。

.. topic:: References参考文献:

    .. [1] O. Ledoit and M. Wolf, "A Well-Conditioned Estimator for Large-Dimensional
           Covariance Matrices", Journal of Multivariate Analysis, Volume 88, Issue 2,
           February 2004, pages 365-411.

.. _oracle_approximating_shrinkage:

Oracle Approximating Shrinkage
Oracle近似收缩（Oracle Approximating Shrinkage）
------------------------------

Under the assumption that the data are Gaussian distributed, Chen et
al. [2]_ derived a formula aimed at choosing a shrinkage coefficient that
yields a smaller Mean Squared Error than the one given by Ledoit and
Wolf's formula. The resulting estimator is known as the Oracle
Shrinkage Approximating estimator of the covariance.
在数据为高斯分布的假设下，Chen et al. 等 [2]_  推导出了一个公式，旨在
产生比 Ledoit 和 Wolf 公式具有更小均方差的收敛系数。
所得到的估计器被称为协方差的Oracle收缩近似估计器。

The OAS estimator of the covariance matrix can be computed on a sample
with the :meth:`oas` function of the `sklearn.covariance`
package, or it can be otherwise obtained by fitting an :class:`OAS`
object to the same sample.
在 `sklearn.covariance` 包中， OAS 估计的协方差可以使用函数 :meth:`oas`
对样本进行计算，或者可以通过将 :class:`OAS` 对象拟合到相同的样本来获得。


.. figure:: ../auto_examples/covariance/images/sphx_glr_plot_covariance_estimation_001.png
   :target: ../auto_examples/covariance/plot_covariance_estimation.html
   :align: center
   :scale: 65%

   Bias-variance trade-off when setting the shrinkage: comparing the
   choices of Ledoit-Wolf and OAS estimators
   设定收缩时的偏差方差权衡：比较 Ledoit-Wolf 和 OAS 估计量的选择

.. topic:: References参考文献:

    .. [2] Chen et al., "Shrinkage Algorithms for MMSE Covariance Estimation",
           IEEE Trans. on Sign. Proc., Volume 58, Issue 10, October 2010.

.. topic:: Examples例子:

   * See :ref:`sphx_glr_auto_examples_covariance_plot_covariance_estimation.py` for
     an example on how to fit an :class:`OAS` object
     to data.

   * See :ref:`sphx_glr_auto_examples_covariance_plot_lw_vs_oas.py` to visualize the
     Mean Squared Error difference between a :class:`LedoitWolf` and
     an :class:`OAS` estimator of the covariance.


.. figure:: ../auto_examples/covariance/images/sphx_glr_plot_lw_vs_oas_001.png
   :target: ../auto_examples/covariance/plot_lw_vs_oas.html
   :align: center
   :scale: 75%


.. _sparse_inverse_covariance:

Sparse inverse covariance
稀疏逆协方差（Sparse inverse covariance）
==========================

The matrix inverse of the covariance matrix, often called the precision
matrix, is proportional to the partial correlation matrix. It gives the
partial independence relationship. In other words, if two features are
independent conditionally on the others, the corresponding coefficient in
the precision matrix will be zero. This is why it makes sense to estimate
a sparse precision matrix: by learning independence relations from the
data, the estimation of the covariance matrix is better conditioned. This
is known as *covariance selection*.
协方差矩阵的逆矩阵，通常称为精度矩阵（precision
matrix），它与部分相关矩阵（partial correlation matrix）成正比。
它给出部分独立性关系。换句话说，如果两个特征在其他特征上有条件地独立，
则精度矩阵中的对应系数将为零。这就是为什么估计一个稀疏精度矩阵是有道理的：
通过从数据中学习独立关系，协方差矩阵的估计能更好处理。这被称为协方差选择。

In the small-samples situation, in which ``n_samples`` is on the order
of ``n_features`` or smaller, sparse inverse covariance estimators tend to work
better than shrunk covariance estimators. However, in the opposite
situation, or for very correlated data, they can be numerically unstable.
In addition, unlike shrinkage estimators, sparse estimators are able to
recover off-diagonal structure.
在小样本的情况，即 ``n_samples`` 是数量级 ``n_features`` 或更小，
稀疏的逆协方差估计往往比收敛的协方差估计更好。
然而，在相反的情况下，或者对于非常相关的数据，它们可能在数值上不稳定。
此外，与收敛估算不同，稀疏估计器能够恢复非对角线结构 （off-diagonal structure）。

The :class:`GraphLasso` estimator uses an l1 penalty to enforce sparsity on
the precision matrix: the higher its ``alpha`` parameter, the more sparse
the precision matrix. The corresponding :class:`GraphLassoCV` object uses
cross-validation to automatically set the ``alpha`` parameter.
:class:`GraphLasso` 估计器使用 L1 惩罚执行关于精度矩阵的稀疏性：
``alpha`` 参数越高，精度矩阵的稀疏性越大。
相应的 :class:`GraphLassoCV` 对象使用交叉验证来自动设置 ``alpha`` 参数。

.. figure:: ../auto_examples/covariance/images/sphx_glr_plot_sparse_cov_001.png
   :target: ../auto_examples/covariance/plot_sparse_cov.html
   :align: center
   :scale: 60%

   *A comparison of maximum likelihood, shrinkage and sparse estimates of
   the covariance and precision matrix in the very small samples
   settings. 在非常小的样本设置中，协方差和精度矩阵的最大似然、收缩和稀疏估计的比较。*

.. note:: **Structure recovery结构恢复**

   Recovering a graphical structure from correlations in the data is a
   challenging thing. If you are interested in such recovery keep in mind
   that:
   从数据中的相关性恢复图形结构是一个具有挑战性的事情。如果您对这种恢复感兴趣，请记住：

   * Recovery is easier from a correlation matrix than a covariance
     matrix: standardize your observations before running :class:`GraphLasso`
     相关矩阵的恢复比协方差矩阵更容易：在运行 :class:`GraphLasso` 前先标准化观察值

   * If the underlying graph has nodes with much more connections than
     the average node, the algorithm will miss some of these connections.
     如果底层图具有比平均节点更多的连接节点，则算法将错过其中一些连接。

   * If your number of observations is not large compared to the number
     of edges in your underlying graph, you will not recover it.
     如果您的观察次数与底层图形中的边数相比不大，则不会恢复。

   * Even if you are in favorable recovery conditions, the alpha
     parameter chosen by cross-validation (e.g. using the
     :class:`GraphLassoCV` object) will lead to selecting too many edges.
     However, the relevant edges will have heavier weights than the
     irrelevant ones.
     即使您具有良好的恢复条件，通过交叉验证
     （例如使用GraphLassoCV对象）选择的 Alpha 参数将导致选择太多边。
     然而，相关边缘将具有比不相关边缘更重的权重。

The mathematical formulation is the following:
数学公式如下：

.. math::

    \hat{K} = \mathrm{argmin}_K \big(
                \mathrm{tr} S K - \mathrm{log} \mathrm{det} K
                + \alpha \|K\|_1
                \big)

Where :math:`K` is the precision matrix to be estimated, and :math:`S` is the
sample covariance matrix. :math:`\|K\|_1` is the sum of the absolute values of
off-diagonal coefficients of :math:`K`. The algorithm employed to solve this
problem is the GLasso algorithm, from the Friedman 2008 Biostatistics
paper. It is the same algorithm as in the R ``glasso`` package.
其中：:math:`K` 是要估计的精度矩阵（precision matrix）， :math:`S` 是样本的协方差矩阵。
:math:`\|K\|_1` 是非对角系数 :math:`K` （off-diagonal coefficients）的绝对值之和。
用于解决这个问题的算法是来自 Friedman 2008 Biostatistics 论文的 GLasso算法。
它与 R语言 ``glasso`` 包中的算法相同。


.. topic:: Examples例子:

   * :ref:`sphx_glr_auto_examples_covariance_plot_sparse_cov.py`: example on synthetic
     data showing some recovery of a structure, and comparing to other
     covariance estimators.合成数据示例，显示结构的一些恢复，并与其他协方差估计器进行比较。

   * :ref:`sphx_glr_auto_examples_applications_plot_stock_market.py`: example on real
     stock market data, finding which symbols are most linked.
     真实股票市场数据示例，查找哪些信号最相关。

.. topic:: References参考文献:

   * Friedman et al, `"Sparse inverse covariance estimation with the
     graphical lasso" <http://biostatistics.oxfordjournals.org/content/9/3/432.short>`_,
     Biostatistics 9, pp 432, 2008

.. _robust_covariance:

Robust Covariance Estimation
Robust协方差估计（Robust Covariance Estimation）
============================

Real data set are often subjects to measurement or recording
errors. Regular but uncommon observations may also appear for a variety
of reason. Every observation which is very uncommon is called an
outlier.
The empirical covariance estimator and the shrunk covariance
estimators presented above are very sensitive to the presence of
outlying observations in the data. Therefore, one should use robust
covariance estimators to estimate the covariance of its real data
sets. Alternatively, robust covariance estimators can be used to
perform outlier detection and discard/downweight some observations
according to further processing of the data.
实际数据集通常是会有测量或记录错误。合格但不常见的观察也可能出于各种原因。
每个不常见的观察称为异常值。
上面提出的经验协方差估计器和收缩协方差估计器对数据中异常观察值非常敏感。
因此，应该使用更好的协方差估计（robust covariance estimators）来估算其真实数据集的协方差。
或者，可以使用更好的协方差估计器（robust covariance estimators）来执行异常值检测，
并根据数据的进一步处理，丢弃/降低某些观察值。


The ``sklearn.covariance`` package implements a robust estimator of covariance,
the Minimum Covariance Determinant [3]_.
``sklearn.covariance`` 包实现了 robust estimator of covariance， 即
 Minimum Covariance Determinant [3]_ 。


Minimum Covariance Determinant
最小协方差决定（Minimum Covariance Determinant）
------------------------------

The Minimum Covariance Determinant estimator is a robust estimator of
a data set's covariance introduced by P.J. Rousseeuw in [3]_.  The idea
is to find a given proportion (h) of "good" observations which are not
outliers and compute their empirical covariance matrix.  This
empirical covariance matrix is then rescaled to compensate the
performed selection of observations ("consistency step").  Having
computed the Minimum Covariance Determinant estimator, one can give
weights to observations according to their Mahalanobis distance,
leading to a reweighted estimate of the covariance matrix of the data
set ("reweighting step").
最小协方差决定（Minimum Covariance Determinant）估计器是
由 P.J. Rousseeuw 在 [3]_ 中引入的数据集协方差的鲁棒估计 (robust estimator)。
这个想法是找出一个给定比例（h）的“好”观察值，它们不是离群值，
且可以计算其经验协方差矩阵。
然后将该经验协方差矩阵重新缩放以补偿所执行的观察选择（“一致性步骤”）。
计算最小协方差决定估计器后，可以根据其马氏距离（Mahalanobis distance）给出观测值的权重，
这导致数据集的协方差矩阵的重新加权估计（“重新加权步骤”，reweighting step）。

Rousseeuw and Van Driessen [4]_ developed the FastMCD algorithm in order
to compute the Minimum Covariance Determinant. This algorithm is used
in scikit-learn when fitting an MCD object to data. The FastMCD
algorithm also computes a robust estimate of the data set location at
the same time.
Rousseeuw 和 Van Driessen  [4]_ 开发了 FastMCD 算法，以计算最小协方差决定因子（Minimum Covariance Determinant）。
在scikit-learn中，该算法在将 MCD 对象拟合到数据时应用。FastMCD 算法同时计算数据集位置的鲁棒估计。

Raw estimates can be accessed as ``raw_location_`` and ``raw_covariance_``
attributes of a :class:`MinCovDet` robust covariance estimator object.
Raw估计可通过 :class:`MinCovDet` 对象的 ``raw_location_`` 和 ``raw_covariance_`` 属性获得。

.. topic:: References参考文献:

    .. [3] P. J. Rousseeuw. Least median of squares regression.
           J. Am Stat Ass, 79:871, 1984.
    .. [4] A Fast Algorithm for the Minimum Covariance Determinant Estimator,
           1999, American Statistical Association and the American Society
           for Quality, TECHNOMETRICS.

.. topic:: Examples例子:

   * See :ref:`sphx_glr_auto_examples_covariance_plot_robust_vs_empirical_covariance.py` for
     an example on how to fit a :class:`MinCovDet` object to data and see how
     the estimate remains accurate despite the presence of outliers.
     关于如何将对象 :class:`MinCovDet` 与数据拟合的示例， 尽管存在异常值，但估计结果仍然比较准确。

   * See :ref:`sphx_glr_auto_examples_covariance_plot_mahalanobis_distances.py` to
     visualize the difference between :class:`EmpiricalCovariance` and
     :class:`MinCovDet` covariance estimators in terms of Mahalanobis distance
     (so we get a better estimate of the precision matrix too).
     就马氏距离（Mahalanobis distance），针对协方差估计器 :class:`EmpiricalCovariance` 和
     :class:`MinCovDet` 之间的差异进行可视化。（所以我们得到了精度矩阵的更好估计）

.. |robust_vs_emp| image:: ../auto_examples/covariance/images/sphx_glr_plot_robust_vs_empirical_covariance_001.png
   :target: ../auto_examples/covariance/plot_robust_vs_empirical_covariance.html
   :scale: 49%

.. |mahalanobis| image:: ../auto_examples/covariance/images/sphx_glr_plot_mahalanobis_distances_001.png
   :target: ../auto_examples/covariance/plot_mahalanobis_distances.html
   :scale: 49%



____

.. list-table::
    :header-rows: 1

    * - Influence of outliers on location and covariance estimates
      - Separating inliers from outliers using a Mahalanobis distance

    * - |robust_vs_emp|
      - |mahalanobis|
