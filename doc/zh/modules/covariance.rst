.. _covariance:

===================================================

协方差估计
===================================================

.. currentmodule:: sklearn.covariance


许多统计问题在某一时刻需要估计一个总体的协方差矩阵，这可以看作是对数据集散点图形状的估计。
大多数情况下，基于样本的估计（基于其属性，如尺寸，结构，均匀性），
对估计质量有很大影响。 `sklearn.covariance` 方法的目的是
提供一个能在各种设置下准确估计总体协方差矩阵的工具。

我们假设观察是独立的，相同分布的 (i.i.d.)。


经验协方差 
================================

已知数据集的协方差矩阵与经典 *maximum likelihood estimator(最大似然估计)* （或 "经验协方差"）
很好地近似，条件是与特征数量（描述观测值的变量）相比，观测数量足够大。
更准确地说，样本的最大似然估计是相应的总体协方差矩阵的无偏估计。

样本的经验协方差矩阵可以使用 :func:`empirical_covariance` 包的函数计算 ，
或者通过 :class:`EmpiricalCovariance` 使用 :meth:`EmpiricalCovariance.fit`
方法将对象与数据样本拟合 。
要注意，取决于数据是否居中，结果会有所不同，所以可能需要准确使用参数 ``assume_centered``。
如果使用 ``assume_centered=False`` ，则结果更准确。且测试集应该具有与训练集相同的均值向量。
如果不是这样，两者都应该使用中心值， ``assume_centered=True`` 应该使用。

.. topic:: 例子:

   * See :ref:`sphx_glr_auto_examples_covariance_plot_covariance_estimation.py` for
     an example on how to fit an :class:`EmpiricalCovariance` object
     to data.


.. _shrunk_covariance:

收敛协方差
==========================

基本收敛
-------------------------------------------

尽管是协方差矩阵的无偏估计，
最大似然估计不是协方差矩阵的特征值的一个很好的估计，
所以从反演得到的精度矩阵是不准确的。
有时，甚至出现数学原因，经验协方差矩阵不能反转。
为了避免这样的反演问题，引入了经验协方差矩阵的一种变换方式：``shrinkage`` 。

在 scikit-learn 中，该变换（具有用户定义的收缩系数）
可以直接应用于使用 :func:`shrunk_covariance` 方法预先计算协方差。
此外，协方差的收缩估计可以用 :class:`ShrunkCovariance` 对象
及其 :meth:`ShrunkCovariance.fit` 方法拟合到数据中。
再次，根据数据是否居中，结果会不同，所以可能要准确使用参数 ``assume_centered`` 。

在数学上，这种收缩在于减少经验协方差矩阵的最小和最大特征值之间的比率。
可以通过简单地根据给定的偏移量移动每个特征值来完成，
这相当于找到协方差矩阵的l2惩罚的最大似然估计器（l2-penalized Maximum
Likelihood Estimator）。在实践中，收缩归结为简单的凸变换：
:math:`\Sigma_{\rm
shrunk} = (1-\alpha)\hat{\Sigma} + \alpha\frac{{\rm
Tr}\hat{\Sigma}}{p}\rm Id`.

选择收缩量， :math:`\alpha` 相当于设置偏差/方差权衡，下面将讨论。

.. topic:: 示例:

   * See :ref:`sphx_glr_auto_examples_covariance_plot_covariance_estimation.py` for
     an example on how to fit a :class:`ShrunkCovariance` object
     to data.


Ledoit-Wolf 收敛
---------------------------------

在他们的 2004 年的论文 [1]_ 中， O.Ledoit 和 M.Wolf 提出了一个公式，
用来计算优化的收敛系数 :math:`\alpha` ，
它使得估计协方差和实际协方差矩阵之间的均方差进行最小化。

在 `sklearn.covariance` 包中，可以使用 :meth:`ledoit_wolf` 函数来计算样本的
基于 Ledoit-Wolf estimator 的协方差， 或者可以针对同样的样本
通过拟合 :class:`LedoitWolf` 对象来获得。

.. topic:: 例子:

   * See :ref:`sphx_glr_auto_examples_covariance_plot_covariance_estimation.py`
     关于如何将 :class:`LedoitWolf` 对象与数据拟合，
     并将 Ledoit-Wolf 估计器的性能进行可视化的示例。

.. topic:: 参考文献:

    .. [1] O. Ledoit and M. Wolf, "A Well-Conditioned Estimator for Large-Dimensional
           Covariance Matrices", Journal of Multivariate Analysis, Volume 88, Issue 2,
           February 2004, pages 365-411.

.. _oracle_approximating_shrinkage:

Oracle 近似收缩
------------------------------------------

在数据为高斯分布的假设下，Chen et al. 等 [2]_  推导出了一个公式，旨在
产生比 Ledoit 和 Wolf 公式具有更小均方差的收敛系数。
所得到的估计器被称为协方差的 Oracle 收缩近似估计器。

在 `sklearn.covariance` 包中， OAS 估计的协方差可以使用函数 :meth:`oas` 对样本进行计算，或者可以通过将 :class:`OAS` 对象拟合到相同的样本来获得。


.. figure:: ../auto_examples/covariance/images/sphx_glr_plot_covariance_estimation_001.png
   :target: ../auto_examples/covariance/plot_covariance_estimation.html
   :align: center
   :scale: 65%

   设定收缩时的偏差方差权衡：比较 Ledoit-Wolf 和 OAS 估计量的选择

.. topic:: 参考文献:

    .. [2] Chen et al., "Shrinkage Algorithms for MMSE Covariance Estimation",
           IEEE Trans. on Sign. Proc., Volume 58, Issue 10, October 2010.

.. topic:: 示例:

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

稀疏逆协方差
======================================

协方差矩阵的逆矩阵，通常称为精度矩阵（precision matrix），它与部分相关矩阵（partial correlation matrix）成正比。
它给出部分独立性关系。换句话说，如果两个特征在其他特征上有条件地独立，
则精度矩阵中的对应系数将为零。这就是为什么估计一个稀疏精度矩阵是有道理的：
通过从数据中学习独立关系，协方差矩阵的估计能更好处理。这被称为协方差选择。

在小样本的情况，即 ``n_samples`` 是数量级 ``n_features`` 或更小，
稀疏的逆协方差估计往往比收敛的协方差估计更好。
然而，在相反的情况下，或者对于非常相关的数据，它们可能在数值上不稳定。
此外，与收敛估算不同，稀疏估计器能够恢复非对角线结构 （off-diagonal structure）。

:class:`GraphLasso` 估计器使用 L1 惩罚执行关于精度矩阵的稀疏性：
``alpha`` 参数越高，精度矩阵的稀疏性越大。
相应的 :class:`GraphLassoCV` 对象使用交叉验证来自动设置 ``alpha`` 参数。

.. figure:: ../auto_examples/covariance/images/sphx_glr_plot_sparse_cov_001.png
   :target: ../auto_examples/covariance/plot_sparse_cov.html
   :align: center
   :scale: 60%

   * 在非常小的样本设置中，协方差和精度矩阵的最大似然、收缩和稀疏估计的比较。*

.. note:: **结构恢复**

   从数据中的相关性恢复图形结构是一个具有挑战性的事情。如果您对这种恢复感兴趣，请记住：

   * 相关矩阵的恢复比协方差矩阵更容易：在运行 :class:`GraphLasso` 前先标准化观察值

   * 如果底层图具有比平均节点更多的连接节点，则算法将错过其中一些连接。

   * 如果您的观察次数与底层图形中的边数相比不大，则不会恢复。

   * 即使您具有良好的恢复条件，通过交叉验证（例如使用GraphLassoCV对象）选择的 Alpha 参数将导致选择太多边。
     然而，相关边缘将具有比不相关边缘更重的权重。

数学公式如下：

.. math::

    \hat{K} = \mathrm{argmin}_K \big(
                \mathrm{tr} S K - \mathrm{log} \mathrm{det} K
                + \alpha \|K\|_1
                \big)

其中：:math:`K` 是要估计的精度矩阵（precision matrix）， :math:`S` 是样本的协方差矩阵。
:math:`\|K\|_1` 是非对角系数 :math:`K` （off-diagonal coefficients）的绝对值之和。
用于解决这个问题的算法是来自 Friedman 2008 Biostatistics 论文的 GLasso 算法。
它与 R 语言 ``glasso`` 包中的算法相同。


.. topic:: 例子:

   * :ref:`sphx_glr_auto_examples_covariance_plot_sparse_cov.py`:
   合成数据示例，显示结构的一些恢复，并与其他协方差估计器进行比较。

   * :ref:`sphx_glr_auto_examples_applications_plot_stock_market.py`:
     真实股票市场数据示例，查找哪些信号最相关。

.. topic:: 参考文献:

   * Friedman et al, `"Sparse inverse covariance estimation with the
     graphical lasso" <http://biostatistics.oxfordjournals.org/content/9/3/432.short>`_,
     Biostatistics 9, pp 432, 2008

.. _robust_covariance:

Robust 协方差估计
=====================================

实际数据集通常是会有测量或记录错误。合格但不常见的观察也可能出于各种原因。
每个不常见的观察称为异常值。
上面提出的经验协方差估计器和收缩协方差估计器对数据中异常观察值非常敏感。
因此，应该使用更好的协方差估计（robust covariance estimators）来估算其真实数据集的协方差。
或者，可以使用更好的协方差估计器（robust covariance estimators）来执行异常值检测，
并根据数据的进一步处理，丢弃/降低某些观察值。


``sklearn.covariance`` 包实现了 robust estimator of covariance， 即 Minimum Covariance Determinant [3]_ 。


最小协方差决定
-----------------------------------

最小协方差决定（Minimum Covariance Determinant）估计器是
由 P.J. Rousseeuw 在 [3]_ 中引入的数据集协方差的鲁棒估计 (robust estimator)。
这个想法是找出一个给定比例（h）的 "好" 观察值，它们不是离群值，
且可以计算其经验协方差矩阵。
然后将该经验协方差矩阵重新缩放以补偿所执行的观察选择（"consistency step(一致性步骤)"）。
计算最小协方差决定估计器后，可以根据其马氏距离（Mahalanobis distance）给出观测值的权重，
这导致数据集的协方差矩阵的重新加权估计（"reweighting step(重新加权步骤)"）。

Rousseeuw 和 Van Driessen  [4]_ 开发了 FastMCD 算法，以计算最小协方差决定因子（Minimum Covariance Determinant）。
在 scikit-learn 中，该算法在将 MCD 对象拟合到数据时应用。FastMCD 算法同时计算数据集位置的鲁棒估计。

Raw估计可通过 :class:`MinCovDet` 对象的 ``raw_location_`` 和 ``raw_covariance_`` 属性获得。

.. topic:: 参考文献:

    .. [3] P. J. Rousseeuw. Least median of squares regression.
           J. Am Stat Ass, 79:871, 1984.
    .. [4] A Fast Algorithm for the Minimum Covariance Determinant Estimator,
           1999, American Statistical Association and the American Society
           for Quality, TECHNOMETRICS.

.. topic:: 例子:

   * See :ref:`sphx_glr_auto_examples_covariance_plot_robust_vs_empirical_covariance.py`
     关于如何将对象 :class:`MinCovDet` 与数据拟合的示例， 尽管存在异常值，但估计结果仍然比较准确。

   * See :ref:`sphx_glr_auto_examples_covariance_plot_mahalanobis_distances.py` 
     马氏距离（Mahalanobis distance），针对协方差估计器 :class:`EmpiricalCovariance` 和
     :class:`MinCovDet` 之间的差异进行可视化。（所以我们得到了精度矩阵的更好估计）

.. |robust_vs_emp| image:: ../auto_examples/covariance/images/sphx_glr_plot_robust_vs_empirical_covariance_001.png
   :target: ../auto_examples/covariance/plot_robust_vs_empirical_covariance.html
   :scale: 49%

.. |mahalanobis| image:: ../auto_examples/covariance/images/sphx_glr_plot_mahalanobis_distances_001.png
   :target: ../auto_examples/covariance/plot_mahalanobis_distances.html
   :scale: 49%


.. list-table::
    :header-rows: 1

    * - Influence of outliers on location and covariance estimates
      - Separating inliers from outliers using a Mahalanobis distance

    * - |robust_vs_emp|
      - |mahalanobis|
