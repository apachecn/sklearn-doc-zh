.. _linear_model:

=========================
广义线性模型
=========================

.. currentmodule:: sklearn.linear_model

下面是一组用于回归的方法，其中目标值 y 是输入变量 x 的线性组合。 在数学概念中，如果 :math:`\hat{y}` 是预测值。

.. math::    \hat{y}(w, x) = w_0 + w_1 x_1 + ... + w_p x_p

在整个模块中，我们定义向量 :math:`w = (w_1,..., w_p)` 作为 ``coef_`` ，定义 :math:`w_0` 作为 ``intercept_`` 。

如果需要使用广义线性模型进行分类，请参阅 :ref:`Logistic_regression` 。


.. _ordinary_least_squares:

普通最小二乘法
=======================

:class:`LinearRegression` 拟合一个带有系数 :math:`w = (w_1, ..., w_p)` 的线性模型，使得数据集实际观测数据和预测数据（估计值）之间的残差平方和最小。其数学表达式为:

.. math:: \underset{w}{min\,} {|| X w - y||_2}^2

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_ols_001.png
   :target: ../auto_examples/linear_model/plot_ols.html
   :align: center
   :scale: 50%

:class:`LinearRegression` 会调用 ``fit`` 方法来拟合数组 X， y，并且将线性模型的系数 :math:`w` 存储在其成员变量 ``coef_`` 中::

    >>> from sklearn import linear_model
    >>> reg = linear_model.LinearRegression()
    >>> reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
    >>> reg.coef_
    array([ 0.5,  0.5])

然而，对于普通最小二乘的系数估计问题，其依赖于模型各项的相互独立性。当各项是相关的，且设计矩阵 :math:`X` 的各列近似线性相关，那么，设计矩阵会趋向于奇异矩阵，这会导致最小二乘估计对于随机误差非常敏感，产生很大的方差。例如，在没有实验设计的情况下收集到的数据，这种多重共线性（multicollinearity）的情况可能真的会出现。

.. topic:: 示例:

   * :ref:`sphx_glr_auto_examples_linear_model_plot_ols.py`


普通最小二乘法复杂度
---------------------------------

该方法使用 X 的奇异值分解来计算最小二乘解。如果 X 是一个 size 为 (n, p) 的矩阵，设 :math:`n \geq p` ，则该方法的复杂度为 :math:`O(n p^2)` 

.. _ridge_regression:

岭回归
================

:class:`Ridge` 回归通过对系数的大小施加惩罚来解决 :ref:`ordinary_least_squares` 的一些问题。 岭系数最小化的是带罚项的残差平方和，


.. math::

   \underset{w}{min\,} {{|| X w - y||_2}^2 + \alpha {||w||_2}^2}


其中， :math:`\alpha \geq 0` 是控制系数收缩量的复杂性参数： :math:`\alpha` 的值越大，收缩量越大，因此系数对共线性的稳定性也更强。

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_ridge_path_001.png
   :target: ../auto_examples/linear_model/plot_ridge_path.html
   :align: center
   :scale: 50%


与其他线性模型一样， :class:`Ridge` 用 ``fit`` 方法将模型系数 :math:`w` 存储在其 ``coef_`` 成员中::

    >>> from sklearn import linear_model
    >>> reg = linear_model.Ridge (alpha = .5)
    >>> reg.fit ([[0, 0], [0, 0], [1, 1]], [0, .1, 1]) # doctest: +NORMALIZE_WHITESPACE
    Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,
          normalize=False, random_state=None, solver='auto', tol=0.001)
    >>> reg.coef_
    array([ 0.34545455,  0.34545455])
    >>> reg.intercept_ #doctest: +ELLIPSIS
    0.13636...


.. topic:: 示例:

   * :ref:`sphx_glr_auto_examples_linear_model_plot_ridge_path.py`( 作为正则化的函数，绘制岭系数 )
   * :ref:`sphx_glr_auto_examples_text_document_classification_20newsgroups.py`( 使用稀疏特征的文本文档分类 )


岭回归的复杂度
----------------

这种方法与 :ref:`ordinary_least_squares` 的复杂度是相同的.

.. FIXME:
.. Not completely true: OLS is solved by an SVD, while Ridge is solved by
.. the method of normal equations (Cholesky), there is a big flop difference
.. between these


设置正则化参数：广义交叉验证
------------------------------------------------------------------

:class:`RidgeCV` 通过内置的 Alpha 参数的交叉验证来实现岭回归。 该对象与 GridSearchCV 的使用方法相同，只是它默认为 Generalized Cross-Validation(广义交叉验证 GCV)，这是一种有效的留一验证方法（LOO-CV）::

    >>> from sklearn import linear_model
    >>> reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
    >>> reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])       # doctest: +SKIP
    RidgeCV(alphas=[0.1, 1.0, 10.0], cv=None, fit_intercept=True, scoring=None,
        normalize=False)
    >>> reg.alpha_                                      # doctest: +SKIP
    0.1

.. topic:: 参考

    * "Notes on Regularized Least Squares", Rifkin & Lippert (`technical report
      <http://cbcl.mit.edu/projects/cbcl/publications/ps/MIT-CSAIL-TR-2007-025.pdf>`_,
      `course slides
      <http://www.mit.edu/~9.520/spring07/Classes/rlsslides.pdf>`_).


.. _lasso:

Lasso
=====

The :class:`Lasso` 是估计稀疏系数的线性模型。 它在一些情况下是有用的，因为它倾向于使用具有较少参数值的情况，有效地减少给定解决方案所依赖变量的数量。 因此，Lasso 及其变体是压缩感知领域的基础。 在一定条件下，它可以恢复一组非零权重的精确集（见 :ref:`sphx_glr_auto_examples_applications_plot_tomography_l1_reconstruction.py` ）。

在数学公式表达上，它由一个带有 :math:`\ell_1` 先验的正则项的线性模型组成。 其最小化的目标函数是:

.. math::  \underset{w}{min\,} { \frac{1}{2n_{samples}} ||X w - y||_2 ^ 2 + \alpha ||w||_1}

lasso estimate 解决了加上罚项 :math:`\alpha ||w||_1` 的最小二乘法的最小化，其中， :math:`\alpha` 是一个常数， :math:`||w||_1` 是参数向量的 :math:`\ell_1`-norm 范数。

:class:`Lasso` 类的实现使用了 coordinate descent （坐标下降算法）来拟合系数。 查看 :ref:`least_angle_regression` ，这是另一种方法::

    >>> from sklearn import linear_model
    >>> reg = linear_model.Lasso(alpha = 0.1)
    >>> reg.fit([[0, 0], [1, 1]], [0, 1])
    Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
       normalize=False, positive=False, precompute=False, random_state=None,
       selection='cyclic', tol=0.0001, warm_start=False)
    >>> reg.predict([[1, 1]])
    array([ 0.8])

对于较低级别的任务，同样有用的是函数 :func:`lasso_path` 。它能够通过搜索所有可能的路径上的值来计算系数。

.. topic:: 举例:

  * :ref:`sphx_glr_auto_examples_linear_model_plot_lasso_and_elasticnet.py` （稀疏信号的 lasso 和弹性网）
  * :ref:`sphx_glr_auto_examples_applications_plot_tomography_l1_reconstruction.py` （压缩感知：L1 先验（Lasso）的断层扫描重建）


.. note:: **Feature selection with Lasso（使用 Lasso 进行特征选择）**

      由于 Lasso 回归产生稀疏模型，因此可以用于执行特征选择，详见
      :ref:`l1_feature_selection` （基于 L1 的特征选择）。


设置正则化参数
--------------------------------

 ``alpha`` 参数控制估计系数的稀疏度。

使用交叉验证
^^^^^^^^^^^^^^^^^^^^^^^

scikit-learn 通过交叉验证来公开设置 Lasso ``alpha`` 参数的对象: :class:`LassoCV` 和 :class:`LassoLarsCV`。
:class:`LassoLarsCV` 是基于下面解释的 :ref:`least_angle_regression` 算法。

对于具有许多线性回归的高维数据集， :class:`LassoCV` 最常见。 然而，:class:`LassoLarsCV` 在寻找 `alpha` 参数值上更具有优势，而且如果样本数量与特征数量相比非常小时，通常 :class:`LassoLarsCV` 比 :class:`LassoCV` 要快。

.. |lasso_cv_1| image:: ../auto_examples/linear_model/images/sphx_glr_plot_lasso_model_selection_002.png
    :target: ../auto_examples/linear_model/plot_lasso_model_selection.html
    :scale: 48%

.. |lasso_cv_2| image:: ../auto_examples/linear_model/images/sphx_glr_plot_lasso_model_selection_003.png
    :target: ../auto_examples/linear_model/plot_lasso_model_selection.html
    :scale: 48%

.. centered:: |lasso_cv_1| |lasso_cv_2|


基于信息标准的模型选择
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

有多种选择时，估计器 :class:`LassoLarsIC` 建议使用 Akaike information criterion （Akaike 信息准则）（AIC）和 Bayes Information criterion （贝叶斯信息准则）（BIC）。 
当使用 k-fold 交叉验证时，正则化路径只计算一次而不是 k + 1 次，所以找到 α 的最优值是一种计算上更便宜的替代方法。 然而，这样的标准需要对解决方案的自由度进行适当的估计，对于大样本（渐近结果）导出，并假设模型是正确的，即数据实际上是由该模型生成的。 当问题严重受限（比样本更多的特征）时，他们也倾向于打破。

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_lasso_model_selection_001.png
    :target: ../auto_examples/linear_model/plot_lasso_model_selection.html
    :align: center
    :scale: 50%


.. topic:: 示例:

  * :ref:`sphx_glr_auto_examples_linear_model_plot_lasso_model_selection.py`(Lasso 型号选择：交叉验证/AIC/BIC)

与 SVM 的正则化参数的比较
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``alpha`` 和 SVM 的正则化参数``C`` 之间的等式关系是 ``alpha = 1 / C`` 或者 ``alpha = 1 / (n_samples * C)`` ，并依赖于估计器和模型优化的确切的目标函数。
.. _multi_task_lasso:

多任务 Lasso
================

 :class:`MultiTaskLasso` 是一个估计多元回归稀疏系数的线性模型： ``y`` 是一个 ``(n_samples, n_tasks)`` 的二维数组，其约束条件和其他回归问题（也称为任务）是一样的，都是所选的特征值。

下图比较了通过使用简单的 Lasso 或 MultiTaskLasso 得到的 W 中非零的位置。 Lasso 估计产生分散的非零值，而 MultiTaskLasso 的一整列都是非零的。

.. |multi_task_lasso_1| image:: ../auto_examples/linear_model/images/sphx_glr_plot_multi_task_lasso_support_001.png
    :target: ../auto_examples/linear_model/plot_multi_task_lasso_support.html
    :scale: 48%

.. |multi_task_lasso_2| image:: ../auto_examples/linear_model/images/sphx_glr_plot_multi_task_lasso_support_002.png
    :target: ../auto_examples/linear_model/plot_multi_task_lasso_support.html
    :scale: 48%

.. centered:: |multi_task_lasso_1| |multi_task_lasso_2|

.. centered:: 拟合 time-series model （时间序列模型），强制任何活动的功能始终处于活动状态。

.. topic:: 示例:

  * :ref:`sphx_glr_auto_examples_linear_model_plot_multi_task_lasso_support.py` （联合功能选择与多任务 Lasso）


在数学上，它由一个线性模型组成，以混合的 :math:`\ell_1` :math:`\ell_2` 作为正则化器进行训练。目标函数最小化是：

.. math::  \underset{w}{min\,} { \frac{1}{2n_{samples}} ||X W - Y||_{Fro} ^ 2 + \alpha ||W||_{21}}

其中 :math:`Fro` 表示 Frobenius 标准：

.. math:: ||A||_{Fro} = \sqrt{\sum_{ij} a_{ij}^2}

并且 :math:`\ell_1` :math:`\ell_2` 读取为:

.. math:: ||A||_{2 1} = \sum_i \sqrt{\sum_j a_{ij}^2}


:class:`MultiTaskLasso` 类的实现使用了坐标下降作为拟合系数的算法。


.. _elastic_net:

弹性网络
===========================
:class:`弹性网络` 是一种使用 L1， L2 范数作为先验正则项训练的线性回归模型。 这种组合允许学习到一个只有少量参数是非零稀疏的模型，就像 :class:`Lasso` 一样，但是它仍然保持
一些像 :class:`Ridge` 的正则性质。我们可利用 ``l1_ratio`` 参数控制 L1 和 L2 的凸组合。

弹性网络在很多特征互相联系的情况下是非常有用的。Lasso 很可能只随机考虑这些特征中的一个，而弹性网络更倾向于选择两个。

在实践中，Lasso 和 Ridge 之间权衡的一个优势是它允许在循环过程（Under rotate）中继承 Ridge 的稳定性。

在这里，最小化的目标函数是

.. math::

    \underset{w}{min\,} { \frac{1}{2n_{samples}} ||X w - y||_2 ^ 2 + \alpha \rho ||w||_1 +
    \frac{\alpha(1-\rho)}{2} ||w||_2 ^ 2}


.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_lasso_coordinate_descent_path_001.png
   :target: ../auto_examples/linear_model/plot_lasso_coordinate_descent_path.html
   :align: center
   :scale: 50%

 :class:`ElasticNetCV` 类可以通过交叉验证来设置参数 ``alpha`` （ :math:`\alpha` ） 和 ``l1_ratio`` （ :math:`\rho` ） 。

.. topic:: 示例:

  * :ref:`sphx_glr_auto_examples_linear_model_plot_lasso_and_elasticnet.py`
  * :ref:`sphx_glr_auto_examples_linear_model_plot_lasso_coordinate_descent_path.py`



.. _multi_task_elastic_net:

多任务弹性网络
======================

 :class:`MultiTaskElasticNet` 是一个对多回归问题估算稀疏参数的弹性网络: ``Y`` 是一个二维数组，形状是 ``(n_samples,n_tasks)``。  其限制条件是和其他回归问题一样，是选择的特征，也称为 tasks 。

从数学上来说， 它包含一个混合的 :math:`\ell_1` :math:`\ell_2` 先验和 :math:`\ell_2` 先验为正则项训练的线性模型
目标函数就是最小化:

.. math::

    \underset{W}{min\,} { \frac{1}{2n_{samples}} ||X W - Y||_{Fro}^2 + \alpha \rho ||W||_{2 1} +
    \frac{\alpha(1-\rho)}{2} ||W||_{Fro}^2}

在 :class:`MultiTaskElasticNet` 类中的实现采用了坐标下降法求解参数。

在 :class:`MultiTaskElasticNetCV` 中可以通过交叉验证来设置参数 ``alpha`` （ :math:`\alpha` ） 和 ``l1_ratio`` （ :math:`\rho` ） 。


.. _least_angle_regression:

最小角回归
======================

最小角回归 （LARS） 是对高维数据的回归算法， 由 Bradley Efron, Trevor Hastie, Iain Johnstone 和 Robert Tibshirani 开发完成。 LARS 和逐步回归很像。在每一步，它寻找与响应最有关联的
预测。当有很多预测有相同的关联时，它没有继续利用相同的预测，而是在这些预测中找出应该等角的方向。

LARS的优点:

  - 当 p >> n，该算法数值运算上非常有效。(例如当维度的数目远超点的个数)

  - 它在计算上和前向选择一样快，和普通最小二乘法有相同的运算复杂度。

  - 它产生了一个完整的分段线性的解决路径，在交叉验证或者其他相似的微调模型的方法上非常有用。

  - 如果两个变量对响应几乎有相等的联系，则它们的系数应该有相似的增长率。因此这个算法和我们直觉
    上的判断一样，而且还更加稳定。

  - 它很容易修改并为其他估算器生成解，比如Lasso。

LARS 的缺点:

  - 因为 LARS 是建立在循环拟合剩余变量上的，所以它对噪声非常敏感。这个问题，在 2004 年统计年鉴的文章由 Weisberg 详细讨论。

LARS 模型可以在 :class:`Lars` ，或者它的底层实现 :func:`lars_path` 中被使用。


LARS Lasso
============================

:class:`LassoLars` 是一个使用 LARS 算法的 lasso 模型，不同于基于坐标下降法的实现，它可以得到一个精确解，也就是一个关于自身参数标准化后的一个分段线性解。

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_lasso_lars_001.png
   :target: ../auto_examples/linear_model/plot_lasso_lars.html
   :align: center
   :scale: 50%

::

   >>> from sklearn import linear_model
   >>> reg = linear_model.LassoLars(alpha=.1)
   >>> reg.fit([[0, 0], [1, 1]], [0, 1])  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
   LassoLars(alpha=0.1, copy_X=True, eps=..., fit_intercept=True,
        fit_path=True, max_iter=500, normalize=True, positive=False,
        precompute='auto', verbose=False)
   >>> reg.coef_    # doctest: +ELLIPSIS
   array([ 0.717157...,  0.        ])

.. topic:: 示例:

 * :ref:`sphx_glr_auto_examples_linear_model_plot_lasso_lars.py`

Lars 算法提供了一个几乎无代价的沿着正则化参数的系数的完整路径，因此常利用函数 :func:`lars_path` 来取回路径。

数学表达式
------------------------

该算法和逐步回归非常相似，但是它没有在每一步包含变量，它估计的参数是根据与
其他剩余变量的联系来增加的。

在 LARS 的解中，没有给出一个向量的结果，而是给出一条曲线，显示参数向量的 L1 范式的每个值的解。
完全的参数路径存在 ``coef_path_`` 下。它的 size 是 (n_features, max_features+1)。 其中第一列通常是全 0 列。

.. topic:: 参考文献:

 * Original Algorithm is detailed in the paper `Least Angle Regression
   <http://www-stat.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf>`_
   by Hastie et al.


.. _omp:

正交匹配追踪法（OMP）
=================================
 :class:`OrthogonalMatchingPursuit` (正交匹配追踪法)和 :func:`orthogonal_mp` 
使用了 OMP 算法近似拟合了一个带限制的线性模型，该限制影响于模型的非 0 系数(例：L0 范数)。

就像最小角回归一样，作为一个前向特征选择方法，正交匹配追踪法可以近似一个固定非 0 元素的最优向量解:

.. math:: \text{arg\,min\,} ||y - X\gamma||_2^2 \text{ subject to } \
    ||\gamma||_0 \leq n_{nonzero\_coefs}

正交匹配追踪法也可以针对一个特殊的误差而不是一个特殊的非零系数的个数。可以表示为:

.. math:: \text{arg\,min\,} ||\gamma||_0 \text{ subject to } ||y-X\gamma||_2^2 \
    \leq \text{tol}


OMP 是基于每一步的贪心算法，其每一步元素都是与当前残差高度相关的。它跟较为简单的匹配追踪（MP）很相似，但是相比 MP 更好，在每一次迭代中，可以利用正交投影到之前选择的字典元素重新计算残差。


.. topic:: 示例:

 * :ref:`sphx_glr_auto_examples_linear_model_plot_omp.py`

.. topic:: 参考文献:

 * http://www.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf

 * `Matching pursuits with time-frequency dictionaries
   <http://blanche.polytechnique.fr/~mallat/papiers/MallatPursuit93.pdf>`_,
   S. G. Mallat, Z. Zhang,


.. _bayesian_regression:

贝叶斯回归
===================

贝叶斯回归可以用于在预估阶段的参数正则化: 正则化参数的选择不是通过人为的选择，而是通过手动调节数据值来实现。

上述过程可以通过引入 `无信息先验 <https://en.wikipedia.org/wiki/Non-informative_prior#Uninformative_priors>`__ 于模型中的超参数来完成。
在 `岭回归` 中使用的 :math:`\ell_{2}` 正则项相当于在 :math:`w` 为高斯先验条件下，且此先验的精确度为 :math:`\lambda^{-1}` 求最大后验估计。在这里，我们没有手工调参数 lambda ，而是让他作为一个变量，通过数据中估计得到。


为了得到一个全概率模型，输出 :math:`y` 也被认为是关于 :math:`X w` 的高斯分布。

.. math::  p(y|X,w,\alpha) = \mathcal{N}(y|X w,\alpha)

Alpha 在这里也是作为一个变量，通过数据中估计得到。

贝叶斯回归有如下几个优点:

    - 它能根据已有的数据进行改变。

    - 它能在估计过程中引入正则项。

贝叶斯回归有如下缺点:

    - 它的推断过程是非常耗时的。


.. topic:: 参考文献

 * 一个对于贝叶斯方法的很好的介绍 C. Bishop: Pattern Recognition and Machine learning

 * 详细介绍原创算法的一本书 `Bayesian learning for neural networks` by Radford M. Neal

.. _bayesian_ridge_regression:

贝叶斯岭回归
-------------------------

 :class:`BayesianRidge` 利用概率模型估算了上述的回归问题，其先验参数 :math:`w` 是由以下球面高斯公式得出的：

.. math:: p(w|\lambda) =
    \mathcal{N}(w|0,\lambda^{-1}\bold{I_{p}})

先验参数 :math:`\alpha` 和 :math:`\lambda` 一般是服从 `gamma 分布 <https://en.wikipedia.org/wiki/Gamma_distribution>`__ ， 这个分布与高斯成共轭先验关系。

得到的模型一般称为 *贝叶斯岭回归*， 并且这个与传统的 :class:`Ridge` 非常相似。参数 :math:`w` ， :math:`\alpha` 和 :math:`\lambda` 是在模型拟合的时候一起被估算出来的。 剩下的超参数就是
关于:math:`\alpha` 和 :math:`\lambda`  的 gamma 分布的先验了。 它们通常被选择为 *无信息先验* 。模型参数的估计一般利用最大 *边缘似然对数估计* 。

默认 :math:`\alpha_1 = \alpha_2 =  \lambda_1 = \lambda_2 = 10^{-6}`.


.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_bayesian_ridge_001.png
   :target: ../auto_examples/linear_model/plot_bayesian_ridge.html
   :align: center
   :scale: 50%


贝叶斯岭回归用来解决回归问题::

    >>> from sklearn import linear_model
    >>> X = [[0., 0.], [1., 1.], [2., 2.], [3., 3.]]
    >>> Y = [0., 1., 2., 3.]
    >>> reg = linear_model.BayesianRidge()
    >>> reg.fit(X, Y)
    BayesianRidge(alpha_1=1e-06, alpha_2=1e-06, compute_score=False, copy_X=True,
           fit_intercept=True, lambda_1=1e-06, lambda_2=1e-06, n_iter=300,
           normalize=False, tol=0.001, verbose=False)

在模型训练完成后，可以用来预测新值::

    >>> reg.predict ([[1, 0.]])
    array([ 0.50000013])


权值 :math:`w` 可以被这样访问::

    >>> reg.coef_
    array([ 0.49999993,  0.49999993])

由于贝叶斯框架的缘故，权值与 :ref:`ordinary_least_squares` 产生的不太一样。
但是，贝叶斯岭回归对病态问题（ill-posed）的鲁棒性要更好。

.. topic:: 示例:

 * :ref:`sphx_glr_auto_examples_linear_model_plot_bayesian_ridge.py`

.. topic:: 参考文献

  * 更多细节可以参考 `Bayesian Interpolation
    <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.27.9072&rep=rep1&type=pdf>`_
    by MacKay, David J. C.



主动相关决策理论 - ARD
---------------------------------------

 :class:`ARDRegression` （主动相关决策理论）和 `Bayesian Ridge Regression`_ 非常相似，
但是会导致一个更加稀疏的权重 :math:`w` [1]_ [2]_ 。
 :class:`ARDRegression` 提出了一个不同的 :math:`w` 的先验假设。具体来说，就是弱化了高斯分布为球形的假设。
它采用 :math:`w` 分布是与轴平行的椭圆高斯分布。

也就是说，每个权值 :math:`w_{i}` 从一个中心在 0 点，精度为 :math:`\lambda_{i}` 的高斯分布中采样得到的。

.. math:: p(w|\lambda) = \mathcal{N}(w|0,A^{-1})

并且 :math:`diag \; (A) = \lambda = \{\lambda_{1},...,\lambda_{p}\}`.

与 `Bayesian Ridge Regression`_ 不同， 每个 :math:`w_{i}` 都有一个标准差 :math:`\lambda_i` 。所有 :math:`\lambda_i` 的先验分布
由超参数 :math:`\lambda_1` 、 :math:`\lambda_2` 确定的相同的 gamma 分布确定。

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_ard_001.png
   :target: ../auto_examples/linear_model/plot_ard.html
   :align: center
   :scale: 50%

ARD 也被称为 *稀疏贝叶斯学习* 或 *相关向量机* [3]_ [4]_ 。

.. topic:: 示例:

  * :ref:`sphx_glr_auto_examples_linear_model_plot_ard.py`

.. topic:: 参考文献:

    .. [1] Christopher M. Bishop: Pattern Recognition and Machine Learning, Chapter 7.2.1

    .. [2] David Wipf and Srikantan Nagarajan: `A new view of automatic relevance determination <http://papers.nips.cc/paper/3372-a-new-view-of-automatic-relevance-determination.pdf>`_

    .. [3] Michael E. Tipping: `Sparse Bayesian Learning and the Relevance Vector Machine <http://www.jmlr.org/papers/volume1/tipping01a/tipping01a.pdf>`_

    .. [4] Tristan Fletcher: `Relevance Vector Machines explained <http://www.tristanfletcher.co.uk/RVM%20Explained.pdf>`_




.. _Logistic_regression:

logistic 回归
===================

logistic 回归，虽然名字里有 "回归" 二字，但实际上是解决分类问题的一类线性模型。在某些文献中，logistic 回归又被称作 logit 回归，maximum-entropy classification（MaxEnt，最大熵分类），或 log-linear classifier（对数线性分类器）。该模型利用函数 `logistic function <https://en.wikipedia.org/wiki/Logistic_function>`_ 
将单次试验（single trial）的可能结果输出为概率。

scikit-learn 中 logistic 回归在 :class:`LogisticRegression` 类中实现了二分类（binary）、一对多分类（one-vs-rest）及多项式 logistic 回归，并带有可选的 L1 和 L2 正则化。

作为优化问题，带 L2 罚项的二分类 logistic 回归要最小化以下代价函数（cost function）：

.. math:: \underset{w, c}{min\,} \frac{1}{2}w^T w + C \sum_{i=1}^n \log(\exp(- y_i (X_i^T w + c)) + 1) .

类似地，带 L1 正则的 logistic 回归解决的是如下优化问题：

.. math:: \underset{w, c}{min\,} \|w\|_1 + C \sum_{i=1}^n \log(\exp(- y_i (X_i^T w + c)) + 1) .

在 :class:`LogisticRegression` 类中实现了这些优化算法: "liblinear"， "newton-cg"， "lbfgs"， "sag" 和 "saga"。

"liblinear" 应用了坐标下降算法（Coordinate Descent, CD），并基于 scikit-learn 内附的高性能 C++ 库 `LIBLINEAR library <http://www.csie.ntu.edu.tw/~cjlin/liblinear/>`_ 实现。不过 CD 算法训练的模型不是真正意义上的多分类模型，而是基于 "one-vs-rest" 思想分解了这个优化问题，为每个类别都训练了一个二元分类器。因为实现在底层使用该求解器的 :class:`LogisticRegression` 实例对象表面上看是一个多元分类器。 :func:`sklearn.svm.l1_min_c` 可以计算使用 L1 罚项时 C 的下界，以避免模型为空（即全部特征分量的权重为零）。

"lbfgs", "sag" 和 "newton-cg" solvers （求解器）只支持 L2 惩罚项，对某些高维数据收敛更快。这些求解器的参数 `multi_class`设为 "multinomial" 即可训练一个真正的多项式 logistic 回归 [5]_ ，其预测的概率比默认的 "one-vs-rest" 设定更为准确。

"sag" 求解器基于平均随机梯度下降算法（Stochastic Average Gradient descent） [6]_。在大数据集上的表现更快，大数据集指样本量大且特征数多。

"saga" 求解器 [7]_ 是 "sag" 的一类变体，它支持非平滑（non-smooth）的 L1 正则选项 ``penalty="l1"`` 。因此对于稀疏多项式 logistic 回归 ，往往选用该求解器。

一言以蔽之，选用求解器可遵循如下规则:

=================================  =====================================
Case                               Solver
=================================  =====================================
L1正则                             	"liblinear" or "saga"
多项式损失（multinomial loss）        	"lbfgs", "sag", "saga" or "newton-cg"
大数据集（`n_samples`）            	"sag" or "saga"
=================================  =====================================

"saga" 一般都是最佳的选择，但出于一些历史遗留原因默认的是 "liblinear" 。

对于大数据集，还可以用 :class:`SGDClassifier` ，并使用对数损失（'log' loss）

.. topic:: 示例：

  * :ref:`sphx_glr_auto_examples_linear_model_plot_logistic_l1_l2_sparsity.py`

  * :ref:`sphx_glr_auto_examples_linear_model_plot_logistic_path.py`

  * :ref:`sphx_glr_auto_examples_linear_model_plot_logistic_multinomial.py`

  * :ref:`sphx_glr_auto_examples_linear_model_plot_sparse_logistic_regression_20newsgroups.py`

  * :ref:`sphx_glr_auto_examples_linear_model_plot_sparse_logistic_regression_mnist.py`

.. _liblinear_differences:

.. topic:: 与 liblinear 的区别:

   当 ``fit_intercept=False`` 拟合得到的 ``coef_`` 或者待预测的数据为零时，用 ``solver=liblinear`` 的 :class:`LogisticRegression` 
   或 :class:`LinearSVC` 与直接使用外部 liblinear 库预测得分会有差异。这是因为，
   对于 ``decision_function`` 为零的样本， :class:`LogisticRegression` 和 :class:`LinearSVC`
   将预测为负类，而 liblinear 预测为正类。
   注意，设定了 ``fit_intercept=False`` ，又有很多样本使得 ``decision_function`` 为零的模型，很可能会欠拟合，其表现往往比较差。建议您设置 ``fit_intercept=True`` 并增大 ``intercept_scaling`` 。

.. note:: **利用稀疏 logistic 回归进行特征选择**

   带 L1 罚项的 logistic 回归 将得到稀疏模型（sparse model），相当于进行了特征选择（feature selection），详情参见 :ref:`l1_feature_selection` 。

 :class:`LogisticRegressionCV` 对 logistic 回归 的实现内置了交叉验证（cross-validation），可以找出最优的参数 C 。"newton-cg"， "sag"， "saga" 和 "lbfgs" 在高维数据上更快，因为采用了热启动（warm-starting）。
 在多分类设定下，若 `multi_class` 设为 "ovr" ，会为每类求一个最佳的 C 值；若 `multi_class` 设为 "multinomial" ，会通过交叉熵损失（cross-entropy loss）求出一个最佳 C 值。

.. topic:: 参考文献：

    .. [5] Christopher M. Bishop: Pattern Recognition and Machine Learning, Chapter 4.3.4

    .. [6] Mark Schmidt, Nicolas Le Roux, and Francis Bach: `Minimizing Finite Sums with the Stochastic Average Gradient. <https://hal.inria.fr/hal-00860051/document>`_

    .. [7] Aaron Defazio, Francis Bach, Simon Lacoste-Julien: `SAGA: A Fast Incremental Gradient Method With Support for Non-Strongly Convex Composite Objectives. <https://arxiv.org/abs/1407.0202>`_

随机梯度下降， SGD
=================================

随机梯度下降是拟合线性模型的一个简单而高效的方法。在样本量（和特征数）很大时尤为有用。
方法 ``partial_fit`` 可用于 online learning （在线学习）或基于 out-of-core learning （外存的学习）

:class:`SGDClassifier` 和 :class:`SGDRegressor` 分别用于拟合分类问题和回归问题的线性模型，可使用不同的（凸）损失函数，支持不同的罚项。
例如，设定 ``loss="log"`` ，则 :class:`SGDClassifier` 拟合一个逻辑斯蒂回归模型，而 ``loss="hinge"`` 拟合线性支持向量机（SVM）。

.. topic:: 参考文献

 * :ref:`sgd`

.. _perceptron:

Perceptron（感知器）
================================

:class:`Perceptron` 是适用于大规模学习的一种简单算法。默认情况下：

    - 不需要设置学习率（learning rate）。

    - 不需要正则化处理。

    - 仅使用错误样本更新模型。

最后一点表明使用合页损失（hinge loss）的感知机比 SGD 略快，所得模型更稀疏。

.. _passive_aggressive:

Passive Aggressive Algorithms（被动攻击算法）
===================================================================

被动攻击算法是大规模学习的一类算法。和感知机类似，它也不需要设置学习率，不过比感知机多出一个正则化参数 ``C`` 。

对于分类问题， :class:`PassiveAggressiveClassifier` 可设定
``loss='hinge'`` （PA-I）或 ``loss='squared_hinge'`` （PA-II）。对于回归问题，
:class:`PassiveAggressiveRegressor` 可设置
``loss='epsilon_insensitive'`` （PA-I）或
``loss='squared_epsilon_insensitive'`` （PA-II）。

.. topic:: 参考文献：


 * `"Online Passive-Aggressive Algorithms"
   <http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf>`_
   K. Crammer, O. Dekel, J. Keshat, S. Shalev-Shwartz, Y. Singer - JMLR 7 (2006)


稳健回归（Robustness regression）: 处理离群点（outliers）和模型错误
===================================================================

稳健回归（robust regression）特别适用于回归模型包含损坏数据（corrupt data）的情况，如离群点或模型中的错误。

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_theilsen_001.png
   :target: ../auto_examples/linear_model/plot_theilsen.html
   :scale: 50%
   :align: center

各种使用场景与相关概念
----------------------------------------

处理包含离群点的数据时牢记以下几点:

.. |y_outliers| image:: ../auto_examples/linear_model/images/sphx_glr_plot_robust_fit_003.png
   :target: ../auto_examples/linear_model/plot_robust_fit.html
   :scale: 60%

.. |X_outliers| image:: ../auto_examples/linear_model/images/sphx_glr_plot_robust_fit_002.png
   :target: ../auto_examples/linear_model/plot_robust_fit.html
   :scale: 60%

.. |large_y_outliers| image:: ../auto_examples/linear_model/images/sphx_glr_plot_robust_fit_005.png
   :target: ../auto_examples/linear_model/plot_robust_fit.html
   :scale: 60%

* **离群值在 X 上还是在 y 方向上**?

  ==================================== ====================================
  离群值在 y 方向上                  	离群值在 X 方向上
  ==================================== ====================================
  |y_outliers|                         |X_outliers|
  ==================================== ====================================

* **离群点的比例 vs. 错误的量级（amplitude）**

  离群点的数量很重要，离群程度也同样重要。

  ==================================== ====================================
  离群值小                            离群值大
  ==================================== ====================================
  |y_outliers|                         |large_y_outliers|
  ==================================== ====================================

稳健拟合（robust fitting）的一个重要概念是崩溃点（breakdown point），即拟合模型（仍准确预测）所能承受的离群值最大比例。

注意，在高维数据条件下（ `n_features` 大），一般而言很难完成稳健拟合，很可能完全不起作用。


.. topic:: **折中： 预测器的选择**

  Scikit-learn提供了三种稳健回归的预测器（estimator）: :ref:`RANSAC <ransac_regression>` ， :ref:`Theil Sen <theil_sen_regression>` 和 :ref:`HuberRegressor <huber_regression>`

  * :ref:`HuberRegressor <huber_regression>` 一般快于 :ref:`RANSAC <ransac_regression>` 和 :ref:`Theil Sen <theil_sen_regression>` ，除非样本数很大，即 ``n_samples`` >> ``n_features`` 。
    这是因为 :ref:`RANSAC <ransac_regression>` 和 :ref:`Theil Sen <theil_sen_regression>` 都是基于数据的较小子集进行拟合。但使用默认参数时， :ref:`Theil Sen <theil_sen_regression>` 和 :ref:`RANSAC <ransac_regression>` 可能不如 :ref:`HuberRegressor <huber_regression>` 鲁棒。

  * :ref:`RANSAC <ransac_regression>` 比 :ref:`Theil Sen <theil_sen_regression>` 更快，在样本数量上的伸缩性（适应性）更好。

  * :ref:`RANSAC <ransac_regression>` 能更好地处理y方向的大值离群点（通常情况下）。

  * :ref:`Theil Sen <theil_sen_regression>` 能更好地处理x方向中等大小的离群点，但在高维情况下无法保证这一特点。

 实在决定不了的话，请使用 :ref:`RANSAC <ransac_regression>`

.. _ransac_regression:

RANSAC： 随机抽样一致性算法（RANdom SAmple Consensus）
--------------------------------------------------------------------

随机抽样一致性算法（RANdom SAmple Consensus， RANSAC）利用全体数据中局内点（inliers）的一个随机子集拟合模型。

RANSAC 是一种非确定性算法，以一定概率输出一个可能的合理结果，依赖于迭代次数（参数 `max_trials` ）。这种算法主要解决线性或非线性回归问题，在计算机视觉摄影测绘领域尤为流行。

算法从全体样本输入中分出一个局内点集合，全体样本可能由于测量错误或对数据的假设错误而含有噪点、离群点。最终的模型仅从这个局内点集合中得出。

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_ransac_001.png
   :target: ../auto_examples/linear_model/plot_ransac.html
   :align: center
   :scale: 50%

算法细节
^^^^^^^^^^^^^^^^^^^^^^^^

每轮迭代执行以下步骤:

1. 从原始数据中抽样 ``min_samples`` 数量的随机样本，检查数据是否合法（见 ``is_data_valid`` ）。
2. 用一个随机子集拟合模型（ ``base_estimator.fit`` ）。检查模型是否合法（见 ``is_model_valid`` ）。
3. 计算预测模型的残差（residual），将全体数据分成局内点和离群点（ ``base_estimator.predict(X) - y`` ）
 - 绝对残差小于 ``residual_threshold`` 的全体数据认为是局内点。
4. 若局内点样本数最大，保存当前模型为最佳模型。以免当前模型离群点数量恰好相等（而出现未定义情况），规定仅当数值大于当前最值时认为是最佳模型。

上述步骤或者迭代到最大次数（ ``max_trials`` ），或者某些终止条件满足时停下（见 ``stop_n_inliers`` 和 ``stop_score`` )。最终模型由之前确定的最佳模型的局内点样本（一致性集合，consensus set）预测。

函数 ``is_data_valid`` 和 ``is_model_valid`` 可以识别出随机样本子集中的退化组合（degenerate combinations）并予以丢弃（reject）。即便不需要考虑退化情况，也会使用 ``is_data_valid`` ，因为在拟合模型之前调用它能得到更高的计算性能。


.. topic:: 示例：

  * :ref:`sphx_glr_auto_examples_linear_model_plot_ransac.py`
  * :ref:`sphx_glr_auto_examples_linear_model_plot_robust_fit.py`

.. topic:: 参考文献：

 * https://en.wikipedia.org/wiki/RANSAC
 * `"Random Sample Consensus: A Paradigm for Model Fitting with Applications to
   Image Analysis and Automated Cartography"
   <http://www.cs.columbia.edu/~belhumeur/courses/compPhoto/ransac.pdf>`_
   Martin A. Fischler and Robert C. Bolles - SRI International (1981)
 * `"Performance Evaluation of RANSAC Family"
   <http://www.bmva.org/bmvc/2009/Papers/Paper355/Paper355.pdf>`_
   Sunglok Choi, Taemin Kim and Wonpil Yu - BMVC (2009)

.. _theil_sen_regression:

Theil-Sen 预估器: 广义中值估计器（generalized-median-based estimator）
-----------------------------------------------------------------------------------------

:class:`TheilSenRegressor` 估计器：使用中位数在多个维度泛化，对多元异常值更具有鲁棒性，但问题是，随着维数的增加，估计器的准确性在迅速下降。准确性的丢失，导致在高维上的估计值比不上普通的最小二乘法。

.. topic:: 示例:

  * :ref:`sphx_glr_auto_examples_linear_model_plot_theilsen.py`
  * :ref:`sphx_glr_auto_examples_linear_model_plot_robust_fit.py`

.. topic:: 参考文献:

 * https://en.wikipedia.org/wiki/Theil%E2%80%93Sen_estimator

算法理论细节
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`TheilSenRegressor` 在渐近效率和无偏估计方面足以媲美 :ref:`Ordinary Least Squares (OLS) <ordinary_least_squares>` （普通最小二乘法（OLS））。与 OLS 不同的是， Theil-Sen 是一种非参数方法，这意味着它没有对底层数据的分布假设。由于 Theil-Sen 是基于中值的估计，它更适合于损坏的数据即离群值。
在单变量的设置中，Theil-Sen 在简单的线性回归的情况下，其崩溃点大约 29.3% ，这意味着它可以容忍任意损坏的数据高达 29.3% 。

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_theilsen_001.png
   :target: ../auto_examples/linear_model/plot_theilsen.html
   :align: center
   :scale: 50%

scikit-learn 中实现的 :class:`TheilSenRegressor` 是多元线性回归模型的推广 [#f1]_ ，利用了空间中值方法，它是多维中值的推广 [#f2]_ 。

关于时间复杂度和空间复杂度，Theil-Sen 的尺度根据

.. math::
    \binom{n_{samples}}{n_{subsamples}}

这使得它不适用于大量样本和特征的问题。因此，可以选择一个亚群的大小来限制时间和空间复杂度，只考虑所有可能组合的随机子集。

.. topic:: 示例:

  * :ref:`sphx_glr_auto_examples_linear_model_plot_theilsen.py`

.. topic:: 参考文献:

    .. [#f1] Xin Dang, Hanxiang Peng, Xueqin Wang and Heping Zhang: `Theil-Sen Estimators in a Multiple Linear Regression Model. <http://home.olemiss.edu/~xdang/papers/MTSE.pdf>`_

    .. [#f2] T. Kärkkäinen and S. Äyrämö: `On Computation of Spatial Median for Robust Data Mining. <http://users.jyu.fi/~samiayr/pdf/ayramo_eurogen05.pdf>`_

.. _huber_regression:

Huber 回归
------------------------------

:class:`HuberRegressor` 与 :class:`Ridge` 不同，因为它对于被分为异常值的样本应用了一个线性损失。如果这个样品的绝对误差小于某一阈值，样品就被分为内围值。
它不同于 :class:`TheilSenRegressor` 和 :class:`RANSACRegressor` ，因为它没有忽略异常值的影响，并分配给它们较小的权重。

.. figure:: /auto_examples/linear_model/images/sphx_glr_plot_huber_vs_ridge_001.png
   :target: ../auto_examples/linear_model/plot_huber_vs_ridge.html
   :align: center
   :scale: 50%

这个 :class:`HuberRegressor` 最小化的损失函数是：

.. math::

  \underset{w, \sigma}{min\,} {\sum_{i=1}^n\left(\sigma + H_m\left(\frac{X_{i}w - y_{i}}{\sigma}\right)\sigma\right) + \alpha {||w||_2}^2}

其中

.. math::

  H_m(z) = \begin{cases}
         z^2, & \text {if } |z| < \epsilon, \\
         2\epsilon|z| - \epsilon^2, & \text{otherwise}
  \end{cases}

建议设置参数 ``epsilon`` 为 1.35 以实现 95% 统计效率。

注意
-----------------
:class:`HuberRegressor` 与将损失设置为 `huber` 的 :class:`SGDRegressor` 并不相同，体现在以下方面的使用方式上。

- :class:`HuberRegressor` 是标度不变性的. 一旦设置了 ``epsilon`` ， 通过不同的值向上或向下缩放 ``X`` 和 ``y`` ，就会跟以前一样对异常值产生同样的鲁棒性。相比 :class:`SGDRegressor` 其中 ``epsilon`` 在 ``X`` 和 ``y`` 被缩放的时候必须再次设置。

- :class:`HuberRegressor` 应该更有效地使用在小样本数据，同时 :class:`SGDRegressor` 需要一些训练数据的 passes 来产生一致的鲁棒性。

.. topic:: 示例:

  * :ref:`sphx_glr_auto_examples_linear_model_plot_huber_vs_ridge.py`

.. topic:: 参考文献:

  * Peter J. Huber, Elvezio M. Ronchetti: Robust Statistics, Concomitant scale estimates, pg 172

另外，这个估计是不同于 R 实现的 Robust Regression (http://www.ats.ucla.edu/stat/r/dae/rreg.htm) ，因为 R 实现加权最小二乘，权重考虑到每个样本并基于残差大于某一阈值的量。

.. _polynomial_regression:

多项式回归：用基函数展开线性模型
===================================================================

.. currentmodule:: sklearn.preprocessing

机器学习中一种常见的模式，是使用线性模型训练数据的非线性函数。这种方法保持了一般快速的线性方法的性能，同时允许它们适应更广泛的数据范围。

例如，可以通过构造系数的 **polynomial features** 来扩展一个简单的线性回归。在标准线性回归的情况下，你可能有一个类似于二维数据的模型: 

.. math::    \hat{y}(w, x) = w_0 + w_1 x_1 + w_2 x_2

如果我们想把抛物面拟合成数据而不是平面，我们可以结合二阶多项式的特征，使模型看起来像这样:

.. math::    \hat{y}(w, x) = w_0 + w_1 x_1 + w_2 x_2 + w_3 x_1 x_2 + w_4 x_1^2 + w_5 x_2^2

观察到这 *还是一个线性模型* （这有时候是令人惊讶的）: 看到这个，想象创造一个新的变量

.. math::  z = [x_1, x_2, x_1 x_2, x_1^2, x_2^2]

有了这些重新标记的数据，我们可以将问题写成

.. math::    \hat{y}(w, x) = w_0 + w_1 z_1 + w_2 z_2 + w_3 z_3 + w_4 z_4 + w_5 z_5

我们看到，所得的 *polynomial regression* 与我们上文所述线性模型是同一类（即关于 :math:`w` 是线性的），因此可以用同样的方法解决。通过用这些基函数建立的高维空间中的线性拟合，该模型具有灵活性，可以适应更广泛的数据范围。

这里是一个例子，使用不同程度的多项式特征将这个想法应用于一维数据:

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_polynomial_interpolation_001.png
   :target: ../auto_examples/linear_model/plot_polynomial_interpolation.html
   :align: center
   :scale: 50%

这个图是使用 :class:`PolynomialFeatures` 预创建。该预处理器将输入数据矩阵转换为给定度的新数据矩阵。使用方法如下::

    >>> from sklearn.preprocessing import PolynomialFeatures
    >>> import numpy as np
    >>> X = np.arange(6).reshape(3, 2)
    >>> X
    array([[0, 1],
           [2, 3],
           [4, 5]])
    >>> poly = PolynomialFeatures(degree=2)
    >>> poly.fit_transform(X)
    array([[  1.,   0.,   1.,   0.,   0.,   1.],
           [  1.,   2.,   3.,   4.,   6.,   9.],
           [  1.,   4.,   5.,  16.,  20.,  25.]])

``X`` 的特征已经从 :math:`[x_1, x_2]` 转换到 :math:`[1, x_1, x_2, x_1^2, x_1 x_2, x_2^2]`, 并且现在可以用在任何线性模型。

这种预处理可以通过 :ref:`Pipeline <pipeline>` 工具进行简化。可以创建一个表示简单多项式回归的单个对象，使用方法如下所示::

    >>> from sklearn.preprocessing import PolynomialFeatures
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.pipeline import Pipeline
    >>> import numpy as np
    >>> model = Pipeline([('poly', PolynomialFeatures(degree=3)),
    ...                   ('linear', LinearRegression(fit_intercept=False))])
    >>> # fit to an order-3 polynomial data
    >>> x = np.arange(5)
    >>> y = 3 - 2 * x + x ** 2 - x ** 3
    >>> model = model.fit(x[:, np.newaxis], y)
    >>> model.named_steps['linear'].coef_
    array([ 3., -2.,  1., -1.])

利用多项式特征训练的线性模型能够准确地恢复输入多项式系数。

在某些情况下，没有必要包含任何单个特征的更高的幂，只需要相乘最多 :math:`d` 个不同的特征即可，所谓 *interaction features（交互特征）* 。这些可通过设定 :class:`PolynomialFeatures` 的 ``interaction_only=True`` 得到。

例如，当处理布尔属性，对于所有 :math:`n`  :math:`x_i^n = x_i` ，因此是无用的；但 :math:`x_i x_j` 代表两布尔结合。这样我们就可以用线性分类器解决异或问题::

    >>> from sklearn.linear_model import Perceptron
    >>> from sklearn.preprocessing import PolynomialFeatures
    >>> import numpy as np
    >>> X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    >>> y = X[:, 0] ^ X[:, 1]
    >>> y
    array([0, 1, 1, 0])
    >>> X = PolynomialFeatures(interaction_only=True).fit_transform(X).astype(int)
    >>> X
    array([[1, 0, 0, 0],
           [1, 0, 1, 0],
           [1, 1, 0, 0],
           [1, 1, 1, 1]])
    >>> clf = Perceptron(fit_intercept=False, max_iter=10, tol=None,
    ...                  shuffle=False).fit(X, y)

分类器的 "predictions" 是完美的::

    >>> clf.predict(X)
    array([0, 1, 1, 0])
    >>> clf.score(X, y)
    1.0
