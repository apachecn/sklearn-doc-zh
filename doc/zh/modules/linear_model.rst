.. _linear_model:

=========================
广义线性模型
=========================

.. currentmodule:: sklearn.linear_model

以下是一组用于回归的方法，其中目标值预期是输入变量的线性组合。 在数学概念中，如果 :math:`\hat{y}` 是预测值
value.

.. math::    \hat{y}(w, x) = w_0 + w_1 x_1 + ... + w_p x_p

在整个模块中，我们指定向量 :math:`w = (w_1,..., w_p)` 作为 ``coef_`` 并且 :math:`w_0` 作为 ``intercept_``.

要使用广义线性模型进行分类，请参阅
:ref:`Logistic_regression`.


.. _ordinary_least_squares:

普通最小二乘法
=======================

:class:`LinearRegression` 适合一个带有系数 :math:`w = (w_1, ..., w_p)` 的线性模型 去最小化 (在数据集中观察到的结果) 和 (通过线性近似值预测的结果) 之间方差的和。 在数学上它解决了一个形式如下的问题：

.. math:: \underset{w}{min\,} {|| X w - y||_2}^2

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_ols_001.png
   :target: ../auto_examples/linear_model/plot_ols.html
   :align: center
   :scale: 50%

:class:`LinearRegression` 将采用其 ``fit`` 拟合方法数组 X, y 并将其线性模型的系数 :math:`w` 存储在其 ``coef_`` 成员中::

    >>> from sklearn import linear_model
    >>> reg = linear_model.LinearRegression()
    >>> reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
    >>> reg.coef_
    array([ 0.5,  0.5])

然而，普通最小二乘的系数估计依赖于模型项的独立性；当多个项有着相互关系并且设计矩阵 :math:`X` 的列具有近似的线性依赖性时，设计的矩阵变会得接近于单一；并且作为观察到的结果，最小二乘的估计值 将会变得对随机的错误非常敏感，并且会产生很大的方差；这种多重共线性的情况可能出现，例如，当收集没有实验设计过的数据时。

.. topic:: 举例:

   * :ref:`sphx_glr_auto_examples_linear_model_plot_ols.py`


普通最小二乘法复杂度
---------------------------------

该方法使用X的奇异值分解来计算最小二乘解。如果X是 size 为 (n, p) 的矩阵，则假设 :math:`n \geq p`
则该方法的成本为 :math:`O(n p^2)`.

.. _ridge_regression:

岭回归
================

:class:`Ridge` 回归通过对系数的大小施加惩罚来解决
:ref:`ordinary_least_squares` (普通最小二乘)的一些问题。 岭系数最小化一个带罚项的残差平方和，


.. math::

   \underset{w}{min\,} {{|| X w - y||_2}^2 + \alpha {||w||_2}^2}


这里， :math:`\alpha \geq 0` 是控制收缩量的复杂性参数： :math:`\alpha`, 的值越大，收缩量越大，因此系数变得对共线性变得更加鲁棒。

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_ridge_path_001.png
   :target: ../auto_examples/linear_model/plot_ridge_path.html
   :align: center
   :scale: 50%


与其他线性模型一样， :class:`Ridge` 将采用其 ``fit`` 将采用其 :math:`w` 存储在其 ``coef_`` 成员中::

    >>> from sklearn import linear_model
    >>> reg = linear_model.Ridge (alpha = .5)
    >>> reg.fit ([[0, 0], [0, 0], [1, 1]], [0, .1, 1]) # doctest: +NORMALIZE_WHITESPACE
    Ridge(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,
          normalize=False, random_state=None, solver='auto', tol=0.001)
    >>> reg.coef_
    array([ 0.34545455,  0.34545455])
    >>> reg.intercept_ #doctest: +ELLIPSIS
    0.13636...


.. topic:: 举例:

   * :ref:`sphx_glr_auto_examples_linear_model_plot_ridge_path.py`( 作为正则化的函数，绘制岭系数 )
   * :ref:`sphx_glr_auto_examples_text_document_classification_20newsgroups.py`( 使用稀疏特征的文本文档分类 )


岭复杂性
----------------

这种方法与 :ref:`ordinary_least_squares`(普通最小二乘方法)的复杂度相同.

.. FIXME:
.. Not completely true: OLS is solved by an SVD, while Ridge is solved by
.. the method of normal equations (Cholesky), there is a big flop difference
.. between these


设置正则化参数：广义交叉验证
------------------------------------------------------------------

:class:`RidgeCV` 通过内置的 Alpha 参数的交叉验证来实现岭回归。  该对象的工作方式与 GridSearchCV 相同，只是它默认为 Generalized Cross-Validation(通用交叉验证 GCV)，这是一种有效的留一交叉验证法::

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

The :class:`Lasso` 是估计稀疏系数的线性模型。 它在一些情况下是有用的，因为它倾向于使用具有较少参数值的解决方案，有效地减少给定解决方案所依赖的变量的数量。 为此，Lasso及其变体是压缩感测领域的基础。 在某些条件下，它可以恢复精确的非零权重集 (见 :ref:`sphx_glr_auto_examples_applications_plot_tomography_l1_reconstruction.py`).

在数学上，它由一个线性模型组成，以 :math:`\ell_1` 为准。 目标函数最小化是:

.. math::  \underset{w}{min\,} { \frac{1}{2n_{samples}} ||X w - y||_2 ^ 2 + \alpha ||w||_1}

因此，lasso estimate 解决了加上罚项 :math:`\alpha ||w||_1` 的最小二乘法的最小化，其中， :math:`\alpha` 是常数， :math:`||w||_1` 是参数向量的 :math:`\ell_1`-norm 范数。

:class:`Lasso` 类中的实现使用 coordinate descent （坐标下降）作为算法来拟合系数。 查看 :ref:`least_angle_regression` 用于另一个实现::

    >>> from sklearn import linear_model
    >>> reg = linear_model.Lasso(alpha = 0.1)
    >>> reg.fit([[0, 0], [1, 1]], [0, 1])
    Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
       normalize=False, positive=False, precompute=False, random_state=None,
       selection='cyclic', tol=0.0001, warm_start=False)
    >>> reg.predict([[1, 1]])
    array([ 0.8])

对于较低级别的任务也很有用的是函数 :func:`lasso_path` 来计算可能值的完整路径上的系数。

.. topic:: 举例:

  * :ref:`sphx_glr_auto_examples_linear_model_plot_lasso_and_elasticnet.py` (稀疏信号的套索和弹性网)
  * :ref:`sphx_glr_auto_examples_applications_plot_tomography_l1_reconstruction.py`(压缩感知：L1先验(Lasso)的断层扫描重建)


.. note:: **Feature selection with Lasso(使用 Lasso 进行 Feature 的选择)**

      由于 Lasso 回归产生稀疏模型，因此可以用于执行特征选择，详见
      :ref:`l1_feature_selection`(基于L1的特征选择).


设置正则化参数
--------------------------------

 ``alpha`` 参数控制估计系数的稀疏度。

使用交叉验证
^^^^^^^^^^^^^^^^^^^^^^^

scikit-learn 通过交叉验证来公开设置 Lasso ``alpha`` 参数的对象: :class:`LassoCV` and :class:`LassoLarsCV`。
:class:`LassoLarsCV` 是基于下面解释的 :ref:`least_angle_regression`(最小角度回归)算法。

对于具有许多线性回归的高维数据集， :class:`LassoCV` 最常见。 然而，:class:`LassoLarsCV` 具有探索更相关的 `alpha` parameter 参数值的优点，并且如果样本数量与特征数量相比非常小，则通常比 :class:`LassoCV` 快。

.. |lasso_cv_1| image:: ../auto_examples/linear_model/images/sphx_glr_plot_lasso_model_selection_002.png
    :target: ../auto_examples/linear_model/plot_lasso_model_selection.html
    :scale: 48%

.. |lasso_cv_2| image:: ../auto_examples/linear_model/images/sphx_glr_plot_lasso_model_selection_003.png
    :target: ../auto_examples/linear_model/plot_lasso_model_selection.html
    :scale: 48%

.. centered:: |lasso_cv_1| |lasso_cv_2|


基于信息标准的模型选择
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

有多种选择时，估计器 :class:`LassoLarsIC` 建议使用 Akaike information criterion （Akaike 信息准则）（AIC）和 Bayes Information criterion （贝叶斯信息准则）（BIC）。 当使用 k-fold 交叉验证时，正则化路径只计算一次而不是k + 1次，所以找到α的最优值是一种计算上更便宜的替代方法。 然而，这样的标准需要对解决方案的自由度进行适当的估计，对于大样本（渐近结果）导出，并假设模型是正确的，即数据实际上是由该模型生成的。 当问题严重受限（比样本更多的特征）时，他们也倾向于打破。

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_lasso_model_selection_001.png
    :target: ../auto_examples/linear_model/plot_lasso_model_selection.html
    :align: center
    :scale: 50%


.. topic:: 举例:

  * :ref:`sphx_glr_auto_examples_linear_model_plot_lasso_model_selection.py`(Lasso 型号选择：交叉验证/AIC/BIC)

与 SVM 的正则化参数进行比较
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

根据估计器和模型优化的精确目标函数，在 ``alpha`` 和 SVM 的正则化参数之间是等值的,其中
``C`` 是通过 ``alpha = 1 / C`` 或者 ``alpha = 1 / (n_samples * C)`` 得到的。

.. _multi_task_lasso:

多任务 Lasso
================

 :class:`MultiTaskLasso` 是一个线性模型，它联合估计多个回归问题的稀疏系数： ``y`` 是 ``(n_samples, n_tasks)`` 的二维数组，
约束是所选的特征对于所有回归问题（也称为任务）是相同的。

下图比较了使用简单 Lasso 或 MultiTaskLasso 获得的 W 中非零的位置。 Lasso 估计产生分散的非零，而 MultiTaskLasso 的非零是全列。

.. |multi_task_lasso_1| image:: ../auto_examples/linear_model/images/sphx_glr_plot_multi_task_lasso_support_001.png
    :target: ../auto_examples/linear_model/plot_multi_task_lasso_support.html
    :scale: 48%

.. |multi_task_lasso_2| image:: ../auto_examples/linear_model/images/sphx_glr_plot_multi_task_lasso_support_002.png
    :target: ../auto_examples/linear_model/plot_multi_task_lasso_support.html
    :scale: 48%

.. centered:: |multi_task_lasso_1| |multi_task_lasso_2|

.. centered:: 拟合 time-series model ( 时间序列模型 )，强制任何活动的功能始终处于活动状态。

.. topic:: 举例:

  * :ref:`sphx_glr_auto_examples_linear_model_plot_multi_task_lasso_support.py`(联合功能选择与多任务Lasso)


在数学上，它由一个线性模型组成，训练有混合的
:math:`\ell_1` :math:`\ell_2` 之前的正则化。目标函数最小化是：

.. math::  \underset{w}{min\,} { \frac{1}{2n_{samples}} ||X W - Y||_{Fro} ^ 2 + \alpha ||W||_{21}}

其中 :math:`Fro` 表示 Frobenius 标准：

.. math:: ||A||_{Fro} = \sqrt{\sum_{ij} a_{ij}^2}

并且 :math:`\ell_1` :math:`\ell_2` 读取为:

.. math:: ||A||_{2 1} = \sum_i \sqrt{\sum_j a_{ij}^2}


:class:`MultiTaskLasso` 类中的实现使用坐标下降作为拟合系数的算法。


.. _elastic_net:

弹性网络
===========
:class:`弹性网络` 是用L1,L2范数作为先验正则项训练的线性回归模型。 这种组合允许学习到一个稀疏模型，这个模型的一些参数是非0的，就像 :class:`Lasso`一样, 但是它仍然保持
一些像 :class:`Ridge`的正则性质。我们可利用 ``l1_ratio`` 参数控制L1和L2的凸组合。

弹性网络在很多特征互相联系的情况下是非常有用的。Lasso很可能只随机考虑这些特征中的一个，但是弹性网络很可能把这些特征全考虑了。

一个在实际中经常采用的权衡Lasso 和Ridge的好处，就是它允许弹性网络继承Ridge的旋转稳定性。

在这里，目标函数就是最小化

.. math::

    \underset{w}{min\,} { \frac{1}{2n_{samples}} ||X w - y||_2 ^ 2 + \alpha \rho ||w||_1 +
    \frac{\alpha(1-\rho)}{2} ||w||_2 ^ 2}


.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_lasso_coordinate_descent_path_001.png
   :target: ../auto_examples/linear_model/plot_lasso_coordinate_descent_path.html
   :align: center
   :scale: 50%

这个类 :class:`ElasticNetCV` 可以通过交叉验证来设置参数
``alpha`` (:math:`\alpha`) 和 ``l1_ratio`` (:math:`\rho`) 。

.. topic:: Examples:

  * :ref:`sphx_glr_auto_examples_linear_model_plot_lasso_and_elasticnet.py`
  * :ref:`sphx_glr_auto_examples_linear_model_plot_lasso_coordinate_descent_path.py`



.. _multi_task_elastic_net:

多任务弹性网络
======================

 :class:`MultiTaskElasticNet` 是一个对多回归问题估算稀疏参数的弹性网络: ``Y`` 是2D的数列，形状是
 ``(参数个数, 参数任务个数)``。 这个模型有个限制，即对所有的回归问题（任务）选择的特征是一样的。

从数学上来说， 它包含一个用
:math:`\ell_1` :math:`\ell_2` 先验 and :math:`\ell_2` 先验为正则项训练的线性模型
目标函数就是最小化:

.. math::

    \underset{W}{min\,} { \frac{1}{2n_{samples}} ||X W - Y||_{Fro}^2 + \alpha \rho ||W||_{2 1} +
    \frac{\alpha(1-\rho)}{2} ||W||_{Fro}^2}

在 :class:`MultiTaskElasticNet`类中的实现采用了坐标下降法求解参数。

在 :class:`MultiTaskElasticNetCV` 中可以通过交叉验证来设置参数
``alpha`` (:math:`\alpha`) 和 ``l1_ratio`` (:math:`\rho`) 。


.. _least_angle_regression:

最小角回归
======================

最小角回归 (LARS) is 是对高维数据的回归算法， 由Bradley Efron, Trevor Hastie, Iain
Johnstone 和 Robert Tibshirani开发完成。 LARS和 逐步回归很像。 在每一步，它寻找与响应最有关联的
预测。当有很多预测由相同的关联时，它没有继续利用相同的预测，而是在这些预测中找出应该等角的方向。

LARS的优点:

  - 当p >> n，该算法数值运算上非常有效。(例子：当维度的数目远超
    点的个数)

  - 它在计算上和前向选择一样快，和普通最小二乘法有相同的运算复杂度。

  - 它产生了一个完整的分段线性的解决路径，这在交叉验证或者其他相似的微调模型的方法上非常有用。

  - 如果两个变量对响应几乎有相等的联系，则它们的系数应该有大约相同的增长率。因而这个算法和我们直觉
    上想得一样，而且也更稳定。

  - 它也很容易改变为为其他估算器提供解，比如Lasso。

LARS的缺点:

  - 因为LARS是建立在循环拟合剩余变量上的，它会对噪声的影响非常敏感。
    这个问题，在2004年统计年鉴的文章由Weisberg详细讨论。

LARS模型可以在:class:`Lars`，或者它的低配实现:func:`lars_path`中被使用。


LARS Lasso
==========

:class:`LassoLars` 是一个使用LARS算法的lasso模型
不同于基于坐标下降法的实现，它得到的是精确解，也就是一个
关于自身参数标准化后的一个分段线性解。

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

.. topic:: 例子:

 * :ref:`sphx_glr_auto_examples_linear_model_plot_lasso_lars.py`

Lars算法提供了一个完整的关于参数的路径，并且几乎无代价的给出了正则化系数，
所以一个常规的操作包括利用函数:func:`lars_path`取回路径。

数学表达式
------------------------

该算法和逐步回归非常相似，但是它没有在每一步包含变量，它估计的参数是根据与
其他剩余变量的联系增加的。

算法没有给出一个向量的结果，LARS的解决方案包括对每一个变量进行总体变量的L1正则化后的显示的一条曲线。
完全的参数路径存在``coef_path_``向量下。它的尺寸是 (特征个数, 最大特征数+1)。 第一列通常是全0列。

.. topic:: 参考文献:

 * Original Algorithm is detailed in the paper `Least Angle Regression
   <http://www-stat.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf>`_
   by Hastie et al.


.. _omp:

正交匹配追踪法（OMP）
=================================
:class:`OrthogonalMatchingPursuit(正交匹配追踪法)` 和 :func:`orthogonal_mp(正交匹配追踪)` 
使用了OMP算法近似拟合了一个带限制的线性模型，该限制限制了模型的非0系数(例：L0范数)。

就像最小角回归一样，作为一个前向特征选择方法，正交匹配追踪法可以近似一个固定非0元素的最优
向量解:

.. math:: \text{arg\,min\,} ||y - X\gamma||_2^2 \text{ subject to } \
    ||\gamma||_0 \leq n_{nonzero\_coefs}

正交匹配追踪法也可以不用特定的非0参数元素个数做限制，它可以用别的特定函数定义其损失函数。
这个可以表示为:

.. math:: \text{arg\,min\,} ||\gamma||_0 \text{ subject to } ||y-X\gamma||_2^2 \
    \leq \text{tol}


OMP是基于每一步的贪心算法，在每一步元素都是与当前剩余量联系最为紧密的。它跟较为简单的匹配追踪
（MP）很像，但是比MP更好，好在每一步循环，剩余量是利用正交投影到之前选择的字典元素重新计算的。


.. topic:: 例子:

 * :ref:`sphx_glr_auto_examples_linear_model_plot_omp.py`

.. topic:: 参考文献:

 * http://www.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf

 * `Matching pursuits with time-frequency dictionaries
   <http://blanche.polytechnique.fr/~mallat/papiers/MallatPursuit93.pdf>`_,
   S. G. Mallat, Z. Zhang,


.. _bayesian_regression:

贝叶斯回归
===================

贝叶斯回归可以用于在预估阶段的参数正则化: 正则化参数的选择不是通过人为的选择，而是通过手边的数据调整而成。

上述过程可以通过引入 `无信息先验
<https://en.wikipedia.org/wiki/Non-informative_prior#Uninformative_priors>`__
于模型中的超参数来完成。
在`岭回归`_中使用的 :math:`\ell_{2}` 正则项相当于在:math:`w` 为高斯先验条件下，且此先验的精确度为 :math:`\lambda^{-1}`
求最大后验估计。在这里，我们没有手工调参数lambda，而是让他作为一个变量，通过数据中估计得到。


为了得到一个全概率模型，输出 :math:`y` 也被认为是关于 :math:`X w`:的高斯分布。

.. math::  p(y|X,w,\alpha) = \mathcal{N}(y|X w,\alpha)

Alpha 在这里也是作为一个变量，通过数据中估计得到.

贝叶斯回归有如下几个优点:

    - 它能根据手边数据进行改变。

    - 它能在估计过程中引入正则项

贝叶斯回归有如下缺点:

    - 它包含的推断过程是非常耗时的。


.. topic:: 参考文献

 * 一个对于贝叶斯方法的很好的介绍 C. Bishop: Pattern
   Recognition and Machine learning

 *详细介绍原创算法的一本书 `Bayesian learning for neural
   networks` by Radford M. Neal

.. _bayesian_ridge_regression:

贝叶斯岭回归
-------------------------

:class:`贝叶斯岭回归` 利用概率模型估算了上述的回归问题
先验参数 :math:`w` 通过球面高斯公式给出

.. math:: p(w|\lambda) =
    \mathcal{N}(w|0,\lambda^{-1}\bold{I_{p}})

先验参数 :math:`\alpha` 和 :math:`\lambda`一般是服从 `gamma
分布 <https://en.wikipedia.org/wiki/Gamma_distribution>`__, 这个分布与高斯成共轭先验关系。

得到的模型一般称为 *贝叶斯岭回归*， 并且这个与传统的 :class:`Ridge` 非常相似。参数 :math:`w`, :math:`\alpha` 和 :math:`\lambda` 是在调模型的时候一起估算出的。 剩下的超参数就是 gamma 分布的先验了。
:math:`\alpha` 和 :math:`\lambda` 。  它们一般被选择为 *没有信息量* 。模型参数的估计一般利用 *最大似然对数估计法* 。

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

等到模型训练完成，可以用来预测新值::

    >>> reg.predict ([[1, 0.]])
    array([ 0.50000013])


权值 :math:`w` 可以被这样访问::

    >>> reg.coef_
    array([ 0.49999993,  0.49999993])

由于贝叶斯框架的缘故，权值与 :ref:`ordinary_least_squares`产生的不太一样。
但是，贝叶斯岭回归对病态问题（ill-posed）鲁棒性要更好。

.. topic:: 例子s:

 * :ref:`sphx_glr_auto_examples_linear_model_plot_bayesian_ridge.py`

.. topic:: 参考文献

  * 更多细节可以参考 `Bayesian Interpolation
    <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.27.9072&rep=rep1&type=pdf>`_
    by MacKay, David J. C.



主动相关决策理论 - ARD
---------------------------------------

:class:`主动相关决策理论`和 `贝叶斯岭回归`_ 非常像，
但是会导致一个更加稀疏的权重 :math:`w` [1]_ [2]_。
:class:`主动相关决策理论` 提出了一个关于:math:`w`的完全不同的先验假设。具体来说，就是放弃了分布为
球状高斯模型的假设。
它采用的，是关于 :math:`w` 轴平行的椭圆高斯分布。

这就是说，每个权值 :math:`w_{i}` 是由0均值，精确度为 :math:`\lambda_{i}` 的分布中采样得到。

.. math:: p(w|\lambda) = \mathcal{N}(w|0,A^{-1})

并且 :math:`diag \; (A) = \lambda = \{\lambda_{1},...,\lambda_{p}\}`.

与 `贝叶斯岭回归`_不同， 每个 :math:`w_{i}`的坐标有它自己的
关于方差的系数 :math:`\lambda_i`。这所有的关于方差的系数
:math:`\lambda_i`  是通过超参数为:math:`\lambda_1` 和 :math:`\lambda_2` 
的gamma分布选择的。

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_ard_001.png
   :target: ../auto_examples/linear_model/plot_ard.html
   :align: center
   :scale: 50%

ARD 也被称为 *稀疏贝叶斯学习* 或
*相关向量机* [3]_ [4]_.

.. topic:: 示例:

  * :ref:`sphx_glr_auto_examples_linear_model_plot_ard.py`

.. topic:: 参考文献:

    .. [1] Christopher M. Bishop: Pattern Recognition and Machine Learning, Chapter 7.2.1

    .. [2] David Wipf and Srikantan Nagarajan: `A new view of automatic relevance determination <http://papers.nips.cc/paper/3372-a-new-view-of-automatic-relevance-determination.pdf>`_

    .. [3] Michael E. Tipping: `Sparse Bayesian Learning and the Relevance Vector Machine <http://www.jmlr.org/papers/volume1/tipping01a/tipping01a.pdf>`_

    .. [4] Tristan Fletcher: `Relevance Vector Machines explained <http://www.tristanfletcher.co.uk/RVM%20Explained.pdf>`_




.. _Logistic_regression:

逻辑回归
===================

逻辑回归，虽然名字里有 "回归" 二字，但实际上是解决分类问题的一类线性模型。在某些文献中，逻辑斯蒂回归又被称作 logit regression（logit 回归），maximum-entropy classification(MaxEnt，最大熵分类)，或 log-linear classifier（线性对数分类器）。该模型利用函数 `logistic function <https://en.wikipedia.org/wiki/Logistic_function>`_ 将单次试验（single trial）的输出转化并描述为概率。

scikit-learn 中 logistic 回归在 :class:`LogisticRegression` 类中实现了二元（binary）、一对余（one-vs-rest）及多元逻辑斯蒂回归，并带有可选的 L1 和 L2 正则化。

若视为一优化问题，带L2罚项的二分类 logistic 回归要最小化以下代价函数（cost function）：

.. math:: \underset{w, c}{min\,} \frac{1}{2}w^T w + C \sum_{i=1}^n \log(\exp(- y_i (X_i^T w + c)) + 1) .

类似地，带 L1 正则的逻辑斯蒂回归需要求解下式：

.. math:: \underset{w, c}{min\,} \|w\|_1 + C \sum_{i=1}^n \log(\exp(- y_i (X_i^T w + c)) + 1) .

在 :class:`LogisticRegression` 类中实现了这些求解器: "liblinear", "newton-cg", "lbfgs", "sag" 和 "saga"。

"liblinear" 应用了坐标下降算法（Coordinate Descent, CD），并基于 scikit-learn 内附的高性能C++库 `LIBLINEAR library <http://www.csie.ntu.edu.tw/~cjlin/liblinear/>`_ 实现。不过CD算法训练的模型不是真正意义上的多分类模型，而是基于 "one-vs-rest" 思想分解了这个优化问题，为每个类别都训练了一个二元分类器。因为实现在底层使用该求解器的 :class:`LogisticRegression` 实例对象表面上看是一个多元分类器。 :func:`sklearn.svm.l1_min_c` 可以计算使用 L1 罚项时 C 的下界，以避免模型为空（即全部特征分量的权重为零）。

"lbfgs", "sag" 和 "newton-cg" solvers （求解器）只支持 L2 罚项，对某些高维数据收敛更快。这些求解器的参数 `multi_class`设为 "multinomial" 即可训练一个真正的多元 logistic 回归 [5]_，其预测的概率比默认的 "one-vs-rest" 设定更为准确。

"sag" 求解器基于平均随机梯度下降算法（Stochastic Average Gradient descent） [6]_。在大数据集上的表现更快，大数据集指样本量大且特征数多。

"saga" solver [7]_ 是 "sag" 的一类变体，它支持非平滑（non-smooth）的 L1 正则选项 ``penalty="l1"`` 。因此对于稀疏多元逻辑回归，往往选用该求解器。

一言以蔽之，选用求解器可遵循如下规则:

=================================  =====================================
Case                               Solver
=================================  =====================================
L1正则                             	"liblinear" or "saga"
多元损失（multinomial loss）        	"lbfgs", "sag", "saga" or "newton-cg"
大数据集（`n_samples`）            	"sag" or "saga"
=================================  =====================================

"saga" 一般都是最佳的选择，但出于一些历史遗留原因默认的是 "liblinear"。

对于大数据集，还可以用 :class:`SGDClassifier` ，并使用对数损失（'log' loss）

.. topic:: 示例：

  * :ref:`sphx_glr_auto_examples_linear_model_plot_logistic_l1_l2_sparsity.py`

  * :ref:`sphx_glr_auto_examples_linear_model_plot_logistic_path.py`

  * :ref:`sphx_glr_auto_examples_linear_model_plot_logistic_multinomial.py`

  * :ref:`sphx_glr_auto_examples_linear_model_plot_sparse_logistic_regression_20newsgroups.py`

  * :ref:`sphx_glr_auto_examples_linear_model_plot_sparse_logistic_regression_mnist.py`

.. _liblinear_differences:

.. topic:: 与 liblinear 的区别:

   当 ``fit_intercept=False`` 、回归得到的 ``coef_`` 以及待预测的数据为零时， :class:`LogisticRegression` 用 ``solver=liblinear``
   及 :class:`LinearSVC` 与直接使用外部liblinear库预测得分会有差异。这是因为，
   对于 ``decision_function`` 为零的样本， :class:`LogisticRegression` 和 :class:`LinearSVC`
   将预测为负类，而liblinear预测为正类。
   注意，设定了 ``fit_intercept=False`` ，又有很多样本使得 ``decision_function`` 为零的模型，很可能会欠拟合，其表现往往比较差。建议您设置 ``fit_intercept=True`` 并增大 ``intercept_scaling``。

.. note:: **利用稀疏逻辑回归（sparse logisitic regression）进行特征选择**

   带 L1 罚项的逻辑斯蒂回归将得到稀疏模型（sparse model），相当于进行了特征选择（feature selection），详情参见 :ref:`l1_feature_selection` 。

 :class:`LogisticRegressionCV` 对逻辑斯蒂回归的实现内置了交叉验证（cross-validation），可以找出最优的参数 C。"newton-cg", "sag", "saga" 和 "lbfgs" 在高维数据上更快，因为采用了热启动（warm-starting）。在多分类设定下，若 `multi_class` 设为"ovr"，会为每类求一个最佳的C值；若 `multi_class` 设为"multinomial"，会通过交叉熵损失（cross-entropy loss）求出一个最佳 C 值。

.. topic:: 参考文献：

    .. [5] Christopher M. Bishop: Pattern Recognition and Machine Learning, Chapter 4.3.4

    .. [6] Mark Schmidt, Nicolas Le Roux, and Francis Bach: `Minimizing Finite Sums with the Stochastic Average Gradient. <https://hal.inria.fr/hal-00860051/document>`_

    .. [7] Aaron Defazio, Francis Bach, Simon Lacoste-Julien: `SAGA: A Fast Incremental Gradient Method With Support for Non-Strongly Convex Composite Objectives. <https://arxiv.org/abs/1407.0202>`_

随机梯度下降, SGD
=================================

随机梯度下降是拟合线性模型的一个简单而高效的方法。在样本量（和特征数）很大时尤为有用。
方法 ``partial_fit`` 可用于 online learning （在线学习）或基于 out-of-core learning （外存的学习）

:class:`SGDClassifier` 和 :class:`SGDRegressor` 分别用于拟合分类问题和回归问题的线性模型，可使用不同的（凸）损失函数，支持不同的罚项。
例如，设定 ``loss="log"`` ，则 :class:`SGDClassifier` 拟合一个逻辑斯蒂回归模型，而 ``loss="hinge"`` 拟合线性支持向量机(SVM).

.. topic:: 参考文献

 * :ref:`sgd`

.. _perceptron:

Perceptron（感知器）
====================

:class:`Perceptron` 是适用于 large scale learning（大规模学习）的一种简单算法。默认地，

    - 不需要设置学习率（learning rate）。

    - 不需要正则化处理。

    - 仅使用错误样本更新模型。

最后一点表明使用合页损失（hinge loss）的感知机比SGD略快，所得模型更稀疏。

.. _passive_aggressive:

Passive Aggressive Algorithms（被动攻击算法）
=============================================

被动攻击算法是大规模学习的一类算法。和感知机类似，它也不需要设置学习率，不过比感知机多出一个正则化参数 ``C`` 。

对于分类问题， :class:`PassiveAggressiveClassifier` 可设定
``loss='hinge'`` (PA-I)或 ``loss='squared_hinge'`` (PA-II)。对于回归问题，
:class:`PassiveAggressiveRegressor` 可设置
``loss='epsilon_insensitive'`` (PA-I)或
``loss='squared_epsilon_insensitive'`` (PA-II).

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

* **离群值在X上还是在y方向上**?

  ==================================== ====================================
  离群值在y方向上                  	离群值在X方向上
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

  Scikit-learn提供了三种稳健回归的预测器（estimator）:
  :ref:`RANSAC <ransac_regression>` ,
  :ref:`Theil Sen <theil_sen_regression>` 和
  :ref:`HuberRegressor <huber_regression>`

  * :ref:`HuberRegressor <huber_regression>` 一般快于
    :ref:`RANSAC <ransac_regression>` 和 :ref:`Theil Sen <theil_sen_regression>` ，
    除非样本数很大，即 ``n_samples`` >> ``n_features`` 。
    这是因为 :ref:`RANSAC <ransac_regression>` 和 :ref:`Theil Sen <theil_sen_regression>`
    都是基于数据的较小子集进行拟合。但使用默认参数时， :ref:`Theil Sen <theil_sen_regression>`
    和 :ref:`RANSAC <ransac_regression>` 可能不如
    :ref:`HuberRegressor <huber_regression>` 鲁棒。

  * :ref:`RANSAC <ransac_regression>` 比 :ref:`Theil Sen <theil_sen_regression>` 更快，在样本数量上的伸缩性（适应性）更好。

  * :ref:`RANSAC <ransac_regression>` 能更好地处理y方向的大值离群点（通常情况下）。

  * :ref:`Theil Sen <theil_sen_regression>` 能更好地处理x方向中等大小的离群点，但在高维情况下无法保证这一特点。

 实在决定不了的话，请使用 :ref:`RANSAC <ransac_regression>`

.. _ransac_regression:

RANSAC： 随机抽样一致性算法（RANdom SAmple Consensus）
------------------------------------------------------

随机抽样一致性算法（RANdom SAmple Consensus, RANSAC）利用全体数据中局内点（inliers）的一个随机子集拟合模型。

RANSAC是一种非确定性算法，以一定概率输出一个可能的合理结果，依赖于迭代次数（参数 `max_trials` ）。这种算法主要解决线性或非线性回归问题，在计算机视觉摄影测量领域尤为流行。

算法从全体样本输入中分出一个局内点集合，全体样本可能由于测量错误或对数据的假设错误而含有噪点、离群点。最终的模型仅从这个局内点集合中得出。

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_ransac_001.png
   :target: ../auto_examples/linear_model/plot_ransac.html
   :align: center
   :scale: 50%

算法细节
^^^^^^^^^^^^^^^^^^^^^^^^

每轮迭代执行以下步骤:

1. 从原始数据中抽样 ``min_samples`` 数量的随机样本，检查数据是否合法（见 ``is_data_valid`` ）.
2. 用一个随机子集拟合模型（ ``base_estimator.fit`` ）。检查模型是否合法（见 ``is_model_valid`` ）。
3. 计算预测模型的残差（residual），将全体数据分成局内点和离群点（ ``base_estimator.predict(X) - y`` ）
 - 绝对残差小于 ``residual_threshold`` 的全体数据认为是局内点。
4. 若局内点样本数最大，保存当前模型为最佳模型。以免当前模型离群点数量恰好相等（而出现未定义情况），规定仅当数值大于当前最值时认为是最佳模型。

上述步骤或者迭代到最大次数（ ``max_trials`` ），或者某些终止条件满足时停下（见 ``stop_n_inliers`` 和 ``stop_score`` )。最终模型由之前确定的最佳模型的局内点样本（一致性集合，consensus
set）预测。

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

Theil-Sen 预估器: 广义中值估计
-----------------------------------------------------------------------------------------

:class:`TheilSenRegressor` 估计器：使用中位数在多个维度推广，因此对多维离散值是有帮助，但问题是，随着维数的增加，估计器的准确性在迅速下降。准确性的丢失，导致在高维上的估计值比不上普通的最小二乘法。

.. topic:: 示例:

  * :ref:`sphx_glr_auto_examples_linear_model_plot_theilsen.py`
  * :ref:`sphx_glr_auto_examples_linear_model_plot_robust_fit.py`

.. topic:: 参考文献:

 * https://en.wikipedia.org/wiki/Theil%E2%80%93Sen_estimator

Theoretical considerations（理论考虑）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`TheilSenRegressor` 媲美 :ref:`Ordinary Least Squares (OLS) <ordinary_least_squares>` （普通最小二乘法（OLS））渐近效率和无偏估计。在对比 OLS, Theil-Sen 是一种非参数方法，这意味着它没有对底层数据的分布假设。由于 Theil-Sen 是基于中位数的估计，它是更适合的对损坏的数据。在单变量的设置，Theil-Sen 在一个简单的线性回归，这意味着它可以容忍任意损坏的数据高达 29.3% 的情况下，约 29.3% 的一个崩溃点。

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_theilsen_001.png
   :target: ../auto_examples/linear_model/plot_theilsen.html
   :align: center
   :scale: 50%

在 scikit-learn 中  :class:`TheilSenRegressor` 实施如下的学习推广到多元线性回归模型 [#f1]_ 利用空间中这是一个概括的中位数多维度 [#f2]_ 。

在时间复杂度和空间复杂度，根据 Theil-Sen 量表

.. math::
    \binom{n_{samples}}{n_{subsamples}}

这使得它不适用于大量样本和特征的问题。因此，可以选择一个亚群的大小来限制时间和空间复杂度，只考虑所有可能组合的随机子集。

.. topic:: 示例:

  * :ref:`sphx_glr_auto_examples_linear_model_plot_theilsen.py`

.. topic:: 参考文献:

    .. [#f1] Xin Dang, Hanxiang Peng, Xueqin Wang and Heping Zhang: `Theil-Sen Estimators in a Multiple Linear Regression Model. <http://home.olemiss.edu/~xdang/papers/MTSE.pdf>`_

    .. [#f2] T. Kärkkäinen and S. Äyrämö: `On Computation of Spatial Median for Robust Data Mining. <http://users.jyu.fi/~samiayr/pdf/ayramo_eurogen05.pdf>`_

.. _huber_regression:

Huber Regression（Huber 回归）
------------------------------

:class:`HuberRegressor` 不同，因为它适用于 :class:`Ridge` 损耗的样品被分类为离群值。如果这个样品的绝对误差小于某一阈值，样品就分为一层。
它不同于 :class:`TheilSenRegressor` 和 :class:`RANSACRegressor` 因为它无法忽略对离群值的影响，但对它们的权重较小。

.. figure:: /auto_examples/linear_model/images/sphx_glr_plot_huber_vs_ridge_001.png
   :target: ../auto_examples/linear_model/plot_huber_vs_ridge.html
   :align: center
   :scale: 50%

这个 :class:`HuberRegressor` 最小化损失函数是由

.. math::

  \underset{w, \sigma}{min\,} {\sum_{i=1}^n\left(\sigma + H_m\left(\frac{X_{i}w - y_{i}}{\sigma}\right)\sigma\right) + \alpha {||w||_2}^2}

其中

.. math::

  H_m(z) = \begin{cases}
         z^2, & \text {if } |z| < \epsilon, \\
         2\epsilon|z| - \epsilon^2, & \text{otherwise}
  \end{cases}

建议设置参数 ``epsilon`` 为 1.35 以实现 95% 统计效率。

Notes
-----
The :class:`HuberRegressor` differs from using :class:`SGDRegressor` with loss set to `huber`
in the following ways.

- :class:`HuberRegressor` is scaling invariant. Once ``epsilon`` is set, scaling ``X`` and ``y``
  down or up by different values would produce the same robustness to outliers as before.
  as compared to :class:`SGDRegressor` where ``epsilon`` has to be set again when ``X`` and ``y`` are
  scaled.

- :class:`HuberRegressor` should be more efficient to use on data with small number of
  samples while :class:`SGDRegressor` needs a number of passes on the training data to
  produce the same robustness.

.. topic:: Examples:

  * :ref:`sphx_glr_auto_examples_linear_model_plot_huber_vs_ridge.py`

.. topic:: References:

  * Peter J. Huber, Elvezio M. Ronchetti: Robust Statistics, Concomitant scale estimates, pg 172

Also, this estimator is different from the R implementation of Robust Regression
(http://www.ats.ucla.edu/stat/r/dae/rreg.htm) because the R implementation does a weighted least
squares implementation with weights given to each sample on the basis of how much the residual is
greater than a certain threshold.

.. _polynomial_regression:

Polynomial regression: extending linear models with basis functions
===================================================================

.. currentmodule:: sklearn.preprocessing

One common pattern within machine learning is to use linear models trained
on nonlinear functions of the data.  This approach maintains the generally
fast performance of linear methods, while allowing them to fit a much wider
range of data.

For example, a simple linear regression can be extended by constructing
**polynomial features** from the coefficients.  In the standard linear
regression case, you might have a model that looks like this for
two-dimensional data:

.. math::    \hat{y}(w, x) = w_0 + w_1 x_1 + w_2 x_2

If we want to fit a paraboloid to the data instead of a plane, we can combine
the features in second-order polynomials, so that the model looks like this:

.. math::    \hat{y}(w, x) = w_0 + w_1 x_1 + w_2 x_2 + w_3 x_1 x_2 + w_4 x_1^2 + w_5 x_2^2

The (sometimes surprising) observation is that this is *still a linear model*:
to see this, imagine creating a new variable

.. math::  z = [x_1, x_2, x_1 x_2, x_1^2, x_2^2]

With this re-labeling of the data, our problem can be written

.. math::    \hat{y}(w, x) = w_0 + w_1 z_1 + w_2 z_2 + w_3 z_3 + w_4 z_4 + w_5 z_5

We see that the resulting *polynomial regression* is in the same class of
linear models we'd considered above (i.e. the model is linear in :math:`w`)
and can be solved by the same techniques.  By considering linear fits within
a higher-dimensional space built with these basis functions, the model has the
flexibility to fit a much broader range of data.

Here is an example of applying this idea to one-dimensional data, using
polynomial features of varying degrees:

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_polynomial_interpolation_001.png
   :target: ../auto_examples/linear_model/plot_polynomial_interpolation.html
   :align: center
   :scale: 50%

This figure is created using the :class:`PolynomialFeatures` preprocessor.
This preprocessor transforms an input data matrix into a new data matrix
of a given degree.  It can be used as follows::

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

The features of ``X`` have been transformed from :math:`[x_1, x_2]` to
:math:`[1, x_1, x_2, x_1^2, x_1 x_2, x_2^2]`, and can now be used within
any linear model.

This sort of preprocessing can be streamlined with the
:ref:`Pipeline <pipeline>` tools. A single object representing a simple
polynomial regression can be created and used as follows::

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

The linear model trained on polynomial features is able to exactly recover
the input polynomial coefficients.

In some cases it's not necessary to include higher powers of any single feature,
but only the so-called *interaction features*
that multiply together at most :math:`d` distinct features.
These can be gotten from :class:`PolynomialFeatures` with the setting
``interaction_only=True``.

For example, when dealing with boolean features,
:math:`x_i^n = x_i` for all :math:`n` and is therefore useless;
but :math:`x_i x_j` represents the conjunction of two booleans.
This way, we can solve the XOR problem with a linear classifier::

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

And the classifier "predictions" are perfect::

    >>> clf.predict(X)
    array([0, 1, 1, 0])
    >>> clf.score(X, y)
    1.0
