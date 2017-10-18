.. _lda_qda:

=============================================================
线性和二次判别分析
=============================================================

.. currentmodule:: sklearn

Linear Discriminant Analysis（线性判别分析）(:class:`discriminant_analysis.LinearDiscriminantAnalysis`) 和 Quadratic Discriminant Analysis （二次判别分析）(:class:`discriminant_analysis.QuadraticDiscriminantAnalysis`) 是两个经典的分类器。
正如他们名字所描述的那样，他们分别适合用于线性和二次方程的决策。

这些分类器十分具有魅力，因为他们拥有十分便于计算的封闭式解决方案，即其天生的多分类特性，已经被证明在实际中运行效果十分好，并且不需要再次调参。

.. |ldaqda| image:: ../auto_examples/classification/images/sphx_glr_plot_lda_qda_001.png
        :target: ../auto_examples/classification/plot_lda_qda.html
        :scale: 80

.. centered:: |ldaqda|

以下这些图像展示了 Linear Discriminant Analysis （线性判别分析）以及 Quadratic Discriminant Analysis （二次判别分析）的决策边界。其中，最底行阐述了线性判别分析只能学习线性边界，
而二次判别分析则可以学习二次函数的边界，因此它会相对而言更加灵活。

.. topic:: 示例:

    :ref:`sphx_glr_auto_examples_classification_plot_lda_qda.py`: 在综合的数据基础上对比LDA和QDA

使用线性判别分析实现降维
====================================================================================

:class:`discriminant_analysis.LinearDiscriminantAnalysis` 可以通过给予包含了最大化不同类别间距的方向的线性子空间（subspace）投放输入数据，
从而用来执行监督下的降维。输出的维度必然会比原来的类别数量更少的。因此它是总体而言十分强大的降维方式，同样也仅仅在多分类环境下才会起作用。

它具体是执行在 :func:`discriminant_analysis.LinearDiscriminantAnalysis.transform` 中.关于维度的数量可以通过n_components来调节 .
值得注意的是，这个参数不会对 :func:`discriminant_analysis.LinearDiscriminantAnalysis.fit` 或者 :func:`discriminant_analysis.LinearDiscriminantAnalysis.predict` 产生影响.

.. topic:: 示例:

    :ref:`sphx_glr_auto_examples_decomposition_plot_pca_vs_lda.py`: 在 Iris 数据集对比了 LDA 和 PCA 之间的降维差异

LDA 和 QDA 分类器的数学公式
================================================================================

LDA 和 QDA 都是源于简单的概率模型，这些模型对于每一个类别 :math:`k` 的类别相关分布 :math:`P(X|y=k)` 都可以通过贝叶斯定理所获得。

.. math::
    P(y=k | X) = \frac{P(X | y=k) P(y=k)}{P(X)} = \frac{P(X | y=k) P(y = k)}{ \sum_{l} P(X | y=l) \cdot P(y=l)}

并且我们选择能够最大化条件概率的类别 :math:`k`.

更详细地，对于线性以及二次判别分析， :math:`P(X|y)` 被塑造成一个多变量的高斯分布密度:

.. math:: p(X | y=k) = \frac{1}{(2\pi)^n |\Sigma_k|^{1/2}}\exp\left(-\frac{1}{2} (X-\mu_k)^t \Sigma_k^{-1} (X-\mu_k)\right)


为了使用该模型作为分类器使用，我们需要通过训练集数据预测更重要的类别 :math:`P(y=k)` （通过每个类 :math:`k` 的实例的概率预测）
类别均值 :math:`\mu_k` （通过经验主义的样本类均值预测）以及协方差矩阵（由经验主义样本类协方差矩阵或常规的预测器：观察收缩损失的部分）.

关于 LDA 的案例，高斯被看作是共享相同协方差矩阵：:math:`\Sigma_k = \Sigma` for all :math:`k`。这会导致线性决策显示介于比较对数概率之比 :math:`\log[P(y=k | X) / P(y=l | X)]` 之间。


.. math::
   \log\left(\frac{P(y=k|X)}{P(y=l | X)}\right) = 0 \Leftrightarrow (\mu_k-\mu_l)\Sigma^{-1} X = \frac{1}{2} (\mu_k^t \Sigma^{-1} \mu_k - \mu_l^t \Sigma^{-1} \mu_l)

对于 QDA 而言，没有关于高斯协方差矩阵 :math:`\Sigma_k` 的假设，可以通过查看 [#1]_ 获取更多信息.

.. note:: **与高斯朴素贝叶斯的关系**

      如果在QDA模型中假设协方差矩阵是对角的，那么在每个类别中的输入数据则被假定是相关依赖的。
      而且结果分类器会和高斯朴素贝叶斯分类器 :class:`naive_bayes.GaussianNB` 相同。

LDA 的降维数学公式
=============================================================================

为了理解 LDA 在降维上的应用，它对于进行 LDA 分类的几何重构是十分有用的。我们用 :math:`K` 表示目标类别的总数。
由于在 LDA 中我们假设所有类别都有相同预测的协方差 :math:`\Sigma` ,我们可重新调节数据从而让让协方差相同。

.. math:: X^* = D^{-1/2}U^t X\text{ with }\Sigma = UDU^t


在缩放后可以分类数据点和找到离数据点最近的欧式距离相同的预测类别均值。但是它可以在投影到 :math:`K-1` 个由所有 :math:`\mu^*_k` 个类生成的仿射子空间
 :math:`H_K` 之后被完成。这也表明，LDA 分类器中存在一个利用线性投影到 :math:`K-1` 个维度空间的降维工具。

我们可以通过投影到可以最大化 :math:`\mu^*_k` 的方差的线性子空间 :math:`H_L` 以更多地减少维度，直到一个选定的 :math:`L` 值
（实际上，我们正在做一个类 PCA 的形式为了实现转换类均值 :math:`\mu^*_k`）
:func:`discriminant_analysis.LinearDiscriminantAnalysis.transform` 方法. 详情参考
[#1]_ 。

Shrinkage（收缩）
=================

收缩是一个在训练样本数量相比特征而言很小的情况下可以提升预测（准确性）的协方差矩阵。
在这个情况下，经验样本协方差是一个很差的预测器。LDA 收缩可以通过设置 :class:`discriminant_analysis.LinearDiscriminantAnalysis` 类的 ``shrinkage`` 参数为 'auto' 以得到应用。

``shrinkage`` parameter （收缩参数）的值同样也可以手动被设置为 0-1 之间。特别地，0 值对应着没有收缩（这意味着经验协方差矩阵将会被使用），
而 1 值则对应着完全使用收缩（意味着方差的对角矩阵将被当作协方差矩阵的估计）。设置该参数在两个极端值之间会估计一个（特定的）协方差矩阵的收缩形式

.. |shrinkage| image:: ../auto_examples/classification/images/sphx_glr_plot_lda_001.png
        :target: ../auto_examples/classification/plot_lda.html
        :scale: 75

.. centered:: |shrinkage|


预估算法
=================================

默认的 solver 是 'svd'。它可以同时执行分类以及 transform (转换),而且它不会依赖于协方差矩阵的计算（结果）。这在特征数量特别大的时候就显得十分具有优势。然而，'svd' solver 无法与 shrinkage （收缩）同时使用。

'lsqr' solver 则是一个高效的算法，它仅仅只能用于分类使用，而且它支持 shrinkage （收缩）。

'eigen'（特征） solver 是基于 class scatter （类散度）与 class scatter ratio （类内散射比）之间的优化。
它既可以被用于分类以及 transform （转换），此外它还同时支持收缩。然而，该解决方案需要计算协方差矩阵，因此它可能不适用于具有大量特征的情况。

.. topic:: Examples:

    :ref:`sphx_glr_auto_examples_classification_plot_lda.py`: Comparison of LDA classifiers
    with and without shrinkage.

.. topic:: References:

   .. [#1] "The Elements of Statistical Learning", Hastie T., Tibshirani R.,
      Friedman J., Section 4.3, p.106-119, 2008.

   .. [#2] Ledoit O, Wolf M. Honey, I Shrunk the Sample Covariance Matrix.
      The Journal of Portfolio Management 30(4), 110-119, 2004.
