.. _mixture:

.. _gmm:

=======================
高斯混合模型
=======================

.. currentmodule:: sklearn.mixture

``sklearn.mixture`` 是一个可以学习高斯混合模型（支持 diagonal（即对角协方差矩阵），spherical（球形，即使用相同的方差值），
tied（连接，是指所有的马尔可夫隐含状态使用相同的完全协方差矩阵），full（完全协方差矩阵）这几种协方差矩阵）的包，
它对数据进行抽样，并且根据数据进行估计。同时也提供了一些有助于决定合适的分量
（component，组件（这里理解为分量）,每一个高斯分布称为一个分量）的数量的设施。

 .. figure:: ../auto_examples/mixture/images/sphx_glr_plot_gmm_pdf_001.png
   :target: ../auto_examples/mixture/plot_gmm_pdf.html
   :align: center
   :scale: 50%

   **Two-component Gaussian mixture model:** *data points, and equi-probability
   surfaces of the model.*

高斯混合模型是一个假设所有的数据点都是生成于一个混合的有限数量的并且未知参数的高斯分布的概率模型。
人们可以将混合模型看作是推广k-means聚类算法，以纳入关于数据的协方差结构以及潜在高斯中心的信息。

对应不同的估算策略，Scikit-learn 实现了不同的类来估算高斯混合模型。
详细描述如下：

高斯混合
================

  :class:`GaussianMixture` 对象实现了用来拟合高斯混合模型的
  :ref:`期望最大化 <expectation_maximization>` (EM)
算法。它还可以为多变量模型画置信椭圆，和计算BIC（Bayesian Information Criterion，贝叶斯信息准则）
来评估数据中聚类的数量。
  :meth:`GaussianMixture.fit` 方法提供了从训练集学习高斯混合模型的方法。 
给定测试集，它可以分配给每个样本最有可能属于使用 :meth:`GaussianMixture.predict` 方法的高斯分布。

..
    Alternatively, the probability of each
    sample belonging to the various Gaussians may be retrieved using the
    :meth:`GaussianMixture.predict_proba` method.

:class:`GaussianMixture` 自带了不同的选项来约束不同估计类的协方差：spherical（球形，即使用相同的方差值）, 
diagonal（即对角协方差）, tied（连接，是指所有的马尔可夫隐含状态使用相同的完全协方差） or 
full（完全协方差）covariance .

.. figure:: ../auto_examples/mixture/images/sphx_glr_plot_gmm_covariances_001.png
   :target: ../auto_examples/mixture/plot_gmm_covariances.html
   :align: center
   :scale: 75%

.. topic:: 示例:

    * 一个在虹膜数据集上用高斯混合作为聚类，请查阅 :ref:`sphx_glr_auto_examples_mixture_plot_gmm_covariances.py` 

    * 一个绘制密度估计的例子，请查阅 :ref:`sphx_glr_auto_examples_mixture_plot_gmm_pdf.py`

优缺点 :class:`GaussianMixture`
-----------------------------------------------

优点
....

:速度: 是学习混合模型最快的算法。

:不可知论: 这个算法仅仅只是最大化可能性，并不能使均值偏向于0，或是使聚类大小偏向于
  可能适用或者可能不适用的特殊的结构。

缺点
....

:奇点: 当每个混合没有足够的点时，估算协方差变得困难起来，同时算法会发散并且找具有无限可能的解，
   除非人为地对协方差进行正则化。
   
:组件的数量: 这个算法将会总是用所有它能用的分量（component，每一个高斯分布称为一个分量），
   所以在没有外部线索的情况下需要留出数据（held-out data，将数据集分为两个互斥的集合，分别是训练集和测试集），
   或者信息理论标准来决定用多少组件。

选择经典高斯混合模型中的分量数
------------------------------------------------------

BIC（Bayesian Information Criterion，贝叶斯信息准则）可以用来高效地选择高斯混合的分量数。
理论上，它可以恢复正确的分量数仅当在近似状态下（即如果有大量数据可用，并且假设这些数据实际上是一个混合高斯模型
生成的）。注意：使用 :ref:`变分贝叶斯高斯混合 <bgmm>`避免高斯混合模型的分量数的规范。

.. figure:: ../auto_examples/mixture/images/sphx_glr_plot_gmm_selection_001.png
   :target: ../auto_examples/mixture/plot_gmm_selection.html
   :align: center
   :scale: 50%
.. topic:: 示例:

    * 一个用典型的高斯混合进行的模型选择的例子，请查阅 :ref:`sphx_glr_auto_examples_mixture_plot_gmm_selection.py` 
.. _expectation_maximization:

估计算法期望最大化（EM）
-----------------------------------------------

在从无标签的数据中学习高斯混合模型主要的困难在于通常不知道哪个点来自哪个潜在分量
（如果可以获取到这些信息，就可以很容易将每个独立的高斯分布去拟合数据集的每个点）。
`期望最大化（Expectation-maximization，EM）
<https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm>`_
是一个有根据的统计算法，通过迭代来解决这个问题。首先，假设一个随机分量
（选择随机地一个中心点，可以用k-means算法得到，或者甚至就直接地随便分布在原点周围），
并且为每个点计算由模型的每个分量生成的概率。然后，调整参数以最大化给出这些分配的数据的可能性。
重复这个过程保证总是收敛到局部最优解。

.. _bgmm:

变分贝叶斯高斯混合
=====================================

 :class:`BayesianGaussianMixture`对象实现了具有变分的高斯混合模型的变体推理算法。
 这个API和 :class:`GaussianMixture`相似。

.. _variational_inference:

估计算法: variational inference
---------------------------------------------

变分推理是期望最大化（EM）的扩展，它最大化模型证据（包括先验）的下界，而不是数据似然。
变分方法的原理与期望最大化相同（二者都是迭代算法，在寻找由混合产生的每个点的概率和
拟合所分配的点之间交替），但是变分方法通过整合先验分布信息来增加正则化。
这避免了期望最大化解决方案中常出现的奇异性，但是也对模型带来了微小的偏差。
推理通常明显较慢，但通常不会慢到无法使用。

由于它的贝叶斯性质，变分算法比预期最大化（EM）需要更多的超参数，其中最重要的就是
浓度参数 ``weight_concentration_prior`` 。指定一个低浓度先验（concentration prior)
将会使模型将大部分的权重放在少数分量上，其余分量的权重则趋近0。而高浓度先验（concentration prior)
将会激活大量在混合中的分量。

 :class:`BayesianGaussianMixture`类的参数实现提出了两种权重分布先验：
 一种是狄利克雷分布（Dirichlet distribution）有限混合模型，
 另一种是狄利克雷过程（Dirichlet Process）无限混合模型。
 在实际应用上，狄利克雷过程推理算法（Dirichlet Process inference algorithm）
 是近似的，并且使用截尾分布（truncated distribution）与固定最大分量数（称之为Stick-breaking representation）。
 使用的分量数实际上总是取决于数据。

下图比较了不同类型的权重浓度先验（weight concentration prior，参数 ``weight_concentration_prior_type`` ）
不同的 ``weight_concentration_prior`` 值获得的结果。
在这里，我们可以发现``weight_concentration_prior``参数的值对获得的有效的激活的分量数有很大影响。
我们也能注意到当先验是'dirichlet_distribution'类型时，大的浓度权重先验（concentration weight prior）
会导致更均匀的权重，然而'dirichlet_process'类型（默认类型）却不是这样。

.. |plot_bgmm| image:: ../auto_examples/mixture/images/sphx_glr_plot_concentration_prior_001.png
   :target: ../auto_examples/mixture/plot_concentration_prior.html
   :scale: 48%

.. |plot_dpgmm| image:: ../auto_examples/mixture/images/sphx_glr_plot_concentration_prior_002.png
   :target: ../auto_examples/mixture/plot_concentration_prior.html
   :scale: 48%

.. centered:: |plot_bgmm| |plot_dpgmm|

下面的例子将具有固定数量的分量的高斯混合模型与
狄利克雷过程先验（Dirichlet process prior）的变分高斯混合模型进行比较。
这里，一个典型的高斯混合在由2个聚类组成的数据集有5个分量。
我们可以看到，具有狄利克雷过程的变分高斯混合可以将自身限制在2个分量，
而高斯混合必须由用户先验设置的固定数量的分量来拟合数据。
在例子中，用户选择了 ``n_components=5`` ，这与真正的toy dataset（有noise的数据）的生成分量不符。
稍微观察就能注意到，狄利克雷过程先验（Dirichlet process prior）的变分高斯混合模型可以采取保守的
立场，仅仅拟合一个分量。

.. figure:: ../auto_examples/mixture/images/sphx_glr_plot_gmm_001.png
   :target: ../auto_examples/mixture/plot_gmm.html
   :align: center
   :scale: 70%


在下图中，我们用高斯混合在拟合一个并不well-depicted的数据集。
调整`BayesianGaussianMixture`类的参数``weight_concentration_prior``，
这个参数决定了用来拟合数据的分量数。我们在最后两个图上展示了从两个混合（resulting mixtures）
产生的随机抽样。

.. figure:: ../auto_examples/mixture/images/sphx_glr_plot_gmm_sin_001.png
   :target: ../auto_examples/mixture/plot_gmm_sin.html
   :align: center
   :scale: 65%



.. topic:: 示例:

    * 一个用 :class:`GaussianMixture` 和 :class:`BayesianGaussianMixture` 绘制置信椭圆体的例子，
      请查阅 :ref:`sphx_glr_auto_examples_mixture_plot_gmm.py` 

    * :ref:`sphx_glr_auto_examples_mixture_plot_gmm_sin.py` 这个例子展示了用 :class:`GaussianMixture`
      和 :class:`BayesianGaussianMixture` 来拟合正弦波。

    * 一个使用不同的 ``weight_concentration_prior_type`` 用以不同的 ``weight_concentration_prior``
      参数值的:class:`BayesianGaussianMixture` 来绘制置信椭圆体的例子。
      请查阅:ref:`sphx_glr_auto_examples_mixture_plot_concentration_prior.py`


 :class:`BayesianGaussianMixture` 下的变分推理的优缺点
----------------------------------------------------------------------------

优点
.....

:自动选择: 当 ``weight_concentration_prior`` 足够小以及
   ``n_components`` 比模型需要的更大，变分贝叶斯混合模型有一个性质可以让一些混合权重值趋近0。
   这让模型可以自动选择合适的有效分量数。仅仅需要提供上限数量。但是请注意，“理想”的激活分量数
   适用于特定的程序，并且在数据挖掘设置中通常不明确。

:对参数数量的敏感度较低: 与总是用尽可以用的分量，因而将为不同数量的组件产生不同的解决方案有限模型不同，
   变分推理狄利克雷过程先验（Dirichlet process prior）变分推理(``weight_concentration_prior_type='dirichlet_process'``)
   改变参数后并不会改变太多，使之更稳定和更少的调谐（tuning）。

:正则化: 由于结合了先验信息，变分解的病理特征（pathological special cases）
   少于期望最大化（EM）的解。


缺点
.....
:速度: 变分推理所需要的额外参数化使推理速度变慢，但是并没有慢很多。

:超参数: 这个算法需要一个可能需要通过交叉验证进行实验调整的额外的超参数

:偏差: 推理算法存在许多隐含的偏差（如果用狄利克雷过程也会有），
   每当这些偏差和数据之间不匹配时，用有限模型可能可以拟合更好的模型。

.. _dirichlet_process:

The Dirichlet Process
-----------------------------


这里我们描述了Dirichlet过程混合的变分推理算法。Dirichlet过程是在*具有
无限大，无限制的分区数的聚类*上的先验概率分布。
相比有限高斯混合模型，变分技术让我们在推理时间几乎没有惩罚（penalty）的情况下
纳入了高斯混合模型的先验结构。

一个重要的问题是Dirichlet过程是如何实现用无限的，无限制的簇数，并且结果仍然是一致的。
完整的解释并不符合本文档，但是你可以看这里`stick breaking process
<https://en.wikipedia.org/wiki/Dirichlet_process#The_stick-breaking_process>`_
来帮助你理解它。stick breaking（折棍）过程是Dirichlet过程的衍生。我们每次从一个单位长度的
stick开始，且每一步都折断剩下的一部分。每次，我们把每个stick的长度关联到落入一组混合的点的比例。
最后，为了表示无限混合，我们关联最后每个stick的剩下的部分到没有落入其他组的点的比例。
每段的长度是随机变量，概率与浓度参数（concentration parameter）成比例。较小的浓度值
将单位长度分成较大的stick段（即定义更集中的分布）。较高的浓度值将生成更小的stick段
（即增加非零权重的分量数）。

Dirichlet过程的变分推理技术仍然可以运用在对该无限混合模型进行有限近似，
而不必事先需要指定用户想要的分量数，用户只需要指定浓度参数和混合分量数的上界
（假定上界高于“真实”的分量数，仅仅影响算法复杂度，而不是实际上使用的分量数）。
