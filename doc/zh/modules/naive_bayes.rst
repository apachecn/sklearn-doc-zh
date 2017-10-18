.. _naive_bayes:

===========
朴素贝叶斯
===========

.. currentmodule:: sklearn.naive_bayes


朴素贝叶斯方法是一组基于贝叶斯定理，即“简单”地假设特征两两相互独立的有监督学习算法。
给定一个类别 :math:`y` 和一个从 :math:`x_1` 到 :math:`x_n` 的相关的特征向量，
贝叶斯定理阐述了以下关系:

.. math::

   P(y \mid x_1, \dots, x_n) = \frac{P(y) P(x_1, \dots x_n \mid y)}
                                    {P(x_1, \dots, x_n)}

使用朴素贝叶斯独立假设，即:

.. math::

   P(x_i | y, x_1, \dots, x_{i-1}, x_{i+1}, \dots, x_n) = P(x_i | y) ,

 对于所有的 :math`i` ，这个关系可以简化为

 .. math::

   P(y \mid x_1, \dots, x_n) = \frac{P(y) \prod_{i=1}^{n} P(x_i \mid y)}
                                    {P(x_1, \dots, x_n)}

由于在给定的输入中 :math:`P(x_1, \dots, x_n)` 是一个常量，我们使用下面的分类规则:

.. math::

   P(y \mid x_1, \dots, x_n) \propto P(y) \prod_{i=1}^{n} P(x_i \mid y)

   \Downarrow

   \hat{y} = \arg\max_y P(y) \prod_{i=1}^{n} P(x_i \mid y),
   
我们可以使用最大后验概率(Maximum A Posteriori, MAP)来估计 :math:`P(y)` 和 :math:`P(x_i \mid y)` ; 前者即为训练集中类别 :math:`y` 的相对频率。
 
不同的朴素贝叶斯分类器的差异大部分来自于处理 :math:`P(x_i \mid y)` 分布时的所做的假设不同。
 

尽管其假设过于简单，朴素贝叶斯在很多实际情况下工作得很好，特别是文档分类和垃圾邮件过滤。这些工作都要求
一个小的训练集来估计必需参数。(至于为什么朴素贝叶斯表现得好的理论原因和它适用于哪些类型的数据，请参见下面的参考。)
 
相比于其他更复杂的方法，朴素贝叶斯学习器和分类器非常快。
分类条件分布的解耦意味着可以独立单独地把每个特征视为一维分布来估计。这样反过来又减轻了维度灾难带来的问题。
 
反过来说，尽管朴素贝叶斯被认为是一种相当不错的分类器，但却不是好的估计器(estimator)，所以不能太过于重视从 ``predict_proba`` 输出的概率。
 
.. topic:: 参考文献:

 * H. Zhang (2004). `The optimality of Naive Bayes.
   <http://www.cs.unb.ca/~hzhang/publications/FLAIRS04ZhangH.pdf>`_
   Proc. FLAIRS.

.. _gaussian_naive_bayes:

高斯朴素贝叶斯
--------------------

:class:`GaussianNB` 实现了运用于分类的高斯朴素贝叶斯算法。特征的可能性(即概率)假设为高斯分布:

.. math::

   P(x_i \mid y) &= \frac{1}{\sqrt{2\pi\sigma^2_y}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma^2_y}\right)
   
参数 :math:`\sigma_y` 和 :math:`\mu_y` 使用最大似然法估计。
 
    >>> from sklearn import datasets
    >>> iris = datasets.load_iris()
    >>> from sklearn.naive_bayes import GaussianNB
    >>> gnb = GaussianNB()
    >>> y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
    >>> print("Number of mislabeled points out of a total %d points : %d"
    ...       % (iris.data.shape[0],(iris.target != y_pred).sum()))
    Number of mislabeled points out of a total 150 points : 6
	
.. _multinomial_naive_bayes:

多项分布朴素贝叶斯
-----------------------

:class:`MultinomialNB` 实现了多项分布数据的朴素贝叶斯算法，是用于文本分类(这个领域中数据往往以词向量表示，尽管在实践中 tf-idf 向量在预测时表现良好)的两大经典朴素贝叶斯变形之一。
分布参数由每类 :math:`y` 的 :math:`\theta_y = (\theta_{y1},\ldots,\theta_{yn})` 向量决定， 式中 :math:`n` 是特征的数量(对于文本分类，是词汇量的大小) :math:`\theta_{yi}` 是样本中属于类 :math:`y` 中特征 :math:`i` 概率 :math:`P(x_i \mid y)` 。

参数 :math:`\theta_y` 使用平滑过的最大似然估计法来估计，即相对频率计数:

.. math::

    \hat{\theta}_{yi} = \frac{ N_{yi} + \alpha}{N_y + \alpha n}
	
式中 :math:`N_{yi} = \sum_{x \in T} x_i` 是 训练集 :math:`T` 中 特征 :math:`i` 在类 :math:`y` 中出现的次数，
 :math:`N_{y} = \sum_{i=1}^{|T|} N_{yi}`  是类:math:`y`中出现所有特征的计数总和。
 
 先验平滑因子 :math:`\alpha \ge 0` 应用于在学习样本中没有出现的特征，以防在将来的计算中出现0概率输出。
 把  :math:`\alpha = 1` 被称为拉普拉斯平滑(Lapalce smoothing)，
 而 :math:`\alpha = 1` 被称为利德斯通(Lidstone smoothing)。
 
 
 .. _bernoulli_naive_bayes:

伯努利朴素贝叶斯
---------------------

:class:`BernoulliNB` 实现了用于多重伯努利分布数据的朴素贝叶斯训练和分类算法，即有多个特征，但每个特征
都假设是一个二元 (Bernoulli, boolean) 变量。
因此，这类算法要求样本以二元值特征向量表示；如果样本含有其他类型的数据， 一个 ``BernoulliNB`` 实例会将其二值化(取决于 ``binarize`` 参数)。

伯努利朴素贝叶斯的决策规则基于

.. math::

    P(x_i \mid y) = P(i \mid y) x_i + (1 - P(i \mid y)) (1 - x_i)
	
与多项分布朴素贝叶斯的规则不同
伯努利朴素贝叶斯明确地惩罚类 :math:`y` 中没有出现作为预测因子的特征 :math:`i`，而多项分布分布朴素贝叶斯只是简单地忽略没出现的特征。

在文本分类的例子中，词频向量(word occurrence vectors)(而非词数向量(word count vectors))可能用于训练和用于这个分类器。 ``BernoulliNB`` 在一些数据集上可能表现得更好，特别是那些更短的文档。
如果时间允许，建议对两个模型都进行评估。

.. topic:: References:

 * C.D. Manning, P. Raghavan and H. Schütze (2008). Introduction to
   Information Retrieval. Cambridge University Press, pp. 234-265.

 * A. McCallum and K. Nigam (1998).
   `A comparison of event models for Naive Bayes text classification.
   <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.46.1529>`_
   Proc. AAAI/ICML-98 Workshop on Learning for Text Categorization, pp. 41-48.

 * V. Metsis, I. Androutsopoulos and G. Paliouras (2006).
   `Spam filtering with Naive Bayes -- Which Naive Bayes?
   <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.61.5542>`_
   3rd Conf. on Email and Anti-Spam (CEAS).
   
堆外朴素贝叶斯模型拟合
-------------------------------------

朴素贝叶斯模型可用于应对整个训练集不能放入内存的大规模分类问题。
为了处理这个问题，
:class:`MultinomialNB`, :class:`BernoulliNB`, 和 :class:`GaussianNB`
公开了一个可增量式使用的 ``partial_fit`` 方法，这个方法的使用方法与其他分类器的一样，使用示例见
:ref:`sphx_glr_auto_examples_applications_plot_out_of_core_classification.py`。所有的朴素贝叶斯分类器都支持样本权重。

与 ``fit`` 方法不同，首次调用 ``partial_fit`` 方法需要传递一个所有期望的类标签的列表。

想要 scikit-learn 中可用策略的概览，另见
:ref:`out-of-core learning <scaling_strategies>` 文档。

.. note::
所有朴素贝叶斯模型调用 ``partial_fit`` 都会引入一些计算开销。推荐让数据快越大越好，其大小与RAM中可用内存大小相同。