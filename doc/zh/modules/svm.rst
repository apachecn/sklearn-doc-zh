.. _svm:

=======================
支持向量机
=======================

.. currentmodule:: sklearn.svm

**支持向量机 (SVMs)** 可用于以下监督学习算法 :ref:`classification（分类） <svm_classification>`, :ref:`regression（回归） <svm_regression>` and  :ref:`outliers detection（异常检测） <svm_outlier_detection>`.

支持向量机的优势在于:

    - 在高维空间中非常高效.

    - 即使在数据维度比样本数量大的情况下仍然有效.

    - 在决策函数（称为支持向量）中使用训练集的子集,因此它也是高效利用内存的.

    - 通用性: 不同的核函数 :ref:`svm_kernels` 与特定的决策函数一一对应.常见的内核已
    经提供,也可以指定定制的内核.

支持向量机的缺点包括:

    - 如果特征数量比样本数量大得多,在选择核函数 :ref:`svm_kernels` 时要避免过拟合,
    而且正则化项是非常重要的.

    - 支持向量机不直接提供概率估计,这些都是使用昂贵的五次交叉验算计算的.
      (详情见 :ref:`Scores and probabilities <scores_probabilities>`, 在下文中).

在 scikit-learn 中,支持向量机提供 dense(``numpy.ndarray`` ,可以通过 ``numpy.asarray`` 进行转换) 和 sparse (任何 ``scipy.sparse``) 样例向量作为输出.然而,要使用支持向量机来对 sparse 数据作预测,它必须已经拟合这样的数据.使用 C 代码的 ``numpy.ndarray`` (dense) 或者带有 ``dtype=float64`` 的 ``scipy.sparse.csr_matrix`` (sparse) 来优化性能.


.. _svm_classification:

分类
==============

:class:`SVC`, :class:`NuSVC` 和 :class:`LinearSVC` 能在数据集中实现多元分类.


.. figure:: ../auto_examples/svm/images/sphx_glr_plot_iris_001.png
   :target: ../auto_examples/svm/plot_iris.html
   :align: center


:class:`SVC` 和 :class:`NuSVC` 是相似的方法, 但是接受稍许不同的参数设置并且有不同的数学方程(在这部分看 :ref:`svm_mathematical_formulation`). 另一方面, :class:`LinearSVC` 是另一个实现线性核函数的支持向量分类. 记住 :class:`LinearSVC` 不接受关键词 ``kernel``, 因为它被假设为线性的. 它也缺少一些:class:`SVC` 和 :class:`NuSVC` 的成员(members) 比如 ``support_``.

和其他分类器一样, :class:`SVC`, :class:`NuSVC` 和 :class:`LinearSVC` 将两个数组作为输入:  ``[n_samples, n_features]`` 大小的数组 X 作为训练样本, ``[n_samples]`` 大小的数组y作为类别标签(字符串或者整数)::


    >>> from sklearn import svm
    >>> X = [[0, 0], [1, 1]]
    >>> y = [0, 1]
    >>> clf = svm.SVC()
    >>> clf.fit(X, y)  # doctest: +NORMALIZE_WHITESPACE
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)

在拟合后, 这个模型可以用来预测新的值::

    >>> clf.predict([[2., 2.]])
    array([1])

SVMs 决策函数取决于训练集的一些子集, 称作支持向量. 这些支持向量的部分特性可以在 ``support_vectors_``, ``support_`` 和 ``n_support`` 找到::

    >>> # 获得支持向量
    >>> clf.support_vectors_
    array([[ 0.,  0.],
           [ 1.,  1.]])
    >>> # 获得支持向量的索引get indices of support vectors
    >>> clf.support_ # doctest: +ELLIPSIS
    array([0, 1]...)
    >>> # 为每一个类别获得支持向量的数量
    >>> clf.n_support_ # doctest: +ELLIPSIS
    array([1, 1]...)

.. _svm_multi_class:

多元分类
--------------------------

:class:`SVC` 和 :class:`NuSVC` 为多元分类实现了 "one-against-one" 的方法 (Knerr et al., 1990) 如果 ``n_class`` 是类别的数量, 那么 ``n_class * (n_class - 1) / 2`` 分类器被重构, 而且每一个从两个类别中训练数据. 为了给其他分类器提供一致的交互, ``decision_function_shape`` 选项允许聚合"one-against-one" 分类器的结果成 ``(n_samples, n_classes)`` 的大小到决策函数::

    >>> X = [[0], [1], [2], [3]]
    >>> Y = [0, 1, 2, 3]
    >>> clf = svm.SVC(decision_function_shape='ovo')
    >>> clf.fit(X, Y) # doctest: +NORMALIZE_WHITESPACE
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)
    >>> dec = clf.decision_function([[1]])
    >>> dec.shape[1] # 4 classes: 4*3/2 = 6
    6
    >>> clf.decision_function_shape = "ovr"
    >>> dec = clf.decision_function([[1]])
    >>> dec.shape[1] # 4 classes
    4

另一方面, :class:`LinearSVC` 实现 "one-vs-the-rest" 多类别策略, 从而训练 n 类别的模型. 如果只有两类, 只训练一个模型.::

    >>> lin_clf = svm.LinearSVC()
    >>> lin_clf.fit(X, Y) # doctest: +NORMALIZE_WHITESPACE
    LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
         intercept_scaling=1, loss='squared_hinge', max_iter=1000,
         multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
         verbose=0)
    >>> dec = lin_clf.decision_function([[1]])
    >>> dec.shape[1]
    4

参见 :ref:`svm_mathematical_formulation` 查看决策函数的完整描述.

记住 :class:`LinearSVC` 也实现了可选择的多类别策略, 通过使用选项 ``multi_class='crammer_singer'``, 所谓的多元 SVM 由 Crammer 和 Singer 明确表达. 这个方法是一致的, 对于 one-vs-rest 是不正确的. 实际上, one-vs-rest 分类通常收到青睐, 因为结果大多数是相似的, 但是运行时间却显著减少.

对于 "one-vs-rest" :class:`LinearSVC`, 属性 ``coef_`` 和 ``intercept_`` 分别具有 ``[n_class, n_features]`` 和 ``[n_class]`` 尺寸. 系数的每一行符合 ``n_class`` 的许多 one-vs-rest 分类器之一, 并且就以这一类的顺序与拦截器(intercepts)相似.

至于 one-vs-one :class:`SVC`, 属性特征的布局(layout)有少多些复杂. 考虑到有一种线性核函数,``coef_`` 和 ``intercept_`` 的布局(layout)与上文描述成 :class:`LinearSVC` 相似, 除了 ``coef_`` 的形状 ``[n_class * (n_class - 1) / 2, n_features]``, 与许多二元的分类器相似. 0到n的类别顺序是 "0 vs 1", "0 vs 2" , ... "0 vs n", "1 vs 2", "1 vs 3", "1 vs n", . . . "n-1 vs n".

``dual_coef_`` 的形状是 ``[n_class-1, n_SV]``, 这个结构有些难以理解.
对应于支持向量的列与 ``n_class * (n_class - 1) / 2`` "one-vs-one" 分类器相关.
每一个支持向量用于 ``n_class - 1`` 分类器中.对于这些分类器,每一行的 ``n_class - 1`` 
条目对应于对偶系数(dual coefficients).

通过这个例子更容易说明:

考虑一个三类的问题,类0有三个支持向量 :math:`v^{0}_0, v^{1}_0, v^{2}_0` 而类 1 和 2 分别有
如下两个支持向量 :math:`v^{0}_1, v^{1}_1` and :math:`v^{0}_2, v^{1}_2`.对于每个支持
向量 :math:`v^{j}_i`, 有两个对偶系数.在类别 :math:`i` 和 :math:`k` :math:`\alpha^{j}_{i,k}` 中,
我们将支持向量的系数记录为 :math:`v^{j}_i` 
那么 ``dual_coef_`` 可以表示为:

+------------------------+------------------------+------------------+
|:math:`\alpha^{0}_{0,1}`|:math:`\alpha^{0}_{0,2}`|Coefficients      |
+------------------------+------------------------+for SVs of class 0|
|:math:`\alpha^{1}_{0,1}`|:math:`\alpha^{1}_{0,2}`|                  |
+------------------------+------------------------+                  |
|:math:`\alpha^{2}_{0,1}`|:math:`\alpha^{2}_{0,2}`|                  |
+------------------------+------------------------+------------------+
|:math:`\alpha^{0}_{1,0}`|:math:`\alpha^{0}_{1,2}`|Coefficients      |
+------------------------+------------------------+for SVs of class 1|
|:math:`\alpha^{1}_{1,0}`|:math:`\alpha^{1}_{1,2}`|                  |
+------------------------+------------------------+------------------+
|:math:`\alpha^{0}_{2,0}`|:math:`\alpha^{0}_{2,1}`|Coefficients      |
+------------------------+------------------------+for SVs of class 2|
|:math:`\alpha^{1}_{2,0}`|:math:`\alpha^{1}_{2,1}`|                  |
+------------------------+------------------------+------------------+


.. _scores_probabilities:

得分和概率
------------------------

:class:`SVC` 方法的 ``decision_function`` 给每一个样例每一个类别分
值(scores)(或者在一个二元类中每一个样例一个分值).
当构造器(constructor)选项 ``probability`` 设置为 ``True``的时候,
类成员可能性评估开启.(来自 ``predict_proba`` 和 ``predict_log_proba`` 方法)
在二元分类中,概率使用Platt scaling进行标准化:在SVM分数上的逻辑回归,在训练集上用额外的交
叉验证来拟合.在多类情况下,这可以扩展为per Wu et al.(2004)

不用说,对于大数据集来说,在Platt scaling中进行交叉验证是一项昂贵的操作.
另外,可能性预测可能与scores不一致,因为scores的"argmax"可能不是可能性的argmax.
(例如,在二元分类中,
一个样本可能被标记为一个有可能性的类``predict`` <½ according to ``predict_proba``.)
Platt的方法也有理论问题.
如果 confidence scores 必要,但是这些没必要是可能性,
那么建议设置 ``probability=False``
并使用 ``decision_function`` 而不是 ``predict_proba``.

.. topic:: 参考:

 * Wu, Lin and Weng,
   `"Probability estimates for multi-class classification by pairwise coupling（成对耦合的多类分类的概率估计）"<http://www.csie.ntu.edu.tw/~cjlin/papers/svmprob/svmprob.pdf>`_, JMLR 5:975-1005, 2004.
 
 
 * Platt
   `"Probabilistic outputs for SVMs and comparisons to regularized likelihood methods（SVMs 的概率输出和与规则化似然方法的比较）"<http://www.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/Platt1999.pdf>`.

非均衡问题
--------------------

这个问题期望给予某一类或某个别样例能使用的关键词 ``class_weight`` 和 ``sample_weight`` 提高权重(importance).

:class:`SVC` (而不是 :class:`NuSVC`) 在 ``fit`` 方法中生成了一个关键词 ``class_weight``. 它是形如``{class_label : value}`` 的字典, value是浮点数大于0的值, 把类 ``class_label`` 的参数``C`` 设置为 ``C * value``.

.. figure:: ../auto_examples/svm/images/sphx_glr_plot_separating_hyperplane_unbalanced_001.png
   :target: ../auto_examples/svm/plot_separating_hyperplane_unbalanced.html
   :align: center
   :scale: 75


:class:`SVC`, :class:`NuSVC`, :class:`SVR`, :class:`NuSVR` 和 :class:`OneClassSVM` 在 ``fit`` 方法中通过关键词 ``sample_weight``  为单一样例实现权重weights.与 ``class_weight`` 相似, 这些把第i个样例的参数 ``C`` 换成 ``C * sample_weight[i]``.


.. figure:: ../auto_examples/svm/images/sphx_glr_plot_weighted_samples_001.png
   :target: ../auto_examples/svm/plot_weighted_samples.html
   :align: center
   :scale: 75


.. topic:: 例子:

 * :ref:`sphx_glr_auto_examples_svm_plot_iris.py`,
 * :ref:`sphx_glr_auto_examples_svm_plot_separating_hyperplane.py`,
 * :ref:`sphx_glr_auto_examples_svm_plot_separating_hyperplane_unbalanced.py`
 * :ref:`sphx_glr_auto_examples_svm_plot_svm_anova.py`,
 * :ref:`sphx_glr_auto_examples_svm_plot_svm_nonlinear.py`
 * :ref:`sphx_glr_auto_examples_svm_plot_weighted_samples.py`,


.. _svm_regression:

回归
==========

支持向量分类的方法可以被扩展用作解决回归问题. 这个方法被称作支持向量回归.

支持向量分类生成的模型(如前描述)只依赖于训练集的子集,因为构建模型的 cost function 不在乎边缘之外的训练点. 类似的,支持向量回归生成的模型只依赖于训练集的子集, 因为构建模型的 cost function 忽略任何接近于模型预测的训练数据.

支持向量分类有三种不同的实现形式: 
:class:`SVR`, :class:`NuSVR` 和 :class:`LinearSVR`. 在只考虑线性核的情况下, :class:`LinearSVR`  比 :class:`SVR` 提供一个更快的实现形式, 然而比起 :class:`SVR` 和 :class:`LinearSVR`, :class:`NuSVR` 实现一个稍微不同的构思(formulation).细节参见 :ref:`svm_implementation_details`.

与分类的类别一样, fit方法会调用参数向量 X, y, 只在 y 是浮点数而不是整数型.::

    >>> from sklearn import svm
    >>> X = [[0, 0], [2, 2]]
    >>> y = [0.5, 2.5]
    >>> clf = svm.SVR()
    >>> clf.fit(X, y) # doctest: +NORMALIZE_WHITESPACE
    SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
        kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
    >>> clf.predict([[1, 1]])
    array([ 1.5])


.. topic:: 样例:

 * :ref:`sphx_glr_auto_examples_svm_plot_svm_regression.py`



.. _svm_outlier_detection:

密度估计, 异常（novelty）检测
=======================================

但类别的 SVM 用于异常检测, 即给予一个样例集, 它会检测这个样例集的 soft boundary 以便给新的数据点分类,
看它是否属于这个样例集. 生成的类称作 :class:`OneClassSVM`.

这种情况下, 因为它属于非监督学习的一类, 没有类标签, fit方法只会考虑输入数组X,.

在章节 :ref:`outlier_detection` 查看这个应用的更多细节.

.. figure:: ../auto_examples/svm/images/sphx_glr_plot_oneclass_001.png
   :target: ../auto_examples/svm/plot_oneclass.html
   :align: center
   :scale: 75


.. topic:: Examples:

 * :ref:`sphx_glr_auto_examples_svm_plot_oneclass.py`
 * :ref:`sphx_glr_auto_examples_applications_plot_species_distribution_modeling.py`


复杂度
==========

支持向量机是个强大的工具，不过它的计算和存储空间要求也会随着要训练向量的数目增加而快速增加。
SVM的核心是一个二次规划问题(Quadratic Programming, QP)，是将支持向量和训练数据的其余部分分离开来。
在实践中(数据集相关)，会根据 `libsvm`_ 的缓存有多效，在 :math:`O(n_{features} \times n_{samples}^2)` 和 
:math:`O(n_{features} \times n_{samples}^3)` 之间基于 `libsvm`_ 的缩放操作才会调用这个 QP解析器。
如果数据是非常稀疏，那 :math:`n_{features}`  就用样本向量中非零特征的平均数量去替换。 

另外请注意，在线性情况下，由 `liblinear`_ 操作的 :class:`LinearSVC` 算法要比由它的 `libsvm`_ 对应的 
:class:`SVC` 更为高效，并且它几乎可以线性缩放到数百万样本或者特征。


使用诀窍
=====================


  * **避免数据复制**: 对于 :class:`SVC`， :class:`SVR`， :class:`NuSVC` 和
    :class:`NuSVR`， 如果数据是通过某些方法而不是用C有序的连续双精度，那它先会调用底层的C命令再复制。
    您可以通过检查它的 ``flags`` 属性，来确定给定的numpy数组是不是C连续的。

    对于 :class:`LinearSVC` (和 :class:`LogisticRegression
    <sklearn.linear_model.LogisticRegression>`) 的任何输入，都会以numpy数组形式，被复制和转换为
    用liblinear内部稀疏数据去表达（双精度浮点型float和非零部分的int32索引）。 
    如果您想要一个适合大规模的线性分类器，又不打算复制一个密集的C-contiguous双精度numpy数组作为输入，
    那我们建议您去使用 :class:`SGDClassifier
    <sklearn.linear_model.SGDClassifier>` 类作为替代。目标函数可以配置为和 :class:`LinearSVC`
    模型差不多相同的。

  * **内核的缓存大小**: 在大规模问题上，对于 :class:`SVC`, :class:`SVR`, :class:`nuSVC` 和
    :class:`NuSVR`, 内核缓存的大小会特别影响到运行时间。如果您有足够可用的RAM，不妨把它的 ``缓存大小`` 
    设得比默认的200(MB)要高，例如为 500(MB) 或者 1000(MB)。

  * **惩罚系数C的设置**:在合理的情况下， ``C`` 的默认选择为 ``1`` 。如果您有很多混杂的观察数据，
    您应该要去调小它。 ``C`` 越小，就能更好地去正规化估计。

  * 支持向量机算法本身不是用来扩大不变性，所以 **我们强烈建议您去扩大数据量**. 举个例子，对于输入向量X，
    规整它的每个数值范围为[0, 1]或[-1, +1]，或者标准化它的为均值为0方差为1的数据分布。请注意，
    相同的缩放标准必须要应用到所有的测试向量，从而获得有意义的结果。 请参考章节
    :ref:`preprocessing` ，那里会提供到更多关于缩放和规整。

  * 在 :class:`NuSVC`/:class:`OneClassSVM`/:class:`NuSVR` 内的参数 ``nu`` ，
    近似是训练误差和支持向量的比值。

  * 在 :class:`SVC`, ，如果分类器的数据不均衡（就是说，很多正例很少负例），设置 ``class_weight='balanced'`` 
    与/或尝试不同的惩罚系数 ``C`` 。	

  * 在拟合模型时，底层 :class:`LinearSVC` 操作使用了随机数生成器去选择特征。
    所以不要感到意外，对于相同的数据输入，也会略有不同的输出结果。如果这个发生了，
    尝试用更小的tol 参数。

  * 使用由 ``LinearSVC(loss='l2', penalty='l1',
    dual=False)`` 提供的L1惩罚去产生稀疏解，也就是说，特征权重的子集不同于零，这样做有助于决策函数。
    随着增加 ``C`` 会产生一个更复杂的模型（要做更多的特征选择）。可以使用 :func:`l1_min_c` 去计算 ``C`` 的数值，去产生一个"null" 模型（所有的权重等于零）。


.. _svm_kernels:

核函数
================

*核函数* 可以是以下任何形式：:

  * 线性: :math:`\langle x, x'\rangle`.

  * 多项式: :math:`(\gamma \langle x, x'\rangle + r)^d`.
    :math:`d` 是关键词 ``degree``, :math:`r` 指定 ``coef0``。

  * rbf: :math:`\exp(-\gamma \|x-x'\|^2)`. :math:`\gamma` 是关键词 ``gamma``, 必须大于0。

  * sigmoid (:math:`\tanh(\gamma \langle x,x'\rangle + r)`),
    其中 :math:`r` 指定 ``coef0``。

初始化时，不同内核由不同的函数名调用::

    >>> linear_svc = svm.SVC(kernel='linear')
    >>> linear_svc.kernel
    'linear'
    >>> rbf_svc = svm.SVC(kernel='rbf')
    >>> rbf_svc.kernel
    'rbf'


自定义核
--------------

您可以自定义自己的核，通过使用python函数作为内核或者通过预计算Gram矩阵。

自定义内核的分类器和别的分类器一样，除了下面这几点:

    * 空间 ``support_vectors_`` 现在不是空的, 只有支持向量的索引被存储在 ``support_``

    * 请把 ``fit()`` 模型中的第一个参数的引用（不是副本）存储为将来的引用。
      如果在 ``fit()`` 和 ``predict()`` 之间有数组发生改变，您将会碰到意料外的结果。


使用 python 函数作为内核
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在构造时，您同样可以通过一个函数传递到关键词 ``kernel`` ，来使用您自己定义的内核。

您的内核必须要以两个矩阵作为参数，大小分别是
``(n_samples_1, n_features)``, ``(n_samples_2, n_features)``
和返回一个内核矩阵，大小是 ``(n_samples_1, n_samples_2)``.

以下代码定义一个线性核，和构造一个使用该内核的分类器例子::

    >>> import numpy as np
    >>> from sklearn import svm
    >>> def my_kernel(X, Y):
    ...     return np.dot(X, Y.T)
    ...
    >>> clf = svm.SVC(kernel=my_kernel)

.. topic:: 例子:

 * :ref:`sphx_glr_auto_examples_svm_plot_custom_kernel.py`.

使用 Gram 矩阵
~~~~~~~~~~~~~~~~~~~~~

在适应算法中，设置 ``kernel='precomputed'`` 和把X替换为Gram矩阵。
此时，必须要提供在 *所有* 训练矢量和测试矢量中的内核值。 

    >>> import numpy as np
    >>> from sklearn import svm
    >>> X = np.array([[0, 0], [1, 1]])
    >>> y = [0, 1]
    >>> clf = svm.SVC(kernel='precomputed')
    >>> # 线性内核计算
    >>> gram = np.dot(X, X.T)
    >>> clf.fit(gram, y) # doctest: +NORMALIZE_WHITESPACE
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='auto',
        kernel='precomputed', max_iter=-1, probability=False,
        random_state=None, shrinking=True, tol=0.001, verbose=False)
    >>> # 预测训练样本
    >>> clf.predict(gram)
    array([0, 1])

RBF内核参数
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

当用 *径向基* (RBF)内核去训练SVM，有两个参数必须要去考虑： ``C`` 惩罚系数和 ``gamma`` 。参数 ``C`` ，
通用在所有SVM内核，与决策表面的简单性相抗衡，可以对训练样本的误分类进行有价转换。
较小的 ``C`` 会使决策表面更平滑，同时较高的 ``C`` 旨在正确地分类所有训练样本。 ``Gamma`` 定义了单一
训练样本能起到多大的影响。较大的 ``gamma`` 会更让其他样本受到影响。

选择合适的 ``C`` 和 ``gamma`` ，对SVM的性能起到很关键的作用。建议一点是
使用  :class:`sklearn.model_selection.GridSearchCV` 与 ``C`` 和 ``gamma`` 相隔
成倍差距从而选择到好的数值。

.. topic:: 例子:

 * :ref:`sphx_glr_auto_examples_svm_plot_rbf_parameters.py`

.. _svm_mathematical_formulation:

数学公式
========================

支持向量机在高维度或无穷维度空间中，构建一个超平面或者一系列的超平面，可以用于分类、回归或者别的任务。
直观地看，借助超平面去实现一个好的分割， 能在任意类别中使最为接近的训练数据点具有最大的间隔距离（即所
谓的函数余量），这样做是因为通常更大的余量能有更低的分类器泛化误差。


.. figure:: ../auto_examples/svm/images/sphx_glr_plot_separating_hyperplane_001.png
   :align: center
   :scale: 75

SVC
---

在两类中，给定训练向量 :math:`x_i \in \mathbb{R}^p`, i=1,..., n, 和一个向量 :math:`y \in \{1, -1\}^n`, SVC能解决
如下主要问题:

.. math::

    \min_ {w, b, \zeta} \frac{1}{2} w^T w + C \sum_{i=1}^{n} \zeta_i



    \textrm {subject to } & y_i (w^T \phi (x_i) + b) \geq 1 - \zeta_i,\\
    & \zeta_i \geq 0, i=1, ..., n

它的对偶是

.. math::

   \min_{\alpha} \frac{1}{2} \alpha^T Q \alpha - e^T \alpha


   \textrm {subject to } & y^T \alpha = 0\\
   & 0 \leq \alpha_i \leq C, i=1, ..., n

其中 :math:`e` 是所有的向量， :math:`C > 0` 是上界，:math:`Q` 是一个 :math:`n` 由 :math:`n` 个半正定矩阵，
而 :math:`Q_{ij} \equiv y_i y_j K(x_i, x_j)` ，其中 :math:`K(x_i, x_j) = \phi (x_i)^T \phi (x_j)` 是内核。所以训练向量是通过函数 :math:`\phi`，间接反映到一个更高维度的（无穷的）空间。


决策函数是:

.. math:: \operatorname{sgn}(\sum_{i=1}^n y_i \alpha_i K(x_i, x) + \rho)

注意::

虽然这些SVM模型是从 `libsvm`_ 和 `liblinear`_ 中派生出来，使用了 ``C`` 作为调整参数，但是大多数的
攻击使用了 ``alpha``。两个模型的正则化量之间的精确等价，取决于模型优化的准确目标函数。举
个例子，当使用的估计器是 :class:`sklearn.linear_model.Ridge <ridge>` 做回归时，他们之间的相关性是 :math:`C = \frac{1}{alpha}`。 



这些参数能通过成员 ``dual_coef_``、 ``support_vectors_`` 、 ``intercept_`` 去访问，这些成员分别控制了输出 :math:`y_i \alpha_i`、支持向量和无关项 :math:`\rho` ： 

.. topic:: 参考文献:

 * `"Automatic Capacity Tuning of Very Large VC-dimension Classifiers"
   <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.17.7215>`_,
   I. Guyon, B. Boser, V. Vapnik - Advances in neural information
   processing 1993.


 * `"Support-vector networks"
   <http://link.springer.com/article/10.1007%2FBF00994018>`_,
   C. Cortes, V. Vapnik - Machine Learning, 20, 273-297 (1995).



NuSVC
-----

我们引入一个新的参数 :math:`\nu` 来控制支持向量的数量和训练误差。参数 :math:`\nu \in (0,
1]` 是训练误差分数的上限和支持向量分数的下限。

可以看出， :math:`\nu`-SVC 公式是 :math:`C`-SVC 的再参数化，所以数学上是等效的。


SVR
---

给定训练向量 :math:`x_i \in \mathbb{R}^p`, i=1,..., n，向量 :math:`y \in \mathbb{R}^n` :math:`\varepsilon`-SVR 
能解决以下的主要问题：


.. math::

    \min_ {w, b, \zeta, \zeta^*} \frac{1}{2} w^T w + C \sum_{i=1}^{n} (\zeta_i + \zeta_i^*)



    \textrm {subject to } & y_i - w^T \phi (x_i) - b \leq \varepsilon + \zeta_i,\\
                          & w^T \phi (x_i) + b - y_i \leq \varepsilon + \zeta_i^*,\\
                          & \zeta_i, \zeta_i^* \geq 0, i=1, ..., n

它的对偶是

.. math::

   \min_{\alpha, \alpha^*} \frac{1}{2} (\alpha - \alpha^*)^T Q (\alpha - \alpha^*) + \varepsilon e^T (\alpha + \alpha^*) - y^T (\alpha - \alpha^*)


   \textrm {subject to } & e^T (\alpha - \alpha^*) = 0\\
   & 0 \leq \alpha_i, \alpha_i^* \leq C, i=1, ..., n


其中 :math:`e` 是所有的向量， :math:`C > 0` 是上界，:math:`Q` 是一个 :math:`n` 由 :math:`n` 个半正定矩阵，
而 :math:`Q_{ij} \equiv K(x_i, x_j) = \phi (x_i)^T \phi (x_j)` 是内核。
所以训练向量是通过函数 :math:`\phi`，间接反映到一个更高维度的（无穷的）空间。


决策函数是:

.. math:: \sum_{i=1}^n (\alpha_i - \alpha_i^*) K(x_i, x) + \rho

这些参数能通过成员 ``dual_coef_``、 ``support_vectors_`` 、 ``intercept_`` 去访问，这些
成员分别控制了不同的 :math:`\alpha_i - \alpha_i^*`、支持向量和无关项 :math:`\rho`：


.. topic:: 参考文献:

 * `"A Tutorial on Support Vector Regression"
   <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.114.4288>`_,
   Alex J. Smola, Bernhard Schölkopf - Statistics and Computing archive
   Volume 14 Issue 3, August 2004, p. 199-222. 


.. _svm_implementation_details:

实现细节
======================

在底层里，我们使用 `libsvm`_ 和 `liblinear`_ 去处理所有的计算。这些库都使用了 C 和 Cython 去包装。


.. _`libsvm`: http://www.csie.ntu.edu.tw/~cjlin/libsvm/
.. _`liblinear`: http://www.csie.ntu.edu.tw/~cjlin/liblinear/

.. topic:: 参考文献:

  有关实现的描述和使用算法的细节，请参考

    - `LIBSVM: A Library for Support Vector Machines
      <http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf>`_.

    - `LIBLINEAR -- A Library for Large Linear Classification
      <http://www.csie.ntu.edu.tw/~cjlin/liblinear/>`_.


