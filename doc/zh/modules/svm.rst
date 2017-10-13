.. _svm:

=======================
Support Vector Machines（支持向量机）
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

Classification（分类）
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

Multi-class classification（多元分类）
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

Scores and 可能性(probability)
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

Unbalanced problems（非均衡问题）
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

Regression（回归）
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

密度估计, 异常(novelty)检测
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

Complexity
==========

Support Vector Machines are powerful tools, but their compute and
storage requirements increase rapidly with the number of training
vectors. The core of an SVM is a quadratic programming problem (QP),
separating support vectors from the rest of the training data. The QP
solver used by this `libsvm`_-based implementation scales between
:math:`O(n_{features} \times n_{samples}^2)` and
:math:`O(n_{features} \times n_{samples}^3)` depending on how efficiently
the `libsvm`_ cache is used in practice (dataset dependent). If the data
is very sparse :math:`n_{features}` should be replaced by the average number
of non-zero features in a sample vector.

Also note that for the linear case, the algorithm used in
:class:`LinearSVC` by the `liblinear`_ implementation is much more
efficient than its `libsvm`_-based :class:`SVC` counterpart and can
scale almost linearly to millions of samples and/or features.


Tips on Practical Use
=====================


  * **Avoiding data copy**: For :class:`SVC`, :class:`SVR`, :class:`NuSVC` and
    :class:`NuSVR`, if the data passed to certain methods is not C-ordered
    contiguous, and double precision, it will be copied before calling the
    underlying C implementation. You can check whether a given numpy array is
    C-contiguous by inspecting its ``flags`` attribute.

    For :class:`LinearSVC` (and :class:`LogisticRegression
    <sklearn.linear_model.LogisticRegression>`) any input passed as a numpy
    array will be copied and converted to the liblinear internal sparse data
    representation (double precision floats and int32 indices of non-zero
    components). If you want to fit a large-scale linear classifier without
    copying a dense numpy C-contiguous double precision array as input we
    suggest to use the :class:`SGDClassifier
    <sklearn.linear_model.SGDClassifier>` class instead.  The objective
    function can be configured to be almost the same as the :class:`LinearSVC`
    model.

  * **Kernel cache size**: For :class:`SVC`, :class:`SVR`, :class:`nuSVC` and
    :class:`NuSVR`, the size of the kernel cache has a strong impact on run
    times for larger problems.  If you have enough RAM available, it is
    recommended to set ``cache_size`` to a higher value than the default of
    200(MB), such as 500(MB) or 1000(MB).

  * **Setting C**: ``C`` is ``1`` by default and it's a reasonable default
    choice.  If you have a lot of noisy observations you should decrease it.
    It corresponds to regularize more the estimation.

  * Support Vector Machine algorithms are not scale invariant, so **it
    is highly recommended to scale your data**. For example, scale each
    attribute on the input vector X to [0,1] or [-1,+1], or standardize it
    to have mean 0 and variance 1. Note that the *same* scaling must be
    applied to the test vector to obtain meaningful results. See section
    :ref:`preprocessing` for more details on scaling and normalization.

  * Parameter ``nu`` in :class:`NuSVC`/:class:`OneClassSVM`/:class:`NuSVR`
    approximates the fraction of training errors and support vectors.

  * In :class:`SVC`, if data for classification are unbalanced (e.g. many
    positive and few negative), set ``class_weight='balanced'`` and/or try
    different penalty parameters ``C``.

  * The underlying :class:`LinearSVC` implementation uses a random
    number generator to select features when fitting the model. It is
    thus not uncommon, to have slightly different results for the same
    input data. If that happens, try with a smaller tol parameter.

  * Using L1 penalization as provided by ``LinearSVC(loss='l2', penalty='l1',
    dual=False)`` yields a sparse solution, i.e. only a subset of feature
    weights is different from zero and contribute to the decision function.
    Increasing ``C`` yields a more complex model (more feature are selected).
    The ``C`` value that yields a "null" model (all weights equal to zero) can
    be calculated using :func:`l1_min_c`.


.. _svm_kernels:

Kernel functions
================

The *kernel function* can be any of the following:

  * linear: :math:`\langle x, x'\rangle`.

  * polynomial: :math:`(\gamma \langle x, x'\rangle + r)^d`.
    :math:`d` is specified by keyword ``degree``, :math:`r` by ``coef0``.

  * rbf: :math:`\exp(-\gamma \|x-x'\|^2)`. :math:`\gamma` is
    specified by keyword ``gamma``, must be greater than 0.

  * sigmoid (:math:`\tanh(\gamma \langle x,x'\rangle + r)`),
    where :math:`r` is specified by ``coef0``.

Different kernels are specified by keyword kernel at initialization::

    >>> linear_svc = svm.SVC(kernel='linear')
    >>> linear_svc.kernel
    'linear'
    >>> rbf_svc = svm.SVC(kernel='rbf')
    >>> rbf_svc.kernel
    'rbf'


Custom Kernels
--------------

You can define your own kernels by either giving the kernel as a
python function or by precomputing the Gram matrix.

Classifiers with custom kernels behave the same way as any other
classifiers, except that:

    * Field ``support_vectors_`` is now empty, only indices of support
      vectors are stored in ``support_``

    * A reference (and not a copy) of the first argument in the ``fit()``
      method is stored for future reference. If that array changes between the
      use of ``fit()`` and ``predict()`` you will have unexpected results.


Using Python functions as kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also use your own defined kernels by passing a function to the
keyword ``kernel`` in the constructor.

Your kernel must take as arguments two matrices of shape
``(n_samples_1, n_features)``, ``(n_samples_2, n_features)``
and return a kernel matrix of shape ``(n_samples_1, n_samples_2)``.

The following code defines a linear kernel and creates a classifier
instance that will use that kernel::

    >>> import numpy as np
    >>> from sklearn import svm
    >>> def my_kernel(X, Y):
    ...     return np.dot(X, Y.T)
    ...
    >>> clf = svm.SVC(kernel=my_kernel)

.. topic:: Examples:

 * :ref:`sphx_glr_auto_examples_svm_plot_custom_kernel.py`.

Using the Gram matrix
~~~~~~~~~~~~~~~~~~~~~

Set ``kernel='precomputed'`` and pass the Gram matrix instead of X in the fit
method. At the moment, the kernel values between *all* training vectors and the
test vectors must be provided.

    >>> import numpy as np
    >>> from sklearn import svm
    >>> X = np.array([[0, 0], [1, 1]])
    >>> y = [0, 1]
    >>> clf = svm.SVC(kernel='precomputed')
    >>> # linear kernel computation
    >>> gram = np.dot(X, X.T)
    >>> clf.fit(gram, y) # doctest: +NORMALIZE_WHITESPACE
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='auto',
        kernel='precomputed', max_iter=-1, probability=False,
        random_state=None, shrinking=True, tol=0.001, verbose=False)
    >>> # predict on training examples
    >>> clf.predict(gram)
    array([0, 1])

Parameters of the RBF Kernel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When training an SVM with the *Radial Basis Function* (RBF) kernel, two
parameters must be considered: ``C`` and ``gamma``.  The parameter ``C``,
common to all SVM kernels, trades off misclassification of training examples
against simplicity of the decision surface. A low ``C`` makes the decision
surface smooth, while a high ``C`` aims at classifying all training examples
correctly.  ``gamma`` defines how much influence a single training example has.
The larger ``gamma`` is, the closer other examples must be to be affected.

Proper choice of ``C`` and ``gamma`` is critical to the SVM's performance.  One
is advised to use :class:`sklearn.model_selection.GridSearchCV` with 
``C`` and ``gamma`` spaced exponentially far apart to choose good values.

.. topic:: Examples:

 * :ref:`sphx_glr_auto_examples_svm_plot_rbf_parameters.py`

.. _svm_mathematical_formulation:

Mathematical formulation
========================

A support vector machine constructs a hyper-plane or set of hyper-planes
in a high or infinite dimensional space, which can be used for
classification, regression or other tasks. Intuitively, a good
separation is achieved by the hyper-plane that has the largest distance
to the nearest training data points of any class (so-called functional
margin), since in general the larger the margin the lower the
generalization error of the classifier.


.. figure:: ../auto_examples/svm/images/sphx_glr_plot_separating_hyperplane_001.png
   :align: center
   :scale: 75

SVC
---

Given training vectors :math:`x_i \in \mathbb{R}^p`, i=1,..., n, in two classes, and a
vector :math:`y \in \{1, -1\}^n`, SVC solves the following primal problem:


.. math::

    \min_ {w, b, \zeta} \frac{1}{2} w^T w + C \sum_{i=1}^{n} \zeta_i



    \textrm {subject to } & y_i (w^T \phi (x_i) + b) \geq 1 - \zeta_i,\\
    & \zeta_i \geq 0, i=1, ..., n

Its dual is

.. math::

   \min_{\alpha} \frac{1}{2} \alpha^T Q \alpha - e^T \alpha


   \textrm {subject to } & y^T \alpha = 0\\
   & 0 \leq \alpha_i \leq C, i=1, ..., n

where :math:`e` is the vector of all ones, :math:`C > 0` is the upper bound,
:math:`Q` is an :math:`n` by :math:`n` positive semidefinite matrix,
:math:`Q_{ij} \equiv y_i y_j K(x_i, x_j)`, where :math:`K(x_i, x_j) = \phi (x_i)^T \phi (x_j)`
is the kernel. Here training vectors are implicitly mapped into a higher
(maybe infinite) dimensional space by the function :math:`\phi`.


The decision function is:

.. math:: \operatorname{sgn}(\sum_{i=1}^n y_i \alpha_i K(x_i, x) + \rho)

.. note::

    While SVM models derived from `libsvm`_ and `liblinear`_ use ``C`` as
    regularization parameter, most other estimators use ``alpha``. The exact
    equivalence between the amount of regularization of two models depends on
    the exact objective function optimized by the model. For example, when the
    estimator used is :class:`sklearn.linear_model.Ridge <ridge>` regression,
    the relation between them is given as :math:`C = \frac{1}{alpha}`.

.. TODO multiclass case ?/

This parameters can be accessed through the members ``dual_coef_``
which holds the product :math:`y_i \alpha_i`, ``support_vectors_`` which
holds the support vectors, and ``intercept_`` which holds the independent
term :math:`\rho` :

.. topic:: References:

 * `"Automatic Capacity Tuning of Very Large VC-dimension Classifiers"
   <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.17.7215>`_,
   I. Guyon, B. Boser, V. Vapnik - Advances in neural information
   processing 1993.


 * `"Support-vector networks"
   <http://link.springer.com/article/10.1007%2FBF00994018>`_,
   C. Cortes, V. Vapnik - Machine Learning, 20, 273-297 (1995).



NuSVC
-----

We introduce a new parameter :math:`\nu` which controls the number of
support vectors and training errors. The parameter :math:`\nu \in (0,
1]` is an upper bound on the fraction of training errors and a lower
bound of the fraction of support vectors.

It can be shown that the :math:`\nu`-SVC formulation is a reparametrization
of the :math:`C`-SVC and therefore mathematically equivalent.


SVR
---

Given training vectors :math:`x_i \in \mathbb{R}^p`, i=1,..., n, and a
vector :math:`y \in \mathbb{R}^n` :math:`\varepsilon`-SVR solves the following primal problem:


.. math::

    \min_ {w, b, \zeta, \zeta^*} \frac{1}{2} w^T w + C \sum_{i=1}^{n} (\zeta_i + \zeta_i^*)



    \textrm {subject to } & y_i - w^T \phi (x_i) - b \leq \varepsilon + \zeta_i,\\
                          & w^T \phi (x_i) + b - y_i \leq \varepsilon + \zeta_i^*,\\
                          & \zeta_i, \zeta_i^* \geq 0, i=1, ..., n

Its dual is

.. math::

   \min_{\alpha, \alpha^*} \frac{1}{2} (\alpha - \alpha^*)^T Q (\alpha - \alpha^*) + \varepsilon e^T (\alpha + \alpha^*) - y^T (\alpha - \alpha^*)


   \textrm {subject to } & e^T (\alpha - \alpha^*) = 0\\
   & 0 \leq \alpha_i, \alpha_i^* \leq C, i=1, ..., n

where :math:`e` is the vector of all ones, :math:`C > 0` is the upper bound,
:math:`Q` is an :math:`n` by :math:`n` positive semidefinite matrix,
:math:`Q_{ij} \equiv K(x_i, x_j) = \phi (x_i)^T \phi (x_j)`
is the kernel. Here training vectors are implicitly mapped into a higher
(maybe infinite) dimensional space by the function :math:`\phi`.

The decision function is:

.. math:: \sum_{i=1}^n (\alpha_i - \alpha_i^*) K(x_i, x) + \rho

These parameters can be accessed through the members ``dual_coef_``
which holds the difference :math:`\alpha_i - \alpha_i^*`, ``support_vectors_`` which
holds the support vectors, and ``intercept_`` which holds the independent
term :math:`\rho`

.. topic:: References:

 * `"A Tutorial on Support Vector Regression"
   <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.114.4288>`_,
   Alex J. Smola, Bernhard Schölkopf - Statistics and Computing archive
   Volume 14 Issue 3, August 2004, p. 199-222. 


.. _svm_implementation_details:

Implementation details
======================

Internally, we use `libsvm`_ and `liblinear`_ to handle all
computations. These libraries are wrapped using C and Cython.

.. _`libsvm`: http://www.csie.ntu.edu.tw/~cjlin/libsvm/
.. _`liblinear`: http://www.csie.ntu.edu.tw/~cjlin/liblinear/

.. topic:: References:

  For a description of the implementation and details of the algorithms
  used, please refer to

    - `LIBSVM: A Library for Support Vector Machines
      <http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf>`_.

    - `LIBLINEAR -- A Library for Large Linear Classification
      <http://www.csie.ntu.edu.tw/~cjlin/liblinear/>`_.


