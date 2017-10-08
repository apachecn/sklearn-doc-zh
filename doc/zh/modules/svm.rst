.. _svm:

=======================
Support Vector Machines
=======================

.. currentmodule:: sklearn.svm

**Support vector machines (SVMs)** are a set of supervised learning
methods used for :ref:`classification <svm_classification>`,
:ref:`regression <svm_regression>` and :ref:`outliers detection
<svm_outlier_detection>`.

The advantages of support vector machines are:

    - Effective in high dimensional spaces.

    - Still effective in cases where number of dimensions is greater
      than the number of samples.

    - Uses a subset of training points in the decision function (called
      support vectors), so it is also memory efficient.

    - Versatile: different :ref:`svm_kernels` can be
      specified for the decision function. Common kernels are
      provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

    - If the number of features is much greater than the number of
      samples, avoid over-fitting in choosing :ref:`svm_kernels` and regularization
      term is crucial.

    - SVMs do not directly provide probability estimates, these are
      calculated using an expensive five-fold cross-validation
      (see :ref:`Scores and probabilities <scores_probabilities>`, below).

The support vector machines in scikit-learn support both dense
(``numpy.ndarray`` and convertible to that by ``numpy.asarray``) and
sparse (any ``scipy.sparse``) sample vectors as input. However, to use
an SVM to make predictions for sparse data, it must have been fit on such
data. For optimal performance, use C-ordered ``numpy.ndarray`` (dense) or
``scipy.sparse.csr_matrix`` (sparse) with ``dtype=float64``.


.. _svm_classification:

Classification
==============

:class:`SVC`, :class:`NuSVC` and :class:`LinearSVC` are classes
capable of performing multi-class classification on a dataset.


.. figure:: ../auto_examples/svm/images/sphx_glr_plot_iris_001.png
   :target: ../auto_examples/svm/plot_iris.html
   :align: center


:class:`SVC` and :class:`NuSVC` are similar methods, but accept
slightly different sets of parameters and have different mathematical
formulations (see section :ref:`svm_mathematical_formulation`). On the
other hand, :class:`LinearSVC` is another implementation of Support
Vector Classification for the case of a linear kernel. Note that
:class:`LinearSVC` does not accept keyword ``kernel``, as this is
assumed to be linear. It also lacks some of the members of
:class:`SVC` and :class:`NuSVC`, like ``support_``.

As other classifiers, :class:`SVC`, :class:`NuSVC` and
:class:`LinearSVC` take as input two arrays: an array X of size ``[n_samples,
n_features]`` holding the training samples, and an array y of class labels
(strings or integers), size ``[n_samples]``::


    >>> from sklearn import svm
    >>> X = [[0, 0], [1, 1]]
    >>> y = [0, 1]
    >>> clf = svm.SVC()
    >>> clf.fit(X, y)  # doctest: +NORMALIZE_WHITESPACE
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)

After being fitted, the model can then be used to predict new values::

    >>> clf.predict([[2., 2.]])
    array([1])

SVMs decision function depends on some subset of the training data,
called the support vectors. Some properties of these support vectors
can be found in members ``support_vectors_``, ``support_`` and
``n_support``::

    >>> # get support vectors
    >>> clf.support_vectors_
    array([[ 0.,  0.],
           [ 1.,  1.]])
    >>> # get indices of support vectors
    >>> clf.support_ # doctest: +ELLIPSIS
    array([0, 1]...)
    >>> # get number of support vectors for each class
    >>> clf.n_support_ # doctest: +ELLIPSIS
    array([1, 1]...)

.. _svm_multi_class:

Multi-class classification
--------------------------

:class:`SVC` and :class:`NuSVC` implement the "one-against-one"
approach (Knerr et al., 1990) for multi- class classification. If
``n_class`` is the number of classes, then ``n_class * (n_class - 1) / 2``
classifiers are constructed and each one trains data from two classes.
To provide a consistent interface with other classifiers, the
``decision_function_shape`` option allows to aggregate the results of the
"one-against-one" classifiers to a decision function of shape ``(n_samples,
n_classes)``::

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

On the other hand, :class:`LinearSVC` implements "one-vs-the-rest"
multi-class strategy, thus training n_class models. If there are only
two classes, only one model is trained::

    >>> lin_clf = svm.LinearSVC()
    >>> lin_clf.fit(X, Y) # doctest: +NORMALIZE_WHITESPACE
    LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
         intercept_scaling=1, loss='squared_hinge', max_iter=1000,
         multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
         verbose=0)
    >>> dec = lin_clf.decision_function([[1]])
    >>> dec.shape[1]
    4

See :ref:`svm_mathematical_formulation` for a complete description of
the decision function.

Note that the :class:`LinearSVC` also implements an alternative multi-class
strategy, the so-called multi-class SVM formulated by Crammer and Singer, by
using the option ``multi_class='crammer_singer'``. This method is consistent,
which is not true for one-vs-rest classification.
In practice, one-vs-rest classification is usually preferred, since the results
are mostly similar, but the runtime is significantly less.

For "one-vs-rest" :class:`LinearSVC` the attributes ``coef_`` and ``intercept_``
have the shape ``[n_class, n_features]`` and ``[n_class]`` respectively.
Each row of the coefficients corresponds to one of the ``n_class`` many
"one-vs-rest" classifiers and similar for the intercepts, in the
order of the "one" class.

In the case of "one-vs-one" :class:`SVC`, the layout of the attributes
is a little more involved. In the case of having a linear kernel,
The layout of ``coef_`` and ``intercept_`` is similar to the one
described for :class:`LinearSVC` described above, except that the shape of
``coef_`` is ``[n_class * (n_class - 1) / 2, n_features]``, corresponding to as
many binary classifiers. The order for classes
0 to n is "0 vs 1", "0 vs 2" , ... "0 vs n", "1 vs 2", "1 vs 3", "1 vs n", . .
. "n-1 vs n".

The shape of ``dual_coef_`` is ``[n_class-1, n_SV]`` with
a somewhat hard to grasp layout.
The columns correspond to the support vectors involved in any
of the ``n_class * (n_class - 1) / 2`` "one-vs-one" classifiers.
Each of the support vectors is used in ``n_class - 1`` classifiers.
The ``n_class - 1`` entries in each row correspond to the dual coefficients
for these classifiers.

This might be made more clear by an example:

Consider a three class problem with class 0 having three support vectors
:math:`v^{0}_0, v^{1}_0, v^{2}_0` and class 1 and 2 having two support vectors
:math:`v^{0}_1, v^{1}_1` and :math:`v^{0}_2, v^{1}_2` respectively.  For each
support vector :math:`v^{j}_i`, there are two dual coefficients.  Let's call
the coefficient of support vector :math:`v^{j}_i` in the classifier between
classes :math:`i` and :math:`k` :math:`\alpha^{j}_{i,k}`.
Then ``dual_coef_`` looks like this:

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

Scores and probabilities
------------------------

The :class:`SVC` method ``decision_function`` gives per-class scores 
for each sample (or a single score per sample in the binary case).
When the constructor option ``probability`` is set to ``True``,
class membership probability estimates
(from the methods ``predict_proba`` and ``predict_log_proba``) are enabled.
In the binary case, the probabilities are calibrated using Platt scaling:
logistic regression on the SVM's scores,
fit by an additional cross-validation on the training data.
In the multiclass case, this is extended as per Wu et al. (2004).

Needless to say, the cross-validation involved in Platt scaling
is an expensive operation for large datasets.
In addition, the probability estimates may be inconsistent with the scores,
in the sense that the "argmax" of the scores
may not be the argmax of the probabilities.
(E.g., in binary classification,
a sample may be labeled by ``predict`` as belonging to a class
that has probability <½ according to ``predict_proba``.)
Platt's method is also known to have theoretical issues.
If confidence scores are required, but these do not have to be probabilities,
then it is advisable to set ``probability=False``
and use ``decision_function`` instead of ``predict_proba``.

.. topic:: References:

 * Wu, Lin and Weng,
   `"Probability estimates for multi-class classification by pairwise coupling"
   <http://www.csie.ntu.edu.tw/~cjlin/papers/svmprob/svmprob.pdf>`_,
   JMLR 5:975-1005, 2004.
 
 
 * Platt
   `"Probabilistic outputs for SVMs and comparisons to regularized likelihood methods"
   <http://www.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/Platt1999.pdf>`.

Unbalanced problems
--------------------

In problems where it is desired to give more importance to certain
classes or certain individual samples keywords ``class_weight`` and
``sample_weight`` can be used.

:class:`SVC` (but not :class:`NuSVC`) implement a keyword
``class_weight`` in the ``fit`` method. It's a dictionary of the form
``{class_label : value}``, where value is a floating point number > 0
that sets the parameter ``C`` of class ``class_label`` to ``C * value``.

.. figure:: ../auto_examples/svm/images/sphx_glr_plot_separating_hyperplane_unbalanced_001.png
   :target: ../auto_examples/svm/plot_separating_hyperplane_unbalanced.html
   :align: center
   :scale: 75


:class:`SVC`, :class:`NuSVC`, :class:`SVR`, :class:`NuSVR` and
:class:`OneClassSVM` implement also weights for individual samples in method
``fit`` through keyword ``sample_weight``. Similar to ``class_weight``, these
set the parameter ``C`` for the i-th example to ``C * sample_weight[i]``.


.. figure:: ../auto_examples/svm/images/sphx_glr_plot_weighted_samples_001.png
   :target: ../auto_examples/svm/plot_weighted_samples.html
   :align: center
   :scale: 75


.. topic:: Examples:

 * :ref:`sphx_glr_auto_examples_svm_plot_iris.py`,
 * :ref:`sphx_glr_auto_examples_svm_plot_separating_hyperplane.py`,
 * :ref:`sphx_glr_auto_examples_svm_plot_separating_hyperplane_unbalanced.py`
 * :ref:`sphx_glr_auto_examples_svm_plot_svm_anova.py`,
 * :ref:`sphx_glr_auto_examples_svm_plot_svm_nonlinear.py`
 * :ref:`sphx_glr_auto_examples_svm_plot_weighted_samples.py`,


.. _svm_regression:

Regression
==========

The method of Support Vector Classification can be extended to solve
regression problems. This method is called Support Vector Regression.

The model produced by support vector classification (as described
above) depends only on a subset of the training data, because the cost
function for building the model does not care about training points
that lie beyond the margin. Analogously, the model produced by Support
Vector Regression depends only on a subset of the training data,
because the cost function for building the model ignores any training
data close to the model prediction.

There are three different implementations of Support Vector Regression: 
:class:`SVR`, :class:`NuSVR` and :class:`LinearSVR`. :class:`LinearSVR` 
provides a faster implementation than :class:`SVR` but only considers
linear kernels, while :class:`NuSVR` implements a slightly different
formulation than :class:`SVR` and :class:`LinearSVR`. See
:ref:`svm_implementation_details` for further details.

As with classification classes, the fit method will take as
argument vectors X, y, only that in this case y is expected to have
floating point values instead of integer values::

    >>> from sklearn import svm
    >>> X = [[0, 0], [2, 2]]
    >>> y = [0.5, 2.5]
    >>> clf = svm.SVR()
    >>> clf.fit(X, y) # doctest: +NORMALIZE_WHITESPACE
    SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
        kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
    >>> clf.predict([[1, 1]])
    array([ 1.5])


.. topic:: Examples:

 * :ref:`sphx_glr_auto_examples_svm_plot_svm_regression.py`



.. _svm_outlier_detection:

Density estimation, novelty detection
=======================================

One-class SVM is used for novelty detection, that is, given a set of
samples, it will detect the soft boundary of that set so as to
classify new points as belonging to that set or not. The class that
implements this is called :class:`OneClassSVM`.

In this case, as it is a type of unsupervised learning, the fit method
will only take as input an array X, as there are no class labels.

See, section :ref:`outlier_detection` for more details on this usage.

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


使用窍门
=====================


  * **避免数据复制**: 对于 :class:`SVC`， :class:`SVR`， :class:`NuSVC` 和
    :class:`NuSVR`， 如果数据是通过某些方法而不是用C有序的连续双精度，那它先会调用底层的C命令再复制。
    你可以通过检查它的 ``flags`` 属性，来确定给定的numpy数组是不是C连续的。

    对于 :class:`LinearSVC` (和 :class:`LogisticRegression
    <sklearn.linear_model.LogisticRegression>`) 的任何输入，都会以numpy数组形式，被复制和转换为
    用liblinear内部稀疏数据去表达（双精度浮点型float和非零部分的int32索引）。 
    如果你想要一个适合大规模的线性分类器，又不打算复制一个密集的C-contiguous双精度numpy数组作为输入，
    那我们建议你去使用 :class:`SGDClassifier
    <sklearn.linear_model.SGDClassifier>` 类作为替代。目标函数可以配置为和 :class:`LinearSVC`
    模型差不多相同的。

  * **内核的缓存大小**: 在大规模问题上，对于 :class:`SVC`, :class:`SVR`, :class:`nuSVC` 和
    :class:`NuSVR`, 内核缓存的大小会特别影响到运行时间。如果你有足够可用的RAM，不妨把它的 ``缓存大小`` 
    设得比默认的200(MB)要高，例如为 500(MB) 或者 1000(MB)。

  * **惩罚系数C的设置**:在合理的情况下， ``C`` 的默认选择为 ``1`` 。如果你有很多混杂的观察数据，
    你应该要去调小它。 ``C`` 越小，就能更好地去正规化估计。

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


