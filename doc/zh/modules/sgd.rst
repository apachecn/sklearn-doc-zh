.. _sgd:

===========================
随机梯度下降
===========================

.. 当前模块:: sklearn.linear_model

**随机梯度下降(SGD)** 是一个简单但又非常有效的方法，主要用于凸损失函数下对线性分类器的学习，例如(线性) `支持向量机
<https://en.wikipedia.org/wiki/Support_vector_machine>`_ 和 `Logistic
回归 <https://en.wikipedia.org/wiki/Logistic_regression>`_。
尽管SGD在机器学习社区已经存在了很长时间, 但是最近在大规模学习方面SGD获得了相当大的关注。

SGD已成功应用于文本分类和自然语言处理中经常遇到的大规模和稀疏的机器学习问题。考虑到数据是稀疏的，本模块的分类器可以
轻易的处理超过10^5的训练样本和超过10^5的特征。

随机梯度下降法的优势:

    + 效率。

    + 易于实现 (有大量优化代码的机会)。

随机梯度下降法的劣势:

    + SGD需要一些超参数，例如正则化
      参数和迭代次数。

    + SGD对特征缩放敏感。

分类
==============

.. 警告::

  在拟合模型前，确保你重新排列了(打乱)你的训练数据，或者
  在每次迭代后用 ``shuffle=True`` 来打乱。

:class:`SGDClassifier` 类实现了一个简单的随机梯度下降学习程序, 支持不同的loss functions（损失函数）和
penalties for classification（分类处罚）。

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_sgd_separating_hyperplane_001.png
   :target: ../auto_examples/linear_model/plot_sgd_separating_hyperplane.html
   :align: center
   :scale: 75

作为其他的分类器, SGD必须拟合两个数组：保存训练样本的大小为[n_samples, n_features]的数组X以及保存训练样本
目标值（类标签）的大小为[n_samples]的数组Y::

    >>> from sklearn.linear_model import SGDClassifier
    >>> X = [[0., 0.], [1., 1.]]
    >>> y = [0, 1]
    >>> clf = SGDClassifier(loss="hinge", penalty="l2")
    >>> clf.fit(X, y)
    SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
           eta0=0.0, fit_intercept=True, l1_ratio=0.15,
           learning_rate='optimal', loss='hinge', max_iter=5, n_iter=None,
           n_jobs=1, penalty='l2', power_t=0.5, random_state=None,
           shuffle=True, tol=None, verbose=0, warm_start=False)


拟合后，可以用该模型来预测新值::

    >>> clf.predict([[2., 2.]])
    array([1])

SGD 通过训练数据来拟合一个线性模型。成员 ``coef_`` 保存
模型参数::

    >>> clf.coef_                                         # doctest: +ELLIPSIS
    array([[ 9.9...,  9.9...]])

成员``intercept_`` 保存intercept（截距） (aka offset or bias(偏移))::

    >>> clf.intercept_                                    # doctest: +ELLIPSIS
    array([-9.9...])

模型是否使用intercept（截距）, i.e. a biased
hyperplane(一个偏置的超平面), 是由参数``fit_intercept``控制的。

使用:meth:`SGDClassifier.decision_function`::来获得与超平面的signed distance(符号距离)。

    >>> clf.decision_function([[2., 2.]])                 # doctest: +ELLIPSIS
    array([ 29.6...])

具体的loss function(损失函数)可以通过``loss``
参数来设置。 :class:`SGDClassifier` 支持以下的loss functions(损失函数)：

  * ``loss="hinge"``: (soft-margin) linear Support Vector Machine ((软-间隔)线性支持向量机)，
  * ``loss="modified_huber"``: smoothed hinge loss  (平滑的hinge损失)，
  * ``loss="log"``: logistic regression (logistic 回归)，
  * and all regression losses below(以及所有的回归损失)。

前两个loss functions（损失函数）是懒惰的，如果一个例子违反了margin constraint（边界约束），它们仅更新模型的参数, 这使得训练非常有效率
,即使在使用L2 penalty（惩罚）也许结果也会是稀疏的模型。

使用 ``loss="log"`` 或者 ``loss="modified_huber"`` 启用
``predict_proba`` 方法, 其给出每个样本 :math:`x` 的概率估计 
:math:`P(y|x)` 的一个向量：

    >>> clf = SGDClassifier(loss="log").fit(X, y)
    >>> clf.predict_proba([[1., 1.]])                      # doctest: +ELLIPSIS
    array([[ 0.00...,  0.99...]])

concrete penalty（具体的惩罚）可以通过 ``penalty`` 参数来设定。
SGD支持以下penalties（惩罚）:

  * ``penalty="l2"``: L2 norm penalty on ``coef_``.
  * ``penalty="l1"``: L1 norm penalty on ``coef_``.
  * ``penalty="elasticnet"``: Convex combination of L2 and L1;
    ``(1 - l1_ratio) * L2 + l1_ratio * L1``.

默认设置为 ``penalty="l2"``。L1 penalty（惩罚）导致稀疏解，使得大多数系数为零。Elastic Net（弹性网）解决了
在高度相关属性上L1 penalty（惩罚）的一些不足。参数 ``l1_ratio`` 控制了L1 和 L2 penalty（惩罚）的凸组合。

:class:`SGDClassifier` 通过在将多个二进制分类器组合在"one versus all" (OVA)方案中来支持多类分类。对于
每一个 :math:`K` 类, 学习了一个二进制分类器来区分自身和其他 :math:`K-1` 个类。在测试阶段，我们计算了每个分类
的confidence score（置信度分数）（也就是与超平面的距离）并选择由最高置信度的类。下图显示了在iris（鸢尾花）数据集上的OVA方法。
虚线表示三个OVA分类器; 背景色显示了由三个分类器引起的绝策面。

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_sgd_iris_001.png
   :target: ../auto_examples/linear_model/plot_sgd_iris.html
   :align: center
   :scale: 75

在多类分类的情况下， ``coef_`` 是 ``shape=[n_classes, n_features]`` 的
一个二维数组， ``intercept_`` is ``shape=[n_classes]`` 的一个一位数组。
``coef_`` 的第i行保存了第i类的OVA分类器的权重向量；
类以升序索引 （参照属性 ``classes_``）。
注意，原则上，由于它们允许创建一个概率模型，所以
``loss="log"`` 和 ``loss="modified_huber"`` 更适合于
one-vs-all 分类。

:class:`SGDClassifier` 通过拟合参数 ``class_weight`` 和 ``sample_weight`` 来支持加权类
和加权实例。更多信息请参照下面的示例和 :meth:`SGDClassifier.fit` 的
文档。

.. topic:: 示例:

 - :ref:`sphx_glr_auto_examples_linear_model_plot_sgd_separating_hyperplane.py`,
 - :ref:`sphx_glr_auto_examples_linear_model_plot_sgd_iris.py`
 - :ref:`sphx_glr_auto_examples_linear_model_plot_sgd_weighted_samples.py`
 - :ref:`sphx_glr_auto_examples_linear_model_plot_sgd_comparison.py`
 - :ref:`sphx_glr_auto_examples_svm_plot_separating_hyperplane_unbalanced.py` (See the `Note`)

:class:`SGDClassifier` 支持 averaged SGD (ASGD)。Averaging（平均值）可以通过
设置 ```average=True``` 来启用。ASGD的工作原理是在一个样本上的每次迭代上将plain SGD（平均SGD）的系数
平均。当使用ASGD时，学习速率可以更大甚至是恒定，主要是在一些数据集上加快训练时间。

对于一个 logistic loss（logistic 损失）的分类，具有averaging strategy（平衡策略）
的SGD的另一变种可用于Stochastic Average Gradient（随即平均梯度）(SAG)
算法，作为 :class:`LogisticRegression` 的（solver）求解器。

Regression（回归）
==========

:class:`SGDRegressor` 类实现了一个简单的随即梯度
下降学习程序，它支持不同的损失函数和
惩罚来拟合线性回归模型。 :class:`SGDRegressor` 是
是非常适用于有大量训练样本（>10.000)的回归
问题,对于其他问题，我们简易使用 :class:`Ridge`，
:class:`Lasso`，或 :class:`ElasticNet`。

具体的损失函数可以通过 ``loss``
参数设置。 :class:`SGDRegressor` 支持一下的损失函数：

  * ``loss="squared_loss"``: Ordinary least squares,
  * ``loss="huber"``: Huber loss for robust regression,
  * ``loss="epsilon_insensitive"``: linear Support Vector Regression.

Huber 和 epsilon-insensitive 损失函数可用于
robust regression（稳健回归）。不敏感区域的宽度必须通过参数
``epsilon`` 来设定。这个参数取决于目标变量的规模。

:class:`SGDRegressor` 支持averaged（平均）SGD作为 :class:`SGDClassifier`。
平均值可以通过设置 ```average=True``` 来启用。

对于一个squared loss（平方损失）和一个l2 penalty（l2惩罚）的回归，具有averaging strategy（平衡策略）
的SGD的另一变种可用于Stochastic Average Gradient（随即平均梯度）(SAG)
算法，作为 :class:`Ridge` 中的solver（求解器）。


Stochastic Gradient Descent for sparse data（稀疏数据的随机梯度下降）
===========================================

.. 注意:: 由于一个对于截距是缩小的学习率，稀疏实现与密集实现相比产生的结果略有不同。

在 `scipy.sparse <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_
支持的格式中，任意矩阵都有对稀疏数据的内置支持。但是，为了获得最好的效率，请使用 `scipy.sparse.csr_matrix
<http://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html>`_中定义的格式.

.. topic:: 示例:

 - :ref:`sphx_glr_auto_examples_text_document_classification_20newsgroups.py`

Complexity（复杂度）
==========

SGD主要的优点在于它的效率，在训练实例
的数量上基本是线性的。假如 X 是大小为(n, p)的矩阵，
训练成本为 :math:`O(k n \bar p)`，其中 k 是迭代
次数， :math:`\bar p` 是每个样本
非零属性的平均数。

但是，最近的理论结果表明，在训练集大小增加时，
运行时得到的一些期望的优化精度不会增加。

Tips on Practical Use（实用小贴士）
=====================

  * 随机梯度下降法对特征缩放很敏感，因此
    强烈建议您缩放您的数据。例如,将输入
    向量X上的每个属性缩放到[0,1]或[- 1，+1]， 或
    将其标准化，使其均值为0，方差为1。请注意，必须将 *相同* 的
    缩放应用于对应的测试向量中，以获得有意义的
    结果。使用 :class:`StandardScaler`: 很容易做到这一点：

      from sklearn.preprocessing import StandardScaler
      scaler = StandardScaler()
      scaler.fit(X_train)  # Don't cheat - fit only on training data
      X_train = scaler.transform(X_train)
      X_test = scaler.transform(X_test)  # apply same transformation to test data

    假如你的属性有一个内在尺度（例如词频或
    指标特征）就不需要缩放。

  * 最好使用 :class:`GridSearchCV` 找到一个合理的
    正则化项 :math:`\alpha` ， 它的范围通常在
     ``10.0**-np.arange(1,7)`` 。

  * 经验性地，我们发现SGD在观察约
    10^6 训练样本后收敛。因此，对于迭代次数的一个
    合理的第一猜想是 ``n_iter = np.ceil(10**6 / n)``，
    其中 ``n`` 训练集的大小。

  * 假如将SGD应用于使用PCA做特征提取，我们发现
    通过常数 `c` 来缩放特征值是明智的，
    这样，训练数据的平均L2平均值等于1。

  * 我们发现 Averaged SGD 在一个更大的特征和一个更高的eta0上工作的最好。
    

.. topic:: References:

 * `"Efficient BackProp" <http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf>`_
   Y. LeCun, L. Bottou, G. Orr, K. Müller - In Neural Networks: Tricks
   of the Trade 1998.

.. _sgd_mathematical_formulation:

Mathematical formulation（数学描述）
========================

给定一组训练样本 :math:`(x_1, y_1), \ldots, (x_n, y_n)` 其中
:math:`x_i \in \mathbf{R}^m` ， :math:`y_i \in \{-1,1\}`， 我们的目标是
一个线性 scoring function（评价函数） :math:`f(x) = w^T x + b` ，其中模型参数
:math:`w \in \mathbf{R}^m` ，截距 :math:`b \in \mathbf{R}`。为了
做预测， 我们只需要看 :math:`f(x)` 的符号。
找到模型参数的一般选择是通过最小化由以下式子给出的
正则化训练误差

.. math::

    E(w,b) = \frac{1}{n}\sum_{i=1}^{n} L(y_i, f(x_i)) + \alpha R(w)

其中 :math:`L` 衡量模型(mis)拟合程度的损失函数，
:math:`R` 是惩罚模型复杂度的正则化项（也叫作惩罚）;
:math:`\alpha > 0` 是一个非负超平面。

:math:`L` 的不同选择需要不同的分类器，例如

   - Hinge: (soft-margin) Support Vector Machines.
   - Hinge: (软-间隔) 支持向量机。
   - Log:   Logistic Regression.
   - Log:   Logistic 回归。
   - Least-Squares: Ridge Regression.
   - Least-Squares: 岭回归。
   - Epsilon-Insensitive: (soft-margin) Support Vector Regression.
   - Epsilon-Insensitive: (软-间隔) 支持向量回归。

所有上述损失函数可以看作是错误分类误差的上限（0 - 1损失），
如下图所示。

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_sgd_loss_functions_001.png
    :target: ../auto_examples/linear_model/plot_sgd_loss_functions.html
    :align: center
    :scale: 75

正则化项 :math:`R` 受欢迎的选择包括：

   - L2 norm: :math:`R(w) := \frac{1}{2} \sum_{i=1}^{n} w_i^2`,
   - L1 norm: :math:`R(w) := \sum_{i=1}^{n} |w_i|`, which leads to sparse
     solutions（）.
   - Elastic Net: :math:`R(w) := \frac{\rho}{2} \sum_{i=1}^{n} w_i^2 + (1-\rho) \sum_{i=1}^{n} |w_i|`, a convex combination of L2 and L1, where :math:`\rho` is given by ``1 - l1_ratio``.

下图显示当 :math:`R(w) = 1` 时参数空间中
不同正则项的轮廓。

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_sgd_penalties_001.png
    :target: ../auto_examples/linear_model/plot_sgd_penalties.html
    :align: center
    :scale: 75

SGD
---

随机梯度下降法一种无约束优化问题的
的优化方法。与（批量）梯度下降法相反，SGD
通过一次只考虑单个训练样本来近似 :math:`E(w,b)` 真实的梯度。

:class:`SGDClassifier` 类s实现了一个一阶SGD学习
程序。 算法在训练样本上遍历，并且对每个样本
根据由以下式子给出的更新规则来更新模型参数

.. math::

    w \leftarrow w - \eta (\alpha \frac{\partial R(w)}{\partial w}
    + \frac{\partial L(w^T x_i + b, y_i)}{\partial w})

其中 :math:`\eta` 是在参数空间中控制步长的学习速率。
截距 :math:`b` 的更新类似但不需要正则化。

学习率 :math:`\eta` 可以是常数或者逐渐减小。对于
分类来说， 默认学习率 schedule（调度） （``learning_rate='optimal'``）
由下式给出。

.. math::

    \eta^{(t)} = \frac {1}{\alpha  (t_0 + t)}

其中 :math:`t` 是时间步长（总共有 `n_samples * n_iter`
时间步长）， :math:`t_0` 是由Léon Bottou提出的启发式决定的，
这样，预期的初始更新可以与权重的期望大小相比较
（这假设训练样本的规范近似1）。
在 :class:`BaseSGD` 中的 ``_init_t`` 中可以找到确切的定义。


对于回归来说，默认的学习率是反向缩放
(``learning_rate='invscaling'``)，由下式给出

.. math::

    \eta^{(t)} = \frac{eta_0}{t^{power\_t}}

其中 :math:`eta_0` 和 :math:`power\_t` 是用户通过 ``eta0`` 和 ``power_t`` 分别选择的超参数。

学习速率常数使用使用 ``learning_rate='constant'`` ，并使用 ``eta0``
来指定学习速率。

模型参数可以通过成员 ``coef_`` and
``intercept_`` 来访问：

     - Member ``coef_`` holds the weights :math:`w`

     - Member ``intercept_`` holds :math:`b`

.. topic:: 参考文献：

 * `"Solving large scale linear prediction problems using stochastic
   gradient descent algorithms"
   <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.58.7377>`_
   T. Zhang - In Proceedings of ICML '04.

 * `"Regularization and variable selection via the elastic net"
   <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.124.4696>`_
   H. Zou, T. Hastie - Journal of the Royal Statistical Society Series B,
   67 (2), 301-320.

 * `"Towards Optimal One Pass Large Scale Learning with
   Averaged Stochastic Gradient Descent"
   <http://arxiv.org/pdf/1107.2490v2.pdf>`_
   Xu, Wei


Implementation details（实现细节）
======================

他对SGD的实现受到了Léon Bottou `Stochastic Gradient SVM
<http://leon.bottou.org/projects/sgd>`_  的影响。类似于SvmSGD，
权值向量表示为在L2正则化的情况下允许有效的
权值更新的标量和向量的乘积。
在稀疏特征向量的情况下，截距是以更小的学习率（乘以0.01）
更新的，导致了它更频繁的更新。
训练样本按顺序选取，每次观察后，学习率降低。
我们从 Shalev-Shwartz 等人那里获得了 learning rate schedule ( 学习率计划表 )。
对于多类分类，使用 “one versus all” 方法。
我们使用 Tsuruoka 等人提出的 truncated gradient algorithm （截断梯度算法）
2009年为L1正则化（和 Elastic Net ）。
代码是用 Cython 编写的。

.. topic:: 参考文献:

 * `"Stochastic Gradient Descent" <http://leon.bottou.org/projects/sgd>`_ L. Bottou - Website, 2010.

 * `"The Tradeoffs of Large Scale Machine Learning" <http://leon.bottou.org/slides/largescale/lstut.pdf>`_ L. Bottou - Website, 2011.

 * `"Pegasos: Primal estimated sub-gradient solver for svm"
   <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.74.8513>`_
   S. Shalev-Shwartz, Y. Singer, N. Srebro - In Proceedings of ICML '07.

 * `"Stochastic gradient descent training for l1-regularized log-linear models with cumulative penalty"
   <http://www.aclweb.org/anthology/P/P09/P09-1054.pdf>`_
   Y. Tsuruoka, J. Tsujii, S. Ananiadou -  In Proceedings of the AFNLP/ACL '09.
