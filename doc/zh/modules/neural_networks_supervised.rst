.. _neural_networks_supervised:

==================================
神经网络模块（监督的）
==================================

.. currentmodule:: sklearn.neural_network


.. warning::

    此实现不适用于大规模应用程序。 特别是 scikit-learn 不支持 GPU。如果想要提高运行速度并使用基于 GPU 的实现以及为构建深度学习架构提供更多灵活性的框架，请参阅 :ref:`related_projects`。

.. _multilayer_perceptron:

多层感知器
======================

**多层感知器（MLP）**是一种监督学习算法，通过训练数据集来学习函数 :math:`f(\cdot): R^m \rightarrow R^o`，其中 :math:`m` 是输入的维数，:math:`o` 是数字的输出维数。 给定一组特征 :math:`X = {x_1, x_2, ..., x_m}` 和目标 :math:`y` ，它可以学习用于分类或回归的非线性函数。 与逻辑回归不同，在输入和输出层之间，可以有一个或多个非线性层，称为隐藏层。 图1 显示了具有标量输出的一个隐藏层 MLP。

.. figure:: ../images/multilayerperceptron_network.png
   :align: center
   :scale: 60%

   **图1：一个隐藏层MLP.**

最左层的输入层由一组神经元组成 :math:`\{x_i | x_1, x_2, ..., x_m\}` 表示输入要素。 每个隐藏层中的神经元将前一层的值转换为加权线性求和 :math:`w_1x_1 + w_2x_2 + ... + w_mx_m` 通过非线性激活函数 :math:`g(\cdot):R \rightarrow R` - 类似于双曲线 tan 函数。 输出层接收来自的值最后一个隐藏层并将其转换为输出值。

该模块包含公共属性 ``coefs_`` 和 ``intercepts_``。``coefs_`` 是权重矩阵的列表，其中权重矩阵的索引 :math:`i` 表示图层 :math:`i` 和图层 :math:`i+1` 的权重。 ``intercepts_`` 是偏移向量的列表，其中的向量索引 :math:`i` 表示添加到图层 :math:`i+1` 的偏差值。

多层感知器的优点是:

    + 学习非线性模型的能力.

    + 能实时学习模型（在线学习） 使用``partial_fit``.


多层感知器（MLP）的缺点包括:

    + 具有隐藏层的 MLP 当多于一个本地最低限度时会存在非凸损失函数。 因此不同的随机重量
     初始化可能导致不同的验证准确性.

    + MLP 需要调整一些超参数，例如隐藏的神经元，层和迭代数量.

    + MLP 对特征缩放很敏感.

请参阅包含一些这样的缺点的 :ref:`实用使用技巧 <mlp_tips>` 部分。


分类
==============

类 :class:`MLPClassifier` 实现使用 `Backpropagation <http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm>`_ 进行训练的多层感知器（MLP）。

MLP 算法列在两个 array 上训练: size 为 (n_samples, n_features) 的 array X，其保存表示为浮点特征向量的训练样本; 和 size (n_samples,) 的 array y，它保存训练样本的目标值（类标签）::

    >>> from sklearn.neural_network import MLPClassifier
    >>> X = [[0., 0.], [1., 1.]]
    >>> y = [0, 1]
    >>> clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
    ...                     hidden_layer_sizes=(5, 2), random_state=1)
    ...
    >>> clf.fit(X, y)                         # doctest: +NORMALIZE_WHITESPACE
    MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
           beta_1=0.9, beta_2=0.999, early_stopping=False,
           epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
           learning_rate_init=0.001, max_iter=200, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
           solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
           warm_start=False)

拟合（训练）后，该模型可以预测新样本的标签::

    >>> clf.predict([[2., 2.], [-1., -2.]])
    array([1, 0])

MLP 可以将非线性模型适用于训练数据。``clf.coefs_`` 包含构成模型参数的权重矩阵::

    >>> [coef.shape for coef in clf.coefs_]
    [(2, 5), (5, 2), (2, 1)]

目前， :class:`MLPClassifier` 只支持运行允许使用概率估计 ``predict_proba`` 方法得到交叉熵损失函数。

MLP 算法使用反向传播的方式。 更准确地说，它训练使用某种形式梯度下降和梯度使用反向传播计算。 对于 classification （分类），它最小化交叉熵损失函数，给出一个向量的概率估计 :math:`P(y|x)` 每个样本 :math:`x`::

    >>> clf.predict_proba([[2., 2.], [1., 2.]])  # doctest: +ELLIPSIS
    array([[  1.967...e-04,   9.998...-01],
           [  1.967...e-04,   9.998...-01]])

:class:`MLPClassifier` 通过应用 `Softmax <https://en.wikipedia.org/wiki/Softmax_activation_function>`_ 作为输出函数来支持多类分类。

此外，该模型支持 :ref:`multi-label classification <multiclass>`，其中样本可以属于多个类。 对于每个类，原始输出通过逻辑函数。 大于或等于 0.5 的值将舍入为 1，否则为 0.对于样本的预测输出，值为 1 的索引表示该样本的分配类别::

    >>> X = [[0., 0.], [1., 1.]]
    >>> y = [[0, 1], [1, 1]]
    >>> clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
    ...                     hidden_layer_sizes=(15,), random_state=1)
    ...
    >>> clf.fit(X, y)                         # doctest: +NORMALIZE_WHITESPACE
    MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
           beta_1=0.9, beta_2=0.999, early_stopping=False,
           epsilon=1e-08, hidden_layer_sizes=(15,), learning_rate='constant',
           learning_rate_init=0.001, max_iter=200, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
           solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
           warm_start=False)
    >>> clf.predict([[1., 2.]])
    array([[1, 1]])
    >>> clf.predict([[0., 0.]])
    array([[0, 1]])

有关更多信息，请参阅下面的示例和文档字符串 :meth:`MLPClassifier.fit`。

.. topic:: 示例:

 * :ref:`sphx_glr_auto_examples_neural_networks_plot_mlp_training_curves.py`
 * :ref:`sphx_glr_auto_examples_neural_networks_plot_mnist_filters.py`

回归
==========

类 :class:`MLPRegressor` 实现了一个多层感知器（MLP），它在输出层中使用没有激活功能的反向传播进行训练，也可以看作是使用身份函数作为激活函数。 因此，它使用平方误差作为损失函数，输出是一组连续值。

:class:`MLPRegressor` 还支持多输出回归，其中样本可以有多个目标。

正则化
==============

Both :class:`MLPRegressor` and :class:`MLPClassifier` use parameter ``alpha``
for regularization (L2 regularization) term which helps in avoiding overfitting
by penalizing weights with large magnitudes. Following plot displays varying
decision function with value of alpha.

:class:`MLPRegressor` 和 :class:`MLPClassifier` 使用参数 ``alpha`` 用于正规化（L2 正则化）术语，有助于避免过度拟合通过惩罚大量的权重。 以下图表显示不同具有 alpha 值的决策函数。

.. figure:: ../auto_examples/neural_networks/images/sphx_glr_plot_mlp_alpha_001.png
   :target: ../auto_examples/neural_networks/plot_mlp_alpha.html
   :align: center
   :scale: 75

有关详细信息，请参阅下面的示例。

.. topic:: 示例:

 * :ref:`sphx_glr_auto_examples_neural_networks_plot_mlp_alpha.py`

算法
==========

MLP 使用 `Stochastic Gradient Descent（随机梯度下降）(SGD)
<https://en.wikipedia.org/wiki/Stochastic_gradient_descent>`_,
`Adam <http://arxiv.org/abs/1412.6980>`_, 或者 `L-BFGS <https://en.wikipedia.org/wiki/Limited-memory_BFGS>`__ 进行训练。
Stochastic Gradient Descent （随机梯度下降）(SGD) 使用渐变梯度更新参数相对于需要适应的参数的损失函数，即.

.. math::

    w \leftarrow w - \eta (\alpha \frac{\partial R(w)}{\partial w}
    + \frac{\partial Loss}{\partial w})

其中 :math:`\eta` 是控制 parameter space search （参数空间搜索）中的 step-size （步长）的 learning rate （学习率）。 :math:`Loss` 是用于网络的 loss function （损失函数）。

更多细节可以在这个文档中找到 `SGD <http://scikit-learn.org/stable/modules/sgd.html>`_

Adam 类似于 SGD，因为它是 stochastic optimizer （随机优化器），但它可以根据低阶矩的自适应估计自动调整更新参数的量。

使用 SGD 或 Adam ，训练支持在线和小批量学习。

L-BFGS 是近似表示函数的二阶偏导数的 Hessian 矩阵的求解器。 此外，它近似于 Hessian 矩阵的逆来执行参数更新。 实现使用 Scipy 版本的 `L-BFGS
<http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html>`_。

如果所选择的求解器是 'L-BFGS'，训练不支持在线或小批量学习。

复杂性
==========

假设有 :math:`n` 训练样本， :math:`m` 特征， :math:`k` 隐藏层，每个包含 :math:`h` 神经元 - 为简单起见， :math:`o` 输出神经元。 反向传播的时间复杂度是 :math:`O(n\cdot m \cdot h^k \cdot o \cdot i)` ，其中 :math:`i` 是数字的迭代。 由于反向传播具有高时间复杂性，因此是可取的从较小数量的隐藏神经元和几个隐藏层开始训练。

数学公式
========================

给出一组训练示例 :math:`(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)` 其中 :math:`x_i \in \mathbf{R}^n` 和 :math:`y_i \in \{0, 1\}`，一个隐藏第一层隐藏神经元 MLP 学习功能 :math:`f(x) = W_2 g(W_1^T x + b_1) + b_2` 其中 :math:`W_1 \in \mathbf{R}^m` 和 :math:`W_2, b_1, b_2 \in \mathbf{R}` 是模型参数.
:math:`W_1, W_2` 表示输入层的权重隐藏层 和 :math:`b_1, b_2` 表示添加的偏见隐藏层和输出层.
:math:`g(\cdot) : R \rightarrow R` 是激活函数，默认设置为双曲线.
它被赋予，

.. math::
      g(z)= \frac{e^z-e^{-z}}{e^z+e^{-z}}

对于二分类， :math:`f(x)` 通过逻辑函数 :math:`g(z)=1/(1+e^{-z})` 来获得 0 到 1 之间的输出值。 一个阈值设置为 0.5，将分配大于或等于 0.5 的输出样本到 positive class （正类），其余的到 negative class （负类）。

如果有两个以上的类，则 :math:`f(x)` 本身将是一个向量 size (n_classes,) 。 而不是通过逻辑功能，它通过通过 softmax 函数，它被写为，

.. math::
      \text{softmax}(z)_i = \frac{\exp(z_i)}{\sum_{l=1}^k\exp(z_l)}

其中 :math:`z_i` 表示中， :math:`i` 是 softmax 输入的元素，它对应于类 :math:`i` ，和 :math:`K` 是类的数量。结果是一个包含可能性的向量 :math:`x` 属于每个类别。 输出是具有最高概率的类。

在回归中，输出依然如下:math:`f(x)`; 因此，输出激活函数只是 identity function 。

MLP 根据问题类型使用不同的 loss functions （损失函数）。 二分类的情况下，loss
function （损失函数）的功能是交叉熵，如下，

.. math::

    Loss(\hat{y},y,W) = -y \ln {\hat{y}} - (1-y) \ln{(1-\hat{y})} + \alpha ||W||_2^2

其中 :math:`\alpha ||W||_2^2` 是一个 L2 正则化术语（又称惩罚）惩罚复杂的模型; 和 :math:`\alpha > 0` 是非负数超参数控制惩罚的大小。

对于回归，MLP 使用平方误差损失函数; 写成，

.. math::

    Loss(\hat{y},y,W) = \frac{1}{2}||\hat{y} - y ||_2^2 + \frac{\alpha}{2} ||W||_2^2

从初始随机权重开始，多层感知器（MLP）最小化通过重复更新这些权重的损失函数。 计算完损失之后，反向传递将其从输出层传播到前一个为每个权重参数提供旨在减少的更新损失值。

在 gradient descent （梯度下降）中，梯度 :math:`\nabla Loss_{W}` 的损失与 respect
to the weights （权重）被计算和扣除 :math:`W`。用公式表示为，

.. math::
    W^{i+1} = W^i - \epsilon \nabla {Loss}_{W}^{i}


其中 :math:`i` 是迭代步骤， :math:`\epsilon` 是学习率值大于 0。

算法要么在达到预设的最大迭代次数时停止; 要么当损失的改善低于一定数量的时候停止。

.. _mlp_tips:

实用技巧
=====================

  * 多层感知器对 feature scaling （特征的缩放）是敏感的，所以它强烈建议您 scale your data （缩放数据）。 例如，缩放每个属性在输入向量 X 到 [0,1] 或 [-1，+1] ，或 standardize （标准化）以使它具有平均值 0 和方差 1.注意，您必须应用 *相同的* 缩放到测试集中以获得有意义的结果。 您可以使用 :class:`StandardScaler` 进行 standardization （标准化）。

      >>> from sklearn.preprocessing import StandardScaler  # doctest: +SKIP
      >>> scaler = StandardScaler()  # doctest: +SKIP
      >>> # Don't cheat - fit only on training data
      >>> scaler.fit(X_train)  # doctest: +SKIP
      >>> X_train = scaler.transform(X_train)  # doctest: +SKIP
      >>> # apply same transformation to test data
      >>> X_test = scaler.transform(X_test)  # doctest: +SKIP

    一个替代和推荐的方案是使用在 :class:`Pipeline` 中的 :class:`StandardScaler`

  * 找到一个合理的正则化参数 :math:`\alpha` ，最好的方法是使用 :class:`GridSearchCV` 通常范围是在 ``10.0 ** -np.arange(1, 7)`` 。

  * 经验上，我们观察到 `L-BFGS` 收敛速度更快并且在小数据集上有更好的解决方案。对于规模相对比较大的数据集，但是，`Adam` 是非常强大的。 它通常会迅速收敛，并提供相当不错的表现。 另一方面，如果 learning rate （学习率）正确调整， 使用 momentum 或 nesterov's momentum 的 `SGD` 可以比这两种算法更好。

*经验上，我们观察到“L-BFGS”收敛速度更快 在小数据集上有更好的解决方案。 对于比较大 数据集，但是，“adam”是非常强大的。 它通常会收敛 迅速，并提供相当不错的表现。 `SGD`有动力或 另一方面，尼斯特罗夫的势头可以比 这两种算法如果学习率正确调整。

使用 warm_start 的更多控制
============================

如果您希望更多地控制 SGD 中的 stopping criteria （停止标准）或 learning rate （学习率），或者想要进行额外的监视，使用 ``warm_start=True`` 和 ``max_iter=1`` 并且自身迭代可能会有所帮助::

    >>> X = [[0., 0.], [1., 1.]]
    >>> y = [0, 1]
    >>> clf = MLPClassifier(hidden_layer_sizes=(15,), random_state=1, max_iter=1, warm_start=True)
    >>> for i in range(10):
    ...     clf.fit(X, y)
    ...     # additional monitoring / inspection # doctest: +ELLIPSIS
    MLPClassifier(...

.. topic:: 参考文献:

    * `"Learning representations by back-propagating errors."
      <http://www.iro.umontreal.ca/~pift6266/A06/refs/backprop_old.pdf>`_
      Rumelhart, David E., Geoffrey E. Hinton, and Ronald J. Williams.

    * `"Stochastic Gradient Descent" <http://leon.bottou.org/projects/sgd>`_ L. Bottou - Website, 2010.

    * `"Backpropagation" <http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm>`_
      Andrew Ng, Jiquan Ngiam, Chuan Yu Foo, Yifan Mai, Caroline Suen - Website, 2011.

    * `"Efficient BackProp" <http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf>`_
      Y. LeCun, L. Bottou, G. Orr, K. Müller - In Neural Networks: Tricks
      of the Trade 1998.

    *  `"Adam: A method for stochastic optimization."
       <http://arxiv.org/pdf/1412.6980v8.pdf>`_
       Kingma, Diederik, and Jimmy Ba. arXiv preprint arXiv:1412.6980 (2014).
