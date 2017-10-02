.. _neural_networks_supervised:
神经网络监督
==================================
Neural network models (supervised)
==================================
神经网络模块（监督）

.. currentmodule:: sklearn.neural_network


.. warning::

    This implementation is not intended for large-scale applications. In particular,
    scikit-learn offers no GPU support. For much faster, GPU-based implementations,
    as well as frameworks offering much more flexibility to build deep learning
    architectures, see  :ref:`related_projects`.
    
    提示
此实现不适用于大规模应用程序。 特别是scikit-learn不支持GPU。如果想要提高运行速度并使用基于GPU的实现以及为构建深度学习架构提供更多灵活性的框架，请参阅：ref：`related_projects`。

.. _multilayer_perceptron:

Multi-layer Perceptron
======================
多层感知器

**Multi-layer Perceptron (MLP)** is a supervised learning algorithm that learns
a function :math:`f(\cdot): R^m \rightarrow R^o` by training on a dataset,
where :math:`m` is the number of dimensions for input and :math:`o` is the
number of dimensions for output. Given a set of features :math:`X = {x_1, x_2, ..., x_m}`
and a target :math:`y`, it can learn a non-linear function approximator for either
classification or regression. It is different from logistic regression, in that
between the input and the output layer, there can be one or more non-linear
layers, called hidden layers. Figure 1 shows a one hidden layer MLP with scalar
output.

多层感知器（MLP）是一种监督学习算法，通过训练数据集来学习函数f（\ cdot）：R ^ m \ rightarrow R ^ o，其中m是输入的维数，o是数字的输出尺寸。 给定一组特征X = {x_1，x_2，...，x_m}和目标y，它可以学习用于分类或回归的非线性函数。 与逻辑回归不同，在输入和输出层之间，可以有一个或多个非线性层，称为隐藏层。 图1显示了具有标量输出的一个隐藏层MLP。

.. figure:: ../images/multilayerperceptron_network.png
   :align: center
   :scale: 60%

   **Figure 1 : One hidden layer MLP.**

图1：一个隐藏层MLP。

The leftmost layer, known as the input layer, consists of a set of neurons
:math:`\{x_i | x_1, x_2, ..., x_m\}` representing the input features. Each
neuron in the hidden layer transforms the values from the previous layer with
a weighted linear summation :math:`w_1x_1 + w_2x_2 + ... + w_mx_m`, followed
by a non-linear activation function :math:`g(\cdot):R \rightarrow R` - like
the hyperbolic tan function. The output layer receives the values from the
last hidden layer and transforms them into output values.

最左层的输入层由一组神经元组成：`\ {x_i | x_1，x_2，...，x_m \}表示输入要素。 每个隐藏层中的神经元将前一层的值转换为加权线性求和：`w_1x_1 + w_2x_2 + ... + w_mx_m`通过非线性激活函数：`g（\ cdot）：R \ rightarrow R` - like双曲线tan函数。 输出层接收来自的值最后一个隐藏层并将其转换为输出值。

The module contains the public attributes ``coefs_`` and ``intercepts_``.
``coefs_`` is a list of weight matrices, where weight matrix at index
:math:`i` represents the weights between layer :math:`i` and layer
:math:`i+1`. ``intercepts_`` is a list of bias vectors, where the vector
at index :math:`i` represents the bias values added to layer :math:`i+1`.

该模块包含公共属性``coefs_``和``intercepts_``。``coefs_``是权重矩阵的列表，其中权重矩阵的索引“i”表示图层“i”和图层`1 +1`的权重。 ``intercepts_``是偏移向量的列表，其中的向量索引`i`表示添加到图层`i + 1`的偏差值。

The advantages of Multi-layer Perceptron are:

    + Capability to learn non-linear models.

    + Capability to learn models in real-time (on-line learning)
      using ``partial_fit``.
多层感知器的优点是：

     学习非线性模型的能力。 

     能实时学习模型（在线学习） 使用``partial_fit``。


The disadvantages of Multi-layer Perceptron (MLP) include:

    + MLP with hidden layers have a non-convex loss function where there exists
      more than one local minimum. Therefore different random weight
      initializations can lead to different validation accuracy.

    + MLP requires tuning a number of hyperparameters such as the number of
      hidden neurons, layers, and iterations.

    + MLP is sensitive to feature scaling.

Please see :ref:`Tips on Practical Use <mlp_tips>` section that addresses
some of these disadvantages.

多层感知器（MLP）的缺点包括：

     具有隐藏层的MLP当多于一个本地最低限度时会存在非凸损失函数。 因此不同的随机重量
     初始化可能导致不同的验证准确性。

     MLP需要调整一些超参数，例如隐藏的神经元，层和迭代数量。
       

      MLP对特征缩放很敏感。

请参阅：“实用使用技巧<mlp_tips>”部分的一些这些缺点。


Classification
==============
分类

Class :class:`MLPClassifier` implements a multi-layer perceptron (MLP) algorithm
that trains using `Backpropagation <http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm>`_.

class类：`MLPClassifier`实现使用Backpropagation进行训练的多层感知器

（MLP）算法列在两个阵列上：大小为（n_samples，n_features）的阵列X，其保存表示为浮点特征向量的训练样本; 和大小（n_samples，）的数组y，它保存训练样本的目标值（类标签）：

MLP trains on two arrays: array X of size (n_samples, n_features), which holds
the training samples represented as floating point feature vectors; and array
y of size (n_samples,), which holds the target values (class labels) for the
training samples::




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

After fitting (training), the model can predict labels for new samples::

拟合（训练）后，该模型可以预测新样本的标签:


    >>> clf.predict([[2., 2.], [-1., -2.]])
    array([1, 0])
    

MLP can fit a non-linear model to the training data. ``clf.coefs_``
contains the weight matrices that constitute the model parameters::

MLP可以将非线性模型适用于训练数据。``clf.coefs_``
包含构成模型参数的权重矩阵:

    >>> [coef.shape for coef in clf.coefs_]
    [(2, 5), (5, 2), (2, 1)]

Currently, :class:`MLPClassifier` supports only the
Cross-Entropy loss function, which allows probability estimates by running the
``predict_proba`` method.

MLP trains using Backpropagation. More precisely, it trains using some form of
gradient descent and the gradients are calculated using Backpropagation. For
classification, it minimizes the Cross-Entropy loss function, giving a vector
of probability estimates :math:`P(y|x)` per sample :math:`x`::

Currently类`MLPClassifier`只支持运行允许使用概率估计``predict_proba``方法得到交叉熵损失函数。

MLP算法使用反向传播的方式。 更准确地说，它训练使用某种形式梯度下降和梯度使用反向传播计算。 对于分类，它最小化交叉熵损失函数，给出一个向量的概率估计：`P（y | x）`样本：`x` ::

    >>> clf.predict_proba([[2., 2.], [1., 2.]])  # doctest: +ELLIPSIS
    array([[  1.967...e-04,   9.998...-01],
           [  1.967...e-04,   9.998...-01]])

:class:`MLPClassifier` supports multi-class classification by
applying `Softmax <https://en.wikipedia.org/wiki/Softmax_activation_function>`_
as the output function.

Further, the model supports :ref:`multi-label classification <multiclass>`
in which a sample can belong to more than one class. For each class, the raw
output passes through the logistic function. Values larger or equal to `0.5`
are rounded to `1`, otherwise to `0`. For a predicted output of a sample, the
indices where the value is `1` represents the assigned classes of that sample::

`MLPClassifier`通过应用Softmax作为输出函数来支持多类分类。

此外，该模型支持：ref：`multi-label classification <multiclass>`，其中样本可以属于多个类。 对于每个类，原始输出通过逻辑函数。 大于或等于0.5的值将舍入为1，否则为0.对于样本的预测输出，值为1的索引表示该样本的分配类别：

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

See the examples below and the doc string of
:meth:`MLPClassifier.fit` for further information.

.. topic:: Examples:

 * :ref:`sphx_glr_auto_examples_neural_networks_plot_mlp_training_curves.py`
 * :ref:`sphx_glr_auto_examples_neural_networks_plot_mnist_filters.py`

有关更多信息，请参阅下面的示例和文档字符串：meth：`MLPClassifier.fit`。

例子：

：REF：`sphx_glr_auto_examples_neural_networks_plot_mlp_training_curves.py`
：REF：`sphx_glr_auto_examples_neural_networks_plot_mnist_filters.py`

Regression
==========
回归

Class :class:`MLPRegressor` implements a multi-layer perceptron (MLP) that
trains using backpropagation with no activation function in the output layer,
which can also be seen as using the identity function as activation function.
Therefore, it uses the square error as the loss function, and the output is a
set of continuous values.

:class:`MLPRegressor` also supports multi-output regression, in
which a sample can have more than one target.

`MLPRegressor`实现了一个多层感知器（MLP），它在输出层中使用没有激活功能的反向传播进行训练，也可以看作是使用身份函数作为激活函数。 因此，它使用平方误差作为损失函数，输出是一组连续值。

`MLPRegressor`还支持多输出回归，其中样本可以有多个目标。

Regularization
==============
正则

Both :class:`MLPRegressor` and :class:`MLPClassifier` use parameter ``alpha``
for regularization (L2 regularization) term which helps in avoiding overfitting
by penalizing weights with large magnitudes. Following plot displays varying
decision function with value of alpha.

`MLPRegressor`和：`MLPClassifier`使用参数``alpha``用于正规化（L2正则化）术语，有助于避免过度拟合通过惩罚大量的权重。 以下图表显示不同具有α值的决策函数。

.. figure:: ../auto_examples/neural_networks/images/sphx_glr_plot_mlp_alpha_001.png
   :target: ../auto_examples/neural_networks/plot_mlp_alpha.html
   :align: center
   :scale: 75

See the examples below for further information.

有关详细信息，请参阅下面的示例。

.. topic:: Examples:

示例


 * :ref:`sphx_glr_auto_examples_neural_networks_plot_mlp_alpha.py`

Algorithms
==========

算法

MLP trains using `Stochastic Gradient Descent
<https://en.wikipedia.org/wiki/Stochastic_gradient_descent>`_,
`Adam <http://arxiv.org/abs/1412.6980>`_, or
`L-BFGS <https://en.wikipedia.org/wiki/Limited-memory_BFGS>`__.
Stochastic Gradient Descent (SGD) updates parameters using the gradient of the
loss function with respect to a parameter that needs adaptation, i.e.

MLP列车使用“随机梯度下降”<https://en.wikipedia.org/wiki/Stochastic_gradient_descent>`_，`_，或_`L-BFGS <https://en.wikipedia.org/wiki/Limited-memory_BFGS>`__。随机梯度下降（SGD）使用渐变梯度更新参数相对于需要适应的参数的损失函数，即

.. math::

    w \leftarrow w - \eta (\alpha \frac{\partial R(w)}{\partial w}
    + \frac{\partial Loss}{\partial w})

where :math:`\eta` is the learning rate which controls the step-size in
the parameter space search.  :math:`Loss` is the loss function used
for the network.

More details can be found in the documentation of
`SGD <http://scikit-learn.org/stable/modules/sgd.html>`_

Adam is similar to SGD in a sense that it is a stochastic optimizer, but it can
automatically adjust the amount to update parameters based on adaptive estimates
of lower-order moments.

With SGD or Adam, training supports online and mini-batch learning.

L-BFGS is a solver that approximates the Hessian matrix which represents the
second-order partial derivative of a function. Further it approximates the
inverse of the Hessian matrix to perform parameter updates. The implementation
uses the Scipy version of `L-BFGS
<http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html>`_.

If the selected solver is 'L-BFGS', training does not support online nor
mini-batch learning.

其中\ eta是控制参数空间搜索中的步长的学习率。 丢失是用于网络的丢失功能。

更多细节请见SGD的文件

Adam类似于SGD，因为它是随机优化器，但它可以根据低阶矩的自适应估计自动调整更新参数的量。

使用SGD或Adam，培训支持在线和小批量学习。

L-BFGS是近似表示函数的二阶偏导数的Hessian矩阵的求解器。 此外，它近似于Hessian矩阵的逆来执行参数更新。 实施使用Scipy版本的L-BFGS。

如果所选择的求解器是“L-BFGS”，不支持在线或小批量学习。

Complexity
==========
复杂性

Suppose there are :math:`n` training samples, :math:`m` features, :math:`k`
hidden layers, each containing :math:`h` neurons - for simplicity, and :math:`o`
output neurons.  The time complexity of backpropagation is
:math:`O(n\cdot m \cdot h^k \cdot o \cdot i)`, where :math:`i` is the number
of iterations. Since backpropagation has a high time complexity, it is advisable
to start with smaller number of hidden neurons and few hidden layers for
training.

假设有：`n`训练样本，：`m`特征，：`k`隐藏层，每个包含：`h`神经元 - 为简单起见，`o`输出神经元。 反向传播的时间复杂度是：'O（n \ cdot m \ cdot h ^ k \ cdot o \ cdot i）`，其中：i是数字的迭代。 由于反向传播具有高时间复杂性，因此是可取的从较小数量的隐藏神经元和几个隐藏层开始训练。

Mathematical formulation
========================
数学表达

Given a set of training examples :math:`(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)`
where :math:`x_i \in \mathbf{R}^n` and :math:`y_i \in \{0, 1\}`, a one hidden
layer one hidden neuron MLP learns the function :math:`f(x) = W_2 g(W_1^T x + b_1) + b_2`
where :math:`W_1 \in \mathbf{R}^m` and :math:`W_2, b_1, b_2 \in \mathbf{R}` are
model parameters. :math:`W_1, W_2` represent the weights of the input layer and
hidden layer, resepctively; and :math:`b_1, b_2` represent the bias added to
the hidden layer and the output layer, respectively.
:math:`g(\cdot) : R \rightarrow R` is the activation function, set by default as
the hyperbolic tan. It is given as,

给出一组训练示例：：`（x_1，y_1），（x_2，y_2），\ ldots，（x_n，y_n）`其中：`x_i \ in \ mathbf {R} ^ n`和：`y_i \ in \ {0，1 \}`，一个隐藏第一层隐藏神经元MLP学习功能：f（x）= W_2 g（W_1 ^ T x + b_1）+ b_2`其中：`W_1 \ in \ mathbf {R} ^ m`和：`W_2，b_1，b_2 \ in \ mathbf {R}`是模型参数。 ：W_1，W_2表示输入层的权重隐藏层 和`b_1，b_2`表示添加的偏见隐藏层和输出层。：`g（\ cdot）：R \ rightarrow R`是激活函数，默认设置为双曲线。 它被赋予，

.. math::
      g(z)= \frac{e^z-e^{-z}}{e^z+e^{-z}}

For binary classification, :math:`f(x)` passes through the logistic function
:math:`g(z)=1/(1+e^{-z})` to obtain output values between zero and one. A
threshold, set to 0.5, would assign samples of outputs larger or equal 0.5
to the positive class, and the rest to the negative class.

If there are more than two classes, :math:`f(x)` itself would be a vector of
size (n_classes,). Instead of passing through logistic function, it passes
through the softmax function, which is written as,

对于二进制分类，`f（x）`通过逻辑函数：`g（z）= 1 /（1 + e ^ { - z}）`来获得0到1之间的输出值。 一个阈值设置为0.5，将分配大于或等于0.5的输出样本到积极的班级，其余的到负面班。如果有两个以上的类，则：“f（x）”本身将是一个向量size（n_classes，）。 而不是通过逻辑功能，它通过通过softmax函数，它被写为，

.. math::
      \text{softmax}(z)_i = \frac{\exp(z_i)}{\sum_{l=1}^k\exp(z_l)}

where :math:`z_i` represents the :math:`i` th element of the input to softmax,
which corresponds to class :math:`i`, and :math:`K` is the number of classes.
The result is a vector containing the probabilities that sample :math:`x`
belong to each class. The output is the class with the highest probability.

In regression, the output remains as :math:`f(x)`; therefore, output activation
function is just the identity function.

MLP uses different loss functions depending on the problem type. The loss
function for classification is Cross-Entropy, which in binary case is given as,

其中：“z_i”表示中，“i”是softmax输入的元素，它对应于类：“i”，和：“K”是类的数量。结果是一个包含可能性的向量：`x`属于每个班级。 输出是具有最高概率的类。在回归中，输出依然如下：`f（x）`; 因此，输出激活功能只是身份功能。MLP根据问题类型使用不同的丢失功能。 二进制的情况下，亏损分类的功能是交叉熵，

.. math::

    Loss(\hat{y},y,W) = -y \ln {\hat{y}} - (1-y) \ln{(1-\hat{y})} + \alpha ||W||_2^2

where :math:`\alpha ||W||_2^2` is an L2-regularization term (aka penalty)
that penalizes complex models; and :math:`\alpha > 0` is a non-negative
hyperparameter that controls the magnitude of the penalty.

For regression, MLP uses the Square Error loss function; written as,

其中：：`\ alpha || W || _2 ^ 2`是一个L2正则化术语（又称惩罚）惩罚复杂的模型; 和：`\ alpha> 0`是非负数超参数控制惩罚的大小。为了回归，MLP使用平方误差损失函数; 写成，

.. math::

    Loss(\hat{y},y,W) = \frac{1}{2}||\hat{y} - y ||_2^2 + \frac{\alpha}{2} ||W||_2^2


Starting from initial random weights, multi-layer perceptron (MLP) minimizes
the loss function by repeatedly updating these weights. After computing the
loss, a backward pass propagates it from the output layer to the previous
layers, providing each weight parameter with an update value meant to decrease
the loss.

In gradient descent, the gradient :math:`\nabla Loss_{W}` of the loss with respect
to the weights is computed and deducted from :math:`W`.
More formally, this is expressed as,

从初始随机权重开始，多层感知器（MLP）最小化通过重复更新这些权重的损失函数。 计算后丢失，反向传递将其从输出层传播到前一个为每个权重参数提供旨在减少的更新值亏损。在渐变下降中，渐变：“\ nabla Loss_ {W}”的损失与尊重权重被计算和扣除：`W`。用公式表示为，

.. math::
    W^{i+1} = W^i - \epsilon \nabla {Loss}_{W}^{i}


where :math:`i` is the iteration step, and :math:`\epsilon` is the learning rate
with a value larger than 0.

The algorithm stops when it reaches a preset maximum number of iterations; or
when the improvement in loss is below a certain, small number.

其中：“i”是迭代步骤，“\ epsilon”是学习率值大于0。算法要么在达到预设的最大迭代次数时停止; 要么当损失的改善低于一定数量的时候停止。

.. _mlp_tips:

Tips on Practical Use
=====================
实用技巧

  * Multi-layer Perceptron is sensitive to feature scaling, so it
    is highly recommended to scale your data. For example, scale each
    attribute on the input vector X to [0, 1] or [-1, +1], or standardize
    it to have mean 0 and variance 1. Note that you must apply the *same*
    scaling to the test set for meaningful results.
    You can use :class:`StandardScaler` for standardization.
    
    多层感知器是敏感的特征缩放，所以它 强烈建议您扩展数据。 例如，缩放每个 属性在输入向量X到[0,1]或[-1，+1]，或标准化 它具有平均值0和方差1.注意，您必须应用*相同的* 缩放到测试集中以获得有意义的结果。 您可以使用：`StandardScaler`进行标准化。

      >>> from sklearn.preprocessing import StandardScaler  # doctest: +SKIP
      >>> scaler = StandardScaler()  # doctest: +SKIP
      >>> # Don't cheat - fit only on training data
      >>> scaler.fit(X_train)  # doctest: +SKIP
      >>> X_train = scaler.transform(X_train)  # doctest: +SKIP
      >>> # apply same transformation to test data
      >>> X_test = scaler.transform(X_test)  # doctest: +SKIP

    An alternative and recommended approach is to use :class:`StandardScaler`
    in a :class:`Pipeline`

  * Finding a reasonable regularization parameter :math:`\alpha` is
    best done using :class:`GridSearchCV`, usually in the
    range ``10.0 ** -np.arange(1, 7)``.

  * Empirically, we observed that `L-BFGS` converges faster and
    with better solutions on small datasets. For relatively large
    datasets, however, `Adam` is very robust. It usually converges
    quickly and gives pretty good performance. `SGD` with momentum or
    nesterov's momentum, on the other hand, can perform better than
    those two algorithms if learning rate is correctly tuned.

一个替代和推荐的方法是使用`StandardScaler` 在`Pipeline` *找到一个合理的正则化参数：`\ alpha`是 最好使用：`GridSearchCV`，通常在 范围``10.0 ** -np.arange（1，7）``。 *经验上，我们观察到“L-BFGS”收敛速度更快 在小数据集上有更好的解决方案。 对于比较大 数据集，但是，“adam”是非常强大的。 它通常会收敛 迅速，并提供相当不错的表现。 `SGD`有动力或 另一方面，尼斯特罗夫的势头可以比 这两种算法如果学习率正确调整。

More control with warm_start
============================
善于控制与warm_start

If you want more control over stopping criteria or learning rate in SGD,
or want to do additional monitoring, using ``warm_start=True`` and
``max_iter=1`` and iterating yourself can be helpful::

如果您希望更多地控制SGD中的停止标准或学习率，或者想要进行额外的监视，使用``warm_start = True``和``max_iter = 1``并且迭代自己可以是有帮助的：

    >>> X = [[0., 0.], [1., 1.]]
    >>> y = [0, 1]
    >>> clf = MLPClassifier(hidden_layer_sizes=(15,), random_state=1, max_iter=1, warm_start=True)
    >>> for i in range(10):
    ...     clf.fit(X, y)
    ...     # additional monitoring / inspection # doctest: +ELLIPSIS
    MLPClassifier(...

.. topic:: References:
参考文献

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
