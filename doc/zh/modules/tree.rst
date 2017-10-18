.. _tree:

==============
决策树
==============

.. currentmodule:: sklearn.tree

**Decision Trees (DTs)**  是一种用来 :ref:`classification <tree_classification>` 和 :ref:`regression <tree_regression>` 的无参监督学习方法。其目的是创建一种模型从数据特征中学习简单的决策规则来预测一个目标变量的值。

例如，在下面的图片中，决策树通过if-then-else的决策规则来学习数据从而估测数一个正弦图像。决策树越深入，决策规则就越复杂并且越符合创建的模型。

.. figure:: ../auto_examples/tree/images/sphx_glr_plot_tree_regression_001.png
   :target: ../auto_examples/tree/plot_tree_regression.html
   :scale: 75
   :align: center

决策树的优势:

   - 理解和说明起来简单。树的结构可以用图表现出来。

    - 需要少量的准备数据。其他的技术则通常需要数据规范化，比如创建所需的虚拟变量和移除空白值,不过请注意，这种模型不支持缺少值。

   - 由于训练决策树的数据点的数量导致了决策树的使用开销呈指数分布。

   - 能够处理数字和分类数据。其他的技术通常只能用来专门分析某一种变量类型的数据集。详情请参阅算法。

   - 能够处理多路输出的问题。

   - 使用白盒模型。如果某种给定的情况在该模型中是可以观察的，那么就可以轻易的通过布尔逻辑来解释这种情况。相比之下，在黑盒模型中的结果就是很难说明清        楚地。

   - 可以通过数值统计测试来验证该模型。这对事解释验证该模型的可靠性成为可能。

   - 即使该模型假设的结果与真实模型所提供的数据有些违反，其表现依旧良好。

决策树的缺点包括:

  - 决策树的学习者可以创造一个过分复杂的决策树却不能很好地概括数据。这被称为过拟合裁剪分支、设置叶节点所需的最小采样数或设置数的最大深度是避免出现      该问题最为有效地方法。

  - 决策树可能是不稳定的，因为数据中的微小变化可能会导致完全不同的树生成。从全局的角度出发来建立决策树可以有效地缓解该问题地产生。

  - 在多方面性能最优和简单化概念的要求下，NP完全是已知的解决解决该问题的决策树创建方法。因此，实际的决策树学习算法是基于启发式算法，例如在每个节点进     行局部最优决策的贪心算法。这样的算法不能保证返回全局最优决策树。学习者可以通过对特征和样本的随机抽取来训练全局决策树从而缓解该问题的发生。
  
  - 因为决策树不能很好地描述某些概念，从而导致这些概念很难被理解。例如XOR，奇偶或者复用器的问题。
  
  - 如果某些类在问题中占主导地位会使决策者创建出有偏见的树。因此，在拟合前因平衡数据集与决策树。


.. _tree_classification:

分类
==============

:class:`DecisionTreeClassifier` 是能够在数据集上执行多类分类的类与其他分类器一样，:class:`DecisionTreeClassifier` 采用输入两个数组：数组X，用 ``[n_samples, n_features]`` 的方式来存放训练样本。整数值数组Y，用 ``[n_samples]`` 来保存训练样本的类标签::

    >>> from sklearn import tree
    >>> X = [[0, 0], [1, 1]]
    >>> Y = [0, 1]
    >>> clf = tree.DecisionTreeClassifier()
    >>> clf = clf.fit(X, Y)

执行通过之后，可以使用该模型来预测样本类别::

    >>> clf.predict([[2., 2.]])
    array([1])

或者，也可以预测每个类的概率，这是叶中相同类的训练样本的分数::

    >>> clf.predict_proba([[2., 2.]])
    array([[ 0.,  1.]])

:class:`DecisionTreeClassifier` 能二分类（其中标签为[-1,1]）和多类（其中标签为[0,...,k-1]）分类。使用Lris数据集，我们可以构造一个决策树，如下所示::

    >>> from sklearn.datasets import load_iris
    >>> from sklearn import tree
    >>> iris = load_iris()
    >>> clf = tree.DecisionTreeClassifier()
    >>> clf = clf.fit(iris.data, iris.target)

经过训练，我们可以使用 :func:`export_graphviz` 导出器以`Graphviz<http://www.graphviz.org/>`_ 格式导出决策树。如果你是用`conda <http://conda.io>`_ 来管理包，那么安装graphviz二进制文件和python包可以用以下指令安装

   conda install python-graphviz

或者，可以从 graphviz 项目主页下载 graphviz 的二进制文件，并从 pypi 安装 Python 包装器，并安装 `pip install graphviz` 。以下是在整个 iris 数据集上训练的上述树的 graphviz 导出示例;其结果被保存在 `iris.pdf` 中::

    >>> import graphviz # doctest: +SKIP
    >>> dot_data = tree.export_graphviz(clf, out_file=None) # doctest: +SKIP
    >>> graph = graphviz.Source(dot_data) # doctest: +SKIP
    >>> graph.render("iris") # doctest: +SKIP

 :func:`export_graphviz` 出导出还支持各种美学选择，包括通过他们的类着色节点（或回归值），如果需要，使用显式变量和类名。Jupyter笔记也可以自动找出相同的模块::

    >>> dot_data = tree.export_graphviz(clf, out_file=None, # doctest: +SKIP
                             feature_names=iris.feature_names,  # doctest: +SKIP
                             class_names=iris.target_names,  # doctest: +SKIP
                             filled=True, rounded=True,  # doctest: +SKIP
                             special_characters=True)  # doctest: +SKIP
    >>> graph = graphviz.Source(dot_data)  # doctest: +SKIP
    >>> graph # doctest: +SKIP

.. only:: html

    .. figure:: ../images/iris.svg
       :align: center

.. only:: latex

    .. figure:: ../images/iris.pdf
       :align: center

执行通过之后，可以使用该模型预测样品类别::

    >>> clf.predict(iris.data[:1, :])
    array([0])

或者，可以根据决策树叶子树里训练样本中的相同类的分数，使得类预测成为可能::

    >>> clf.predict_proba(iris.data[:1, :])
    array([[ 1.,  0.,  0.]])

.. figure:: ../auto_examples/tree/images/sphx_glr_plot_iris_001.png
   :target: ../auto_examples/tree/plot_iris.html
   :align: center
   :scale: 75

.. topic:: 示例:

 * :ref:`sphx_glr_auto_examples_tree_plot_iris.py`


.. _tree_regression:

回归
==========

.. figure:: ../auto_examples/tree/images/sphx_glr_plot_tree_regression_001.png
   :target: ../auto_examples/tree/plot_tree_regression.html
   :scale: 75
   :align: center

决策树通过使用 :class:`DecisionTreeRegressor` 类也可以用来解决回归问题。如在分类设置中，拟合方法将数组X和数组y作为参数，只有在这种情况下，y数组预期才是浮点值::

    >>> from sklearn import tree
    >>> X = [[0, 0], [2, 2]]
    >>> y = [0.5, 2.5]
    >>> clf = tree.DecisionTreeRegressor()
    >>> clf = clf.fit(X, y)
    >>> clf.predict([[1, 1]])
    array([ 0.5])

.. topic:: 示例:

 * :ref:`sphx_glr_auto_examples_tree_plot_tree_regression.py`


.. _tree_multioutput:

多值输出问题
=====================

一个多值输出问题是一个类似当 Y 是大小为当 Y 是大小为 ``[n_samples, n_outputs]`` 的2d数组时，有多个输出值需要预测的监督学习问题。

当输出值之间没有关联时，一个很简单的处理该类型的方法是建立一个n独立模型，即每个输出一个，然后使用这些模型来独立地预测n个输出中的每一个。然而，由于可能与相同输入相关的输出值本身是相关的，所以通常更好的方法是构建能够同时预测所有n个输出的单个模型。首先，由于单个估计的建立使用，该模型需要更少的训练时间。第二，关于结果估计的概括准确性需要被提升。关于决策树，这一策略可以很容易地用于支持多输出问题。 这需要以下更改：

  - 在叶中存储n个输出值，而且输出值不为1;
  - 使用分割标准来计算所有n个输出的平均减少量

该模块通过在 :class:`DecisionTreeClassifier `和 :class:`DecisionTreeRegressor` 中实现该策略来支持多输出问题。如果决策树与大小为 ``[n_samples, n_outputs]`` 的输出数组Y向匹配，则得到的估计器将::

  * ``predict`` 是输出n_output的值

  * 在 ``predict_proba`` 上输出 n_output 数组列表

用多输出决策树进行回归分析 :ref:`sphx_glr_auto_examples_tree_plot_tree_regression_multioutput.py` 。 在该示例中，输入X是单个实数值，并且输出Y是X的正弦和余弦。

.. figure:: ../auto_examples/tree/images/sphx_glr_plot_tree_regression_multioutput_001.png
   :target: ../auto_examples/tree/plot_tree_regression_multioutput.html
   :scale: 75
   :align: center

使用多输出树进行分类，在 :ref:`sphx_glr_auto_examples_plot_multioutput_face_completion.py` 中进行了演示。 在该示例中，输入X是面的上半部分的像素，并且输出Y是这些面的下半部分的像素。

.. figure:: ../auto_examples/images/sphx_glr_plot_multioutput_face_completion_001.png
   :target: ../auto_examples/plot_multioutput_face_completion.html
   :scale: 75
   :align: center

.. topic:: 示例:

 * :ref:`sphx_glr_auto_examples_tree_plot_tree_regression_multioutput.py`
 * :ref:`sphx_glr_auto_examples_plot_multioutput_face_completion.py`

.. topic:: 参考:

 * M. Dumont et al,  `Fast multi-class image annotation with random subwindows
   and multiple output randomized trees
   <http://www.montefiore.ulg.ac.be/services/stochastic/pubs/2009/DMWG09/dumont-visapp09-shortpaper.pdf>`_, International Conference on
   Computer Vision Theory and Applications 2009

.. _tree_complexity:

复杂度分析
==========

总体来说，用来构建平衡二叉树的运行时间为 :math:`O(n_{samples}n_{features}\log(n_{samples}))` 并且执行时间为 :math:`O(\log(n_{samples}))` 。尽管树的构造算法尝试生成平衡树，但它们并不总能保持平衡。假设子树能大概保持平衡，每个节点的成本包括通过 :math:`O(n_{features})` 搜索来找到提供最大熵减小的特征。每个节点的花费为 :math:`O(n_{features}n_{samples}\log(n_{samples}))` ，导致整个决策树的构造成本为 :math:`O(n_{features}n_{samples}^{2}\log(n_{samples}))` 。Scikit-learn提供了更多有效的方法来创建决策树。

Scikit-learn提供了更多有效的方法来创建决策树。初始实现（如上所述）将重新计算沿着给定特征的每个新分割点的类标签直方图（用于分类）或平均值（用于回归）。与分类所有的样本特征，然后再次训练时运行标签计数，可将每个节点的复杂度降低为 :math:`O(n_{features}\log(n_{samples}))` ，则总的成本花费为 :math:`O(n_{features}n_{samples}\log(n_{samples}))` 。这是一种对所有基于树的算法的改进选项。默认情况下，它会打开梯度提升，一般来说，训练速度更快。但关闭所有其他算法，则训练深树时往往会减慢训练速度。


实际使用技巧
=====================

  * 决策树在分析数据的大量特征时往往会出现过拟合的现象。获得一个合适的样本和特征数量之间的比值是十分重要的，
    由于树在高维空间中只有少量的样本，所以可能会导致树的过拟合。

  * 考虑事先进行降维( :ref:`PCA <PCA>` , :ref:`ICA <ICA>` ，使您的树更好地找到具有分辨性的特征。
  
  * 通过 ``export`` 功能可以可视化您的决策树。使用 ``max_depth=3`` 作为初始树深度，让决策树知道如何适应您的数据，然后再增加树的深度。

  * 请记住，填充树的样本数量会增加树的每个附加级别。使用 ``max_depth`` 来控制输的大小防止过拟合。
  
  * 通过使用 ``min_samples_split`` 和 ``min_samples_leaf`` 来控制叶节点上的样本数量。一个非常小的数字也可能会导致决策树的过拟合，然而大的数据不利于决策树的学习。所以尝试 ``min_samples_leaf=5`` 作为初始值。如果样本的变化量很大，浮点数可以用作这两个参数中的百分比。两者之间的主要区别在于 ``min_samples_leaf`` 保证叶片中最少的采样数，而 ``min_samples_split`` 可以创建任意小的叶子，尽管在文献中 ``min_samples_split`` 更常见。

  * 在训练之前平衡您的数据集，以防止决策树偏向于主导类可以通过从每个类中抽取相等数量的样本来进行类平衡，或者优选地通过将每个类的样本权重 (``sample_weight``) 的和归一化为相同的值。还要注意的是，基于权重的预修剪标准 (``min_weight_fraction_leaf``) 对于显性类别的偏倚偏小，而不是不了解样本权重的标准，如 ``min_samples_leaf`` 。

 * 如果样本被加权，则使用基于权重的预修剪标准 ``min_weight_fraction_leaf`` 来优化树结构将更容易，这确保叶节点包含样本权重的总和的至少一部分。
 
 * 所有的决策树内部使用 ``np.float32`` 数组 ，如果训练数据不是这种格式，将会复制数据集。

 * 如果输入的矩阵X为稀疏矩阵，在调用预测之前，建议您在调用fit和sparse ``csc_matrix`` 之前将其转换为稀疏 ``csr_matrix`` 。当特征在大多数样本中具有零值时，与密集矩阵相比，稀疏矩阵输入的训练时间可以快几个数量级。



.. _tree_algorithms:

决策树算法: ID3, C4.5, C5.0 和 CART
==========================================

什么是决策树算法以及它们之间的区别？那么 scikit-learn 中实现何种算法呢？

ID3（Iterative Dichotomiser 3）由 Ross Quinlan 在1986年提出。该算法创建一个多路树，找到每个节点（即以贪心的方式）分类特征，这将产生分类目标的最大信息增益。决策树发展到其最大尺寸，然后通常应用精简步骤来提高树的概括未知的数据的能力。

C4.5 是 ID3 的后继者，并且通过动态定义将连续属性值分割成一组离散的间隔的离散属性（基于数字变量），消除了特征必须被明确分类的限制。C4.5 将训练的树（即，ID3算法的输出）转换成 if-then 规则的集合。然后评估每个规则的这些准确性，以确定应用它们的顺序。如果规则的准确性没有改变，则需要决策树的树枝来解决。

C5.0 是 Quinlan 根据专有许可证发布的最新版本。它使用更少的内存，并建立比 C4.5 更小的规则集，同时更准确。

CART（Classification and Regression Trees （分类和回归树））与 C4.5 非常相似，但它不同之处在于它支持数值目标变量（回归），并且不计算规则集。CART 使用在每个节点产生最大信息增益的特征和阈值来构造二叉树。

scikit-learn 使用 CART 算法的优化版本。

.. _ID3: https://en.wikipedia.org/wiki/ID3_algorithm
.. _CART: https://en.wikipedia.org/wiki/Predictive_analytics#Classification_and_regression_trees_.28CART.29


.. _tree_mathematical_formulation:

数学表达
========================

给定训练向量 :math:`x_i \in R^n`, i=1,..., l 和标签向量 :math:`y \in R^l`。决策树递归地分割空间，例如将有相同标签的样本归为一组。

将 :math:`m` 节点上的数据用 :math:`Q` 来表示。每一个候选组 :math:`\theta = (j, t_m)` 包含一个特征 :math:`j` 和阈值 :math:`t_m` 将,数据分成 :math:`Q_{left}(\theta)` 和 :math:`Q_{right}(\theta)` 子集。

.. math::

    Q_{left}(\theta) = {(x, y) | x_j <= t_m}

    Q_{right}(\theta) = Q \setminus Q_{left}(\theta)

使用杂质函数 :math:`H()` 计算 :math:`m` 处的杂质,其选择取决于正在解决的任务（分类或回归）

.. math::

   G(Q, \theta) = \frac{n_{left}}{N_m} H(Q_{left}(\theta))
   + \frac{n_{right}}{N_m} H(Q_{right}(\theta))

选择最小化杂质的参数

.. math::

    \theta^* = \operatorname{argmin}_\theta  G(Q, \theta)

重新计算子集 :math:`Q_{left}(\theta^*)` 和 :math:`Q_{right}(\theta^*)` ，直到达到最大允许深度，:math:`N_m < \min_{samples}` 或 :math:`N_m = 1`。

分类标准
-----------------------

如果目标是采用值 0,1,...,K-1 的分类结果，对于节点 :math:`m` ，表示具有 :math:`N_m` 个观测值的区域 :math:`R_m` ，让

.. math::

    p_{mk} = 1/ N_m \sum_{x_i \in R_m} I(y_i = k)

是节点 :math:`m` 中k类观测的比例通常用来处理杂质的方法是Gini

.. math::

    H(X_m) = \sum_k p_{mk} (1 - p_{mk})

Cross-Entropy （交叉熵）

.. math::

    H(X_m) = - \sum_k p_{mk} \log(p_{mk})

和 Misclassification （错误分类）

.. math::

    H(X_m) = 1 - \max(p_{mk})

在 :math:`X_m` 训练 :math:`m` 节点上的数据时。

回归标准
-------------------

如果目标是连续性的值，那么对于节点 :math:`m` ,表示具有 :math:`N_m` 个观测值的区域 :math:`R_m` ，最小化用于确定未来拆分的位置的通用标准是均方误差，其使用终端节点处的平均值来最小化L2误差，并且平均绝对误差，其使用终端节点处的中值来最小化 L1 误差。

Mean Squared Error （均方误差）:

.. math::

    c_m = \frac{1}{N_m} \sum_{i \in N_m} y_i

    H(X_m) = \frac{1}{N_m} \sum_{i \in N_m} (y_i - c_m)^2

Mean Absolute Error（平均绝对误差）:

.. math::

    \bar{y_m} = \frac{1}{N_m} \sum_{i \in N_m} y_i

    H(X_m) = \frac{1}{N_m} \sum_{i \in N_m} |y_i - \bar{y_m}|

在 :math:`X_m` 训练 :math:`m` 节点上的数据时。

.. topic:: 示例:

    * https://en.wikipedia.org/wiki/Decision_tree_learning

    * https://en.wikipedia.org/wiki/Predictive_analytics

    * L. Breiman, J. Friedman, R. Olshen, and C. Stone. Classification and
      Regression Trees. Wadsworth, Belmont, CA, 1984.

    * J.R. Quinlan. C4. 5: programs for machine learning. Morgan Kaufmann, 1993.

    * T. Hastie, R. Tibshirani and J. Friedman.
      Elements of Statistical Learning, Springer, 2009.

