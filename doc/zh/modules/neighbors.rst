.. _neighbors:

=================
最近邻
=================

.. sectionauthor:: Jake Vanderplas <vanderplas@astro.washington.edu>

.. currentmodule:: sklearn.neighbors

:mod:`sklearn.neighbors` 提供了 neighbors-based (基于邻居的) 无监督学习以及监督学习方法的功能。
无监督的最近邻是许多其它学习方法的基础，尤其是 manifold learning (流行学习) 和 spectral clustering (谱聚类)。
受监督的 neighbors-based (基于邻居的) 学习分为两种： `分类`_ 针对的是具有离散标签的数据，`回归`_ 针对的是具有连续标签的数据。

最近邻方法的原理是从训练样本中找到与新点在距离上最近的预定数量的几个点，并从这些点中预测标签。
这些点的数量可以是用户自定义的常量（K最近邻学习），
或是基于当前点的局部密度（基于半径的最近邻学习）。距离通常可以通过任何方式来度量：standard Euclidean
distance（标准欧式距离）是最常见的选择。Neighbors-based（基于邻居的）方法被称为 *非泛化* 机器学习方法，
因为它们只是简单地"考虑"其所有的训练数据（正因如此可能需要加速，
将其转换为一个快速索引结构，如: ref:`Ball Tree <ball_tree>` 或 :ref:`KD Tree <kd_tree>`）。

尽管它很简单，但最近邻算法已经成功地适用于很多的分类和回归问题，例如手写数字或卫星图像的场景。
作为一个 non-parametric（非参数化）方法，它经常成功地应用于决策边界非常不规则的情景下。

:mod:`sklearn.neighbors` 可以处理 Numpy 数组或 `scipy.sparse` 矩阵中的任何一个作为其输入。
对于密集矩阵，大多数可能距离的矩阵都是支持的。对于稀疏矩阵，任何 Minkowski 矩阵都支持被搜索。

有许多学习方法都是依赖最近邻作为其核心。一个例子是：:ref:`kernel density estimation <kernel_density>`（核密度估计）,
在:ref:`density estimation <density_estimation>`（密度估计）章节中有讨论。

.. _unsupervised_neighbors:

无监督最近邻
==============================

:class:`NearestNeighbors`（最近邻）实现了 unsupervised nearest neighbors learning（无监督的最近邻学习）。
它为三种不同的最近邻算法提供统一的接口：:class:`BallTree`, :class:`KDTree`, 还有基于 :mod:`sklearn.metrics.pairwise`
的 brute-force 算法。选择算法时可通过关键字 ``'algorithm'`` 来控制，
并指定为 ``['auto', 'ball_tree', 'kd_tree', 'brute']`` 其中的一个即可。当默认值设置为 ``'auto'``
时，算法会尝试从训练数据中确定最佳方法。有关上述每个选项的优缺点，参见 `Nearest Neighbor Algorithms`_。


    .. warning::

        关于最近邻算法，如果邻居 :math:`k+1` 和邻居 :math:`k` 具有相同的距离，但具有不同的标签，
        结果将取决于训练数据的顺序。

找到最近邻
-----------------------------
为了完成找到两组数据集中最近邻点的简单任务, 可以使用 :mod:`sklearn.neighbors` 中的无监督算法:

    >>> from sklearn.neighbors import NearestNeighbors
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
    >>> distances, indices = nbrs.kneighbors(X)
    >>> indices                                           # doctest: +ELLIPSIS
    array([[0, 1],
           [1, 0],
           [2, 1],
           [3, 4],
           [4, 3],
           [5, 4]]...)
    >>> distances
    array([[ 0.        ,  1.        ],
           [ 0.        ,  1.        ],
           [ 0.        ,  1.41421356],
           [ 0.        ,  1.        ],
           [ 0.        ,  1.        ],
           [ 0.        ,  1.41421356]])

因为查询集匹配训练集，每个点的最近邻点是其自身，距离为0。

还可以有效地生成一个稀疏图来标识相连点之间的连接情况：

    >>> nbrs.kneighbors_graph(X).toarray()
    array([[ 1.,  1.,  0.,  0.,  0.,  0.],
           [ 1.,  1.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  1.,  0.],
           [ 0.,  0.,  0.,  1.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  1.,  1.]])

我们的数据集是结构化的，因此附近索引顺序的点就在参数空间附近，从而生成了近似 K-nearest neighbors（K近邻）的块对角矩阵。
这种稀疏图在各种情况下都是有用的，它利用点之间的空间关系进行无监督学习：特别地可参见 :class:`sklearn.manifold.Isomap`,
:class:`sklearn.manifold.LocallyLinearEmbedding`, 和 :class:`sklearn.cluster.SpectralClustering`。

KDTree 和 BallTree 类
---------------------------
或者，可以使用 :class:`KDTree` 或 :class:`BallTree` 类来找最近邻。
这是上文使用过的 :class:`NearestNeighbors` 类所包含的功能。
:class:`KDTree` 和 :class:`BallTree` 具有相同的接口；
我们将在这里展示使用 :class:`KDTree` 的例子：

    >>> from sklearn.neighbors import KDTree
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> kdt = KDTree(X, leaf_size=30, metric='euclidean')
    >>> kdt.query(X, k=2, return_distance=False)          # doctest: +ELLIPSIS
    array([[0, 1],
           [1, 0],
           [2, 1],
           [3, 4],
           [4, 3],
           [5, 4]]...)

对于近邻搜索中选项的更多信息，包括各种度量距离的查询策略的说明等，请参阅 :class:`KDTree` 和 :class:`BallTree` 类文档。
关于可用度量距离的列表，请参阅 :class:`DistanceMetric` 类。

.. _classification:

最近邻分类
================================

最近邻分类属于基于样本的学习或非泛化学习：它不试图去构造一个泛化的内部模型，而是简单地存储训练数据的实例。
分类是由每个点的最近邻的简单多数投票中计算得到的：一个查询点的数据类型是由它最近邻点中最具代表性的数据类型来决定的。

scikit-learn 实现了两种不同的最近邻分类器：:class:`KNeighborsClassifier` 基于每个查询点的 :math:`k` 个最近邻实现，
其中 :math:`k` 是用户指定的整数值。:class:`RadiusNeighborsClassifier` 基于每个查询点的固定半径 :math:`r` 内的邻居数量实现，
其中 :math:`r` 是用户指定的浮点数值。

:class:`KNeighborsClassifier` 中 :math:`k`-邻居分类是两种技术中更常使用的。:math:`k` 值的最佳选择是高度数据依赖的：
通常较大的 :math:`k` 抑制噪声的影响，但是使得分类界限不明显。

如果数据是不均匀采样的，那么 :class:`RadiusNeighborsClassifier` 中的基于半径的近邻分类可能是更好的选择。
用户指定一个固定半径 :math:`r`，使得稀疏邻居中的点使用较少的最近邻来分类。
对于高维参数空间，这个方法会由于所谓的“维度惩罚”而变得不那么有效。

基本的最近邻分类使用统一的权重：分配给查询点的值是由一个简单的最近邻中的多数投票方法计算而来的。
在某些环境下，最好对邻居进行加权，使得近邻更有利于拟合。这可以通过 ``weights`` 关键字来完成。
默认值 ``weights = 'uniform'`` 为每个近邻分配统一的权重。而 ``weights = 'distance'`` 分配与查询点距离的倒数呈比例的权重。
或者，用户定义的距离函数也可应用于权重的计算之中。

.. |classification_1| image:: ../auto_examples/neighbors/images/sphx_glr_plot_classification_001.png
   :target: ../auto_examples/neighbors/plot_classification.html
   :scale: 50

.. |classification_2| image:: ../auto_examples/neighbors/images/sphx_glr_plot_classification_002.png
   :target: ../auto_examples/neighbors/plot_classification.html
   :scale: 50

.. centered:: |classification_1| |classification_2|

.. topic:: 示例:

  * :ref:`sphx_glr_auto_examples_neighbors_plot_classification.py`: 使用最近邻进行分类的示例。

.. _regression:

最近邻回归
============================

最近邻回归是用在数据标签为连续变量，而不是离散变量的情况下。分配给查询点的标签是由它的最近邻标签的均值计算而来的。

scikit-learn 实现了两种不同的最近邻回归：:class:`KNeighborsRegressor` 基于每个查询点的 :math:`k` 个最近邻实现，
其中 :math:`k` 是用户指定的整数值。:class:`RadiusNeighborsRegressor` 基于每个查询点的固定半径 :math:`r` 内的邻居数量实现，
其中 :math:`r` 是用户指定的浮点数值。

在某些环境下，增加权重可能是有利的，使得附近点对于回归所作出的贡献多于远处点。
这可以通过 ``weights`` 关键字来实现。默认值 ``weights = 'uniform'`` 为所有点分配同等权重。
而 ``weights = 'distance'`` 分配与查询点距离的倒数呈比例的权重。
或者，用户定义的距离函数也可应用于权重的计算之中。

.. figure:: ../auto_examples/neighbors/images/sphx_glr_plot_regression_001.png
   :target: ../auto_examples/neighbors/plot_regression.html
   :align: center
   :scale: 75

使用多输出的最近邻进行回归在此演示 :ref:`sphx_glr_auto_examples_plot_multioutput_face_completion.py`。
在这个示例中，输入 X 是脸部的上半部分像素，输出 Y 是脸部的下半部分像素。

.. figure:: ../auto_examples/images/sphx_glr_plot_multioutput_face_completion_001.png
   :target: ../auto_examples/plot_multioutput_face_completion.html
   :scale: 75
   :align: center


.. topic:: 示例:

  * :ref:`sphx_glr_auto_examples_neighbors_plot_regression.py`: 使用最近邻进行回归的示例。

  * :ref:`sphx_glr_auto_examples_plot_multioutput_face_completion.py`: 使用最近邻进行多输出回归的示例。


最近邻算法
===========================

.. _brute_force:

暴力计算
-----------

最近邻的快速计算是机器学习中一个活跃的研究领域。最简单的近邻搜索涉及数据集中所有成对点之间距离的暴力计算：
对于 :math:`D` 维度中的 :math:`N` 个样本来说, 这个方法的范围是 :math:`O[D N^2]`。
对于小数据样本，有效的暴力近邻搜索是非常有竞争力的。
然而，随着样本数 :math:`N` 的增长，暴力方法迅速变得不可行。在 :mod:`sklearn.neighbors` 类中，
暴力近邻搜索通过关键字 ``algorithm = 'brute'`` 来指定，并通过 :mod:`sklearn.metrics.pairwise` 中的例程来进行计算。

.. _kd_tree:

K-D 树
--------

为了解决暴力搜索方法的效率底下，已经发明了大量的基于树的数据结构。总的来说，
这些结构通过有效地对样本中聚合距离信息的编码，来试图减少距离所需的距离计算的数量。
基本思想是，若 :math:`A` 点距离 :math:`B` 点非常远，:math:`B` 点距离 :math:`C` 点非常近，
可知 :math:`A` 点与 :math:`C` 点很遥远，*不需要明确计算它们的距离*。
通过这样的方式，近邻搜索的计算成本可以降低为 :math:`O[D N \log(N)]` 或更好。
这是对于暴力搜索在大样本数 large :math:`N` 中表现的显著改善。


利用这种聚合信息的早期方法是 *KD tree* 数据结构（* K-dimensional tree* 的简写）,
它将二维 *Quad-trees* 和三维 *Oct-trees 概括为任意数量的维度.
KD 树是二元的树结构, 它沿着数据轴递归地划分参数空间, 将其划分成嵌套的原点区域, 数据点被归档到其中.
KD 树的构造非常快：因为只能沿数据轴执行分区, 无需计算 :math:`D`-dimensional 距离.
一旦构建完成, 查询点的最近邻可以仅使用 :math:`O[\log(N)]` 距离计算来确定.
虽然 KD 树的方法对于低维度 (:math:`D < 20`) 近邻搜索非常快, 当 :math:`D` 增长到很大时,
效率开始降低: 这就是所谓的 "维度灾难" 的一种体现.
在 scikit-learn 中, KD 树近邻搜索可以使用关键字 ``algorithm = 'kd_tree'`` 来指定,
并且使用类 :class:`KDTree` 来计算.

.. topic:: References:

   * `"Multidimensional binary search trees used for associative searching"
     <http://dl.acm.org/citation.cfm?doid=361002.361007>`_,
     Bentley, J.L., Communications of the ACM (1975)


.. _ball_tree:

Ball 树
---------

为了解决 KD 树在高维上的低效率问题, 开发了 *ball 树* 数据结构.
其中 KD 树沿笛卡尔轴分割数据, bakk 树在沿着一系列的 hyper-spheres 来分割数据.
这样在树构建的过程比 KD 树消耗的时间更多,
但其结果是数据结构对于高度结构化的数据可能非常有效, 即使是在非常高的维度上也一样.

ball 树将数据递归地划分为由质心 :math:`C` 和半径 :math:`r` 定义的节点,
使得节点中的每个点位于由 :math:`r` 和 :math:`C` 定义的 hyper-sphere 内.
通过使用 *triangle inequality（三角不等式）* 减少近邻搜索的候选点数:

.. math::   |x+y| \leq |x| + |y|

通过这种设置, 测试点和质心之间的单一距离计算足以确定距节点内所有点的距离的下限和上限.
由于 ball 树节点的球形几何, 它可以在高维度上执行 *KD-tree*, 尽管实际的性能高度依赖于训练数据的结构.
在 scikit-learn 中, 基于 ball 树的近邻搜索可以使用关键字 ``algorithm = 'ball_tree'`` 来指定,
并且使用类 :class:`sklearn.neighbors.BallTree` 来计算.
或者, 用户可以直接使用 :class:`BallTree` 类.

.. topic:: 参考:

   * `"Five balltree construction algorithms"
     <http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.91.8209>`_,
     Omohundro, S.M., International Computer Science Institute
     Technical Report (1989)

最近邻居算法的选择（跪求大佬代为装逼 。。。）
---------------------------------------------
对于给定数据集的最优算法是一个复杂的选择, 并且取决于多个因素:

* 样本数量 :math:`N` (i.e. ``n_samples``) 和维度 :math:`D` (例如. ``n_features``).

  * *Brute force* query time grows as :math:`O[D N]`
  * *Ball tree* query time grows as approximately :math:`O[D \log(N)]`
  * *KD tree* query time changes with :math:`D` in a way that is difficult
    to precisely characterise.  For small :math:`D` (less than 20 or so)
    the cost is approximately :math:`O[D\log(N)]`, and the KD tree
    query can be very efficient.
    For larger :math:`D`, the cost increases to nearly :math:`O[DN]`, and
    the overhead due to the tree
    structure can lead to queries which are slower than brute force.

  For small data sets (:math:`N` less than 30 or so), :math:`\log(N)` is
  comparable to :math:`N`, and brute force algorithms can be more efficient
  than a tree-based approach.  Both :class:`KDTree` and :class:`BallTree`
  address this through providing a *leaf size* parameter: this controls the
  number of samples at which a query switches to brute-force.  This allows both
  algorithms to approach the efficiency of a brute-force computation for small
  :math:`N`.

* 数据结构: *intrinsic dimensionality* of the data and/or *sparsity*
  of the data. Intrinsic dimensionality refers to the dimension
  :math:`d \le D` of a manifold on which the data lies, which can be linearly
  or non-linearly embedded in the parameter space. Sparsity refers to the
  degree to which the data fills the parameter space (this is to be
  distinguished from the concept as used in "sparse" matrices.  The data
  matrix may have no zero entries, but the **structure** can still be
  "sparse" in this sense).

  * *Brute force* query time is unchanged by data structure.
  * *Ball tree* and *KD tree* query times can be greatly influenced
    by data structure.  In general, sparser data with a smaller intrinsic
    dimensionality leads to faster query times.  Because the KD tree
    internal representation is aligned with the parameter axes, it will not
    generally show as much improvement as ball tree for arbitrarily
    structured data.

  Datasets used in machine learning tend to be very structured, and are
  very well-suited for tree-based queries.

* 近邻数 :math:`k` 请求 query point（查询点）.

  * *Brute force* query time is largely unaffected by the value of :math:`k`
  * *Ball tree* and *KD tree* query time will become slower as :math:`k`
    increases.  This is due to two effects: first, a larger :math:`k` leads
    to the necessity to search a larger portion of the parameter space.
    Second, using :math:`k > 1` requires internal queueing of results
    as the tree is traversed.

  As :math:`k` becomes large compared to :math:`N`, the ability to prune
  branches in a tree-based query is reduced.  In this situation, Brute force
  queries can be more efficient.

* query points（查询点）数.  Both the ball tree and the KD Tree
  require a construction phase.  The cost of this construction becomes
  negligible when amortized over many queries.  If only a small number of
  queries will be performed, however, the construction can make up
  a significant fraction of the total cost.  If very few query points
  will be required, brute force is better than a tree-based method.

当前, ``algorithm = 'auto'`` 选择 ``'kd_tree'`` 如果 :math:`k < N/2`
并且 ``'effective_metric_'`` 在 ``'kd_tree'`` 的列表 ``'VALID_METRICS'`` 中. 
它选择 ``'ball_tree'`` 如果 :math:`k < N/2` 并且
``'effective_metric_'`` 在 ``'ball_tree'`` 的列表 ``'VALID_METRICS'`` 中.
它选择 ``'brute'`` 如果 :math:`k < N/2` 并且
``'effective_metric_'`` 不在 ``'kd_tree'`` 或 ``'ball_tree'`` 的列表 ``'VALID_METRICS'`` 中.
它选择 ``'brute'`` 如果 :math:`k >= N/2`.

这种选择基于以下假设: 查询点的数量与训练点的数量至少相同, 并且 ``leaf_size`` 接近其默认值 ``30``.

``leaf_size`` 的影响
-----------------------
如上所述, 在样本很小的情况下, 暴力搜索可以比基于树的查询更有效.
这个事实在 ball 树和 KD 树中通过内部切换到叶节点内的暴力搜索来解释.
该开关的级别可以使用参数 ``leaf_size`` 来指定.
这个参数选择有很多的效果:

**构造时间**
  更大的 ``leaf_size`` 会导致更快的树构建时间, 因为需要创建更少的节点.

**查询时间**
  一个大或小的 ``leaf_size`` 可能会导致次优查询成本.
  当 ``leaf_size`` 接近 1 时, 遍历节点所涉及的开销大大减慢了查询时间.
  当 ``leaf_size``, 接近训练集的大小，查询变得本质上是暴力的.
  这些之间的一个很好的妥协是 ``leaf_size = 30``, 这是该参数的默认值.

**内存**
  针对 :class:`BallTree` 所需的存储空间近似于 ``1 / leaf_size`` 乘以训练集的大小.

``leaf_size`` 不被 brute force queries（暴力查询）所引用.

.. _nearest_centroid_classifier:

最近质心分类
===========================

该 :class:`NearestCentroid` 分类器是一个简单的算法, 它表示每个类都通过其成员的质心组成.
实际上, 这使得它类似于 :class:`sklearn.KMeans` 算法的标签更新阶段.
它也没有参数选择, 使其成为良好的基准分类器.
然而, 它确实存在非凸类, 以及当类具有截然不同的方差时, 假设在所有维度上均等.
对于没有做出这个假设的更复杂的方法, 请参阅线性判别分析 (:class:`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`)
和二次判别分析 (:class:`sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis`).
默认的 :class:`NearestCentroid` 用法示例如下:

    >>> from sklearn.neighbors.nearest_centroid import NearestCentroid
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> clf = NearestCentroid()
    >>> clf.fit(X, y)
    NearestCentroid(metric='euclidean', shrink_threshold=None)
    >>> print(clf.predict([[-0.8, -1]]))
    [1]


Nearest Shrunken Centroid
-------------------------

该 :class:`NearestCentroid` 分类器有一个 ``shrink_threshold`` 参数,
它实现了 nearest shrunken centroid 分类器.
实际上, 每个质心的每个特征的值除以该特征的类中的方差.
然后通过 ``shrink_threshold`` 来减小特征值.
最值得注意的是, 如果特定特征值越过零, 则将其设置为零.
实际上, 这将从影响的分类上删除该特征.
这是有用的, 例如, 去除噪声特征.

在以下例子中, 使用一个较小的 shrink 阀值将模型的准确度从 0.81 提高到 0.82.

.. |nearest_centroid_1| image:: ../auto_examples/neighbors/images/sphx_glr_plot_nearest_centroid_001.png
   :target: ../auto_examples/neighbors/plot_nearest_centroid.html
   :scale: 50

.. |nearest_centroid_2| image:: ../auto_examples/neighbors/images/sphx_glr_plot_nearest_centroid_002.png
   :target: ../auto_examples/neighbors/plot_nearest_centroid.html
   :scale: 50

.. centered:: |nearest_centroid_1| |nearest_centroid_2|

.. topic:: 例子:

  * :ref:`sphx_glr_auto_examples_neighbors_plot_nearest_centroid.py`: 一个分类的例子, 它使用了不同 shrink 阀值的最近质心.
