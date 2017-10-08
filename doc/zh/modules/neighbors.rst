.. _neighbors:

=================
Nearest Neighbors
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

Unsupervised Nearest Neighbors
==============================

:class:`NearestNeighbors`（最近邻）实现了 unsupervised nearest neighbors learning（无监督的最近邻学习）。
它为三种不同的最近邻算法提供统一的接口：:class:`BallTree`, :class:`KDTree`, 还有基于 :mod:`sklearn.metrics.pairwise`
的 brute-force 算法。选择算法时可通过关键字 ``'algorithm'`` 来控制，
并指定为 ``['auto', 'ball_tree', 'kd_tree', 'brute']`` 其中的一个即可。当默认值设置为 ``'auto'``
时，算法会尝试从训练数据中确定最佳方法。有关上述每个选项的优缺点，参见 `Nearest Neighbor Algorithms`_。


    .. warning::

        关于最近邻算法，如果邻居 :math:`k+1` 和邻居 :math:`k` 具有相同的距离，但具有不同的标签，
        结果将取决于训练数据的顺序。

Finding the Nearest Neighbors
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

KDTree and BallTree Classes
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

Nearest Neighbors Classification
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

Nearest Neighbors Regression
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


Nearest Neighbor Algorithms
===========================

.. _brute_force:

Brute Force
-----------

Fast computation of nearest neighbors is an active area of research in
machine learning.  The most naive neighbor search implementation involves
the brute-force computation of distances between all pairs of points in the
dataset: for :math:`N` samples in :math:`D` dimensions, this approach scales
as :math:`O[D N^2]`.  Efficient brute-force neighbors searches can be very
competitive for small data samples.
However, as the number of samples :math:`N` grows, the brute-force
approach quickly becomes infeasible.  In the classes within
:mod:`sklearn.neighbors`, brute-force neighbors searches are specified
using the keyword ``algorithm = 'brute'``, and are computed using the
routines available in :mod:`sklearn.metrics.pairwise`.

.. _kd_tree:

K-D Tree
--------

To address the computational inefficiencies of the brute-force approach, a
variety of tree-based data structures have been invented.  In general, these
structures attempt to reduce the required number of distance calculations
by efficiently encoding aggregate distance information for the sample.
The basic idea is that if point :math:`A` is very distant from point
:math:`B`, and point :math:`B` is very close to point :math:`C`,
then we know that points :math:`A` and :math:`C`
are very distant, *without having to explicitly calculate their distance*.
In this way, the computational cost of a nearest neighbors search can be
reduced to :math:`O[D N \log(N)]` or better.  This is a significant
improvement over brute-force for large :math:`N`.

An early approach to taking advantage of this aggregate information was
the *KD tree* data structure (short for *K-dimensional tree*), which
generalizes two-dimensional *Quad-trees* and 3-dimensional *Oct-trees*
to an arbitrary number of dimensions.  The KD tree is a binary tree
structure which recursively partitions the parameter space along the data
axes, dividing it into nested orthotopic regions into which data points
are filed.  The construction of a KD tree is very fast: because partitioning
is performed only along the data axes, no :math:`D`-dimensional distances
need to be computed.  Once constructed, the nearest neighbor of a query
point can be determined with only :math:`O[\log(N)]` distance computations.
Though the KD tree approach is very fast for low-dimensional (:math:`D < 20`)
neighbors searches, it becomes inefficient as :math:`D` grows very large:
this is one manifestation of the so-called "curse of dimensionality".
In scikit-learn, KD tree neighbors searches are specified using the
keyword ``algorithm = 'kd_tree'``, and are computed using the class
:class:`KDTree`.


.. topic:: References:

   * `"Multidimensional binary search trees used for associative searching"
     <http://dl.acm.org/citation.cfm?doid=361002.361007>`_,
     Bentley, J.L., Communications of the ACM (1975)


.. _ball_tree:

Ball Tree
---------

To address the inefficiencies of KD Trees in higher dimensions, the *ball tree*
data structure was developed.  Where KD trees partition data along
Cartesian axes, ball trees partition data in a series of nesting
hyper-spheres.  This makes tree construction more costly than that of the
KD tree, but
results in a data structure which can be very efficient on highly-structured
data, even in very high dimensions.

A ball tree recursively divides the data into
nodes defined by a centroid :math:`C` and radius :math:`r`, such that each
point in the node lies within the hyper-sphere defined by :math:`r` and
:math:`C`. The number of candidate points for a neighbor search
is reduced through use of the *triangle inequality*:

.. math::   |x+y| \leq |x| + |y|

With this setup, a single distance calculation between a test point and
the centroid is sufficient to determine a lower and upper bound on the
distance to all points within the node.
Because of the spherical geometry of the ball tree nodes, it can out-perform
a *KD-tree* in high dimensions, though the actual performance is highly
dependent on the structure of the training data.
In scikit-learn, ball-tree-based
neighbors searches are specified using the keyword ``algorithm = 'ball_tree'``,
and are computed using the class :class:`sklearn.neighbors.BallTree`.
Alternatively, the user can work with the :class:`BallTree` class directly.

.. topic:: References:

   * `"Five balltree construction algorithms"
     <http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.91.8209>`_,
     Omohundro, S.M., International Computer Science Institute
     Technical Report (1989)

Choice of Nearest Neighbors Algorithm
-------------------------------------
The optimal algorithm for a given dataset is a complicated choice, and
depends on a number of factors:

* number of samples :math:`N` (i.e. ``n_samples``) and dimensionality
  :math:`D` (i.e. ``n_features``).

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

* data structure: *intrinsic dimensionality* of the data and/or *sparsity*
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

* number of neighbors :math:`k` requested for a query point.

  * *Brute force* query time is largely unaffected by the value of :math:`k`
  * *Ball tree* and *KD tree* query time will become slower as :math:`k`
    increases.  This is due to two effects: first, a larger :math:`k` leads
    to the necessity to search a larger portion of the parameter space.
    Second, using :math:`k > 1` requires internal queueing of results
    as the tree is traversed.

  As :math:`k` becomes large compared to :math:`N`, the ability to prune
  branches in a tree-based query is reduced.  In this situation, Brute force
  queries can be more efficient.

* number of query points.  Both the ball tree and the KD Tree
  require a construction phase.  The cost of this construction becomes
  negligible when amortized over many queries.  If only a small number of
  queries will be performed, however, the construction can make up
  a significant fraction of the total cost.  If very few query points
  will be required, brute force is better than a tree-based method.

Currently, ``algorithm = 'auto'`` selects ``'kd_tree'`` if :math:`k < N/2`
and the ``'effective_metric_'`` is in the ``'VALID_METRICS'`` list of
``'kd_tree'``. It selects ``'ball_tree'`` if :math:`k < N/2` and the
``'effective_metric_'`` is in the ``'VALID_METRICS'`` list of
``'ball_tree'``. It selects ``'brute'`` if :math:`k < N/2` and the
``'effective_metric_'`` is not in the ``'VALID_METRICS'`` list of
``'kd_tree'`` or ``'ball_tree'``. It selects ``'brute'`` if :math:`k >= N/2`.
This choice is based on the assumption that the number of query points is at
least the same order as the number of training points, and that ``leaf_size``
is close to its default value of ``30``.

Effect of ``leaf_size``
-----------------------
As noted above, for small sample sizes a brute force search can be more
efficient than a tree-based query.  This fact is accounted for in the ball
tree and KD tree by internally switching to brute force searches within
leaf nodes.  The level of this switch can be specified with the parameter
``leaf_size``.  This parameter choice has many effects:

**construction time**
  A larger ``leaf_size`` leads to a faster tree construction time, because
  fewer nodes need to be created

**query time**
  Both a large or small ``leaf_size`` can lead to suboptimal query cost.
  For ``leaf_size`` approaching 1, the overhead involved in traversing
  nodes can significantly slow query times.  For ``leaf_size`` approaching
  the size of the training set, queries become essentially brute force.
  A good compromise between these is ``leaf_size = 30``, the default value
  of the parameter.

**memory**
  As ``leaf_size`` increases, the memory required to store a tree structure
  decreases.  This is especially important in the case of ball tree, which
  stores a :math:`D`-dimensional centroid for each node.  The required
  storage space for :class:`BallTree` is approximately ``1 / leaf_size`` times
  the size of the training set.

``leaf_size`` is not referenced for brute force queries.

.. _nearest_centroid_classifier:

Nearest Centroid Classifier
===========================

The :class:`NearestCentroid` classifier is a simple algorithm that represents
each class by the centroid of its members. In effect, this makes it
similar to the label updating phase of the :class:`sklearn.KMeans` algorithm.
It also has no parameters to choose, making it a good baseline classifier. It
does, however, suffer on non-convex classes, as well as when classes have
drastically different variances, as equal variance in all dimensions is
assumed. See Linear Discriminant Analysis (:class:`sklearn.discriminant_analysis.LinearDiscriminantAnalysis`)
and Quadratic Discriminant Analysis (:class:`sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis`)
for more complex methods that do not make this assumption. Usage of the default
:class:`NearestCentroid` is simple:

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

The :class:`NearestCentroid` classifier has a ``shrink_threshold`` parameter,
which implements the nearest shrunken centroid classifier. In effect, the value
of each feature for each centroid is divided by the within-class variance of
that feature. The feature values are then reduced by ``shrink_threshold``. Most
notably, if a particular feature value crosses zero, it is set
to zero. In effect, this removes the feature from affecting the classification.
This is useful, for example, for removing noisy features.

In the example below, using a small shrink threshold increases the accuracy of
the model from 0.81 to 0.82.

.. |nearest_centroid_1| image:: ../auto_examples/neighbors/images/sphx_glr_plot_nearest_centroid_001.png
   :target: ../auto_examples/neighbors/plot_nearest_centroid.html
   :scale: 50

.. |nearest_centroid_2| image:: ../auto_examples/neighbors/images/sphx_glr_plot_nearest_centroid_002.png
   :target: ../auto_examples/neighbors/plot_nearest_centroid.html
   :scale: 50

.. centered:: |nearest_centroid_1| |nearest_centroid_2|

.. topic:: Examples:

  * :ref:`sphx_glr_auto_examples_neighbors_plot_nearest_centroid.py`: an example of
    classification using nearest centroid with different shrink thresholds.
