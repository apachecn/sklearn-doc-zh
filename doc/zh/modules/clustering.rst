.. _clustering:

==========
Clustering（聚类）
==========

未标记的数据的 `Clustering（聚类） <https://en.wikipedia.org/wiki/Cluster_analysis>`__ 可以使用模块 :mod:`sklearn.cluster` 来实现。

每个 clustering algorithm （聚类算法）有两个变体: 一个是 class, 它实现了 ``fit`` 方法来学习 train data（训练数据）的 clusters（聚类），还有一个 function（函数），是给定 train data（训练数据），返回与不同 clusters（聚类）对应的整数标签 array（数组）。对于 class（类），training data（训练数据）上的标签可以在 ``labels_`` 属性中找到。

.. currentmodule:: sklearn.cluster

.. topic:: 输入数据

    需要注意的一点是，该模块中实现的算法可以采用不同种类的 matrix （矩阵）作为输入。所有这些都接受 shape ``[n_samples, n_features]`` 的标准数据矩阵。
    这些可以从以下的 :mod:`sklearn.feature_extraction` 模块的 classes （类）中获得。对于 :class:`AffinityPropagation`, :class:`SpectralClustering` 和 :class:`DBSCAN` 也可以输入 shape ``[n_samples, n_samples]`` 的相似矩阵。这些可以从 :mod:`sklearn.metrics.pairwise` 模块中的函数获得。

Clustering （聚类）方法概述
===============================

.. figure:: ../auto_examples/cluster/images/sphx_glr_plot_cluster_comparison_001.png
   :target: ../auto_examples/cluster/plot_cluster_comparison.html
   :align: center
   :scale: 50

   在 scikit-learn 中的 clustering algorithms （聚类算法）的比较


.. list-table::
   :header-rows: 1
   :widths: 14 15 19 25 20

   * - Method name（方法名称）
     - Parameters（参数）
     - Scalability（可扩展性）
     - Usecase（使用场景）
     - Geometry (metric used)（几何图形（公制使用））

   * - :ref:`K-Means（K-均值） <k_means>`
     - number of clusters（聚类形成的簇的个数）
     - 非常大的 ``n_samples``, 中等的 ``n_clusters`` 使用
       :ref:`MiniBatch code（MiniBatch 代码） <mini_batch_kmeans>`
     - 通用, 均匀的 cluster size（簇大小）, flat geometry（平面几何）, 不是太多的 clusters（簇）
     - Distances between points（点之间的距离）

   * - :ref:`Affinity propagation <affinity_propagation>`
     - damping（阻尼）, sample preference（样本偏好）
     - Not scalable with n_samples（n_samples 不可扩展）
     - Many clusters, uneven cluster size, non-flat geometry（许多簇，不均匀的簇大小，非平面几何）
     - Graph distance (e.g. nearest-neighbor graph)（图形距离（例如，最近邻图））

   * - :ref:`Mean-shift <mean_shift>`
     - bandwidth（带宽）
     - Not scalable with ``n_samples`` （不可扩展的 ``n_samples``）
     - Many clusters, uneven cluster size, non-flat geometry（许多簇，不均匀的簇大小，非平面几何）
     - Distances between points（点之间的距离）

   * - :ref:`Spectral clustering <spectral_clustering>`
     - number of clusters（簇的个数）
     - 中等的 ``n_samples``, 小的 ``n_clusters``
     - Few clusters, even cluster size, non-flat geometry（几个簇，均匀的簇大小，非平面几何）
     - Graph distance (e.g. nearest-neighbor graph)（图形距离（例如最近邻图））

   * - :ref:`Ward hierarchical clustering <hierarchical_clustering>`
     - number of clusters（簇的个数）
     - 大的 ``n_samples`` 和 ``n_clusters``
     - Many clusters, possibly connectivity constraints（很多的簇，可能连接限制）
     - Distances between points（点之间的距离）

   * - :ref:`Agglomerative clustering <hierarchical_clustering>`
     - number of clusters（簇的个数）, linkage type（链接类型）, distance（距离）
     - 大的 ``n_samples`` 和 ``n_clusters``
     - Many clusters, possibly connectivity constraints, non Euclidean distances（很多簇，可能连接限制，非欧几里得距离）
     - Any pairwise distance（任意成对距离）

   * - :ref:`DBSCAN <dbscan>`
     - neighborhood size（neighborhood 的大小）
     - 非常大的 ``n_samples``, 中等的 ``n_clusters``
     - Non-flat geometry, uneven cluster sizes（非平面几何，不均匀的簇大小）
     - Distances between nearest points（最近点之间的距离）

   * - :ref:`Gaussian mixtures（高斯混合） <mixture>`
     - many（很多）
     - Not scalable（不可扩展）
     - Flat geometry, good for density estimation（平面几何，适用于密度估计）
     - Mahalanobis distances to  centers（Mahalanobis 与中心的距离）

   * - :ref:`Birch`
     - branching factor（分支因子）, threshold（阈值）, optional global clusterer（可选全局簇）.
     - 大的 ``n_clusters`` 和 ``n_samples``
     - Large dataset, outlier removal, data reduction.（大数据集，异常值去除，数据简化）
     - Euclidean distance between points（点之间的欧式距离）

当 clusters （簇）具有 specific shape （特殊的形状），即 non-flat manifold（非平面 manifold），并且标准欧几里得距离不是正确的 metric （度量标准）时，Non-flat geometry clustering （非平面几何聚类）是非常有用的。这种情况出现在上图的两个顶行中。

用于 clustering （聚类）的 Gaussian mixture models （高斯混合模型），专用于 mixture models （混合模型）描述在 :ref:`文档的另一章节 <mixture>` 。可以将 KMeans 视为具有 equal covariance per component （每个分量相等协方差）的 Gaussian mixture model （高斯混合模型）的特殊情况。

.. _k_means:

K-means（K-均值）
=======

:class:`KMeans` 算法通过试图分离 n groups of equal variance（n 个相等方差组）的样本来聚集数据，minimizing （最小化）称为 `inertia <inertia>`_ 或者 within-cluster sum-of-squares （簇内和平方）的 criterion （标准）。
该算法需要指定 number of clusters （簇的数量）。它可以很好地 scales （扩展）到 large number of samples（大量样本），并已经被广泛应用于许多不同领域的应用领域。

k-means 算法将一组 :math:`N` 样本 :math:`X` 划分成 :math:`K` 不相交的 clusters （簇） :math:`C`, 每个都用 cluster （该簇）中的样本的均值 :math:`\mu_j` 描述。
这个 means （均值）通常被称为 cluster（簇）的 "centroids（质心）"; 注意，它们一般不是从 :math:`X` 中挑选出的点，虽然它们是处在同一个 space（空间）。
K-means（K-均值）算法旨在选择最小化 *inertia（惯性）* 或  within-cluster sum of squared（簇内和的平方和）的标准的 centroids（质心）:

.. math:: \sum_{i=0}^{n}\min_{\mu_j \in C}(||x_j - \mu_i||^2)

Inertia（惯性）, 或 the within-cluster sum of squares（簇内和平方差） criterion（标准）,可以被认为是 internally coherent clusters （内部想干聚类）的 measure （度量）。
它有各种缺点: 

- Inertia（惯性）假设 clusters （簇）是 convex（凸）的和 isotropic （各项同性），这并不是总是这样。它对 elongated clusters （细长的簇）或具有不规则形状的 manifolds 反应不佳。

- Inertia（惯性）不是一个 normalized metric（归一化度量）: 我们只知道 lower values （较低的值）是更好的，并且 零 是最优的。但是在 very high-dimensional spaces （非常高维的空间）中，欧几里得距离往往会变得 inflated （膨胀）（这就是所谓的 "curse of dimensionality （维度诅咒/维度惩罚）"）。在 k-means 聚类之前运行诸如 `PCA <PCA>`_ 之类的 dimensionality reduction algorithm （降维算法）可以减轻这个问题并加快计算速度。 

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_kmeans_assumptions_001.png
   :target: ../auto_examples/cluster/plot_kmeans_assumptions.html
   :align: center
   :scale: 50

K-means 通常被称为 Lloyd's algorithm（劳埃德算法）。在基本术语中，算法有三个步骤。、
第一步是选择 initial centroids （初始质心），最基本的方法是从 :math:`X` 数据集中选择 :math:`k` 个样本。初始化完成后，K-means 由两个其他步骤之间的循环组成。
第一步将每个样本分配到其 nearest centroid （最近的质心）。第二步通过取分配给每个先前质心的所有样本的平均值来创建新的质心。计算旧的和新的质心之间的差异，并且算法重复这些最后的两个步骤，直到该值小于阈值。换句话说，算法重复这个步骤，直到质心不再显著移动。

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_kmeans_digits_001.png
   :target: ../auto_examples/cluster/plot_kmeans_digits.html
   :align: right
   :scale: 35

K-means 相当于具有 small, all-equal, diagonal covariance matrix （小的全对称协方差矩阵）的 expectation-maximization algorithm （期望最大化算法）。

该算法也可以通过 `Voronoi diagrams（Voronoi图）<https://en.wikipedia.org/wiki/Voronoi_diagram>`_ 的概念来理解。首先使用 current centroids （当前质心）计算点的 Voronoi 图。
Voronoi 图中的每个 segment （段）都成为一个 separate cluster （单独的簇）。其次，centroids（质心）被更新为每个 segment （段）的 mean（平均值）。然后，该算法重复此操作，直到满足停止条件。
通常情况下，当 iterations （迭代）之间的 objective function （目标函数）的相对减小小于给定的 tolerance value （公差值）时，算法停止。在此实现中不是这样: 当质心移动小于 tolerance （公差）时，迭代停止。

给定足够的时间，K-means 将总是收敛的，但这可能是 local minimum （局部最小）的。这很大程度上取决于 initialization of the centroids （质心的初始化）。
因此，通常会进行几次 different initializations of the centroids （初始化不同质心）的计算。帮助解决这个问题的一种方法是 k-means++ 初始化方案，它已经在 scikit-learn 中实现（使用 ``init='k-means++'`` 参数）。
这将初始化 centroids （质心）（通常）彼此远离，导致比随机初始化更好的结果，如参考文献所示。

可以给出一个参数，以允许 K-means 并行运行，称为 ``n_jobs``。给这个参数一个正值使用许多处理器（默认值: 1）。值 -1 使用所有可用的处理器，-2 使用一个，等等。Parallelization （并行化）通常以 cost of memory（内存的代价）加速计算（在这种情况下，需要存储多个质心副本，每个作业使用一个）。

.. warning::

    当 `numpy` 使用 `Accelerate` 框架时，K-Means 的并行版本在 OS X 上损坏。这是 expected behavior （预期的行为）: `Accelerate` 可以在 fork 之后调用，但是您需要使用 Python binary（二进制）（该多进程在 posix 下不执行）来执行子进程。

K-means 可用于 vector quantization （矢量量化）。这是使用以下类型的 trained model （训练模型）的变换方法实现的 :class:`KMeans` 。

.. topic:: 示例:

 * :ref:`sphx_glr_auto_examples_cluster_plot_kmeans_assumptions.py`: 演示 k-means 是否 performs intuitively （直观执行），何时不执行
 * :ref:`sphx_glr_auto_examples_cluster_plot_kmeans_digits.py`: 聚类手写数字

.. topic:: 参考:

 * `"k-means++: The advantages of careful seeding"
   <http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf>`_
   Arthur, David, and Sergei Vassilvitskii,
   *Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete
   algorithms*, Society for Industrial and Applied Mathematics (2007)

.. _mini_batch_kmeans:

Mini Batch K-Means（小批量 K-Means）
------------------

:class:`MiniBatchKMeans` 是 :class:`KMeans` 算法的一个变体，它使用 mini-batches （小批量）来减少计算时间，同时仍然尝试优化相同的 objective function （目标函数）。
Mini-batches（小批量）是输入数据的子集，在每次 training iteration （训练迭代）中 randomly sampled （随机抽样）。这些小批量大大减少了融合到本地解决方案所需的计算量。
与其他降低 k-means 收敛时间的算法相反，mini-batch k-means 产生的结果通常只比标准算法略差。

该算法在两个主要步骤之间进行迭代，类似于 vanilla k-means 。
在第一步， :math:`b` 样本是从数据集中随机抽取的，形成一个 mini-batch （小批量）。然后将它们分配到最近的 centroid（质心）。
在第二步，centroids （质心）被更新。与 k-means 相反，这是在每个样本的基础上完成的。对于 mini-batch （小批量）中的每个样本，通过取样本的 streaming average （流平均值）和分配给该质心的所有先前样本来更新分配的质心。
这具有随时间降低 centroid （质心）的 rate （变化率）的效果。执行这些步骤直到达到收敛或达到预定次数的迭代。

:class:`MiniBatchKMeans` 收敛速度比 :class:`KMeans` ，但是结果的质量会降低。在实践中，质量差异可能相当小，如示例和引用的参考。

.. figure:: ../auto_examples/cluster/images/sphx_glr_plot_mini_batch_kmeans_001.png
   :target: ../auto_examples/cluster/plot_mini_batch_kmeans.html
   :align: center
   :scale: 100


.. topic:: 示例:

 * :ref:`sphx_glr_auto_examples_cluster_plot_mini_batch_kmeans.py`: KMeans 与 MiniBatchKMeans 的比较

 * :ref:`sphx_glr_auto_examples_text_document_clustering.py`: 使用 sparse MiniBatchKMeans （稀疏 MiniBatchKMeans）的文档聚类

 * :ref:`sphx_glr_auto_examples_cluster_plot_dict_face_patches.py`


.. topic:: 参考:

 * `"Web Scale K-Means clustering"
   <http://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf>`_
   D. Sculley, *Proceedings of the 19th international conference on World
   wide web* (2010)

.. _affinity_propagation:

Affinity Propagation
====================

:class:`AffinityPropagation` creates clusters by sending messages between
pairs of samples until convergence. A dataset is then described using a small
number of exemplars, which are identified as those most representative of other
samples. The messages sent between pairs represent the suitability for one
sample to be the exemplar of the other, which is updated in response to the
values from other pairs. This updating happens iteratively until convergence,
at which point the final exemplars are chosen, and hence the final clustering
is given.

.. figure:: ../auto_examples/cluster/images/sphx_glr_plot_affinity_propagation_001.png
   :target: ../auto_examples/cluster/plot_affinity_propagation.html
   :align: center
   :scale: 50


Affinity Propagation can be interesting as it chooses the number of
clusters based on the data provided. For this purpose, the two important
parameters are the *preference*, which controls how many exemplars are
used, and the *damping factor* which damps the responsibility and 
availability messages to avoid numerical oscillations when updating these
messages.

The main drawback of Affinity Propagation is its complexity. The
algorithm has a time complexity of the order :math:`O(N^2 T)`, where :math:`N`
is the number of samples and :math:`T` is the number of iterations until
convergence. Further, the memory complexity is of the order
:math:`O(N^2)` if a dense similarity matrix is used, but reducible if a
sparse similarity matrix is used. This makes Affinity Propagation most
appropriate for small to medium sized datasets.

.. topic:: Examples:

 * :ref:`sphx_glr_auto_examples_cluster_plot_affinity_propagation.py`: Affinity
   Propagation on a synthetic 2D datasets with 3 classes.

 * :ref:`sphx_glr_auto_examples_applications_plot_stock_market.py` Affinity Propagation on
   Financial time series to find groups of companies


**Algorithm description:**
The messages sent between points belong to one of two categories. The first is
the responsibility :math:`r(i, k)`,
which is the accumulated evidence that sample :math:`k`
should be the exemplar for sample :math:`i`.
The second is the availability :math:`a(i, k)`
which is the accumulated evidence that sample :math:`i`
should choose sample :math:`k` to be its exemplar,
and considers the values for all other samples that :math:`k` should
be an exemplar. In this way, exemplars are chosen by samples if they are (1)
similar enough to many samples and (2) chosen by many samples to be
representative of themselves.

More formally, the responsibility of a sample :math:`k`
to be the exemplar of sample :math:`i` is given by:

.. math::

    r(i, k) \leftarrow s(i, k) - max [ a(i, k') + s(i, k') \forall k' \neq k ]

Where :math:`s(i, k)` is the similarity between samples :math:`i` and :math:`k`.
The availability of sample :math:`k`
to be the exemplar of sample :math:`i` is given by:

.. math::

    a(i, k) \leftarrow min [0, r(k, k) + \sum_{i'~s.t.~i' \notin \{i, k\}}{r(i', k)}]

To begin with, all values for :math:`r` and :math:`a` are set to zero,
and the calculation of each iterates until convergence.
As discussed above, in order to avoid numerical oscillations when updating the 
messages, the damping factor :math:`\lambda` is introduced to iteration process:

.. math:: r_{t+1}(i, k) = \lambda\cdot r_{t}(i, k) + (1-\lambda)\cdot r_{t+1}(i, k)
.. math:: a_{t+1}(i, k) = \lambda\cdot a_{t}(i, k) + (1-\lambda)\cdot a_{t+1}(i, k)

where :math:`t` indicates the iteration times.

.. _mean_shift:

Mean Shift
==========
:class:`MeanShift` clustering aims to discover *blobs* in a smooth density of
samples. It is a centroid based algorithm, which works by updating candidates
for centroids to be the mean of the points within a given region. These
candidates are then filtered in a post-processing stage to eliminate
near-duplicates to form the final set of centroids.

Given a candidate centroid :math:`x_i` for iteration :math:`t`, the candidate
is updated according to the following equation:

.. math::

    x_i^{t+1} = x_i^t + m(x_i^t)

Where :math:`N(x_i)` is the neighborhood of samples within a given distance
around :math:`x_i` and :math:`m` is the *mean shift* vector that is computed for each
centroid that points towards a region of the maximum increase in the density of points.
This is computed using the following equation, effectively updating a centroid
to be the mean of the samples within its neighborhood:

.. math::

    m(x_i) = \frac{\sum_{x_j \in N(x_i)}K(x_j - x_i)x_j}{\sum_{x_j \in N(x_i)}K(x_j - x_i)}

The algorithm automatically sets the number of clusters, instead of relying on a
parameter ``bandwidth``, which dictates the size of the region to search through.
This parameter can be set manually, but can be estimated using the provided
``estimate_bandwidth`` function, which is called if the bandwidth is not set.

The algorithm is not highly scalable, as it requires multiple nearest neighbor
searches during the execution of the algorithm. The algorithm is guaranteed to
converge, however the algorithm will stop iterating when the change in centroids
is small.

Labelling a new sample is performed by finding the nearest centroid for a
given sample.


.. figure:: ../auto_examples/cluster/images/sphx_glr_plot_mean_shift_001.png
   :target: ../auto_examples/cluster/plot_mean_shift.html
   :align: center
   :scale: 50


.. topic:: Examples:

 * :ref:`sphx_glr_auto_examples_cluster_plot_mean_shift.py`: Mean Shift clustering
   on a synthetic 2D datasets with 3 classes.

.. topic:: References:

 * `"Mean shift: A robust approach toward feature space analysis."
   <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.76.8968&rep=rep1&type=pdf>`_
   D. Comaniciu and P. Meer, *IEEE Transactions on Pattern Analysis and Machine Intelligence* (2002)


.. _spectral_clustering:

Spectral clustering
===================

:class:`SpectralClustering` does a low-dimension embedding of the
affinity matrix between samples, followed by a KMeans in the low
dimensional space. It is especially efficient if the affinity matrix is
sparse and the `pyamg <http://pyamg.org/>`_ module is installed.
SpectralClustering requires the number of clusters to be specified. It
works well for a small number of clusters but is not advised when using
many clusters.

For two clusters, it solves a convex relaxation of the `normalised
cuts <http://people.eecs.berkeley.edu/~malik/papers/SM-ncut.pdf>`_ problem on
the similarity graph: cutting the graph in two so that the weight of the
edges cut is small compared to the weights of the edges inside each
cluster. This criteria is especially interesting when working on images:
graph vertices are pixels, and edges of the similarity graph are a
function of the gradient of the image.


.. |noisy_img| image:: ../auto_examples/cluster/images/sphx_glr_plot_segmentation_toy_001.png
    :target: ../auto_examples/cluster/plot_segmentation_toy.html
    :scale: 50

.. |segmented_img| image:: ../auto_examples/cluster/images/sphx_glr_plot_segmentation_toy_002.png
    :target: ../auto_examples/cluster/plot_segmentation_toy.html
    :scale: 50

.. centered:: |noisy_img| |segmented_img|

.. warning:: Transforming distance to well-behaved similarities

    Note that if the values of your similarity matrix are not well
    distributed, e.g. with negative values or with a distance matrix
    rather than a similarity, the spectral problem will be singular and
    the problem not solvable. In which case it is advised to apply a
    transformation to the entries of the matrix. For instance, in the
    case of a signed distance matrix, is common to apply a heat kernel::

        similarity = np.exp(-beta * distance / distance.std())

    See the examples for such an application.

.. topic:: Examples:

 * :ref:`sphx_glr_auto_examples_cluster_plot_segmentation_toy.py`: Segmenting objects
   from a noisy background using spectral clustering.

 * :ref:`sphx_glr_auto_examples_cluster_plot_face_segmentation.py`: Spectral clustering
   to split the image of the raccoon face in regions.

.. |face_kmeans| image:: ../auto_examples/cluster/images/sphx_glr_plot_face_segmentation_001.png
    :target: ../auto_examples/cluster/plot_face_segmentation.html
    :scale: 65

.. |face_discretize| image:: ../auto_examples/cluster/images/sphx_glr_plot_face_segmentation_002.png
    :target: ../auto_examples/cluster/plot_face_segmentation.html
    :scale: 65

Different label assignment strategies
---------------------------------------

Different label assignment strategies can be used, corresponding to the
``assign_labels`` parameter of :class:`SpectralClustering`.
The ``"kmeans"`` strategy can match finer details of the data, but it can be
more unstable. In particular, unless you control the ``random_state``, it
may not be reproducible from run-to-run, as it depends on a random
initialization. On the other hand, the ``"discretize"`` strategy is 100%
reproducible, but it tends to create parcels of fairly even and
geometrical shape.

=====================================  =====================================
 ``assign_labels="kmeans"``              ``assign_labels="discretize"``
=====================================  =====================================
|face_kmeans|                          |face_discretize|
=====================================  =====================================


.. topic:: References:

 * `"A Tutorial on Spectral Clustering"
   <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.165.9323>`_
   Ulrike von Luxburg, 2007

 * `"Normalized cuts and image segmentation"
   <http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.160.2324>`_
   Jianbo Shi, Jitendra Malik, 2000

 * `"A Random Walks View of Spectral Segmentation"
   <http://citeseer.ist.psu.edu/viewdoc/summary?doi=10.1.1.33.1501>`_
   Marina Meila, Jianbo Shi, 2001

 * `"On Spectral Clustering: Analysis and an algorithm"
   <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.19.8100>`_
   Andrew Y. Ng, Michael I. Jordan, Yair Weiss, 2001


.. _hierarchical_clustering:

Hierarchical clustering
=======================

Hierarchical clustering is a general family of clustering algorithms that
build nested clusters by merging or splitting them successively. This
hierarchy of clusters is represented as a tree (or dendrogram). The root of the
tree is the unique cluster that gathers all the samples, the leaves being the
clusters with only one sample. See the `Wikipedia page
<https://en.wikipedia.org/wiki/Hierarchical_clustering>`_ for more details.

The :class:`AgglomerativeClustering` object performs a hierarchical clustering
using a bottom up approach: each observation starts in its own cluster, and
clusters are successively merged together. The linkage criteria determines the
metric used for the merge strategy:

- **Ward** minimizes the sum of squared differences within all clusters. It is a
  variance-minimizing approach and in this sense is similar to the k-means
  objective function but tackled with an agglomerative hierarchical
  approach.
- **Maximum** or **complete linkage** minimizes the maximum distance between
  observations of pairs of clusters.
- **Average linkage** minimizes the average of the distances between all
  observations of pairs of clusters.

:class:`AgglomerativeClustering` can also scale to large number of samples
when it is used jointly with a connectivity matrix, but is computationally
expensive when no connectivity constraints are added between samples: it
considers at each step all the possible merges.

.. topic:: :class:`FeatureAgglomeration`

   The :class:`FeatureAgglomeration` uses agglomerative clustering to
   group together features that look very similar, thus decreasing the
   number of features. It is a dimensionality reduction tool, see
   :ref:`data_reduction`.

Different linkage type: Ward, complete and average linkage
-----------------------------------------------------------

:class:`AgglomerativeClustering` supports Ward, average, and complete
linkage strategies.

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_digits_linkage_001.png
    :target: ../auto_examples/cluster/plot_digits_linkage.html
    :scale: 43

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_digits_linkage_002.png
    :target: ../auto_examples/cluster/plot_digits_linkage.html
    :scale: 43

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_digits_linkage_003.png
    :target: ../auto_examples/cluster/plot_digits_linkage.html
    :scale: 43


Agglomerative cluster has a "rich get richer" behavior that leads to
uneven cluster sizes. In this regard, complete linkage is the worst
strategy, and Ward gives the most regular sizes. However, the affinity
(or distance used in clustering) cannot be varied with Ward, thus for non
Euclidean metrics, average linkage is a good alternative.

.. topic:: Examples:

 * :ref:`sphx_glr_auto_examples_cluster_plot_digits_linkage.py`: exploration of the
   different linkage strategies in a real dataset.


Adding connectivity constraints
-------------------------------

An interesting aspect of :class:`AgglomerativeClustering` is that
connectivity constraints can be added to this algorithm (only adjacent
clusters can be merged together), through a connectivity matrix that defines
for each sample the neighboring samples following a given structure of the
data. For instance, in the swiss-roll example below, the connectivity
constraints forbid the merging of points that are not adjacent on the swiss
roll, and thus avoid forming clusters that extend across overlapping folds of
the roll.

.. |unstructured| image:: ../auto_examples/cluster/images/sphx_glr_plot_ward_structured_vs_unstructured_001.png
        :target: ../auto_examples/cluster/plot_ward_structured_vs_unstructured.html
        :scale: 49

.. |structured| image:: ../auto_examples/cluster/images/sphx_glr_plot_ward_structured_vs_unstructured_002.png
        :target: ../auto_examples/cluster/plot_ward_structured_vs_unstructured.html
        :scale: 49

.. centered:: |unstructured| |structured|

These constraint are useful to impose a certain local structure, but they
also make the algorithm faster, especially when the number of the samples
is high.

The connectivity constraints are imposed via an connectivity matrix: a
scipy sparse matrix that has elements only at the intersection of a row
and a column with indices of the dataset that should be connected. This
matrix can be constructed from a-priori information: for instance, you
may wish to cluster web pages by only merging pages with a link pointing
from one to another. It can also be learned from the data, for instance
using :func:`sklearn.neighbors.kneighbors_graph` to restrict
merging to nearest neighbors as in :ref:`this example
<sphx_glr_auto_examples_cluster_plot_agglomerative_clustering.py>`, or
using :func:`sklearn.feature_extraction.image.grid_to_graph` to
enable only merging of neighboring pixels on an image, as in the
:ref:`raccoon face <sphx_glr_auto_examples_cluster_plot_face_ward_segmentation.py>` example.

.. topic:: Examples:

 * :ref:`sphx_glr_auto_examples_cluster_plot_face_ward_segmentation.py`: Ward clustering
   to split the image of a raccoon face in regions.

 * :ref:`sphx_glr_auto_examples_cluster_plot_ward_structured_vs_unstructured.py`: Example of
   Ward algorithm on a swiss-roll, comparison of structured approaches
   versus unstructured approaches.

 * :ref:`sphx_glr_auto_examples_cluster_plot_feature_agglomeration_vs_univariate_selection.py`:
   Example of dimensionality reduction with feature agglomeration based on
   Ward hierarchical clustering.

 * :ref:`sphx_glr_auto_examples_cluster_plot_agglomerative_clustering.py`

.. warning:: **Connectivity constraints with average and complete linkage**

    Connectivity constraints and complete or average linkage can enhance
    the 'rich getting richer' aspect of agglomerative clustering,
    particularly so if they are built with
    :func:`sklearn.neighbors.kneighbors_graph`. In the limit of a small
    number of clusters, they tend to give a few macroscopically occupied
    clusters and almost empty ones. (see the discussion in
    :ref:`sphx_glr_auto_examples_cluster_plot_agglomerative_clustering.py`).

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_agglomerative_clustering_001.png
    :target: ../auto_examples/cluster/plot_agglomerative_clustering.html
    :scale: 38

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_agglomerative_clustering_002.png
    :target: ../auto_examples/cluster/plot_agglomerative_clustering.html
    :scale: 38

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_agglomerative_clustering_003.png
    :target: ../auto_examples/cluster/plot_agglomerative_clustering.html
    :scale: 38

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_agglomerative_clustering_004.png
    :target: ../auto_examples/cluster/plot_agglomerative_clustering.html
    :scale: 38


Varying the metric
-------------------

Average and complete linkage can be used with a variety of distances (or
affinities), in particular Euclidean distance (*l2*), Manhattan distance
(or Cityblock, or *l1*), cosine distance, or any precomputed affinity
matrix.

* *l1* distance is often good for sparse features, or sparse noise: ie
  many of the features are zero, as in text mining using occurrences of
  rare words.

* *cosine* distance is interesting because it is invariant to global
  scalings of the signal.

The guidelines for choosing a metric is to use one that maximizes the
distance between samples in different classes, and minimizes that within
each class.

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_agglomerative_clustering_metrics_005.png
    :target: ../auto_examples/cluster/plot_agglomerative_clustering_metrics.html
    :scale: 32

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_agglomerative_clustering_metrics_006.png
    :target: ../auto_examples/cluster/plot_agglomerative_clustering_metrics.html
    :scale: 32

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_agglomerative_clustering_metrics_007.png
    :target: ../auto_examples/cluster/plot_agglomerative_clustering_metrics.html
    :scale: 32

.. topic:: Examples:

 * :ref:`sphx_glr_auto_examples_cluster_plot_agglomerative_clustering_metrics.py`


.. _dbscan:

DBSCAN
======

The :class:`DBSCAN` algorithm views clusters as areas of high density
separated by areas of low density. Due to this rather generic view, clusters
found by DBSCAN can be any shape, as opposed to k-means which assumes that
clusters are convex shaped. The central component to the DBSCAN is the concept
of *core samples*, which are samples that are in areas of high density. A
cluster is therefore a set of core samples, each close to each other
(measured by some distance measure)
and a set of non-core samples that are close to a core sample (but are not
themselves core samples). There are two parameters to the algorithm,
``min_samples`` and ``eps``,
which define formally what we mean when we say *dense*.
Higher ``min_samples`` or lower ``eps``
indicate higher density necessary to form a cluster.

More formally, we define a core sample as being a sample in the dataset such
that there exist ``min_samples`` other samples within a distance of
``eps``, which are defined as *neighbors* of the core sample. This tells
us that the core sample is in a dense area of the vector space. A cluster
is a set of core samples that can be built by recursively taking a core
sample, finding all of its neighbors that are core samples, finding all of
*their* neighbors that are core samples, and so on. A cluster also has a
set of non-core samples, which are samples that are neighbors of a core sample
in the cluster but are not themselves core samples. Intuitively, these samples
are on the fringes of a cluster.

Any core sample is part of a cluster, by definition. Any sample that is not a 
core sample, and is at least ``eps`` in distance from any core sample, is 
considered an outlier by the algorithm.

In the figure below, the color indicates cluster membership, with large circles
indicating core samples found by the algorithm. Smaller circles are non-core
samples that are still part of a cluster. Moreover, the outliers are indicated
by black points below.

.. |dbscan_results| image:: ../auto_examples/cluster/images/sphx_glr_plot_dbscan_001.png
        :target: ../auto_examples/cluster/plot_dbscan.html
        :scale: 50

.. centered:: |dbscan_results|

.. topic:: Examples:

    * :ref:`sphx_glr_auto_examples_cluster_plot_dbscan.py`

.. topic:: Implementation

    The DBSCAN algorithm is deterministic, always generating the same clusters 
    when given the same data in the same order.  However, the results can differ when
    data is provided in a different order. First, even though the core samples 
    will always be assigned to the same clusters, the labels of those clusters
    will depend on the order in which those samples are encountered in the data.
    Second and more importantly, the clusters to which non-core samples are assigned
    can differ depending on the data order.  This would happen when a non-core sample
    has a distance lower than ``eps`` to two core samples in different clusters. By the
    triangular inequality, those two core samples must be more distant than
    ``eps`` from each other, or they would be in the same cluster. The non-core
    sample is assigned to whichever cluster is generated first in a pass
    through the data, and so the results will depend on the data ordering.

    The current implementation uses ball trees and kd-trees
    to determine the neighborhood of points,
    which avoids calculating the full distance matrix
    (as was done in scikit-learn versions before 0.14).
    The possibility to use custom metrics is retained;
    for details, see :class:`NearestNeighbors`.

.. topic:: Memory consumption for large sample sizes

    This implementation is by default not memory efficient because it constructs
    a full pairwise similarity matrix in the case where kd-trees or ball-trees cannot
    be used (e.g. with sparse matrices). This matrix will consume n^2 floats.
    A couple of mechanisms for getting around this are:

    - A sparse radius neighborhood graph (where missing
      entries are presumed to be out of eps) can be precomputed in a memory-efficient
      way and dbscan can be run over this with ``metric='precomputed'``.

    - The dataset can be compressed, either by removing exact duplicates if
      these occur in your data, or by using BIRCH. Then you only have a
      relatively small number of representatives for a large number of points.
      You can then provide a ``sample_weight`` when fitting DBSCAN.

.. topic:: References:

 * "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases
   with Noise"
   Ester, M., H. P. Kriegel, J. Sander, and X. Xu,
   In Proceedings of the 2nd International Conference on Knowledge Discovery
   and Data Mining, Portland, OR, AAAI Press, pp. 226–231. 1996

.. _birch:

Birch
=====

The :class:`Birch` builds a tree called the Characteristic Feature Tree (CFT)
for the given data. The data is essentially lossy compressed to a set of
Characteristic Feature nodes (CF Nodes). The CF Nodes have a number of
subclusters called Characteristic Feature subclusters (CF Subclusters)
and these CF Subclusters located in the non-terminal CF Nodes
can have CF Nodes as children.

The CF Subclusters hold the necessary information for clustering which prevents
the need to hold the entire input data in memory. This information includes:

- Number of samples in a subcluster.
- Linear Sum - A n-dimensional vector holding the sum of all samples
- Squared Sum - Sum of the squared L2 norm of all samples.
- Centroids - To avoid recalculation linear sum / n_samples.
- Squared norm of the centroids.

The Birch algorithm has two parameters, the threshold and the branching factor.
The branching factor limits the number of subclusters in a node and the
threshold limits the distance between the entering sample and the existing
subclusters.

This algorithm can be viewed as an instance or data reduction method,
since it reduces the input data to a set of subclusters which are obtained directly
from the leaves of the CFT. This reduced data can be further processed by feeding
it into a global clusterer. This global clusterer can be set by ``n_clusters``.
If ``n_clusters`` is set to None, the subclusters from the leaves are directly
read off, otherwise a global clustering step labels these subclusters into global
clusters (labels) and the samples are mapped to the global label of the nearest subcluster.

**Algorithm description:**

- A new sample is inserted into the root of the CF Tree which is a CF Node.
  It is then merged with the subcluster of the root, that has the smallest
  radius after merging, constrained by the threshold and branching factor conditions.
  If the subcluster has any child node, then this is done repeatedly till it reaches
  a leaf. After finding the nearest subcluster in the leaf, the properties of this
  subcluster and the parent subclusters are recursively updated.

- If the radius of the subcluster obtained by merging the new sample and the
  nearest subcluster is greater than the square of the threshold and if the
  number of subclusters is greater than the branching factor, then a space is temporarily
  allocated to this new sample. The two farthest subclusters are taken and
  the subclusters are divided into two groups on the basis of the distance
  between these subclusters.

- If this split node has a parent subcluster and there is room
  for a new subcluster, then the parent is split into two. If there is no room,
  then this node is again split into two and the process is continued
  recursively, till it reaches the root.

**Birch or MiniBatchKMeans?**

 - Birch does not scale very well to high dimensional data. As a rule of thumb if
   ``n_features`` is greater than twenty, it is generally better to use MiniBatchKMeans.
 - If the number of instances of data needs to be reduced, or if one wants a
   large number of subclusters either as a preprocessing step or otherwise,
   Birch is more useful than MiniBatchKMeans.


**How to use partial_fit?**

To avoid the computation of global clustering, for every call of ``partial_fit``
the user is advised

 1. To set ``n_clusters=None`` initially
 2. Train all data by multiple calls to partial_fit.
 3. Set ``n_clusters`` to a required value using
    ``brc.set_params(n_clusters=n_clusters)``.
 4. Call ``partial_fit`` finally with no arguments, i.e ``brc.partial_fit()``
    which performs the global clustering.

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_birch_vs_minibatchkmeans_001.png
    :target: ../auto_examples/cluster/plot_birch_vs_minibatchkmeans.html

.. topic:: References:

 * Tian Zhang, Raghu Ramakrishnan, Maron Livny
   BIRCH: An efficient data clustering method for large databases.
   http://www.cs.sfu.ca/CourseCentral/459/han/papers/zhang96.pdf

 * Roberto Perdisci
   JBirch - Java implementation of BIRCH clustering algorithm
   https://code.google.com/archive/p/jbirch


.. _clustering_evaluation:

Clustering performance evaluation（聚类性能评估）
=================================

评估聚类算法的性能不像 counting the number of errors （计数错误数量）或监督分类算法的精度和调用那样微不足道。
特别地，任何 evaluation metric （评估度量）不应该考虑到 cluster labels （簇标签）的绝对值，而是如果这个聚类定义类似于 some ground truth set of classes or satisfying some assumption （某些基本真值集合的数据的分离或者满足一些假设），使得属于同一个类的成员更类似于根据某些 similarity metric （相似性度量）的不同类的成员。

.. currentmodule:: sklearn.metrics

.. _adjusted_rand_score:

Adjusted Rand index（调整后的 Rand index）
-------------------

考虑到 the ground truth class 赋值 ``labels_true`` 和相同样本 ``labels_pred`` 的聚类算法分配的知识，**adjusted Rand index** 是一个函数，用于测量两个 assignments （作业）的 **similarity（相似度）** ，忽略 permutations （置换）和 **with chance normalization（使用机会规范化）**::

  >>> from sklearn import metrics
  >>> labels_true = [0, 0, 0, 1, 1, 1]
  >>> labels_pred = [0, 0, 1, 1, 2, 2]

  >>> metrics.adjusted_rand_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  0.24...

可以在预测的标签中 permute （排列） 0 和 1，重命名为 2 到 3， 得到相同的分数::

  >>> labels_pred = [1, 1, 0, 0, 3, 3]
  >>> metrics.adjusted_rand_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  0.24...

另外，:func:`adjusted_rand_score` 是 **symmetric（对称的）**: 交换参数不会改变 score （得分）。它可以作为 **consensus measure（共识度量）**::

  >>> metrics.adjusted_rand_score(labels_pred, labels_true)  # doctest: +ELLIPSIS
  0.24...

完美的标签得分为 1.0::

  >>> labels_pred = labels_true[:]
  >>> metrics.adjusted_rand_score(labels_true, labels_pred)
  1.0

坏 (e.g. independent labelings（独立标签）) 有负数 or 接近于 0.0 分::

  >>> labels_true = [0, 1, 2, 0, 3, 4, 5, 1]
  >>> labels_pred = [1, 1, 0, 0, 2, 2, 2, 2]
  >>> metrics.adjusted_rand_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  -0.12...


Advantages（优点）
~~~~~~~~~~

- **Random (uniform) label assignments have a ARI score close to 0.0（随机（统一）标签分配的 ARI 评分接近于 0.0）**
  对于 ``n_clusters`` 和 ``n_samples`` 的任何值（这不是原始的 Rand index 或者 V-measure 的情况）。

- **Bounded range（有界范围） [-1, 1]**: negative values （负值）是坏的 (独立性标签), 类似的聚类有一个 positive ARI （正的 ARI）， 1.0 是完美的匹配得分。

- **No assumption is made on the cluster structure（对簇的结构没有作出任何假设）**: 可以用于比较聚类算法，例如 k-means，其假定 isotropic blob shapes 与可以找到具有 "folded" shapes 的聚类的 spectral clustering algorithms（频谱聚类算法）的结果。


Drawbacks（缺点）
~~~~~~~~~

- 与 inertia 相反，**ARI requires knowledge of the ground truth classes（ARI需要了解 ground truth classes）** ，而在实践中几乎不可用，或者需要人工标注者手动分配（如在监督学习环境中）。

  然而，ARI 还可以在 purely unsupervised setting （纯粹无监督的设置中）作为可用于 聚类模型选择（TODO）的共识索引的构建块。


.. topic:: 示例:

 * :ref:`sphx_glr_auto_examples_cluster_plot_adjusted_for_chance_measures.py`: 分析数据集大小对随机分配聚类度量值的影响。


Mathematical formulation（数学表达）
~~~~~~~~~~~~~~~~~~~~~~~~

如果 C 是一个 ground truth class assignment （标定过的真实数据类分配）和 K 个簇，就让我们定义 :math:`a` 和 :math:`b` 如:

- :math:`a`, 在 C 中的相同集合的与 K 中的相同集合中的元素的对数

- :math:`b`, 在 C 中的不同集合与 K 中的不同集合中的元素的对数

原始的（unadjusted（未调整的）） Rand index 则由下式给出: 

.. math:: \text{RI} = \frac{a + b}{C_2^{n_{samples}}}

其中 :math:`C_2^{n_{samples}}` 是数据集中可能的 pairs （对）的总数（不排序）。

然而，RI 评分不能保证 random label assignments （随机标签分配）将获得接近零的值（特别是如果聚类的数量与采样数量相同的数量级）。

为了抵消这种影响，我们可以通过定义 adjusted Rand index （调整后的 Rand index）来 discount 随机标签的预期 RI :math:`E[\text{RI}]` ,如下所示:

.. math:: \text{ARI} = \frac{\text{RI} - E[\text{RI}]}{\max(\text{RI}) - E[\text{RI}]}

.. topic:: 参考

 * `Comparing Partitions
   <http://link.springer.com/article/10.1007%2FBF01908075>`_
   L. Hubert and P. Arabie, Journal of Classification 1985

 * `Wikipedia entry for the adjusted Rand index
   <https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index>`_

.. _mutual_info_score:

Mutual Information based scores （基于 Mutual Information 的分数）
-------------------------------

考虑到 ground truth class assignments （标定过的真实数据类分配） ``labels_true`` 的知识和相同样本 ``labels_pred`` 的聚类算法分配， **Mutual Information** 是测量两者 **agreement** 分配的函数，忽略 permutations（排列）。
这种测量方案的两个不同的标准化版本可用，**Normalized Mutual Information(NMI)** 和 **Adjusted Mutual Information(AMI)**。NMI 经常在文献中使用，而 AMI 最近被提出，并且 **normalized against chance**::

  >>> from sklearn import metrics
  >>> labels_true = [0, 0, 0, 1, 1, 1]
  >>> labels_pred = [0, 0, 1, 1, 2, 2]

  >>> metrics.adjusted_mutual_info_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  0.22504...

可以在 predicted labels （预测的标签）中 permute （排列） 0 和 1, 重命名为 2 到 3 并得到相同的得分::

  >>> labels_pred = [1, 1, 0, 0, 3, 3]
  >>> metrics.adjusted_mutual_info_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  0.22504...

全部的，:func:`mutual_info_score`, :func:`adjusted_mutual_info_score` 和 :func:`normalized_mutual_info_score` 是 symmetric（对称的）: 交换参数不会更改分数。因此，它们可以用作 **consensus measure**::

  >>> metrics.adjusted_mutual_info_score(labels_pred, labels_true)  # doctest: +ELLIPSIS
  0.22504...

完美标签得分是 1.0::

  >>> labels_pred = labels_true[:]
  >>> metrics.adjusted_mutual_info_score(labels_true, labels_pred)
  1.0

  >>> metrics.normalized_mutual_info_score(labels_true, labels_pred)
  1.0

这对于 ``mutual_info_score`` 是不正确的，因此更难判断::

  >>> metrics.mutual_info_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  0.69...

坏的 (例如 independent labelings（独立标签）) 具有非正分数::

  >>> labels_true = [0, 1, 2, 0, 3, 4, 5, 1]
  >>> labels_pred = [1, 1, 0, 0, 2, 2, 2, 2]
  >>> metrics.adjusted_mutual_info_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  -0.10526...


Advantages（优点）
~~~~~~~~~~

- **Random (uniform) label assignments have a AMI score close to 0.0（随机（统一）标签分配的AMI评分接近0.0）**
  对于 ``n_clusters`` 和 ``n_samples`` 的任何值（这不是原始 Mutual Information 或者 V-measure 的情况）。

- **Bounded range（有界范围） [0, 1]**:  接近 0 的值表示两个主要独立的标签分配，而接近 1 的值表示重要的一致性。此外，正好 0 的值表示 **purely（纯粹）** 独立标签分配，正好为 1 的 AMI 表示两个标签分配相等（有或者没有 permutation）。

- **No assumption is made on the cluster structure（对簇的结构没有作出任何假设）**: 可以用于比较聚类算法，例如 k-means，其假定 isotropic blob shapes 与可以找到具有 "folded" shapes 的聚类的 spectral clustering algorithms （频谱聚类算法）的结果。


Drawbacks（缺点）
~~~~~~~~~

- 与 inertia 相反，**MI-based measures require the knowledge of the ground truth classes（MI-based measures 需要了解 ground truth classes）** ，而在实践中几乎不可用，或者需要人工标注者手动分配（如在监督学习环境中）。

  然而，基于 MI-based measures （基于 MI 的测量方式）也可用于纯无人监控的设置，作为可用于聚类模型选择的 Consensus Index （共识索引）的构建块。

- NMI 和 MI 没有调整机会。


.. topic:: 示例:

 * :ref:`sphx_glr_auto_examples_cluster_plot_adjusted_for_chance_measures.py`: 分析数据集大小对随机分配聚类度量值的影响。 此示例还包括 Adjusted Rand Index。


Mathematical formulation（数学表达）
~~~~~~~~~~~~~~~~~~~~~~~~

假设两个标签分配（相同的 N 个对象），:math:`U` 和 :math:`V`。
它们的 entropy （熵）是一个 partition set （分区集合）的不确定性量，定义如下:

.. math:: H(U) = - \sum_{i=1}^{|U|}P(i)\log(P(i))

其中 :math:`P(i) = |U_i| / N` 是从 :math:`U` 中随机选取的对象到类 :math:`U_i` 的概率。同样对于 :math:`V`:

.. math:: H(V) = - \sum_{j=1}^{|V|}P'(j)\log(P'(j))

使用 :math:`P'(j) = |V_j| / N`. :math:`U` 和 :math:`V` 之间的 mutual information (MI) 由下式计算: 

.. math:: \text{MI}(U, V) = \sum_{i=1}^{|U|}\sum_{j=1}^{|V|}P(i, j)\log\left(\frac{P(i,j)}{P(i)P'(j)}\right)
 
其中 :math:`P(i, j) = |U_i \cap V_j| / N` 是随机选择的对象落入两个类的概率 :math:`U_i` 和 :math:`V_j` 。

也可以用设定的基数表达式表示: 

.. math:: \text{MI}(U, V) = \sum_{i=1}^{|U|} \sum_{j=1}^{|V|} \frac{|U_i \cap V_j|}{N}\log\left(\frac{N|U_i \cap V_j|}{|U_i||V_j|}\right)

normalized (归一化) mutual information 被定义为

.. math:: \text{NMI}(U, V) = \frac{\text{MI}(U, V)}{\sqrt{H(U)H(V)}}

mutual information 的价值以及 normalized variant （标准化变量）的值不会因 chance （机会）而被调整，随着不同标签（clusters（簇））的数量的增加，不管标签分配之间的 "mutual information" 的实际数量如何，都会趋向于增加。

mutual information 的期望值可以用 Vinh, Epps 和 Bailey,(2009) 的以下公式来计算。在这个方程式中,
:math:`a_i = |U_i|` (:math:`U_i` 中元素的数量) 和
:math:`b_j = |V_j|` (:math:`V_j` 中元素的数量).


.. math:: E[\text{MI}(U,V)]=\sum_{i=1}^|U| \sum_{j=1}^|V| \sum_{n_{ij}=(a_i+b_j-N)^+
   }^{\min(a_i, b_j)} \frac{n_{ij}}{N}\log \left( \frac{ N.n_{ij}}{a_i b_j}\right)
   \frac{a_i!b_j!(N-a_i)!(N-b_j)!}{N!n_{ij}!(a_i-n_{ij})!(b_j-n_{ij})!
   (N-a_i-b_j+n_{ij})!}

使用 expected value (期望值), 然后可以使用与 adjusted Rand index 相似的形式来计算调整后的 mutual information:

.. math:: \text{AMI} = \frac{\text{MI} - E[\text{MI}]}{\max(H(U), H(V)) - E[\text{MI}]}

.. topic:: 参考

 * Strehl, Alexander, and Joydeep Ghosh (2002). "Cluster ensembles – a
   knowledge reuse framework for combining multiple partitions". Journal of
   Machine Learning Research 3: 583–617.
   `doi:10.1162/153244303321897735 <http://strehl.com/download/strehl-jmlr02.pdf>`_.

 * Vinh, Epps, and Bailey, (2009). "Information theoretic measures
   for clusterings comparison". Proceedings of the 26th Annual International
   Conference on Machine Learning - ICML '09.
   `doi:10.1145/1553374.1553511 <https://dl.acm.org/citation.cfm?doid=1553374.1553511>`_.
   ISBN 9781605585161.

 * Vinh, Epps, and Bailey, (2010). Information Theoretic Measures for
   Clusterings Comparison: Variants, Properties, Normalization and
   Correction for Chance, JMLR
   http://jmlr.csail.mit.edu/papers/volume11/vinh10a/vinh10a.pdf

 * `Wikipedia entry for the (normalized) Mutual Information
   <https://en.wikipedia.org/wiki/Mutual_Information>`_

 * `Wikipedia entry for the Adjusted Mutual Information
   <https://en.wikipedia.org/wiki/Adjusted_Mutual_Information>`_

.. _homogeneity_completeness:

Homogeneity, completeness and V-measure（同质性，完整性和 V-measure）
---------------------------------------

鉴于样本的 ground truth class assignments （标定过的真实数据类分配）的知识，可以使用 conditional entropy （条件熵）分析来定义一些 intuitive metric（直观的度量）。

特别是 Rosenberg 和 Hirschberg (2007) 为任何 cluster （簇）分配定义了以下两个理想的目标:

- **homogeneity(同质性)**: 每个簇只包含一个类的成员

- **completeness(完整性)**: 给定类的所有成员都分配给同一个簇。

我们可以把这些概念作为分数 :func:`homogeneity_score` 和 :func:`completeness_score` 。两者均在 0.0 以下 和 1.0 以上（越高越好）::

  >>> from sklearn import metrics
  >>> labels_true = [0, 0, 0, 1, 1, 1]
  >>> labels_pred = [0, 0, 1, 1, 2, 2]

  >>> metrics.homogeneity_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  0.66...

  >>> metrics.completeness_score(labels_true, labels_pred) # doctest: +ELLIPSIS
  0.42...

称为 **V-measure** 的 harmonic mean 由以下函数计算 :func:`v_measure_score`::

  >>> metrics.v_measure_score(labels_true, labels_pred)    # doctest: +ELLIPSIS
  0.51...

V-measure 实际上等于上面讨论的 mutual information (NMI) 由 label entropies [B2011]_ （标准熵 [B2011]_） 的总和 normalized （归一化）。

Homogeneity（同质性）, completeness（完整性） and V-measure 可以立即计算 :func:`homogeneity_completeness_v_measure` 如下:: 

  >>> metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)
  ...                                                      # doctest: +ELLIPSIS
  (0.66..., 0.42..., 0.51...)

以下聚类分配稍微好一些，因为它是同构但不完整的::

  >>> labels_pred = [0, 0, 0, 1, 2, 2]
  >>> metrics.homogeneity_completeness_v_measure(labels_true, labels_pred)
  ...                                                      # doctest: +ELLIPSIS
  (1.0, 0.68..., 0.81...)

.. note::

  :func:`v_measure_score` 是 **symmetric（对称的）**: 它可以用于评估同一数据集上两个 independent assignments （独立赋值）的 **agreement（协议）**。

  这不是这样的 :func:`completeness_score` 和 :func:`homogeneity_score`: 两者的关系是被这样约束着::

    homogeneity_score(a, b) == completeness_score(b, a)


Advantages（优点）
~~~~~~~~~~

- **Bounded scores（有界分数）**: 0.0 是最坏的, 1.0 是一个完美的分数.

- Intuitive interpretation（直觉解释）: 具有不良 V-measure 的聚类可以在 **qualitatively analyzed in terms of homogeneity and completeness（在同质性和完整性方面进行定性分析）** 以更好地感知到作业完成的错误类型。

- **No assumption is made on the cluster structure（对簇的结构没有作出任何假设）**: 可以用于比较聚类算法，例如 k-means ，其假定 isotropic blob shapes 与可以找到具有 "folded" shapes 的聚类的 spectral clustering algorithms （频谱聚类算法）的结果。


Drawbacks（缺点）
~~~~~~~~~

- 以前引入的 metrics （度量标准）**not normalized with regards to random labeling（并不是随机标记的标准化的）**: 这意味着，根据 number of samples （样本数量），clusters （簇）和 ground truth classes （标定过的真实数据类），完全随机的标签并不总是产生 homogeneity （同质性），completeness（完整性）和 hence v-measure 的相同值。特别是 **random labeling won't yield zero scores especially when the number of clusters is large（随机标记不会产生零分，特别是当集群数量大时）**。

  当样本数量超过一千，簇的数量小于 10 时，可以安全地忽略此问题。**For smaller sample sizes or larger number of clusters it is safer to use an adjusted index such as the Adjusted Rand Index (ARI)（对于较小的样本数量或者较大数量的簇，使用 adjusted index 例如 Adjusted Rand Index (ARI)）**。

.. figure:: ../auto_examples/cluster/images/sphx_glr_plot_adjusted_for_chance_measures_001.png
   :target: ../auto_examples/cluster/plot_adjusted_for_chance_measures.html
   :align: center
   :scale: 100

- 这些 metrics （指标） **require the knowledge of the ground truth classes（需要标定过的真实数据类的知识）**，而在实践中几乎不可用，或需要人工标注来人工分配（如在受监督的学习环境中）。


.. topic:: 示例:

 * :ref:`sphx_glr_auto_examples_cluster_plot_adjusted_for_chance_measures.py`: 分析数据集大小对随机分配聚类度量值的影响。


Mathematical formulation（数学表达）
~~~~~~~~~~~~~~~~~~~~~~~~

Homogeneity（同质性） and completeness（完整性） 的得分由下面公式给出:

.. math:: h = 1 - \frac{H(C|K)}{H(C)}

.. math:: c = 1 - \frac{H(K|C)}{H(K)}

其中 :math:`H(C|K)` 是 **给定簇分配的类的 conditional entropy （条件熵）** ，由下式给出:

.. math:: H(C|K) = - \sum_{c=1}^{|C|} \sum_{k=1}^{|K|} \frac{n_{c,k}}{n}
          \cdot \log\left(\frac{n_{c,k}}{n_k}\right)

并且 :math:`H(C)` 是 **entropy of the classes（类的熵）**，并且由下式给出:

.. math:: H(C) = - \sum_{c=1}^{|C|} \frac{n_c}{n} \cdot \log\left(\frac{n_c}{n}\right)

:math:`n` 个样本总数， :math:`n_c` 和 :math:`n_k` 分别属于 :math:`c` 类和簇 :math:`k` 的样本数，最后 :math:`n_{c,k}` 分配给簇 :math:`k` 的类 :math:`c` 的样本数。

**conditional entropy of clusters given class（给定类的条件熵）** :math:`H(K|C)` 和 **entropy of clusters（类的熵）** :math:`H(K)` 以 symmetric manner （对称方式）定义。

Rosenberg 和 Hirschberg 进一步定义 **V-measure** 作为 **harmonic mean of homogeneity and completeness（同质性和完整性的 harmonic mean）**:

.. math:: v = 2 \cdot \frac{h \cdot c}{h + c}

.. topic:: 参考

 * `V-Measure: A conditional entropy-based external cluster evaluation
   measure <http://aclweb.org/anthology/D/D07/D07-1043.pdf>`_
   Andrew Rosenberg and Julia Hirschberg, 2007

 .. [B2011] `Identication and Characterization of Events in Social Media
   <http://www.cs.columbia.edu/~hila/hila-thesis-distributed.pdf>`_, Hila
   Becker, PhD Thesis.

.. _fowlkes_mallows_scores:

Fowlkes-Mallows scores（Fowlkes-Mallows 得分）
----------------------

当样本的已标定的真实数据的类别分配已知时，可以使用 Fowlkes-Mallows index （Fowlkes-Mallows 指数）(:func:`sklearn.metrics.fowlkes_mallows_score`) 。Fowlkes-Mallows 得分 FMI 被定义为 geometric mean of the pairwise precision （成对精度）和 recall （召回）的几何平均值:

.. math:: \text{FMI} = \frac{\text{TP}}{\sqrt{(\text{TP} + \text{FP}) (\text{TP} + \text{FN})}}

其中的 ``TP`` 是 **True Positive（正确的正）** 的数量（即，真实标签和预测标签中属于相同簇的点对数），``FP`` 是 **False Positive（错误的正）** （即，在真实标签中属于同一簇的点对数，而不在预测标签中），``FN`` 是 **False Negative（错误的负）** 的数量（即，预测标签中属于同一簇的点对数，而不在真实标签中）。

score （分数）范围为 0 到 1。较高的值表示两个簇之间的良好相似性。

  >>> from sklearn import metrics
  >>> labels_true = [0, 0, 0, 1, 1, 1]
  >>> labels_pred = [0, 0, 1, 1, 2, 2]

  >>> metrics.fowlkes_mallows_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  0.47140...

可以在 predicted labels （预测的标签）中 permute （排列） 0 和 1 ，重命名为 2 到 3 并得到相同的得分::

  >>> labels_pred = [1, 1, 0, 0, 3, 3]

  >>> metrics.fowlkes_mallows_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  0.47140...

完美的标签得分是 1.0::

  >>> labels_pred = labels_true[:]
  >>> metrics.fowlkes_mallows_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  1.0

坏的（例如 independent labelings （独立标签））的标签得分为 0:: 

  >>> labels_true = [0, 1, 2, 0, 3, 4, 5, 1]
  >>> labels_pred = [1, 1, 0, 0, 2, 2, 2, 2]
  >>> metrics.fowlkes_mallows_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  0.0

Advantages（优点）
~~~~~~~~~~

- **Random (uniform) label assignments have a FMI score close to 0.0（随机（统一）标签分配 FMI 得分接近于 0.0）** 对于 ``n_clusters`` 和 ``n_samples`` 的任何值（对于原始 Mutual Information 或例如 V-measure 而言）。

- **Bounded range（有界范围） [0, 1]**:  接近于 0 的值表示两个标签分配在很大程度上是独立的，而接近于 1 的值表示 significant agreement 。此外，正好为 0 的值表示 **purely** 独立标签分配，正好为 1 的 AMI 表示两个标签分配相等（有或者没有 permutation （排列））。

- **No assumption is made on the cluster structure（对簇的结构没有作出任何假设）**: 可以用于比较诸如 k-means 的聚类算法，其将假设 isotropic blob shapes 与能够找到具有 "folded" shapes 的簇的 spectral clustering algorithms （频谱聚类算法）的结果相结合。


Drawbacks（缺点）
~~~~~~~~~

- 与 inertia（惯性）相反，**FMI-based measures require the knowledge of the ground truth classes（基于 FMI 的测量方案需要了解已标注的真是数据的类）** ，而在实践中几乎不用，或需要人工标注者的人工分配（如在监督学习的学习环境中）。

.. topic:: 参考

  * E. B. Fowkles and C. L. Mallows, 1983. "A method for comparing two
    hierarchical clusterings". Journal of the American Statistical Association.
    http://wildfire.stat.ucla.edu/pdflibrary/fowlkes.pdf

  * `Wikipedia entry for the Fowlkes-Mallows Index
    <https://en.wikipedia.org/wiki/Fowlkes-Mallows_index>`_

.. _silhouette_coefficient:

Silhouette Coefficient
----------------------

如果标注过的真实数据的标签不知道，则必须使用模型本身进行 evaluation （评估）。Silhouette Coefficient (:func:`sklearn.metrics.silhouette_score`) 是一个这样的评估的例子，其中较高的 Silhouette Coefficient 得分与具有更好定义的聚类的模型相关。Silhouette Coefficient 是为每个样本定义的，由两个得分组成:

- **a**: 样本与同一类别中所有其他点之间的平均距离。

- **b**: 样本与 *next nearest cluster（下一个最近的簇）* 中的所有其他点之间的平均距离。

然后将单个样本的 Silhouette Coefficient *s* 给出为:

.. math:: s = \frac{b - a}{max(a, b)}

给定一组样本的 Silhouette Coefficient 作为每个样本的 Silhouette Coefficient 的平均值。


  >>> from sklearn import metrics
  >>> from sklearn.metrics import pairwise_distances
  >>> from sklearn import datasets
  >>> dataset = datasets.load_iris()
  >>> X = dataset.data
  >>> y = dataset.target

在正常使用情况下，将 Silhouette Coefficient 应用于聚类分析的结果。

  >>> import numpy as np
  >>> from sklearn.cluster import KMeans
  >>> kmeans_model = KMeans(n_clusters=3, random_state=1).fit(X)
  >>> labels = kmeans_model.labels_
  >>> metrics.silhouette_score(X, labels, metric='euclidean')
  ...                                                      # doctest: +ELLIPSIS
  0.55...

.. topic:: 参考

 * Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
   Interpretation and Validation of Cluster Analysis". Computational
   and Applied Mathematics 20: 53–65.
   `doi:10.1016/0377-0427(87)90125-7 <http://dx.doi.org/10.1016/0377-0427(87)90125-7>`_.


Advantages（优点）
~~~~~~~~~~

- 对于不正确的 clustering （聚类），分数为 -1 ， highly dense clustering （高密度聚类）为 +1 。零点附近的分数表示 overlapping clusters （重叠的聚类）。

- 当 clusters （簇）密集且分离较好时，分数更高，这与 cluster （簇）的标准概念有关。


Drawbacks（缺点）
~~~~~~~~~

- convex clusters（凸集）的 Silhouette Coefficient 通常比其他 cluster （簇）的概念更高，例如通过 DBSCAN 获得的基于密度的 cluster（簇）。

.. topic:: 示例:

 * :ref:`sphx_glr_auto_examples_cluster_plot_kmeans_silhouette_analysis.py` : 在这个例子中，silhouette 分析用于为 n_clusters 选择最佳值.

.. _calinski_harabaz_index:

Calinski-Harabaz Index（Calinski-Harabaz 指数）
----------------------

如果不知道真实数据的类别标签，则可以使用 Calinski-Harabaz index (:func:`sklearn.metrics.calinski_harabaz_score`) 来评估模型，其中较高的 Calinski-Harabaz 的得分与具有更好定义的聚类的模型相关。

对于 :math:`k` 簇，Calinski-Harabaz 得分 :math:`s` 是作为 between-clusters dispersion mean （簇间色散平均值）与 within-cluster dispersion（群内色散之间）的比值给出的:

.. math::
  s(k) = \frac{\mathrm{Tr}(B_k)}{\mathrm{Tr}(W_k)} \times \frac{N - k}{k - 1}

其中 :math:`B_K` 是 between group dispersion matrix （组间色散矩阵）， :math:`W_K` 是由以下定义的 within-cluster dispersion matrix （群内色散矩阵）: 

.. math:: W_k = \sum_{q=1}^k \sum_{x \in C_q} (x - c_q) (x - c_q)^T

.. math:: B_k = \sum_q n_q (c_q - c) (c_q - c)^T

:math:`N` 为数据中的点数，:math:`C_q` 为 cluster （簇） :math:`q` 中的点集， :math:`c_q` 为 cluster（簇） :math:`q` 的中心， :math:`c` 为 :math:`E` 的中心， :math:`n_q` 为 cluster（簇） :math:`q` 中的点数。 


  >>> from sklearn import metrics
  >>> from sklearn.metrics import pairwise_distances
  >>> from sklearn import datasets
  >>> dataset = datasets.load_iris()
  >>> X = dataset.data
  >>> y = dataset.target

在正常使用情况下，将 Calinski-Harabaz index （Calinski-Harabaz 指数）应用于 cluster analysis （聚类分析）的结果。

  >>> import numpy as np
  >>> from sklearn.cluster import KMeans
  >>> kmeans_model = KMeans(n_clusters=3, random_state=1).fit(X)
  >>> labels = kmeans_model.labels_
  >>> metrics.calinski_harabaz_score(X, labels)  # doctest: +ELLIPSIS
  560.39...


Advantages（优点）
~~~~~~~~~~

- 当 cluster （簇）密集且分离较好时，分数更高，这与 cluster（簇）的标准概念有关。

- 得分计算很快


Drawbacks（缺点）
~~~~~~~~~

- 凸集的 Calinski-Harabaz index（Calinski-Harabaz 指数）通常高于 cluster （簇） 的其他概念，例如通过 DBSCAN 获得的基于密度的 cluster（簇）。

.. topic:: 参考

 *  Caliński, T., & Harabasz, J. (1974). "A dendrite method for cluster
    analysis". Communications in Statistics-theory and Methods 3: 1-27.
    `doi:10.1080/03610926.2011.560741 <http://dx.doi.org/10.1080/03610926.2011.560741>`_.
