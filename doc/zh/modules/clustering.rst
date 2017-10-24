.. _clustering:

==========
聚类
==========

未标记的数据的 `Clustering（聚类） <https://en.wikipedia.org/wiki/Cluster_analysis>`__ 可以使用模块 :mod:`sklearn.cluster` 来实现。

每个 clustering algorithm （聚类算法）有两个变体: 一个是 class, 它实现了 ``fit`` 方法来学习 train data（训练数据）的 clusters（聚类），还有一个 function（函数），是给定 train data（训练数据），返回与不同 clusters（聚类）对应的整数标签 array（数组）。对于 class（类），training data（训练数据）上的标签可以在 ``labels_`` 属性中找到。

.. currentmodule:: sklearn.cluster

.. topic:: 输入数据

    需要注意的一点是，该模块中实现的算法可以采用不同种类的 matrix （矩阵）作为输入。所有这些都接受 shape ``[n_samples, n_features]`` 的标准数据矩阵。
    这些可以从以下的 :mod:`sklearn.feature_extraction` 模块的 classes （类）中获得。对于 :class:`AffinityPropagation`, :class:`SpectralClustering` 和 :class:`DBSCAN` 也可以输入 shape ``[n_samples, n_samples]`` 的相似矩阵。这些可以从 :mod:`sklearn.metrics.pairwise` 模块中的函数获得。

聚类方法概述
=================

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

K-means
========

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

小批量 K-Means
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
========================

:class:`AffinityPropagation` AP聚类是通过在样本对之间发送消息直到收敛来创建聚类。然后使用少量示例样本作为聚类中心来描述数据集，
聚类中心是数据集中最能代表一类数据的样本。在样本对之间发送的消息表示一个样本作为另一个样本的示例样本的
适合程度，适合程度值在根据通信的反馈不断更新。更新迭代直到收敛，完成聚类中心的选取，因此也给出了最终聚类。

.. figure:: ../auto_examples/cluster/images/sphx_glr_plot_affinity_propagation_001.png
   :target: ../auto_examples/cluster/plot_affinity_propagation.html
   :align: center
   :scale: 50


Affinity Propagation 算法比较有趣的是可以根据提供的数据决定聚类的数目。 因此有两个比较重要的参数
 *preference*, 决定使用多少个示例样本  *damping factor*（阻尼因子） 减少吸引信息和归属信息以防止
 更新减少吸引度和归属度信息时数据振荡。

AP聚类算法主要的缺点是算法的复杂度. AP聚类算法的时间复杂度是 :math:`O(N^2 T)`, 其中 :math:`N`
是样本的个数 ， :math:`T` 是收敛之前迭代的次数. 如果使用密集的相似性矩阵空间复杂度是
:math:`O(N^2)` 如果使用稀疏的相似性矩阵空间复杂度可以降低。 这使得AP聚类最适合中小型数据集。

.. topic:: 示例:

 * :ref:`sphx_glr_auto_examples_cluster_plot_affinity_propagation.py`: Affinity
   Propagation on a synthetic 2D datasets with 3 classes.

 * :ref:`sphx_glr_auto_examples_applications_plot_stock_market.py` Affinity Propagation on
   Financial time series to find groups of companies


**Algorithm description(算法描述):**
样本之间传递的信息有两种。 第一种是 responsibility(吸引信息) :math:`r(i, k)`, 样本 :math:`k` 适合作为样本 :math:`i` 的聚类中心的程度。

第二种是 availability(归属信息) :math:`a(i, k)` 样本 :math:`i` 选择样本 :math:`k` 作为聚类中心的适合程度,并且考虑其他所有样本选取 :math:`k` 做为聚类中心的合适程度。
通过这个方法，选取示例样本作为聚类中心如果 (1) 该样本与其许多样本相似，并且 (2) 被许多样本选取
为它们自己的示例样本。

样本 :math:`k` 对样本 :math:`i` 吸引度计算公式:

.. math::

    r(i, k) \leftarrow s(i, k) - max [ a(i, k') + s(i, k') \forall k' \neq k ]

其中 :math:`s(i, k)` 是样本 :math:`i` 和样本 :math:`k` 之间的相似度。
样本 :math:`k` 作为样本 :math:`i` 的示例样本的合适程度:

.. math::

    a(i, k) \leftarrow min [0, r(k, k) + \sum_{i'~s.t.~i' \notin \{i, k\}}{r(i', k)}]

算法开始时 :math:`r` 和 :math:`a` 都被置 0,然后开始迭代计算直到收敛。
为了防止更新数据时出现数据振荡，在迭代过程中引入阻尼因子 :math:`\lambda` :

.. math:: r_{t+1}(i, k) = \lambda\cdot r_{t}(i, k) + (1-\lambda)\cdot r_{t+1}(i, k)
.. math:: a_{t+1}(i, k) = \lambda\cdot a_{t}(i, k) + (1-\lambda)\cdot a_{t+1}(i, k)

其中 :math:`t` 迭代的次数。

.. _mean_shift:

Mean Shift
============
:class:`MeanShift` 算法旨在于发现一个样本密度平滑的 *blobs* 。
均值漂移算法是基于质心的算法，通过更新质心的候选位置为所选定区域的偏移均值。
然后，这些候选者在后处理阶段被过滤以消除近似重复，从而形成最终质心集合。

给定第 :math:`t` 次迭代中的候选质心 :math:`x_i` , 候选质心的位置将被安装如下公式更新:

.. math::

    x_i^{t+1} = x_i^t + m(x_i^t)

其中 :math:`N(x_i)` 是围绕 :math:`x_i` 周围一个给定距离范围内的样本空间 
and :math:`m` 是  *mean shift* vector（均值偏移向量） 是所有质心中指向
点密度增加最多的区域的偏移向量。使用以下等式计算，有效地将质心更新为其邻域内样本的平均值:

.. math::

    m(x_i) = \frac{\sum_{x_j \in N(x_i)}K(x_j - x_i)x_j}{\sum_{x_j \in N(x_i)}K(x_j - x_i)}

算法自动设定聚类的数目，取代依赖参数 ``bandwidth``（带宽）,带宽是决定搜索区域的size的参数。
这个参数可以手动设置，但是如果没有设置，可以使用提供的评估函数 ``estimate_bandwidth`` 进行评估。

该算法不是高度可扩展的，因为在执行算法期间需要执行多个最近邻搜索。 该算法保证收敛，但是当
质心的变化较小时，算法将停止迭代。

通过找到给定样本的最近质心来给新样本打上标签。


.. figure:: ../auto_examples/cluster/images/sphx_glr_plot_mean_shift_001.png
   :target: ../auto_examples/cluster/plot_mean_shift.html
   :align: center
   :scale: 50


.. topic:: 示例:

 * :ref:`sphx_glr_auto_examples_cluster_plot_mean_shift.py`: Mean Shift clustering
   on a synthetic 2D datasets with 3 classes.

.. topic:: 参考:

 * `"Mean shift: A robust approach toward feature space analysis."
   <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.76.8968&rep=rep1&type=pdf>`_
   D. Comaniciu and P. Meer, *IEEE Transactions on Pattern Analysis and Machine Intelligence* (2002)


.. _spectral_clustering:

Spectral clustering
======================

:class:`SpectralClustering` 是在样本之间进行亲和力矩阵的低维度嵌入，其实是低维空间中的 KMeans。
如果亲和度矩阵稀疏，则这是非常有效的并且 `pyamg <http://pyamg.org/>`_ module 以及安装好。
SpectralClustering 需要指定聚类数。这个算法适用于聚类数少时，在聚类数多是不建议使用。

对于两个聚类，它解决了相似图上的 `normalised cuts <http://people.eecs.berkeley.edu/~malik/papers/SM-ncut.pdf>`_ 问题:
将图形切割成两个，使得切割的边缘的重量比每个簇内的边缘的权重小。在图像处理时，这个标准是特别有趣的:
图像的顶点是像素，相似图的边缘是图像的渐变函数。


.. |noisy_img| image:: ../auto_examples/cluster/images/sphx_glr_plot_segmentation_toy_001.png
    :target: ../auto_examples/cluster/plot_segmentation_toy.html
    :scale: 50

.. |segmented_img| image:: ../auto_examples/cluster/images/sphx_glr_plot_segmentation_toy_002.png
    :target: ../auto_examples/cluster/plot_segmentation_toy.html
    :scale: 50

.. centered:: |noisy_img| |segmented_img|

.. warning:: Transforming distance to well-behaved similarities
 
    请注意，如果你的相似矩阵的值分布不均匀，例如:存在负值或者距离矩阵并不表示相似性
    spectral problem 将会变得奇异，并且不能解决。
    在这种情况下，建议对矩阵的 entries 进行转换。比如在符号距离有符号的情况下通常使用 heat kernel::

        similarity = np.exp(-beta * distance / distance.std())

    请看这样一个应用程序的例子。

.. topic:: 示例:

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

不同的标记分配策略
------------------------

可以使用不同的分配策略, 对应于 ``assign_labels`` 参数 :class:`SpectralClustering`。
``"kmeans"`` 可以匹配更精细的数据细节，但是可能更加不稳定。 特别是，除非你设置
``random_state`` 否则可能无法复现运行的结果 ，因为它取决于随机初始化。另一方，
使用 ``"discretize"`` 策略是 100% 可以复现的，但是它往往会产生相当均匀的几何形状的边缘。


=====================================  =====================================
 ``assign_labels="kmeans"``              ``assign_labels="discretize"``
=====================================  =====================================
|face_kmeans|                          |face_discretize|
=====================================  =====================================


.. topic:: 参考:

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

层次聚类
==================

Hierarchical clustering 是一个常用的聚类算法，它通过不断的合并或者分割来构建聚类。
聚类的层次被表示成树（或者 dendrogram（树形图））。树根是拥有所有样本的唯一聚类，叶子是仅有一个样本的聚类。
请参照 `Wikipedia page <https://en.wikipedia.org/wiki/Hierarchical_clustering>`_ 查看更多细节。

The :class:`AgglomerativeClustering` 使用自下而上的方法进行层次聚类:开始是每一个对象是一个聚类，
并且聚类别相继合并在一起。 linkage criteria 确定用于合并的策略的度量:

- **Ward** 最小化所有聚类内的平方差总和。这是一种 variance-minimizing （方差最小化）的优化方向，
  这是与k-means 的目标函数相似的优化方法，但是用 agglomerative hierarchical（聚类分层）的方法处理。

- **Maximum** 或 **complete linkage** 最小化聚类对两个样本之间的最大距离。

- **Average linkage** 最小化聚类两个聚类中样本距离的平均值。

:class:`AgglomerativeClustering` 在于连接矩阵联合使用时，也可以扩大到大量的样本，但是
在样本之间没有添加连接约束时，计算代价很大:每一个步骤都要考虑所有可能的合并。

.. topic:: :class:`FeatureAgglomeration`

   The :class:`FeatureAgglomeration` 使用 agglomerative clustering 将看上去相似的
   特征组合在一起，从而减少特征的数量。这是一个降维工具, 请参照 :ref:`data_reduction`。

Different linkage type: Ward, complete and average linkage
-----------------------------------------------------------

:class:`AgglomerativeClustering` 支持 Ward, average, and complete
linkage 策略.

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_digits_linkage_001.png
    :target: ../auto_examples/cluster/plot_digits_linkage.html
    :scale: 43

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_digits_linkage_002.png
    :target: ../auto_examples/cluster/plot_digits_linkage.html
    :scale: 43

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_digits_linkage_003.png
    :target: ../auto_examples/cluster/plot_digits_linkage.html
    :scale: 43


Agglomerative cluster 存在 "rich get richer" 现象导致聚类大小不均匀。这方面 complete linkage
是最坏的策略，Ward 给出了最规则的大小。然而，在 Ward 中 affinity (or distance used in clustering) 
不能被改变，对于 non Euclidean metrics 来说 average linkage 是一个好的选择。

.. topic:: 示例:

 * :ref:`sphx_glr_auto_examples_cluster_plot_digits_linkage.py`: exploration of the
   different linkage strategies in a real dataset.


添加连接约束
--------------------

:class:`AgglomerativeClustering` 中一个有趣的特点是可以使用 connectivity matrix（连接矩阵）
将连接约束添加到算法中（只有相邻的聚类可以合并到一起），连接矩阵为每一个样本给定了相邻的样本。
例如，在 swiss-roll 的例子中，连接约束禁止在不相邻的 swiss roll 上合并，从而防止形成在 roll 上
重复折叠的聚类。

.. |unstructured| image:: ../auto_examples/cluster/images/sphx_glr_plot_ward_structured_vs_unstructured_001.png
        :target: ../auto_examples/cluster/plot_ward_structured_vs_unstructured.html
        :scale: 49

.. |structured| image:: ../auto_examples/cluster/images/sphx_glr_plot_ward_structured_vs_unstructured_002.png
        :target: ../auto_examples/cluster/plot_ward_structured_vs_unstructured.html
        :scale: 49

.. centered:: |unstructured| |structured|

这些约束对于强加一定的局部结构是很有用的，但是这也使得算法更快，特别是当样本数量巨大时。

连通性的限制是通过连接矩阵来实现的:一个 scipy sparse matrix（稀疏矩阵），仅在一行和
一列的交集处具有应该连接在一起的数据集的索引。这个矩阵可以通过 a-priori information （先验信息）
构建:例如，你可能通过仅仅将从一个连接指向另一个的链接合并页面来聚类页面。也可以从数据中学习到,
 例如使用 :func:`sklearn.neighbors.kneighbors_graph` 限制与最临近的合并 :ref:`this example
<sphx_glr_auto_examples_cluster_plot_agglomerative_clustering.py>`, 或者使用
 :func:`sklearn.feature_extraction.image.grid_to_graph` 仅合并图像上相邻的像素点，
 例如 :ref:`raccoon face <sphx_glr_auto_examples_cluster_plot_face_ward_segmentation.py>` 。

.. topic:: 示例:

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

    Connectivity constraints 和 complete or average linkage 可以增强 agglomerative clustering 中的
    'rich getting richer' 现象。特别是，如果建立的是 :func:`sklearn.neighbors.kneighbors_graph`。
    在少量聚类的限制中, 更倾向于给出一些 macroscopically occupied clusters 并且几乎是空的 (讨论内容请查看
    :ref:`sphx_glr_auto_examples_cluster_plot_agglomerative_clustering.py`)。

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

Average and complete linkage 可以使用各种距离 (or affinities), 特别是 Euclidean distance (*l2*), 
Manhattan distance（曼哈顿距离）(or Cityblock（城市区块距离）, or *l1*), cosine distance(余弦距离),
 或者任何预先计算的 affinity matrix（亲和度矩阵）.

* *l1* distance 有利于稀疏特征或者稀疏噪声: 例如很多特征都是0，就想在文本挖掘中使用 rare words 一样。

* *cosine* distance 非常有趣因为它对全局放缩是一样的。 

选择度量标准的方针是使得不同类样本之间距离最大化，并且最小化同类样本之间的距离。

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_agglomerative_clustering_metrics_005.png
    :target: ../auto_examples/cluster/plot_agglomerative_clustering_metrics.html
    :scale: 32

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_agglomerative_clustering_metrics_006.png
    :target: ../auto_examples/cluster/plot_agglomerative_clustering_metrics.html
    :scale: 32

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_agglomerative_clustering_metrics_007.png
    :target: ../auto_examples/cluster/plot_agglomerative_clustering_metrics.html
    :scale: 32

.. topic:: 示例:

 * :ref:`sphx_glr_auto_examples_cluster_plot_agglomerative_clustering_metrics.py`


.. _dbscan:

DBSCAN
======

The :class:`DBSCAN` 算法将聚类视为被低密度区域分隔的高密度区域。由于这个相当普遍的观点，
DBSCAN发现的聚类可以是任何形状的，与假设聚类是 convex shaped 的 K-means 相反。 
DBSCAN 的核心概念是 *core samples*, 是指位于高密度区域的样本。
因此一个聚类是一组核心样本，每个核心样本彼此靠近（通过一定距离度量测量）
和一组接近核心样本的非核心样本（但本身不是核心样本）。算法中的两个参数, ``min_samples`` 
和 ``eps``,正式的定义了我们所说的 *dense*（稠密）。较高的 ``min_samples`` 或者
 较低的 ``eps``表示形成聚类所需的较高密度。

更正式的,我们定义核心样本是指数据集中的一个样本，存在 ``min_samples`` 个其他样本在 ``eps`` 
距离范围内，这些样本被定为为核心样本的邻居 *neighbors* 。这告诉我们核心样本在向量空间稠密的区域。
一个聚类是一个核心样本的集合，可以通过递归来构建，选取一个核心样本，查找它所有的 neighbors （邻居样本）
中的核心样本，然后查找 *their* （新获取的核心样本）的 neighbors （邻居样本）中的核心样本，递归这个过程。
聚类中还具有一组非核心样本，它们是集群中核心样本的邻居的样本，但本身并不是核心样本。
显然，这些样本位于聚类的边缘。

根据定义，任何核心样本都是聚类的一部分，任何不是核心样本并且和任意一个核心样本距离都大于
 ``eps`` 的样本将被视为异常值。

在下图中，颜色表示聚类成员属性，大圆圈表示算法发现的核心样本。 较小的圈子是仍然是群集的
一部分的非核心样本。 此外，异常值由下面的黑点表示。

.. |dbscan_results| image:: ../auto_examples/cluster/images/sphx_glr_plot_dbscan_001.png
        :target: ../auto_examples/cluster/plot_dbscan.html
        :scale: 50

.. centered:: |dbscan_results|

.. topic:: 示例:

    * :ref:`sphx_glr_auto_examples_cluster_plot_dbscan.py`

.. topic:: 实现

    DBSCAN 算法是具有确定性的，当以相同的顺序给出相同的数据时总是形成相同的聚类。
    然而，当以不同的顺序提供数据时聚类的结果可能不相同。首先，即使核心样本总是被
    分配给相同的聚类，这些集群的标签将取决于数据中遇到这些样本的顺序。第二个更重
    要的是，非核心样本的聚类可能因数据顺序而有所不同。
    当一个非核心样本距离两个核心样本的距离都小于 ``eps`` 时，就会发生这种情况。 
    通过三角不等式可知，这两个核心样本距离一定大于 ``eps`` 或者处于同一个聚类中。
    非核心样本将被非配到首先查找到改样本的类别，因此结果将取决于数据的顺序。

    当前版本使用 ball trees 和 kd-trees 来确定领域，这样避免了计算全部的距离矩阵
    （0.14 之前的 scikit-learn 版本计算全部的距离矩阵）。保留使用 custom metrics 
    （自定义指标）的可能性。细节请参照 :class:`NearestNeighbors`。

.. topic:: 大量样本的内存消耗

    默认的实现方式并没有充分利用内存，因为在不使用 kd-trees 或者 ball-trees 的情况下构建一个
    完整的相似度矩阵（e.g. 使用稀疏矩阵）。这个矩阵将消耗 n^2 个浮点数。
    解决这个问题的几种机制:

    - A sparse radius neighborhood graph （稀疏半径邻域图）(其中缺少条目被假定为距离超出eps)
      可以以高效的方式预先编译，并且可以使用 ``metric='precomputed'`` 来运行 dbscan。

    - 数据可以压缩，当数据中存在准确的重复时，可以删除这些重复的数据，或者使用BIRCH。
      任何。然后仅需要使用相对少量的样本来表示大量的点。当训练DBSCAN时，可以提供一个
       ``sample_weight`` 参数。

.. topic:: 引用:

 * "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases
   with Noise"
   Ester, M., H. P. Kriegel, J. Sander, and X. Xu,
   In Proceedings of the 2nd International Conference on Knowledge Discovery
   and Data Mining, Portland, OR, AAAI Press, pp. 226–231. 1996

.. _birch:

Birch
=====

The :class:`Birch` 为提供的数据构建一颗 Characteristic Feature Tree (CFT，聚类特征树)。
数据实质上是被有损压缩成一组 Characteristic Feature nodes (CF Nodes，聚类特征节点)。
CF Nodes 中有一部分子聚类被称为 Characteristic Feature subclusters (CF Subclusters)，
并且这些位于非终端位置的CF Subclusters 可以拥有 CF Nodes 作为孩子节点。


CF Subclusters 保存用于聚类的必要信息，防止将整个输入数据保存在内存中。
这些信息包括:

- Number of samples in a subcluster（子聚类中样本数）.
- Linear Sum - A n-dimensional vector holding the sum of all samples（保存所有样本和的n维向量）
- Squared Sum - Sum of the squared L2 norm of all samples（所有样本的L2 norm的平方和）.
- Centroids - To avoid recalculation linear sum / n_samples（为了防止重复计算 linear sum / n_samples）.
- Squared norm of the centroids（质心的 Squared norm ）.

Birch 算法有两个参数，即 threshold （阈值）和 branching factor 分支因子。Branching factor （分支因子）
限制了一个节点中的子集群的数量 ，threshold （簇半径阈值）限制了新加入的样本和存在与现有子集群中样本的最大距离。

该算法可以视为将一个实例或者数据简化的方法，因为它将输入的数据简化到可以直接从CFT的叶子结点中获取的一组子聚类。
这种简化的数据可以通过将其馈送到global clusterer（全局聚类）来进一步处理。Global clusterer（全局聚类）可以
通过 ``n_clusters``来设置。

如果 ``n_clusters`` 被设置为 None，将直接读取叶子结点中的子聚类，否则，global clustering（全局聚类）
将逐步标记他的 subclusters 到 global clusters (labels) 中，样本将被映射到 距离最近的子聚类的global label中。 


**算法描述:**

- 一个新的样本作为一个CF Node 被插入到 CF Tree 的根节点。然后将其合并到根节点的子聚类中去，使得合并后子聚类
  拥有最小的半径，子聚类的选取受 threshold 和 branching factor 的约束。如果子聚类也拥有孩子节点，则重复执
  行这个步骤直到到达叶子结点。在叶子结点中找到最近的子聚类以后，递归的更新这个子聚类及其父聚类的属性。

- 如果合并了新样本和最近的子聚类获得的子聚类半径大约square of the threshold（阈值的平方），
  并且子聚类的数量大于branching factor（分支因子），则将为这个样本分配一个临时空间。
  最远的两个子聚类被选取，剩下的子聚类按照之间的距离分为两组作为被选取的两个子聚类的孩子节点。
  
- If this split node has a parent subcluster and there is room
  for a new subcluster, then the parent is split into two. If there is no room,
  then this node is again split into two and the process is continued
  recursively, till it reaches the root.
  如果拆分的节点有一个 parent subcluster ，并且有一个容纳一个新的子聚类的空间，那么父聚类拆分成两个。
  如果没有空间容纳一个新的聚类，那么这个节点将被再次拆分，依次向上检查父节点是否需要分裂，
  如果需要按叶子节点方式相同分裂。

**Birch or MiniBatchKMeans?**

 - Birch 在高维数据上表现不好。一条经验法则，如果 ``n_features`` 大于20，通常使用 MiniBatchKMeans 更好。

 - 如果需要减少数据实例的数量，或者如果需要大量的子聚类作为预处理步骤或者其他， Birch 比 MiniBatchKMeans 更有用。


**How to use partial_fit?**

为了避免对 global clustering 的计算，每次调用建议使用  ``partial_fit`` 。


 1. 初始化 ``n_clusters=None`` 。
 2. 通过多次调用 partial_fit 训练所以的数据。
 3. 设置 ``n_clusters`` 为所需值，通过使用 ``brc.set_params(n_clusters=n_clusters)`` 。
 4. 最后不需要参数调用 ``partial_fit`` , 例如 ``brc.partial_fit()`` 执行全局聚类。

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_birch_vs_minibatchkmeans_001.png
    :target: ../auto_examples/cluster/plot_birch_vs_minibatchkmeans.html

.. topic:: 参考:

 * Tian Zhang, Raghu Ramakrishnan, Maron Livny
   BIRCH: An efficient data clustering method for large databases.
   http://www.cs.sfu.ca/CourseCentral/459/han/papers/zhang96.pdf

 * Roberto Perdisci
   JBirch - Java implementation of BIRCH clustering algorithm
   https://code.google.com/archive/p/jbirch


.. _clustering_evaluation:

聚类性能评估
=====================

评估聚类算法的性能不像 counting the number of errors （计数错误数量）或监督分类算法的精度和调用那样微不足道。
特别地，任何 evaluation metric （评估度量）不应该考虑到 cluster labels （簇标签）的绝对值，而是如果这个聚类定义类似于 some ground truth set of classes or satisfying some assumption （某些基本真值集合的数据的分离或者满足一些假设），使得属于同一个类的成员更类似于根据某些 similarity metric （相似性度量）的不同类的成员。

.. currentmodule:: sklearn.metrics

.. _adjusted_rand_score:

调整后的 Rand index
-----------------------

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

另外， :func:`adjusted_rand_score` 是 **symmetric（对称的）** : 交换参数不会改变 score （得分）。它可以作为 **consensus measure（共识度量）**::

  >>> metrics.adjusted_rand_score(labels_pred, labels_true)  # doctest: +ELLIPSIS
  0.24...

完美的标签得分为 1.0 ::

  >>> labels_pred = labels_true[:]
  >>> metrics.adjusted_rand_score(labels_true, labels_pred)
  1.0

坏 (e.g. independent labelings（独立标签）) 有负数 或 接近于 0.0 分::

  >>> labels_true = [0, 1, 2, 0, 3, 4, 5, 1]
  >>> labels_pred = [1, 1, 0, 0, 2, 2, 2, 2]
  >>> metrics.adjusted_rand_score(labels_true, labels_pred)  # doctest: +ELLIPSIS
  -0.12...


优点
~~~~~~~~~~

- **Random (uniform) label assignments have a ARI score close to 0.0（随机（统一）标签分配的 ARI 评分接近于 0.0）**
  对于 ``n_clusters`` 和 ``n_samples`` 的任何值（这不是原始的 Rand index 或者 V-measure 的情况）。

- **Bounded range（有界范围） [-1, 1]**: negative values （负值）是坏的 (独立性标签), 类似的聚类有一个 positive ARI （正的 ARI）， 1.0 是完美的匹配得分。

- **No assumption is made on the cluster structure（对簇的结构没有作出任何假设）**: 可以用于比较聚类算法，例如 k-means，其假定 isotropic blob shapes 与可以找到具有 "folded" shapes 的聚类的 spectral clustering algorithms（频谱聚类算法）的结果。


缺点
~~~~~~~~~

- 与 inertia 相反，**ARI requires knowledge of the ground truth classes（ARI需要了解 ground truth classes）** ，而在实践中几乎不可用，或者需要人工标注者手动分配（如在监督学习环境中）。

  然而，ARI 还可以在 purely unsupervised setting （纯粹无监督的设置中）作为可用于 聚类模型选择（TODO）的共识索引的构建块。


.. topic:: 示例:

 * :ref:`sphx_glr_auto_examples_cluster_plot_adjusted_for_chance_measures.py`: 分析数据集大小对随机分配聚类度量值的影响。


数学表达
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

基于 Mutual Information 的分数
------------------------------------

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


优点
~~~~~~~~~~

- **Random (uniform) label assignments have a AMI score close to 0.0（随机（统一）标签分配的AMI评分接近0.0）**
  对于 ``n_clusters`` 和 ``n_samples`` 的任何值（这不是原始 Mutual Information 或者 V-measure 的情况）。

- **Bounded range（有界范围） [0, 1]**:  接近 0 的值表示两个主要独立的标签分配，而接近 1 的值表示重要的一致性。此外，正好 0 的值表示 **purely（纯粹）** 独立标签分配，正好为 1 的 AMI 表示两个标签分配相等（有或者没有 permutation）。

- **No assumption is made on the cluster structure（对簇的结构没有作出任何假设）**: 可以用于比较聚类算法，例如 k-means，其假定 isotropic blob shapes 与可以找到具有 "folded" shapes 的聚类的 spectral clustering algorithms （频谱聚类算法）的结果。


缺点
~~~~~~~~~

- 与 inertia 相反，**MI-based measures require the knowledge of the ground truth classes（MI-based measures 需要了解 ground truth classes）** ，而在实践中几乎不可用，或者需要人工标注者手动分配（如在监督学习环境中）。

  然而，基于 MI-based measures （基于 MI 的测量方式）也可用于纯无人监控的设置，作为可用于聚类模型选择的 Consensus Index （共识索引）的构建块。

- NMI 和 MI 没有调整机会。


.. topic:: 示例:

 * :ref:`sphx_glr_auto_examples_cluster_plot_adjusted_for_chance_measures.py`: 分析数据集大小对随机分配聚类度量值的影响。 此示例还包括 Adjusted Rand Index。


数学表达
~~~~~~~~~~~~~~~

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

同质性，完整性和 V-measure
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


优点
~~~~~~~~~~

- **Bounded scores（有界分数）**: 0.0 是最坏的, 1.0 是一个完美的分数.

- Intuitive interpretation（直觉解释）: 具有不良 V-measure 的聚类可以在 **qualitatively analyzed in terms of homogeneity and completeness（在同质性和完整性方面进行定性分析）** 以更好地感知到作业完成的错误类型。

- **No assumption is made on the cluster structure（对簇的结构没有作出任何假设）**: 可以用于比较聚类算法，例如 k-means ，其假定 isotropic blob shapes 与可以找到具有 "folded" shapes 的聚类的 spectral clustering algorithms （频谱聚类算法）的结果。


缺点
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


数学表达
~~~~~~~~~~~~~~~~

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

Fowlkes-Mallows 得分
-------------------------

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

优点
~~~~~~~~~~

- **Random (uniform) label assignments have a FMI score close to 0.0（随机（统一）标签分配 FMI 得分接近于 0.0）** 对于 ``n_clusters`` 和 ``n_samples`` 的任何值（对于原始 Mutual Information 或例如 V-measure 而言）。

- **Bounded range（有界范围） [0, 1]**:  接近于 0 的值表示两个标签分配在很大程度上是独立的，而接近于 1 的值表示 significant agreement 。此外，正好为 0 的值表示 **purely** 独立标签分配，正好为 1 的 AMI 表示两个标签分配相等（有或者没有 permutation （排列））。

- **No assumption is made on the cluster structure（对簇的结构没有作出任何假设）**: 可以用于比较诸如 k-means 的聚类算法，其将假设 isotropic blob shapes 与能够找到具有 "folded" shapes 的簇的 spectral clustering algorithms （频谱聚类算法）的结果相结合。


缺点
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


优点
~~~~~~~~~~

- 对于不正确的 clustering （聚类），分数为 -1 ， highly dense clustering （高密度聚类）为 +1 。零点附近的分数表示 overlapping clusters （重叠的聚类）。

- 当 clusters （簇）密集且分离较好时，分数更高，这与 cluster （簇）的标准概念有关。


缺点
~~~~~~~~~

- convex clusters（凸集）的 Silhouette Coefficient 通常比其他 cluster （簇）的概念更高，例如通过 DBSCAN 获得的基于密度的 cluster（簇）。

.. topic:: 示例:

 * :ref:`sphx_glr_auto_examples_cluster_plot_kmeans_silhouette_analysis.py` : 在这个例子中，silhouette 分析用于为 n_clusters 选择最佳值.

.. _calinski_harabaz_index:

Calinski-Harabaz 指数
--------------------------

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


优点
~~~~~~~~~~

- 当 cluster （簇）密集且分离较好时，分数更高，这与 cluster（簇）的标准概念有关。

- 得分计算很快


缺点
~~~~~~~~~

- 凸集的 Calinski-Harabaz index（Calinski-Harabaz 指数）通常高于 cluster （簇） 的其他概念，例如通过 DBSCAN 获得的基于密度的 cluster（簇）。

.. topic:: 参考

 *  Caliński, T., & Harabasz, J. (1974). "A dendrite method for cluster
    analysis". Communications in Statistics-theory and Methods 3: 1-27.
    `doi:10.1080/03610926.2011.560741 <http://dx.doi.org/10.1080/03610926.2011.560741>`_.
