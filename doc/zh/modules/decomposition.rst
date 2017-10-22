.. _decompositions:


=================================================================
分解成分中的信号（矩阵分解问题）
=================================================================

.. currentmodule:: sklearn.decomposition


.. _PCA:


主成分分析（PCA）
==================================

准确的PCA和概率解释（Exact PCA and probabilistic interpretation）
------------------------------------------


PCA 通过引入最大数量的变量来对一组连续正交分量中的多变量数据集进行降维。
在 scikit-learn 中，:class:`PCA` 被实现为一个变换对象， 通过 ``fit`` 方法可以降维成 `n` 个成分，
并且可以将新的数据投射(project, 亦可理解为 分解)到这些成分中。

可选参数 ``whiten=True`` 使得可以将数据投影到单个空间上，同时将每个成分缩放到单位方差。
如果下游模型对信号的各向同性作出强烈的假设，这通常是有用的。
例如，使用RBF内核的 SVM(Support Vector Machines) 算法和 K-Means 聚类算法。

以下是iris数据集的一个示例， 该数据集包含4个特征， 通过PCA降维后投影到二维空间上：

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_pca_vs_lda_001.png
    :target: ../auto_examples/decomposition/plot_pca_vs_lda.html
    :align: center
    :scale: 75%


:class:`PCA` 对象还提供了 PCA 的概率解释， 其可以基于降维后的变量数量来给出数据的可能性。
可以通过在 cross-validation 中使用 `score` 方法来实现：

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_pca_vs_fa_model_selection_001.png
    :target: ../auto_examples/decomposition/plot_pca_vs_fa_model_selection.html
    :align: center
    :scale: 75%


.. topic:: 例子:

    * :ref:`sphx_glr_auto_examples_decomposition_plot_pca_vs_lda.py`
    * :ref:`sphx_glr_auto_examples_decomposition_plot_pca_vs_fa_model_selection.py`


.. _IncrementalPCA:

增量PCA (Incremental PCA)
---------------

:class:`PCA` 对象非常有用, 但对大型数据集有一定的限制。
最大的限制是 :class:`PCA` 仅支持批处理，这意味着所有要处理的数据必须适合主内存。
:class:`IncrementalPCA` 对象使用不同的处理形式，并且允许部分计算，
几乎完全匹配 :class:`PCA` 以小型批处理方式处理数据的结果。
:class:`IncrementalPCA` 可以通过以下方式实现核心主成分分析：

 * 使用 ``partial_fit`` 方法从本地硬盘或网络数据库中以此获取数据块。

 * 通过 ``numpy.memmap`` 在一个 memory mapped file 上使用 fit 方法。

为了逐步更新 ``explained_variance_ratio_``，  :class:`IncrementalPCA` 仅存储成分和噪声变量的估算。
这就是为什么内存使用取决于每个批次的样本数，而不是数据集中要处理的样本数。

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_incremental_pca_001.png
    :target: ../auto_examples/decomposition/plot_incremental_pca.html
    :align: center
    :scale: 75%

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_incremental_pca_002.png
    :target: ../auto_examples/decomposition/plot_incremental_pca.html
    :align: center
    :scale: 75%


.. topic:: Examples:

    * :ref:`sphx_glr_auto_examples_decomposition_plot_incremental_pca.py`


.. _RandomizedPCA:


PCA 使用随机SVD
------------------------

通过丢弃具有较低奇异值的奇异向量成分，将数据降维到低维空间并保留大部分变量属性是非常有意义的。

例如，如果我们使用64x64像素的灰度级图像进行面部识别，数据的维数为4096，
并且在这样大的数据上训练含RBF内核的支持向量机是很慢的。
此外，我们知道数据的固有维度远低于4096，因为人脸的所有照片都看起来有点相似。
样本依赖于多种低维度（例如约200种）。
PCA算法可以用于线性变换数据，同时降低维数并同时保留大部分变量。

在这种情况下，具有可选参数 ``svd_solver='randomized'`` 的 :class:`PCA` 是非常有用的。
因为我们将要丢弃大部分奇异值，所以将计算限制为单个向量的近似估计更有效，我们将保持实际执行变换。

例如：以下显示了来自 Olivetti 数据集的 16 个样本肖像（以 0.0 为中心）。
右侧是重画为肖像的前 16 个奇异向量。因为我们只需要使数据集的前 16 个奇异向量
:math:`n_{samples} = 400`
和 :math:`n_{features} = 64 \times 64 = 4096`, 计算时间小于 1 秒。

.. |orig_img| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_001.png
   :target: ../auto_examples/decomposition/plot_faces_decomposition.html
   :scale: 60%

.. |pca_img| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_002.png
   :target: ../auto_examples/decomposition/plot_faces_decomposition.html
   :scale: 60%

.. centered:: |orig_img| |pca_img|

注意：使用可选参数 ``svd_solver='randomized'`` ，
在 :class:`PCA` 中我们必须输入低维度参数 ``n_components`` 。

如果我们注意到：:math:`n_{\max} = \max(n_{\mathrm{samples}}, n_{\mathrm{features}})` 且
:math:`n_{\min} = \min(n_{\mathrm{samples}}, n_{\mathrm{features}})`,
随机 :class:`PCA` 的时间复杂度是：:math:`O(n_{\max}^2 \cdot n_{\mathrm{components}})` ，
而不是 :math:`O(n_{\max}^2 \cdot n_{\min})` 。

在准确情形下，随机 :class:`PCA` 的内存占用量正比于 :math:`2 \cdot n_{\max} \cdot n_{\mathrm{components}}` ，
而不是 :math:`n_{\max}\cdot n_{\min}`

注意：选择参数 ``svd_solver='randomized'`` 的 :class:`PCA`，
在执行 ``inverse_transform`` 时， 并不是 ``transform`` 的确切的逆变换操作
（即使 参数设置为默认的 ``whiten=False``）

.. topic:: 例子:

    * :ref:`sphx_glr_auto_examples_applications_plot_face_recognition.py`
    * :ref:`sphx_glr_auto_examples_decomposition_plot_faces_decomposition.py`

.. topic:: 参考文献:

    * `"Finding structure with randomness: Stochastic algorithms for
      constructing approximate matrix decompositions"
      <http://arxiv.org/abs/0909.4061>`_
      Halko, et al., 2009


.. _kernel_PCA:

Kernel PCA
----------

:class:`KernelPCA` 是 PCA 的扩展， 通过使用内核（参见 :ref:`metrics` ）实现非线性维数降低。
它具有许多应用，包括去噪，压缩和结构预测（内核依赖估计）。
:class:`KernelPCA` 支持 ``transform`` 和 ``inverse_transform``。

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_kernel_pca_001.png
    :target: ../auto_examples/decomposition/plot_kernel_pca.html
    :align: center
    :scale: 75%

.. topic:: 例子:

    * :ref:`sphx_glr_auto_examples_decomposition_plot_kernel_pca.py`


.. _SparsePCA:

稀疏主成分分析（SparsePCA和MiniBatchSparsePCA）
-----------------------------------------------------------------------

:class:`SparsePCA` 是PCA的一个变体，其目标是提取最能重建数据的稀疏成分集合。

Mini-batch sparse PCA (:class:`MiniBatchSparsePCA`) 是 :class:`SparsePCA`
 的一种变体， 它速度更快但准确度有所降低。
 对于给定的迭代次数，通过迭代该组特征的小数据块来达到速度的增加。

主成分分析（PCA）具有以下缺点：通过该方法提取的成分具有唯一的密集表达式。
即当以原始变量的线性组合表示时，它们具有非零系数。这可以使解释变得困难。
在许多情况下，真正的基础成分可以更自然地想象为稀疏向量;
例如在面部识别中，组件可能自然地映射到部分脸部。

稀疏的主成分产生更简洁，可解释的，明确强调哪些原始特征有助于区分样本之间的差异。

以下示例说明了使用sparse PCA 从Olivetti数据集中提取的16个成分。
可以看出重整后如何引发许多零。此外，数据的自然结构导致非零系数垂直相邻。
该模型不会在数学上强制执行：每个成分都是一个向量 :math:`h \in \mathbf{R}^{4096}` ,
并且没有垂直相邻性的概念，除非人性化地可视化为64x64像素图像。
下面显示的成分出现局部的事实是数据的固有结构的影响，这使得这种局部模式使重建误差最小化。
存在考虑到邻接和不同类型结构的稀疏诱导规范; 参见 [Jen09]_ 。

更多关于 Sparse PCA 使用的内容，参见示例部分，如下：

.. |spca_img| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_005.png
   :target: ../auto_examples/decomposition/plot_faces_decomposition.html
   :scale: 60%

.. centered:: |pca_img| |spca_img|

请注意：有多种计算 Sparse PCA 问题的公式。 这里使用的是基于 [Mrl09]_ 。
优化问题的解决是一个 PCA 问题， 其针对成分使用 :math:`\ell_1` 惩罚：

.. math::
   (U^*, V^*) = \underset{U, V}{\operatorname{arg\,min\,}} & \frac{1}{2}
                ||X-UV||_2^2+\alpha||V||_1 \\
                \text{subject to\,} & ||U_k||_2 = 1 \text{ for all }
                0 \leq k < n_{components}


当训练数据样本数量较少时，稀疏-诱导(sparsity-inducing) :math:`\ell_1` 规范同样会阻止从噪声中学习成分。
惩罚程度（因而稀疏度）可以通过超参数 ``alpha`` 进行调整。
小值导致温和的调整，而较大的值会将许多系数缩小到零。

.. note::

  虽然本着在线算法的精神，该类 :class:`MiniBatchSparsePCA` 不能实现 ``partial_fit`` 。
  因为算法沿着特征方向在线，而不是样本方向。


.. topic:: 例子:

   * :ref:`sphx_glr_auto_examples_decomposition_plot_faces_decomposition.py`

.. topic:: s参考文献:

  .. [Mrl09] `"Online Dictionary Learning for Sparse Coding"
     <http://www.di.ens.fr/sierra/pdfs/icml09.pdf>`_
     J. Mairal, F. Bach, J. Ponce, G. Sapiro, 2009
  .. [Jen09] `"Structured Sparse Principal Component Analysis"
     <www.di.ens.fr/~fbach/sspca_AISTATS2010.pdf>`_
     R. Jenatton, G. Obozinski, F. Bach, 2009


.. _LSA:

截断的奇异值分解和潜在语义分析
===================================================================

:class:`TruncatedSVD` 实现仅计算 :math:`k` 最大奇异值的奇异值分解（SVD）的变体，
其中 :math:`k` 是用户指定的参数。

当 truncated SVD 应用于 term-document matrices (由 ``CountVectorizer`` 或 ``TfidfVectorizer`` 返回)时，
该变换被成为 LSA（`latent semantic analysis <http://nlp.stanford.edu/IR-book/pdf/18lsi.pdf>`_ ，潜在语义分析）。
因为它将这样的矩阵转换为低维度的“语义”空间。
特别地，LSA 能够抵抗同义词和多义词的影响（这两者大致意味着每个单词有多重含义），
这导致 term-document matrices 过度稀疏，并且在诸如余弦相似性的度量下表现出差的相似性。

.. note::
    LSA也被称为潜在语义索引LSI，尽管严格地说它是指在持久索引中用于信息检索的目的。

在数学上， truncated SVD 用于训练样本 :math:`X`， 产生低阶逼近 :math:`X` ：

.. math::
    X \approx X_k = U_k \Sigma_k V_k^\top

在这个操作后，:math:`U_k \Sigma_k^\top` 具有 :math:`k` 特征的变换训练集
（在 API 中称为 ``n_components`` ）

用于转换测试集 :math:`X` 时， 将他与 :math:`V_k` 相乘：

.. math::
    X' = X V_k

.. note::
    自然语言处理（NLP）和信息检索（IR）文献中， 大多数 LSA 的处理是交换 矩阵 :math:`X`
     的轴，使其具有形状 ``n_features`` × ``n_samples``。
    我们描述 LSA 时以不同的方式进行阐述，以与 Scikit-learn API 更好的匹配，
    ，但是找到的奇异值是相同的。

:class:`TruncatedSVD` 与 :class:`PCA` 非常相似，
但不同之处在于，它直接作用于样本矩阵 :math:`X` 而不是协方差矩阵。
当 矩阵 :math:`X` 从特征值中减去列均值时，truncated SVD 得到的结果矩阵与 PCA 相同。
实际上，这意味着 :class:`TruncatedSVD` 变换接受 ``scipy.sparse`` 矩阵，
而不需要对它们进行致密化，因为即使对于中型文档集合，致密化也可能填满内存。

当 :class:`TruncatedSVD` 变换与任何（稀疏）特征矩阵一起使用时，
建议在 LSA/文档 处理设置中的原始频率计数上使用tf-idf矩阵。
特别地，应该打开子线性缩放和逆文档频率 (``sublinear_tf=True, use_idf=True``)
以使特征值更接近高斯分布，从而补偿 LSA 对文本数据的错误假设。

.. topic:: 例子:

   * :ref:`sphx_glr_auto_examples_text_document_clustering.py`

.. topic:: 参考文献:

  * Christopher D. Manning, Prabhakar Raghavan and Hinrich Schütze (2008),
    *Introduction to Information Retrieval*, Cambridge University Press,
    chapter 18: `Matrix decompositions & latent semantic indexing
    <http://nlp.stanford.edu/IR-book/pdf/18lsi.pdf>`_


.. _DictionaryLearning:


词典学习（Dictionary Learning）
===================

.. _SparseCoder:

用预先计算的字典稀疏编码（Sparse coding with a precomputed dictionary）
-------------------------------------------

:class:`SparseCoder` 对象是可从固定的、预先计算的字典
（例如离散小波基，discrete wavelet basis）中将信号转换成
原子的稀疏线性组合的估计器（estimator）。
因此，该对象不实现 ``fit`` 方法。该转换相当于稀疏编码问题：
将数据的表示为尽可能少的字典原子的线性组合。
以下所有字典学习变量实现的变换方法，可通过 ``transform_method`` 初始化参数进行控制：

* 正交匹配追踪（Orthogonal matching pursuit） (:ref:`omp`)

* 最小角度回归（Least-angle regression） (:ref:`least_angle_regression`)

* Lasso通过最小角度回归计算

* Lasso使用坐标下降 (:ref:`lasso`)

* Thresholding 阈值

阈值非常快，但不能产生精确的重建。它们在文本分类任务中已被证明是有用的。
对于图像重建任务，正交匹配追求产生最准确，无偏见的重建。

字典学习对象通过 ``split_code`` 参数提供在稀疏编码结果中分离正值和负值的可能性。
当使用字典学习来提取将用于监督学习的特征时，这是有用的，
因为它允许学习算法将不同的权重从相应的正负载中
分配给特定的负负荷（negative loading）原子。

单个样本的分割代码具有长度 ``2 * n_components`` ， 并使用以下规则构建：
首先计算常规代码的长度 ``n_components`` 。然后，第一个 ``n_components`` 的
``split_code`` 填充了常规代码向量的正向部分。另一半分割代码的填充有负向的代码向量，只有一个正号。
因此，split_code是非负数。

.. topic:: 例子:

    * :ref:`sphx_glr_auto_examples_decomposition_plot_sparse_coding.py`



通用字典学习（Generic dictionary learning）
---------------------------

词典学习（:class:`DictionaryLearning`）是一个矩阵因式分解问题，
相当于找到一个（通常是过于完备的）字典，它将对拟合的数据进行良好的稀疏编码。

将数据表示为来自过度完整字典的原子稀疏组合被认为是哺乳动物初级视觉皮层的工作方式。
因此，应用于图像补丁的字典学习已被证明在诸如图像完成，
修复和去噪以及监督识别任务的图像处理任务中给出良好的结果。

词典学习是通过交替更新稀疏代码解决的优化问题，
作为解决多个Lasso问题的一个解决方案，考虑到字典固定，然后更新字典以最适合稀疏代码。

.. math::
   (U^*, V^*) = \underset{U, V}{\operatorname{arg\,min\,}} & \frac{1}{2}
                ||X-UV||_2^2+\alpha||U||_1 \\
                \text{subject to\,} & ||V_k||_2 = 1 \text{ for all }
                0 \leq k < n_{\mathrm{atoms}}


.. |pca_img2| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_002.png
   :target: ../auto_examples/decomposition/plot_faces_decomposition.html
   :scale: 60%

.. |dict_img2| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_006.png
   :target: ../auto_examples/decomposition/plot_faces_decomposition.html
   :scale: 60%

.. centered:: |pca_img2| |dict_img2|


在使用这样一个过程来拟合字典之后，变换只是一个稀疏编码步骤，它与所有字典学习对象共享相同的实现。
（参见 :ref:`SparseCoder`）


以下图像显示了字典学习是如何从浣熊脸部图像中提取的4x4像素图像补丁中进行实现的。

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_image_denoising_001.png
    :target: ../auto_examples/decomposition/plot_image_denoising.html
    :align: center
    :scale: 50%


.. topic:: 例子:

  * :ref:`sphx_glr_auto_examples_decomposition_plot_image_denoising.py`


.. topic:: 参考文献:

  * `"Online dictionary learning for sparse coding"
    <http://www.di.ens.fr/sierra/pdfs/icml09.pdf>`_
    J. Mairal, F. Bach, J. Ponce, G. Sapiro, 2009

.. _MiniBatchDictionaryLearning:

Mini-batch 字典学习 （Mini-batch dictionary learning）
------------------------------

:class:`MiniBatchDictionaryLearning` 实现更适合大型数据集的字典学习算法，
其运行速度更快，但准确度有所降低。

默认情况下，:class:`MiniBatchDictionaryLearning` 将数据分成小批量，
并通过在指定次数的迭代中循环使用小批量，以在线方式进行优化。但是，目前它没有实现停止条件。

该方法还实现了 ``partial_fit`` ， 它通过在一个 mini-batch 中迭代一次来更新字典。
当数据从一开始就不容易获得，或数据不适合内存时，这可以用于在线学习。

.. currentmodule:: sklearn.cluster

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_dict_face_patches_001.png
    :target: ../auto_examples/cluster/plot_dict_face_patches.html
    :scale: 50%
    :align: right

.. topic:: **Clustering for dictionary learning字典学习聚类**

   请注意，当使用字典学习提取文档（例如稀疏编码）时，聚类可以成为学习字典的好代理。
   例如，:class:`MiniBatchKMeans` 估计器在计算上是有效的，
   并通过一种  ``partial_fit`` 方法实现在线学习 。

    例子: :ref:`sphx_glr_auto_examples_cluster_plot_dict_face_patches.py`

.. currentmodule:: sklearn.decomposition

.. _FA:

因子分析
===============

在无监督的学习中，我们只有一个数据集 :math:`X = \{x_1, x_2, \dots, x_n
\}` 。这个数据集如何在数学上描述？
一个非常简单的连续潜变量 `continuous latent variable` 的模型 :math:`X` 是：

.. math:: x_i = W h_i + \mu + \epsilon

向量 :math:`h_i` 被称为 潜在 （"latent"）， 因为它是不可观察的。
:math:`\epsilon` 被认为是根据具有平均值0和协方差
:math:`\Psi` (即 :math:`\epsilon \sim \mathcal{N}(0, \Psi)`) 的高斯分布的噪声项，
:math:`\mu` 是任意的偏移向量。这样的模型被称为生成("generative")，
因为它描述了如何从 :math:`h_i` 中生成 :math:`x_i`。
如果我们使用所有的 :math:`x_i` 列作为矩阵 :math:`\mathbf{X}` ，
并将所有的 :math:`h_i` 列作为矩阵 :math:`\mathbf{H}` 的列，
那么我们可以写（适当地定义 :math:`\mathbf{M}` 和 :math:`\mathbf{E}`）：

.. math::
    \mathbf{X} = W \mathbf{H} + \mathbf{M} + \mathbf{E}

换句话说，我们 *分解* 矩阵 :math:`\mathbf{X}` 。

如果 :math:`h_i` 给出，上述方程自动暗示以下概率解释：

.. math:: p(x_i|h_i) = \mathcal{N}(Wh_i + \mu, \Psi)

对于一个完整的概率模型，我们还需要潜在变量 :math:`h` 的先前分布。
最直接的假设（基于高斯分布的良好属性）是 :math:`h \sim \mathcal{N}(0,
\mathbf{I})` 。这产生一个高斯作为 :math:`x` 的边际分布：


.. math:: p(x) = \mathcal{N}(\mu, WW^T + \Psi)

现在，没有任何进一步的假设，具有潜在变量 :math:`h` 的想法将是多余的
 -- :math:`x` 可以用平均和协方差来完全建模。
我们需要对这两个参数之一施加一些更具体的结构。
一个简单的附加假设是误差协方差的结构 :math:`\Psi` ：


* :math:`\Psi = \sigma^2 \mathbf{I}`: 这个假设导致了概率模型 :class:`PCA` 。

* :math:`\Psi = \mathrm{diag}(\psi_1, \psi_2, \dots, \psi_n)`:
  这个模型叫做 :class:`FactorAnalysis` ，是一个经典的统计模型。
  矩阵 W 有时也称为“因子加载矩阵” （"factor loading matrix"）。

两个模型基本上估计出具有低阶协方差矩阵的高斯。
因为这两个模型都是概率性的，所以它们可以集成到更复杂的模型中，例如混合因子分析。
如果假设潜在变量上的非高斯先验，则得到非常不同的模型（例如 :class:`FastICA` ）。


因子分析可以产生类似于 :class:`PCA` 的成分（其加载矩阵的列）。
然而，不能对这些成分做出任何一般性的说明（例如它们是否是正交的）：

.. |pca_img3| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_002.png
    :target: ../auto_examples/decomposition/plot_faces_decomposition.html
    :scale: 60%

.. |fa_img3| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_009.png
    :target: ../auto_examples/decomposition/plot_faces_decomposition.html
    :scale: 60%

.. centered:: |pca_img3| |fa_img3|

因子分析的主要优于 :class:`PCA` 的地方是
它可以独立地对输入空间的每个方向的模型进行建模（异方差噪声）：

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_008.png
    :target: ../auto_examples/decomposition/plot_faces_decomposition.html
    :align: center
    :scale: 75%

在异方差噪声的存在下，这样可以比概率PCA更好的模型选择：

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_pca_vs_fa_model_selection_002.png
    :target: ../auto_examples/decomposition/plot_pca_vs_fa_model_selection.html
    :align: center
    :scale: 75%


.. topic:: 例子:

    * :ref:`sphx_glr_auto_examples_decomposition_plot_pca_vs_fa_model_selection.py`

.. _ICA:

独立成分分析 （Independent component analysis, ICA）
====================================

独立成分分析将多变量信号分解为最大独立的附加子成分。
在scikit-learn中使用 :class:`Fast ICA <FastICA>` 算法来实现。
通常，ICA不用于降低维度，而是用于分离叠加信号。
由于ICA模型不包括噪声项，因此，为了使模型正确，必须首先进行预处理。
这可以在内部使用whiten参数或手动使用其中一种PCA变体进行。


通常用于分离混合信号（称为盲源分离的问题， *blind source separation* ），如下例所示：

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_ica_blind_source_separation_001.png
    :target: ../auto_examples/decomposition/plot_ica_blind_source_separation.html
    :align: center
    :scale: 60%


ICA也可以用作另一种非线性分解，可以找到一些稀疏成分：

.. |pca_img4| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_002.png
    :target: ../auto_examples/decomposition/plot_faces_decomposition.html
    :scale: 60%

.. |ica_img4| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_004.png
    :target: ../auto_examples/decomposition/plot_faces_decomposition.html
    :scale: 60%

.. centered:: |pca_img4| |ica_img4|

.. topic:: 例子:

    * :ref:`sphx_glr_auto_examples_decomposition_plot_ica_blind_source_separation.py`
    * :ref:`sphx_glr_auto_examples_decomposition_plot_ica_vs_pca.py`
    * :ref:`sphx_glr_auto_examples_decomposition_plot_faces_decomposition.py`


.. _NMF:

非负矩阵分解 (Non-negative matrix factorization, NMF or NNMF)
===============================================

具有 Frobenius 范数的 NMF
---------------------------

:class:`NMF` [1]_ 是分解的另一种方法，它假定数据和成分是非负的。
在数据矩阵不包含负值的情况下， :class:`NMF` 可以插入，用来替代 :class:`PCA` 或其变体。
它通过优化 :math:`d` 和 :math:`X` 之间的距离以及 矩阵 :math:`WH` ，
来将样本 :math:`X `分解成为两个非负元素的矩阵 :math:`W` 和 :math:`H`。
使用最广泛的距离函数是 平方Frobenius范数 (squared Frobenius norm)，
它是是 欧几里得范数 (Euclidean norm) 到矩阵的一个明显的扩展 ：


.. math::
    d_{\mathrm{Fro}}(X, Y) = \frac{1}{2} ||X - Y||_{\mathrm{Fro}}^2 = \frac{1}{2} \sum_{i,j} (X_{ij} - {Y}_{ij})^2

不同于 :class:`PCA` 的是，通过叠加成分而不加减去，以附加的方式获得向量的表示。
这种附加模型对于表示图像和文本是有效的。

在 [Hoyer, 2004] [2]_ 中已经观察到，当被仔细约束时，
 :class:`NMF` 可以产生数据集的基于部分的表示，导致可解释的模型。
以下示例显示了使用 :class:`NMF` 从Olivetti面数据集中的图像中找到的16个稀疏成分，
并与 PCA 得到的结果进行对比。

.. |pca_img5| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_002.png
    :target: ../auto_examples/decomposition/plot_faces_decomposition.html
    :scale: 60%

.. |nmf_img5| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_003.png
    :target: ../auto_examples/decomposition/plot_faces_decomposition.html
    :scale: 60%

.. centered:: |pca_img5| |nmf_img5|


 :attr:`init` 属性决定了应用的初始化方法，对方法的性能有很大的影响。
 :class:`NMF` 实现非负双奇异值分解方法。
  NNDSVD [4]_ 基于两个 SVD 过程，一个近似于数据矩阵，
  另一个近似于 使用单位秩矩阵的代数性质，得到的部分SVD因子的正部分。
基本的 NNDSVD 算法更适合稀疏分解。
其变体 NNDSVDa（其中所有零被设置为等于数据的所有元素的平均值）
和 NNDSVDar（其中将零设置为小于数据平均值的随机扰动除以100）推荐在密集情况下使用。

请注意：乘法更新方法 （Multiplicative Update ('mu') solver）不能在初始化中更新零值，
因此，当其与基础的 NNDSVD 算法（包含大量的零值）一起使用时，它将导致比较差的结果。
在这种情况下，使用 NNDSVDa 或 NNDSVDar 会更好。

:class:`NMF` 也可以通过设置正确缩放的随机非负矩阵（设定，:attr:`init="random"`）进行初始化。
整数种子 或 ``RandomState`` 也可以传递 给 :attr:`random_state` ，
以保证结果可以重现。

在NMF L1和L2先验可以添加到损失函数，以使模型正规化。
L2之前使用Frobenius标准，而L1先验使用元素L1范数。
如ElasticNet，我们控制L1和L2的与所述组合l1_ratio（）参数，
并与正则化强度alpha （）参数。那么先修课程是：

.. math::
    \alpha \rho ||W||_1 + \alpha \rho ||H||_1
    + \frac{\alpha(1-\rho)}{2} ||W||_{\mathrm{Fro}} ^ 2
    + \frac{\alpha(1-\rho)}{2} ||H||_{\mathrm{Fro}} ^ 2

正则化的目标函数是：

.. math::
    d_{\mathrm{Fro}}(X, WH)
    + \alpha \rho ||W||_1 + \alpha \rho ||H||_1
    + \frac{\alpha(1-\rho)}{2} ||W||_{\mathrm{Fro}} ^ 2
    + \frac{\alpha(1-\rho)}{2} ||H||_{\mathrm{Fro}} ^ 2

:class:`NMF` 正则化 W 和 H 。公共函数 :func:`non_negative_factorization`
允许通过 :attr:`regularization` 属性进行更精细的控制 ，并且可以仅将 W ，仅 H 或两者正规化。


beta-分离的 NMF （NMF with a beta-divergence）
--------------------------

如前所述，最广泛使用的距离函数是Frobenius范数（the squared Frobenius norm）。
，这是欧氏距离（Euclidean norm，欧几里德范数）一个最广泛的扩展：

.. math::
    d_{\mathrm{Fro}}(X, Y) = \frac{1}{2} ||X - Y||_{Fro}^2 = \frac{1}{2} \sum_{i,j} (X_{ij} - {Y}_{ij})^2

其他能在 NMF 中使用的距离函数，例如 the (generalized)
Kullback-Leibler (KL) divergence，也称为 I-divergence ：

.. math::
    d_{KL}(X, Y) = \sum_{i,j} (X_{ij} \log(\frac{X_{ij}}{Y_{ij}}) - X_{ij} + Y_{ij})

Or, the Itakura-Saito (IS) divergence:
或者, the Itakura-Saito (IS) divergence:

.. math::
    d_{IS}(X, Y) = \sum_{i,j} (\frac{X_{ij}}{Y_{ij}} - \log(\frac{X_{ij}}{Y_{ij}}) - 1)

这三种距离函数都属于 beta-divergence 家族， 其参数分别为 :math:`\beta = 2, 1, 0` [6]_。
beta-divergence 定义为：

.. math::
    d_{\beta}(X, Y) = \sum_{i,j} \frac{1}{\beta(\beta - 1)}(X_{ij}^\beta + (\beta-1)Y_{ij}^\beta - \beta X_{ij} Y_{ij}^{\beta - 1})

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_beta_divergence_001.png
    :target: ../auto_examples/decomposition/plot_beta_divergence.html
    :align: center
    :scale: 75%

请注意： 当 :math:`\beta \in (0; 1)` 时，这个定义是无效的， 然而，它可以是分别是 定义
:math:`d_{KL}` 和 :math:`d_{IS}` 的连续延伸。

:class:`NMF` 可以使用两种解决方案， Coordinate Descent ('cd') [5]_， 和
Multiplicative Update ('mu') [6]_。'mu' 方法可以优化每种 beta-divergence， 包括
Frobenius norm (:math:`\beta=2`), the
(generalized) Kullback-Leibler divergence (:math:`\beta=1`) 以及 the
Itakura-Saito divergence (:math:`\beta=0`)。
请注意： 当 :math:`\beta \in (1; 2)` 时,  'mu' 方法 明显快于 其他的 :math:`\beta` 值。
同样请注意， 负的 :math:`\beta` 值 （或，0， 如'itakura-saito'），
矩阵的输入值不能包含零值。


'cd' 方法仅能优化 Frobenius norm。因为 NMF 可能是非凸的 （Due to the
underlying non-convexity of NMF）， 不同的方法可能得到不同的最小值，
即使在优化时使用的是相同的距离函数。


当使用 ``fit_transform`` 方法时，NMF是最好的，它返回矩阵 W。
矩阵 H 则保存在  fitted model 的 ``components_`` 属性里。
``transform`` 方法将基于这些保存的成分分解出一个新的矩阵 X_new 。

    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> from sklearn.decomposition import NMF
    >>> model = NMF(n_components=2, init='random', random_state=0)
    >>> W = model.fit_transform(X)
    >>> H = model.components_
    >>> X_new = np.array([[1, 0], [1, 6.1], [1, 0], [1, 4], [3.2, 1], [0, 4]])
    >>> W_new = model.transform(X_new)

.. topic:: 例子:

    * :ref:`sphx_glr_auto_examples_decomposition_plot_faces_decomposition.py`
    * :ref:`sphx_glr_auto_examples_applications_plot_topics_extraction_with_nmf_lda.py`
    * :ref:`sphx_glr_auto_examples_decomposition_plot_beta_divergence.py`

.. topic:: 参考文献:

    .. [1] `"Learning the parts of objects by non-negative matrix factorization"
      <http://www.columbia.edu/~jwp2128/Teaching/W4721/papers/nmf_nature.pdf>`_
      D. Lee, S. Seung, 1999

    .. [2] `"Non-negative Matrix Factorization with Sparseness Constraints"
      <http://www.jmlr.org/papers/volume5/hoyer04a/hoyer04a.pdf>`_
      P. Hoyer, 2004

    .. [4] `"SVD based initialization: A head start for nonnegative
      matrix factorization"
      <http://scgroup.hpclab.ceid.upatras.gr/faculty/stratis/Papers/HPCLAB020107.pdf>`_
      C. Boutsidis, E. Gallopoulos, 2008

    .. [5] `"Fast local algorithms for large scale nonnegative matrix and tensor
      factorizations."
      <http://www.bsp.brain.riken.jp/publications/2009/Cichocki-Phan-IEICE_col.pdf>`_
      A. Cichocki, P. Anh-Huy, 2009

    .. [6] `"Algorithms for nonnegative matrix factorization with the beta-divergence"
      <http://http://arxiv.org/pdf/1010.1763v3.pdf>`_
      C. Fevotte, J. Idier, 2011


.. _LatentDirichletAllocation:


潜在 Dirichlet 分配 （Latent Dirichlet Allocation，LDA）
=================================

Latent Dirichlet Allocation 是离散数据集（如文本语料库）的集合的生成概率模型。
它也是一个主题模型，用于从文档集合中发现摘要内容。

LDA的图形模型是一个三级贝叶斯模型 （three-level Bayesian model）：

.. image:: ../images/lda_model_graph.png
   :align: center

当建模文本语料库时，该模型对文档语料库 :math:`D` 和 主题语料库 :math:`K`
有以下假设：

  1. 针对每个主题 :math:`k`， 绘制 :math:`\beta_k \sim \mathrm{Dirichlet}(\eta),\: k =1...K`

  2. 针对每个文档 :math:`d`， 绘制 :math:`\theta_d \sim \mathrm{Dirichlet}(\alpha), \: d=1...D`

  3. 针对文档 :math:`d` 中的每个单词 :math:`i` ：

    a. 绘制主题索引 :math:`z_{di} \sim \mathrm{Multinomial}(\theta_d)`
    b. 绘制出观察到的单词： :math:`w_{ij} \sim \mathrm{Multinomial}(beta_{z_{di}}.)`




对于参数估计，后验分布为：


.. math::
  p(z, \theta, \beta |w, \alpha, \eta) =
    \frac{p(z, \theta, \beta|\alpha, \eta)}{p(w|\alpha, \eta)}

由于后验分布难以处理，经变化的贝叶斯方法（variational Bayesian method）使用一个简单的分布
 :math:`q(z,\theta,\beta | \lambda, \phi, \gamma)` 来处理， 这些变化的参数
 :math:`\lambda`, :math:`\phi`, :math:`\gamma`
 经优化后使 Evidence Lower Bound (ELBO) 得到最大化。

.. math::
  \log\: P(w | \alpha, \eta) \geq L(w,\phi,\gamma,\lambda) \overset{\triangle}{=}
    E_{q}[\log\:p(w,z,\theta,\beta|\alpha,\eta)] - E_{q}[\log\:q(z, \theta, \beta)]

最大化 ELBO 等同于 最小化 Kullback-Leibler(KL) 的 :math:`q(z,\theta,\beta)`
与 :math:`p(z, \theta, \beta |w, \alpha, \eta)` 之间的差异。

:class:`LatentDirichletAllocation` 实现了在线 变分贝叶斯算法 （variational Bayes algorithm），
并支持在线和批量更新方法。
批处理方法在每次完全传递数据后更新变分变量，在线方法从小批量数据点更新变分变量。

.. note::

  注意: 虽然在线方法保证收敛到局部最优点，
  但最优点的质量和收敛速度可能取决于小批量大小和学习率设置相关的属性。

当 :class:`LatentDirichletAllocation` 应用于 "document-term" 矩阵时， 该矩阵将会被分解
称为 "topic-term" 矩阵和 "document-topic"  矩阵。 在模型中， "topic-term" 矩阵 以参数
:attr:`components_` 的形式保存， 而 "document-topic"  矩阵 可以从 ``transform`` 方法中
计算出来。

:class:`LatentDirichletAllocation` 也实现了 ``partial_fit`` 方法。当数据可以顺序提取时使用。

.. topic:: 例子:

    * :ref:`sphx_glr_auto_examples_applications_plot_topics_extraction_with_nmf_lda.py`

.. topic:: 参考文献:

    * `"Latent Dirichlet Allocation"
      <https://www.cs.princeton.edu/~blei/papers/BleiNgJordan2003.pdf>`_
      D. Blei, A. Ng, M. Jordan, 2003

    * `"Online Learning for Latent Dirichlet Allocation”
      <https://www.cs.princeton.edu/~blei/papers/HoffmanBleiBach2010b.pdf>`_
      M. Hoffman, D. Blei, F. Bach, 2010

    * `"Stochastic Variational Inference"
      <http://www.columbia.edu/~jwp2128/Papers/HoffmanBleiWangPaisley2013.pdf>`_
      M. Hoffman, D. Blei, C. Wang, J. Paisley, 2013
