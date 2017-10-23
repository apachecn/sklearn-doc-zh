.. _decompositions:


==============================================
分解成分中的信号（矩阵分解问题）
==============================================

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

内核 PCA
----------------

:class:`KernelPCA` 是 PCA 的扩展，通过使用内核实现非线性 dimensionality reduction（降维） (参阅 :ref:`metrics`)。它具有许多应用，包括 denoising（去噪）, compression（压缩） 和 structured prediction（结构预测） (kernel dependency estimation（内核依赖估计）)。 :class:`KernelPCA` 支持 ``transform`` 和 ``inverse_transform`` 。

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_kernel_pca_001.png
    :target: ../auto_examples/decomposition/plot_kernel_pca.html
    :align: center
    :scale: 75%

.. topic:: 示例:

    * :ref:`sphx_glr_auto_examples_decomposition_plot_kernel_pca.py`


.. _SparsePCA:

稀疏主成分分析 ( SparsePCA 和 MiniBatchSparsePCA )
--------------------------------------------------------------------

:class:`SparsePCA` 是 PCA 的一个变体，目的是提取能重建数据的 sparse components （稀疏组件）集合。

Mini-batch sparse PCA（小批量稀疏 PCA） (:class:`MiniBatchSparsePCA`) 是一个 :class:`SparsePCA` 的变种，它速度更快但准确度有所降低。对于给定的迭代次数，通过迭代该组特征的 small chunks （小块）来达到速度的增加。


Principal component analysis（主成分分析） (:class:`PCA`) 的缺点在于，通过该方法提取的成分具有唯一的密集表达式，即当表示为原始变量的线性组合时，它们具有非零系数。这可以使解释变得困难。在许多情况下，真正的基础组件可以更自然地想象为稀疏向量; 例如在面部识别中，组件可能自然地映射到面部的部分。

Sparse principal components（稀疏的主成分）产生更简洁，可解释的表示，明确强调哪些 original features （原始特征）有助于样本之间的差异。

以下示例说明了使用 Olivetti faces dataset 中的稀疏 PCA 提取的 16 个 components （组件）。可以看出 regularization term （正则化术语）如何引发许多零。此外，数据的 natural structure （自然结构）导致非零系数 vertically adjacent （垂直相邻）。该模型不会在数学上强制执行: 每个 component （组件）都是一个向量  :math:`h \in \mathbf{R}^{4096}`, 没有 vertical adjacency （垂直相邻性）的概念，除了人类友好的可视化视图为 64x64 像素图像。下面显示的组件出现局部的事实是数据的固有结构的影响，这使得这种局部模式使重建误差最小化。存在考虑到邻接和不同类型结构的稀疏诱导规范; 参见 [Jen09]_ 对这种方法进行审查。
有关如何使用稀疏 PCA 的更多详细信息，请参阅下面的示例部分。

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

请注意，有多种计算 Sparse PCA 问题的公式。 这里使用的是基于 [Mrl09]_ 。解决的优化问题是一个 PCA 问题（dictionary learning（字典学习）），具有 :math:`\ell_1` 对 components （组件）的 penalty （惩罚）:

.. math::
   (U^*, V^*) = \underset{U, V}{\operatorname{arg\,min\,}} & \frac{1}{2}
                ||X-UV||_2^2+\alpha||V||_1 \\
                \text{subject to\,} & ||U_k||_2 = 1 \text{ for all }
                0 \leq k < n_{components}


sparsity-inducing（稀疏性诱导） :math:`\ell_1` 规范也可以避免学习成分的噪音降低，而 training samples （训练样本）很少。可以通过超参数 ``alpha`` 来调整惩罚程度（从而减少稀疏度）。小值导致了温和的正则化因式分解，而较大的值将许多系数缩小到零。

.. note::

  虽然本着在线算法的精神， :class:`MiniBatchSparsePCA` 类不实现 ``partial_fit`` , 因为算法沿特征方向在线，而不是样本方向。

.. topic:: 示例:

   * :ref:`sphx_glr_auto_examples_decomposition_plot_faces_decomposition.py`

.. topic:: 参考文献:

  .. [Mrl09] `"Online Dictionary Learning for Sparse Coding"
     <http://www.di.ens.fr/sierra/pdfs/icml09.pdf>`_
     J. Mairal, F. Bach, J. Ponce, G. Sapiro, 2009
  .. [Jen09] `"Structured Sparse Principal Component Analysis"
     <www.di.ens.fr/~fbach/sspca_AISTATS2010.pdf>`_
     R. Jenatton, G. Obozinski, F. Bach, 2009


.. _LSA:


截断奇异值分解和潜在语义分析
=========================================================

:class:`TruncatedSVD` 实现了一个奇异值分解（SVD）的变体，它只能计算 :math:`k` 最大的奇异值，其中 :math:`k` 是用户指定的参数。

当 truncated SVD （截断的 SVD） 被应用于 term-document matrices （由 ``CountVectorizer`` 或 ``TfidfVectorizer`` 返回）时，这种转换被称为 `latent semantic analysis <http://nlp.stanford.edu/IR-book/pdf/18lsi.pdf>`_ (LSA), 因为它将这样的矩阵转换为低纬度的 "semantic（语义）" 空间。
特别地， LSA 已知能够抵抗同义词和多义词的影响（两者大致意味着每个单词有多重含义），这导致 term-document matrices 过度稀疏，并且在诸如余弦相似性的度量下表现出差的相似性。

.. note::
    LSA 也被称为潜在语义索引 LSI，尽管严格地说它是指在 persistent indexes （持久索引）中用于 information retrieval （信息检索）的目的。

数学表示中， truncated SVD 应用于训练样本 :math:`X` 产生一个 low-rank approximation （低阶逼近于） :math:`X`:

.. math::
    X \approx X_k = U_k \Sigma_k V_k^\top

在这个操作之后，:math:`U_k \Sigma_k^\top` 是转换后的训练集，其中包括 :math:`k` 个特征（在 API 中被称为 ``n_components`` ）。

还需要转换一个测试集 :math:`X`, 我们乘以 :math:`V_k`: 

.. math::
    X' = X V_k

.. note::
    
    自然语言处理(NLP) 和信息检索(IR) 文献中的 LSA 的大多数处理方式交换 axes of the :math:`X` matrix （:math:`X` 矩阵的轴）,使其具有形状 ``n_features`` × ``n_samples`` 。
    我们以 scikit-learn API 相匹配的不同方式呈现 LSA, 但是找到的奇异值是相同的。

:class:`TruncatedSVD` 非常类似于 :class:`PCA`, 但不同之处在于它适用于示例矩阵 :math:`X` 而不是它们的协方差矩阵。
当从特征值中减去 :math:`X` 的列（每个特征）手段时，得到的矩阵上的 truncated SVD 相当于 PCA 。
实际上，这意味着 :class:`TruncatedSVD` transformer（转换器）接受 ``scipy.sparse`` 矩阵，而不需要对它们进行加密，因为即使对于中型文档集合，densifying （密集）也可能填满内存。

虽然 :class:`TruncatedSVD` transformer（转换器）与任何（sparse（稀疏））特征矩阵一起工作，建议在 LSA/document 处理设置中使用它在 tf–idf 矩阵上的原始频率计数。
特别地，应该打开 sublinear scaling （子线性缩放）和 inverse document frequency （逆文档频率） (``sublinear_tf=True, use_idf=True``) 以使特征值更接近于高斯分布，补偿 LSA 对文本数据的错误假设。

.. topic:: 示例:

   * :ref:`sphx_glr_auto_examples_text_document_clustering.py`

.. topic:: 参考文献:

  * Christopher D. Manning, Prabhakar Raghavan and Hinrich Schütze (2008),
    *Introduction to Information Retrieval*, Cambridge University Press,
    chapter 18: `Matrix decompositions & latent semantic indexing
    <http://nlp.stanford.edu/IR-book/pdf/18lsi.pdf>`_


.. _DictionaryLearning:

字典学习
==================

.. _SparseCoder:

稀疏编码与预先计算的字典
---------------------------------

:class:`SparseCoder` 对象是一个 estimator（估计器），可以用来将信号转换成固定的预先计算的字典的 sparse linear combination （稀疏线性组合），如 discrete wavelet basis 。因此，该对象不实现 ``fit`` 方法。该转换相当于 sparse coding problem （稀疏编码问题）: 将数据的表示尽可能少的 dictionary atoms （字典原子）的线性组合。字典学习的所有变体实现以下变换方法，可以通过 ``transform_method`` 初始化参数进行控制: 

* Orthogonal matching pursuit(正交匹配 pursuit ) (:ref:`omp`)

* Least-angle regression (最小角度回归)(:ref:`least_angle_regression`)

* Lasso computed by least-angle regression(Lasso 通过最小角度回归计算)

* Lasso using coordinate descent (Lasso 使用梯度下降)(:ref:`lasso`)

* Thresholding(阈值)

Thresholding （阈值）非常快，但是不能产生精确的 reconstructions（重建）。
它们在分类任务的文献中已被证明是有用的。对于 image reconstruction tasks （图像重建任务）， orthogonal matching pursuit 产生最精确，unbiased（无偏）的重建。 

dictionary learning（字典学习）对象通过 ``split_code`` 参数提供分类稀疏编码结果中的正值和负值的可能性。当使用 dictionary learning （字典学习）来提取将用于监督学习的特征时，这是有用的，因为它允许学习算法将不同的权重分配给 particular atom （特定原子）的 negative loadings （负的负荷），从相应的 positive loading （正加载）。

split code for a single sample（单个样本的分割代码）具有长度 ``2 * n_components`` ，并使用以下规则构造: 首先，计算长度为 ``n_components`` 的常规代码。然后， ``split_code`` 的第一个 ``n_components`` 条目将用正常代码向量的正部分填充。分割代码的下半部分填充有代码矢量的负部分，只有一个正号。因此， split_code 是非负的。 


.. topic:: 示例:

    * :ref:`sphx_glr_auto_examples_decomposition_plot_sparse_coding.py`


通用字典学习
--------------------

Dictionary learning（字典学习） (:class:`DictionaryLearning`) 是一个矩阵因式分解问题，相当于找到一个（通常是不完整的）字典，它将在拟合数据的稀疏编码中表现良好。

将数据表示为来自 overcomplete dictionary（过度完整字典）的稀疏组合的原子被认为是  mammal primary visual cortex works（哺乳动物初级视觉皮层的工作方式）. 
因此，应用于 image patches （图像补丁）的 dictionary learning （字典学习）已被证明在诸如 image completion ，inpainting（修复） and denoising（去噪）以及监督识别的图像处理任务中给出良好的结果。

Dictionary learning（字典学习）是通过交替更新稀疏代码来解决的优化问题，作为解决多个 Lasso 问题的一个解决方案，考虑到 dictionary fixed （字典固定），然后更新字典以最适合 sparse code （稀疏代码）。

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


在使用这样一个过程来 fit the dictionary （拟合字典）之后，变换只是一个稀疏的编码步骤，与所有的字典学习对象共享相同的实现。(参见 :ref:`SparseCoder`)。

以下图像显示从 raccoon face （浣熊脸部）图像中提取的 4x4 像素图像补丁中学习的字典如何。

以下图像显示了字典学习是如何从浣熊脸部图像中提取的4x4像素图像补丁中进行实现的。

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_image_denoising_001.png
    :target: ../auto_examples/decomposition/plot_image_denoising.html
    :align: center
    :scale: 50%


.. topic:: 示例:

  * :ref:`sphx_glr_auto_examples_decomposition_plot_image_denoising.py`


.. topic:: 参考文献:

  * `"Online dictionary learning for sparse coding"
    <http://www.di.ens.fr/sierra/pdfs/icml09.pdf>`_
    J. Mairal, F. Bach, J. Ponce, G. Sapiro, 2009

.. _MiniBatchDictionaryLearning:

Mini-batch 字典学习
----------------------

:class:`MiniBatchDictionaryLearning` 实现了更快、更适合大型数据集的字典学习算法，其运行速度更快，但准确度有所降低。

默认情况下，:class:`MiniBatchDictionaryLearning` 将数据分成小批量，并通过在指定次数的迭代中循环使用小批量，以在线方式进行优化。但是，目前它没有实现停止条件。

估计器还实现了  ``partial_fit``, 它通过在一个迷你批处理中仅迭代一次来更新字典。 当数据从一开始就不容易获得，或者当数据不适合内存时，这可以用于在线学习。

.. currentmodule:: sklearn.cluster

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_dict_face_patches_001.png
    :target: ../auto_examples/cluster/plot_dict_face_patches.html
    :scale: 50%
    :align: right

.. topic:: **字典学习聚类**

   注意，当使用字典学习来提取表示（例如，用于稀疏编码）时，聚类可以是学习字典的良好代理。 
   例如，:class:`MiniBatchKMeans` 估计器在计算上是有效的，并使用 ``partial_fit`` 方法实现在线学习。

   示例: 在线学习面部部分的字典 :ref:`sphx_glr_auto_examples_cluster_plot_dict_face_patches.py`

.. currentmodule:: sklearn.decomposition

.. _FA:

因子分析
===============

在无监督的学习中，我们只有一个数据集 :math:`X = \{x_1, x_2, \dots, x_n\}`. 
这个数据集如何在数学上描述？ :math:`X` 的一个非常简单的连续潜变量模型

.. math:: x_i = W h_i + \mu + \epsilon

矢量 :math:`h_i` 被称为 "潜在"，因为它是不可观察的。 
:math:`\epsilon` 被认为是根据高斯分布的噪声项，平均值为 0，协方差为 :math:`\Psi` （即 :math:`\epsilon \sim \mathcal{N}(0, \Psi)`）， 
:math:`\mu` 是一些任意的偏移向量。 这样一个模型被称为 "生成"，因为它描述了如何从 :math:`h_i` 生成 :math:`x_i` 。
如果我们使用所有的 :math:`x_i` 作为列来形成一个矩阵 :math:`\mathbf{X}` ，并将所有的 :math:`h_i` 作为矩阵 :math:`\mathbf{H}` 的列，
那么我们可以写（适当定义的 :math:`\mathbf{M}` 和 :math:`\mathbf{E}` ）:

.. math::
    \mathbf{X} = W \mathbf{H} + \mathbf{M} + \mathbf{E}

换句话说，我们 *分解* 矩阵 :math:`\mathbf{X}`.
如果给出 :math:`h_i`，上述方程自动地表示以下概率解释：

.. math:: p(x_i|h_i) = \mathcal{N}(Wh_i + \mu, \Psi)

对于一个完整的概率模型，我们还需要一个隐变量 :math:`h` 的先验分布。 
最直接的假设（基于高斯分布的不同属性）是:math:`h \sim \mathcal{N}(0, \mathbf{I})`. 这产生一个高斯作为 :math:`x` 的边际分布:

.. math:: p(x) = \mathcal{N}(\mu, WW^T + \Psi)

现在，没有任何进一步的假设，具有隐变量 :math:`h` 的想法将是多余的 -- :math:`x` 可以用均值和协方差来完全建模。 
我们需要对这两个参数之一施加一些更具体的结构。 一个简单的附加假设是误差协方差的结构 :math:`\Psi`:

* :math:`\Psi = \sigma^2 \mathbf{I}`: 这个假设导致 :class:`PCA` 的概率模型。

* :math:`\Psi = \mathrm{diag}(\psi_1, \psi_2, \dots, \psi_n)`: 这个模型称为 :class:`FactorAnalysis`, 一个经典的统计模型。 矩阵W有时称为 "因子加载矩阵"。

两个模型基本上估计出具有低阶协方差矩阵的高斯。 
因为这两个模型都是概率性的，所以它们可以集成到更复杂的模型中，
例如 因子分析仪的混合物 如果假设潜在变量上的非高斯先验，则获得非常不同的模型（例如， :class:`FastICA` ）。

因子分析 *可以* 产生类似的组件（其加载矩阵的列）到 :class:`PCA`。 
然而，不能对这些组件做出任何一般性的陈述（例如它们是否是正交的）:

.. |pca_img3| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_002.png
    :target: ../auto_examples/decomposition/plot_faces_decomposition.html
    :scale: 60%

.. |fa_img3| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_009.png
    :target: ../auto_examples/decomposition/plot_faces_decomposition.html
    :scale: 60%

.. centered:: |pca_img3| |fa_img3|

因子分析(:class:`PCA`) 的主要优点是可以独立地对输入空间的每个方向（异方差噪声）建模方差:

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_008.png
    :target: ../auto_examples/decomposition/plot_faces_decomposition.html
    :align: center
    :scale: 75%

在异方差噪声存在的情况下，这可以比概率 PCA 更好的模型选择:

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_pca_vs_fa_model_selection_002.png
    :target: ../auto_examples/decomposition/plot_pca_vs_fa_model_selection.html
    :align: center
    :scale: 75%


.. topic:: 示例:

    * :ref:`sphx_glr_auto_examples_decomposition_plot_pca_vs_fa_model_selection.py`

.. _ICA:

独立成分分析（ICA）
=========================

独立分量分析将多变量信号分解为最大独立的加性子组件。 
它使用 :class:`Fast ICA <FastICA>` 算法在 scikit-learn 中实现。 
通常，ICA 不用于降低维度，而是用于分离叠加信号。 
由于 ICA 模型不包括噪声项，因此要使模型正确，必须应用美白。 
这可以在内部使用 whiten 参数或手动使用其中一种 PCA 变体进行。

通常用于分离混合信号（称为 *盲源分离* 的问题），如下例所示:

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_ica_blind_source_separation_001.png
    :target: ../auto_examples/decomposition/plot_ica_blind_source_separation.html
    :align: center
    :scale: 60%


ICA 也可以被用作发现具有一些稀疏性的组件的另一个非线性分解:

.. |pca_img4| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_002.png
    :target: ../auto_examples/decomposition/plot_faces_decomposition.html
    :scale: 60%

.. |ica_img4| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_004.png
    :target: ../auto_examples/decomposition/plot_faces_decomposition.html
    :scale: 60%

.. centered:: |pca_img4| |ica_img4|

.. topic:: 示例:

    * :ref:`sphx_glr_auto_examples_decomposition_plot_ica_blind_source_separation.py`
    * :ref:`sphx_glr_auto_examples_decomposition_plot_ica_vs_pca.py`
    * :ref:`sphx_glr_auto_examples_decomposition_plot_faces_decomposition.py`


.. _NMF:

非负矩阵分解(NMF 或 NNMF)
===================================

NMF 与 Frobenius 规范
---------------------------

:class:`NMF` [1]_ 是一种替代的分解方法，假设数据和分量是非负数的。 
在数据矩阵不包含负值的情况下，可以插入 :class:`NMF` 而不是 :class:`PCA` 或其变体。 
通过优化 :math:`X` 与矩阵乘积 :math:`WH` 之间的距离 :math:`d` ，可以将样本 :math:`X` 分解为非负元素的两个矩阵 :math:`W` 和 :math:`H`。 
最广泛使用的距离函数是 Frobenius 方程的平方，这是欧几里德范数到矩阵的广泛的扩展:

.. math::
    d_{\mathrm{Fro}}(X, Y) = \frac{1}{2} ||X - Y||_{\mathrm{Fro}}^2 = \frac{1}{2} \sum_{i,j} (X_{ij} - {Y}_{ij})^2

与 :class:`PCA` 不同，通过叠加分量而不减去，以加法方式获得向量的表示。这种添加剂模型对于表示图像和文本是有效的。

在 [Hoyer, 2004] [2]_ 中已经观察到，当精心约束时，:class:`NMF` 可以产生数据集的基于零件的表示，导致可解释的模型。 
以下示例显示了与 PCA 特征面相比， :class:`NMF` 从 Olivetti 面数据集中的图像中发现的16个稀疏组件。

Unlike :class:`PCA`, the representation of a vector is obtained in an additive
fashion, by superimposing the components, without subtracting. Such additive
models are efficient for representing images and text.

.. |pca_img5| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_002.png
    :target: ../auto_examples/decomposition/plot_faces_decomposition.html
    :scale: 60%

.. |nmf_img5| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_003.png
    :target: ../auto_examples/decomposition/plot_faces_decomposition.html
    :scale: 60%

.. centered:: |pca_img5| |nmf_img5|

:attr:`init` 属性确定应用的初始化方法，这对方法的性能有很大的影响。 
:class:`NMF` 实现了非负双奇异值分解方法。NNDSVD [4]_ 基于两个 SVD 过程，一个近似数据矩阵，
使用单位秩矩阵的代数性质，得到的部分SVD因子的其他近似正部分。
基本的 NNDSVD 算法更适合稀疏分解。其变体 NNDSVDa（其中全部零设置为等于数据的所有元素的平均值）和 
NNDSVDar（其中零被设置为小于数据平均值的随机扰动除以100）在密集案件。

请注意，乘法更新 ('mu') 求解器无法更新初始化中存在的零，因此当与引入大量零的基本 NNDSVD 算法联合使用时，
会导致较差的结果; 在这种情况下，应优先使用 NNDSVDa 或 NNDSVDar。

也可以通过设置 :attr:`init="random"`，使用正确缩放的随机非负矩阵初始化 :class:`NMF` 。
整数种子或 ``RandomState`` 也可以传递给 :attr:`random_state` 以控制重现性。

在 :class:`NMF` 中，L1 和 L2 先验可以被添加到损失函数中以使模型正规化。 
L2 之前使用 Frobenius 范数，而L1 先验使用元素 L1 范数。与 :class:`ElasticNet` 一样，
我们使用 :attr:`l1_ratio` (:math:`\rho`) 参数和 :attr:`alpha` (:math:`\alpha`) 参数的正则化强度来控制 L1 和 L2 的组合。那么先修课程是:

.. math::
    \alpha \rho ||W||_1 + \alpha \rho ||H||_1
    + \frac{\alpha(1-\rho)}{2} ||W||_{\mathrm{Fro}} ^ 2
    + \frac{\alpha(1-\rho)}{2} ||H||_{\mathrm{Fro}} ^ 2

正则化目标函数为:

.. math::
    d_{\mathrm{Fro}}(X, WH)
    + \alpha \rho ||W||_1 + \alpha \rho ||H||_1
    + \frac{\alpha(1-\rho)}{2} ||W||_{\mathrm{Fro}} ^ 2
    + \frac{\alpha(1-\rho)}{2} ||H||_{\mathrm{Fro}} ^ 2

:class:`NMF` 正规化 W 和 H . 公共函数 :func:`non_negative_factorization` 允许通过 :attr:`regularization` 属性进行更精细的控制，并且可以仅将 W ，仅 H 或两者正规化。

NMF 具有 beta-divergence
------------------------------

如前所述，最广泛使用的距离函数是平方 Frobenius 范数，这是欧几里得范数到矩阵的明显延伸:

.. math::
    d_{\mathrm{Fro}}(X, Y) = \frac{1}{2} ||X - Y||_{Fro}^2 = \frac{1}{2} \sum_{i,j} (X_{ij} - {Y}_{ij})^2

其他距离函数可用于 NMF，例如（广义） Kullback-Leibler(KL) 发散，也称为 I-divergence:

.. math::
    d_{KL}(X, Y) = \sum_{i,j} (X_{ij} \log(\frac{X_{ij}}{Y_{ij}}) - X_{ij} + Y_{ij})

或者， Itakura-Saito(IS) 分歧:

.. math::
    d_{IS}(X, Y) = \sum_{i,j} (\frac{X_{ij}}{Y_{ij}} - \log(\frac{X_{ij}}{Y_{ij}}) - 1)

这三个距离函数是 beta-divergence 家族的特殊情况，其参数分别为 :math:`\beta = 2, 1, 0` [6]_ 。 beta-divergence 定义如下:

.. math::
    d_{\beta}(X, Y) = \sum_{i,j} \frac{1}{\beta(\beta - 1)}(X_{ij}^\beta + (\beta-1)Y_{ij}^\beta - \beta X_{ij} Y_{ij}^{\beta - 1})

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_beta_divergence_001.png
    :target: ../auto_examples/decomposition/plot_beta_divergence.html
    :align: center
    :scale: 75%

请注意，如果在 :math:`\beta \in (0; 1)` ，但是它可以分别连续扩展到 :math:`d_{KL}` 
和 :math:`d_{IS}` 的定义，则此定义无效。

:class:`NMF` 使用 Coordinate Descent ('cd') [5]_ 和乘法更新 ('mu') [6]_ 来实现两个求解器。 
'mu' 求解器可以优化每个 beta-divergence，包括 Frobenius 范数 (:math:`\beta=2`) ，
（广义） Kullback-Leibler 分歧 (:math:`\beta=1`) 和Itakura-Saito分歧（\ beta = 0） ）。
请注意，对于 :math:`\beta \in (1; 2)`，'mu' 求解器明显快于 :math:`\beta` 的其他值。
还要注意，使用负数（或0，即 'itakura-saito' ） :math:`\beta`，输入矩阵不能包含零值。

'cd' 求解器只能优化 Frobenius 规范。由于 NMF 的潜在非凸性，即使优化相同的距离函数，
不同的求解器也可能会收敛到不同的最小值。

NMF最适用于 ``fit_transform`` 方法，该方法返回矩阵W.矩阵 H 在 ``components_`` 属性中存储到拟合模型中;
方法 ``变换`` 将基于这些存储的组件分解新的矩阵 X_new::

    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> from sklearn.decomposition import NMF
    >>> model = NMF(n_components=2, init='random', random_state=0)
    >>> W = model.fit_transform(X)
    >>> H = model.components_
    >>> X_new = np.array([[1, 0], [1, 6.1], [1, 0], [1, 4], [3.2, 1], [0, 4]])
    >>> W_new = model.transform(X_new)

.. topic:: 示例:

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

潜在 Dirichlet 分配（LDA）
=================================

潜在 Dirichlet 分配是离散数据集（如文本语料库）的集合的生成概率模型。 
它也是一个主题模型，用于从文档集合中发现抽象主题。

LDA 的图形模型是一个 three-level 贝叶斯模型:

.. image:: ../images/lda_model_graph.png
   :align: center

当建模文本语料库时，该模型假设具有 :math:`D` 文档和 :math:`K` 主题的语料库的以下生成过程:

  1. 对于每个主题 :math:`k`，绘制 :math:`\beta_k \sim \mathrm{Dirichlet}(\eta),\: k =1...K`

  2. 对于每个文档 :math:`d`，绘制 :math:`\theta_d \sim \mathrm{Dirichlet}(\alpha), \: d=1...D`

  3. 对于文档 :math:`d` 中的每个单词 :math:`i`:

    a. 绘制主题索引 :math:`z_{di} \sim \mathrm{Multinomial}(\theta_d)`
    b. 绘制观察词 :math:`w_{ij} \sim \mathrm{Multinomial}(beta_{z_{di}}.)`

对于参数估计，后验分布为:

.. math::
  p(z, \theta, \beta |w, \alpha, \eta) =
    \frac{p(z, \theta, \beta|\alpha, \eta)}{p(w|\alpha, \eta)}

由于后验分布难以处理，变分贝叶斯方法使用更简单的分布 :math:`q(z,\theta,\beta | \lambda, \phi, \gamma)` 近似，
并且优化了这些变分参数  :math:`\lambda`, :math:`\phi`, :math:`\gamma` 最大化证据下限 (ELBO):

.. math::
  \log\: P(w | \alpha, \eta) \geq L(w,\phi,\gamma,\lambda) \overset{\triangle}{=}
    E_{q}[\log\:p(w,z,\theta,\beta|\alpha,\eta)] - E_{q}[\log\:q(z, \theta, \beta)]

最大化 ELBO 相当于最小化 :math:`q(z,\theta,\beta)` 和真实后 :math:`p(z, \theta, \beta |w, \alpha, \eta)` 之间的 Kullback-Leibler(KL) 发散。

:class:`LatentDirichletAllocation` 实现在线变分贝叶斯算法，支持在线和批量更新方法。
批处理方法在每次完全传递数据后更新变分变量，联机方法从小批量数据点更新变分变量。

.. note::
  虽然在线方法保证收敛到局部最优点，最优点的质量和收敛速度可能取决于小批量大小和学习率设置相关的属性。

当 :class:`LatentDirichletAllocation` 应用于 "document-term" 矩阵时，矩阵将被分解为 "topic-term" 矩阵和 "document-topic" 矩阵。
虽然 "topic-term" 矩阵在模型中被存储为 :attr:`components_` ，但是可以通过变换方法计算 "document-topic" 矩阵。

:class:`LatentDirichletAllocation` 还实现了  ``partial_fit`` 方法。当数据可以顺序提取时使用.

.. topic:: 示例:

    * :ref:`sphx_glr_auto_examples_applications_plot_topics_extraction_with_nmf_lda.py`

.. topic:: 参考:

    * `"Latent Dirichlet Allocation"
      <https://www.cs.princeton.edu/~blei/papers/BleiNgJordan2003.pdf>`_
      D. Blei, A. Ng, M. Jordan, 2003

    * `"Online Learning for Latent Dirichlet Allocation”
      <https://www.cs.princeton.edu/~blei/papers/HoffmanBleiBach2010b.pdf>`_
      M. Hoffman, D. Blei, F. Bach, 2010

    * `"Stochastic Variational Inference"
      <http://www.columbia.edu/~jwp2128/Papers/HoffmanBleiWangPaisley2013.pdf>`_
      M. Hoffman, D. Blei, C. Wang, J. Paisley, 2013
