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


PCA 用于对一组连续正交分量中的多变量数据集进行方差最大方向的分解。
在 scikit-learn 中， :class:`PCA` 被实现为一个变换对象， 通过 ``fit`` 方法可以降维成 `n` 个成分，
并且可以将新的数据投影(project, 亦可理解为分解)到这些成分中。

可选参数 ``whiten=True`` 使得可以将数据投影到奇异（singular）空间上，同时将每个成分缩放到单位方差。
如果下游模型对信号的各向同性作出强烈的假设，这通常是有用的，例如，使用RBF内核的 SVM 算法和 K-Means 聚类算法。

以下是iris数据集的一个示例，该数据集包含4个特征， 通过PCA降维后投影到方差最大的二维空间上：

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_pca_vs_lda_001.png
    :target: ../auto_examples/decomposition/plot_pca_vs_lda.html
    :align: center
    :scale: 75%


 :class:`PCA` 对象还提供了 PCA 的概率解释， 其可以基于其解释的方差量给出数据的可能性。
可以通过在交叉验证（cross-validation）中使用 `score` 方法来实现：

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
:class:`IncrementalPCA` 对象使用不同的处理形式使之允许部分计算，
这一形式几乎和 :class:`PCA` 以小型批处理方式处理数据的方法完全匹配。
:class:`IncrementalPCA` 可以通过以下方式实现核外（out-of-core）主成分分析：

 * 使用 ``partial_fit`` 方法从本地硬盘或网络数据库中以此获取数据块。

 * 通过 ``numpy.memmap`` 在一个 memory mapped file 上使用 fit 方法。

 :class:`IncrementalPCA` 仅存储成分和噪声方差的估计值，并按顺序递增地更新解释方差比（explained_variance_ratio_）。
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

通过丢弃具有较低奇异值的奇异向量成分，将数据降维到低维空间并保留大部分方差是非常有意义的。

例如，如果我们使用64x64像素的灰度级图像进行人脸识别，数据的维数为4096，
在这样大的数据上训练含RBF内核的支持向量机是很慢的。
此外我们知道数据本质上的维度远低于4096，因为人脸的所有照片都看起来有点相似。
样本位于许多的很低维度（例如约200维）。PCA算法可以用于线性变换数据，同时降低维数并同时保留大部分方差。

在这种情况下，使用可选参数 ``svd_solver='randomized'`` 的 :class:`PCA` 是非常有用的。
因为我们将要丢弃大部分奇异值，所以对我们将保留并实际执行变换的奇异向量进行近似估计的有限的计算更有效。

例如：以下显示了来自 Olivetti 数据集的 16 个样本肖像（以 0.0 为中心）。
右侧是前 16 个奇异向量重画为肖像。因为我们只需要使用大小为 :math:`n_{samples} = 400`
和 :math:`n_{features} = 64 \times 64 = 4096` 的数据集的前 16 个奇异向量, 使得计算时间小于 1 秒。

.. |orig_img| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_001.png
   :target: ../auto_examples/decomposition/plot_faces_decomposition.html
   :scale: 60%

.. |pca_img| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_002.png
   :target: ../auto_examples/decomposition/plot_faces_decomposition.html
   :scale: 60%

.. centered:: |orig_img| |pca_img|

注意：使用可选参数 ``svd_solver='randomized'`` ，在 :class:`PCA` 中我们还需要给出输入低维空间大小 ``n_components`` 。

如果我们注意到： :math:`n_{\max} = \max(n_{\mathrm{samples}}, n_{\mathrm{features}})` 且
:math:`n_{\min} = \min(n_{\mathrm{samples}}, n_{\mathrm{features}})`,
对于PCA中实施的确切方式，随机 :class:`PCA` 的时间复杂度是：:math:`O(n_{\max}^2 \cdot n_{\mathrm{components}})` ，
而不是 :math:`O(n_{\max}^2 \cdot n_{\min})` 。

对于确切的方式，随机 :class:`PCA` 的内存占用量正比于 :math:`2 \cdot n_{\max} \cdot n_{\mathrm{components}}` ，
而不是 :math:`n_{\max}\cdot n_{\min}`

注意：选择参数 ``svd_solver='randomized'`` 的 :class:`PCA`，在执行 ``inverse_transform`` 时，
并不是 ``transform`` 的确切的逆变换操作（即使 参数设置为默认的 ``whiten=False``）

.. topic:: 例子:

    * :ref:`sphx_glr_auto_examples_applications_plot_face_recognition.py`
    * :ref:`sphx_glr_auto_examples_decomposition_plot_faces_decomposition.py`

.. topic:: 参考文献:

    * `"Finding structure with randomness: Stochastic algorithms for
      constructing approximate matrix decompositions"
      <http://arxiv.org/abs/0909.4061>`_
      Halko, et al., 2009


.. _kernel_PCA:

核 PCA
----------------

:class:`KernelPCA` 是 PCA 的扩展，通过使用核方法实现非线性降维（dimensionality reduction） (参阅 :ref:`metrics`)。
它具有许多应用，包括去噪, 压缩和结构化预测（ structured prediction ） (kernel dependency estimation（内核依赖估计）)。
 :class:`KernelPCA` 支持 ``transform`` 和 ``inverse_transform`` 。

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_kernel_pca_001.png
    :target: ../auto_examples/decomposition/plot_kernel_pca.html
    :align: center
    :scale: 75%

.. topic:: 示例:

    * :ref:`sphx_glr_auto_examples_decomposition_plot_kernel_pca.py`


.. _SparsePCA:

稀疏主成分分析 ( SparsePCA 和 MiniBatchSparsePCA )
--------------------------------------------------------------------

:class:`SparsePCA` 是 PCA 的一个变体，目的是提取能最好地重建数据的稀疏组分集合。

小批量稀疏 PCA ( :class:`MiniBatchSparsePCA` ) 是一个 :class:`SparsePCA` 的变种，它速度更快但准确度有所降低。对于给定的迭代次数，通过迭代该组特征的小块来达到速度的增加。

Principal component analysis（主成分分析） (:class:`PCA`) 的缺点在于，通过该方法提取的成分具有唯一的密度表达式，即当表示为原始变量的线性组合时，它们具有非零系数，使之难以解释。在许多情况下，真正的基础组件可以更自然地想象为稀疏向量; 例如在面部识别中，每个组件可能自然地映射到面部的某个部分。

稀疏的主成分产生更简洁、可解释的表达式，明确强调了样本之间的差异性来自哪些原始特征。

以下示例说明了使用稀疏 PCA 提取 Olivetti 人脸数据集中的 16 个组分。可以看出正则化项产生了许多零。此外，数据的自然结构导致了非零系数垂直相邻 （vertically adjacent）。该模型不会在数学上强制执行: 每个组分都是一个向量  :math:`h \in \mathbf{R}^{4096}`,除非人性化地的可视化为 64x64 像素的图像，否则没有垂直相邻性的概念。
下面显示的组分看起来局部化（appear local)是数据的内在结构的影响，这种局部模式使重建误差最小化。有一种考虑到邻接性和不同结构类型的导致稀疏的规范（sparsity-inducing norms）,参见 [Jen09]_ 对这种方法进行了解。
有关如何使用稀疏 PCA 的更多详细信息，请参阅下面的示例部分。
更多关于 Sparse PCA 使用的内容，参见示例部分，如下：


.. |spca_img| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_005.png
   :target: ../auto_examples/decomposition/plot_faces_decomposition.html
   :scale: 60%

.. centered:: |pca_img| |spca_img|

请注意，有多种不同的计算稀疏PCA 问题的公式。 这里使用的方法基于 [Mrl09]_ 。优化问题的解决是一个带有惩罚项（L1范数的） :math:`\ell_1` 的一个 PCA 问题（dictionary learning（字典学习））:

.. math::
   (U^*, V^*) = \underset{U, V}{\operatorname{arg\,min\,}} & \frac{1}{2}
                ||X-UV||_2^2+\alpha||V||_1 \\
                \text{subject to\,} & ||U_k||_2 = 1 \text{ for all }
                0 \leq k < n_{components}


导致稀疏（sparsity-inducing）的 :math:`\ell_1` 规范也可以避免当训练样本很少时从噪声中学习成分。可以通过超参数 ``alpha`` 来调整惩罚程度（从而减少稀疏度）。值较小会导致温和的正则化因式分解，而较大的值将许多系数缩小到零。

.. note::

  虽然本着在线算法的精神， :class:`MiniBatchSparsePCA` 类不实现 ``partial_fit`` , 因为在线算法沿特征方向，而不是样本方向。

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


截断奇异值分解和隐语义分析
=========================================================

:class:`TruncatedSVD` 实现了一个奇异值分解（SVD）的变体，它只计算 :math:`k` 个最大的奇异值，其中 :math:`k` 是用户指定的参数。

当截断的 SVD被应用于 term-document矩阵（由 ``CountVectorizer`` 或 ``TfidfVectorizer`` 返回）时，这种转换被称为 `latent semantic analysis <http://nlp.stanford.edu/IR-book/pdf/18lsi.pdf>`_ (LSA), 因为它将这样的矩阵转换为低纬度的 "semantic（语义）" 空间。
特别地是 LSA 能够抵抗同义词和多义词的影响（两者大致意味着每个单词有多重含义），这导致 term-document 矩阵过度稀疏，并且在诸如余弦相似性的度量下表现出差的相似性。

.. note::
    LSA 也被称为隐语义索引 LSI，尽管严格地说它是指在持久索引（persistent indexes）中用于信息检索的目的。

数学表示中， 训练样本 :math:`X` 用截断的SVD产生一个低秩的（ low-rank）近似值 :math:`X` :

.. math::
    X \approx X_k = U_k \Sigma_k V_k^\top

在这个操作之后，:math:`U_k \Sigma_k^\top` 是转换后的训练集，其中包括 :math:`k` 个特征（在 API 中被称为 ``n_components`` ）。

还需要转换一个测试集 :math:`X`, 我们乘以 :math:`V_k`: 

.. math::
    X' = X V_k

.. note::
    
    自然语言处理(NLP) 和信息检索(IR) 文献中的 LSA 的大多数处理方式是交换矩阵 :math:`X` 的坐标轴,使其具有 ``n_features`` × ``n_samples`` 的形状。
    我们以 scikit-learn API 相匹配的不同方式呈现 LSA, 但是找到的奇异值是相同的。

:class:`TruncatedSVD` 非常类似于 :class:`PCA`, 但不同之处在于它工作在样本矩阵 :math:`X` 而不是它们的协方差矩阵。
当从特征值中减去 :math:`X` 的每列（每个特征per-feature）的均值时，在得到的矩阵上应用 truncated SVD 相当于 PCA 。
实际上，这意味着 :class:`TruncatedSVD` 转换器（transformer）接受 ``scipy.sparse`` 矩阵，而不需要对它们进行密集（density），因为即使对于中型大小文档的集合，密集化 （densifying）也可能填满内存。

虽然 :class:`TruncatedSVD` 转换器（transformer）可以在任何（稀疏的）特征矩阵上工作，但还是建议在 LSA/document 处理设置中，在 tf–idf 矩阵上的原始频率计数使用它。
特别地，应该打开子线性缩放（sublinear scaling）和逆文档频率（inverse document frequency） (``sublinear_tf=True, use_idf=True``) 以使特征值更接近于高斯分布，补偿 LSA 对文本数据的错误假设。

.. topic:: 示例:

   * :ref:`sphx_glr_auto_examples_text_document_clustering.py`

.. topic:: 参考文献:

  * Christopher D. Manning, Prabhakar Raghavan and Hinrich Schütze (2008),
    *Introduction to Information Retrieval*, Cambridge University Press,
    chapter 18: `Matrix decompositions & latent semantic indexing
    <http://nlp.stanford.edu/IR-book/pdf/18lsi.pdf>`_


.. _DictionaryLearning:

词典学习
==================

.. _SparseCoder:

带有预计算词典的稀疏编码
---------------------------------

:class:`SparseCoder` 对象是一个估计器 （estimator），可以用来将信号转换成一个固定的预计算的词典内原子（atoms）的稀疏线性组合（sparse linear combination），如离散小波基（ discrete wavelet basis ） 。
因此，该对象不实现 ``fit`` 方法。该转换相当于一个稀疏编码问题: 将数据的表示为尽可能少的词典原子的线性组合。
词典学习的所有变体实现以下变换方法，可以通过 ``transform_method`` 初始化参数进行控制: 

* Orthogonal matching pursuit(追求正交匹配) (:ref:`omp`)

* Least-angle regression (最小角度回归)(:ref:`least_angle_regression`)

* Lasso computed by least-angle regression(最小角度回归的Lasso 计算)

* Lasso using coordinate descent ( 使用坐标下降的Lasso)(:ref:`lasso`)

* Thresholding(阈值)

阈值方法速度非常快，但是不能产生精确的重建。
它们在分类任务的文献中已被证明是有用的。对于图像重建任务，追求正交匹配可以产生最精确、无偏的重建。 

词典学习对象通过 ``split_code`` 参数提供稀疏编码结果中的正值和负值分离的可能性。当使用词典学习来提取将用于监督学习的特征时，这是有用的，因为它允许学习算法将不同的权重从正加载（loading）分配给相应的负加载的特定原子。

单个样本的分割编码具有长度 ``2 * n_components`` ，并使用以下规则构造: 首先，计算长度为 ``n_components`` 的常规编码。然后， ``split_code`` 的第一个 ``n_components`` 条目将用正常编码向量的正部分填充。分割编码的第二部分用编码向量的负部分填充，只有一个正号。因此， split_code 是非负的。 


.. topic:: 示例:

    * :ref:`sphx_glr_auto_examples_decomposition_plot_sparse_coding.py`


通用词典学习
--------------------

词典学习( :class:`DictionaryLearning` ) 是一个矩阵因式分解问题，相当于找到一个在拟合数据的稀疏编码中表现良好的（通常是过完备的（overcomplete））词典。

将数据表示为来自过完备词典的原子的稀疏组合被认为是哺乳动物初级视觉皮层的工作方式。
因此，应用于图像补丁的词典学习已被证明在诸如图像完成、修复和去噪，以及有监督的识别图像处理任务中表现良好的结果。

词典学习是通过交替更新稀疏编码来解决的优化问题，作为解决多个 Lasso 问题的一个解决方案，考虑到字典固定，然后更新字典以最好地适合稀疏编码。

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


在使用这样一个过程来拟合词典之后，变换只是一个稀疏的编码步骤，与所有的词典学习对象共享相同的实现。(参见 :ref:`SparseCoder`)。

以下图像显示了字典学习是如何从浣熊脸部的部分图像中提取的4x4像素图像补丁中进行词典学习的。

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

小批量字典学习
----------------------

:class:`MiniBatchDictionaryLearning` 实现了更快、更适合大型数据集的字典学习算法，其运行速度更快，但准确度有所降低。

默认情况下，:class:`MiniBatchDictionaryLearning` 将数据分成小批量，并通过在指定次数的迭代中循环使用小批量，以在线方式进行优化。但是，目前它没有实现停止条件。

估计器还实现了  ``partial_fit``, 它通过在一个小批处理中仅迭代一次来更新字典。 当在线学习的数据从一开始就不容易获得，或者数据超出内存时，可以使用这种迭代方法。

.. currentmodule:: sklearn.cluster

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_dict_face_patches_001.png
    :target: ../auto_examples/cluster/plot_dict_face_patches.html
    :scale: 50%
    :align: right

.. topic:: **字典学习聚类**

   注意，当使用字典学习来提取表示（例如，用于稀疏编码）时，聚类可以是学习字典的良好中间方法。 
   例如，:class:`MiniBatchKMeans` 估计器能高效计算并使用 ``partial_fit`` 方法实现在线学习。

   示例: 在线学习面部部分的字典 :ref:`sphx_glr_auto_examples_cluster_plot_dict_face_patches.py`

.. currentmodule:: sklearn.decomposition

.. _FA:

因子分析
===============

在无监督的学习中，我们只有一个数据集 :math:`X = \{x_1, x_2, \dots, x_n\}`. 
这个数据集如何在数学上描述？ :math:`X` 的一个非常简单的连续隐变量模型

.. math:: x_i = W h_i + \mu + \epsilon

矢量 :math:`h_i` 被称为 "隐性的"，因为它是不可观察的。 
:math:`\epsilon` 被认为是符合高斯分布的噪声项，平均值为 0，协方差为 :math:`\Psi` （即 :math:`\epsilon \sim \mathcal{N}(0, \Psi)`）， 
:math:`\mu` 是偏移向量。 这样一个模型被称为 "生成的"，因为它描述了如何从 :math:`h_i` 生成 :math:`x_i` 。
如果我们使用所有的 :math:`x_i` 作为列来形成一个矩阵 :math:`\mathbf{X}` ，并将所有的 :math:`h_i` 作为矩阵 :math:`\mathbf{H}` 的列，
那么我们可以写（适当定义的 :math:`\mathbf{M}` 和 :math:`\mathbf{E}` ）:

.. math::
    \mathbf{X} = W \mathbf{H} + \mathbf{M} + \mathbf{E}

换句话说，我们 *分解* 矩阵 :math:`\mathbf{X}`.
如果给出 :math:`h_i`，上述方程自动地表示以下概率解释：

.. math:: p(x_i|h_i) = \mathcal{N}(Wh_i + \mu, \Psi)

对于一个完整的概率模型，我们还需要隐变量 :math:`h` 的先验分布。 
最直接的假设（基于高斯分布的良好性质）是 :math:`h \sim \mathcal{N}(0, \mathbf{I})`. 这产生一个高斯分布作为 :math:`x` 的边际分布:

.. math:: p(x) = \mathcal{N}(\mu, WW^T + \Psi)

现在，在没有任何进一步假设的前提下，隐变量 :math:`h` 是多余的 -- :math:`x` 完全可以用均值和协方差来建模。 
我们需要对这两个参数之一进行更具体的构造。 一个简单的附加假设是将误差协方差 :math:`\Psi` 构造成如下:

* :math:`\Psi = \sigma^2 \mathbf{I}`: 这个假设能推导出 :class:`PCA` 的概率模型。

* :math:`\Psi = \mathrm{diag}(\psi_1, \psi_2, \dots, \psi_n)`: 这个模型称为 :class:`FactorAnalysis`, 一个经典的统计模型。 矩阵W有时称为 "因子加载矩阵"。

两个模型基都基于高斯分布是低阶协方差矩阵的假设。 
因为这两个模型都是概率性的，所以它们可以集成到更复杂的模型中，
例如因子分析器的混合。如果隐变量基于非高斯分布，则得到完全不同的模型（例如， :class:`FastICA` ）。

因子分析 *可以* 产生与 :class:`PCA`类似的成分（例如其加载矩阵的列）。 
然而，这些成分没有通用的性质（例如它们是否是正交的）:

.. |pca_img3| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_002.png
    :target: ../auto_examples/decomposition/plot_faces_decomposition.html
    :scale: 60%

.. |fa_img3| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_009.png
    :target: ../auto_examples/decomposition/plot_faces_decomposition.html
    :scale: 60%

.. centered:: |pca_img3| |fa_img3|

因子分析( :class:`PCA` ) 的主要优点是可以独立地对输入空间的每个方向（异方差噪声）的方差建模:

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_008.png
    :target: ../auto_examples/decomposition/plot_faces_decomposition.html
    :align: center
    :scale: 75%

在异方差噪声存在的情况下，这可以比概率 PCA 作出更好的模型选择:

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_pca_vs_fa_model_selection_002.png
    :target: ../auto_examples/decomposition/plot_pca_vs_fa_model_selection.html
    :align: center
    :scale: 75%


.. topic:: 示例:

    * :ref:`sphx_glr_auto_examples_decomposition_plot_pca_vs_fa_model_selection.py`

.. _ICA:

独立成分分析（ICA）
=========================

独立分量分析将多变量信号分解为独立性最强的加性子组件。 
它通过 :class:`Fast ICA <FastICA>` 算法在 scikit-learn 中实现。 
ICA 通常不用于降低维度，而是用于分离叠加信号。 
由于 ICA 模型不包括噪声项，因此要使模型正确，必须使用白化。 
这可以在内部调节白化参数或手动使用 PCA 的一种变体。

ICA 通常用于分离混合信号（称为 *盲源分离* 的问题），如下例所示:

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_ica_blind_source_separation_001.png
    :target: ../auto_examples/decomposition/plot_ica_blind_source_separation.html
    :align: center
    :scale: 60%


ICA 也可以用于具有稀疏子成分的非线性分解:

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

NMF 与 Frobenius 范数
---------------------------

:class:`NMF` [1]_ 是在数据和分量是非负情况下的另一种降维方法。 
在数据矩阵不包含负值的情况下，可以插入 :class:`NMF` 而不是 :class:`PCA` 或其变体。 
通过优化 :math:`X` 与矩阵乘积 :math:`WH` 之间的距离 :math:`d` ，可以将样本 :math:`X` 分解为两个非负矩阵 :math:`W` 和 :math:`H`。 
最广泛使用的距离函数是 Frobenius 平方范数，它是欧几里德范数到矩阵的推广:

.. math::
    d_{\mathrm{Fro}}(X, Y) = \frac{1}{2} ||X - Y||_{\mathrm{Fro}}^2 = \frac{1}{2} \sum_{i,j} (X_{ij} - {Y}_{ij})^2

与 :class:`PCA` 不同，通过叠加分量而不减去，以加法方式获得向量的表示。这种加性模型对于表示图像和文本是有效的。

 [Hoyer, 2004] [2]_ 研究表明，当处于一定约束时，:class:`NMF` 可以产生数据集基于某子部分的表示，从而获得可解释的模型。 
以下示例展示了与 PCA 特征面相比， :class:`NMF` 从 Olivetti 面部数据集中的图像中发现的16个稀疏组件。

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

:attr:`init` 属性确定了应用的初始化方法，这对方法的性能有很大的影响。 
:class:`NMF` 实现了非负双奇异值分解方法。NNDSVD [4]_ 基于两个 SVD 过程，一个近似数据矩阵，
使用单位秩矩阵的代数性质，得到的部分SVD因子的其他近似正部分。
基本的 NNDSVD 算法更适合稀疏分解。其变体 NNDSVDa（全部零值替换为所有元素的平均值）和 
NNDSVDar（零值替换为比数据平均值除以100小的随机扰动）在稠密情况时推荐使用。

请注意，乘法更新 ('mu') 求解器无法更新初始化中存在的零，因此当与引入大量零的基本 NNDSVD 算法联合使用时，
会导致较差的结果; 在这种情况下，应优先使用 NNDSVDa 或 NNDSVDar。

也可以通过设置 :attr:`init="random"`，使用正确缩放的随机非负矩阵初始化 :class:`NMF` 。
整数种子或 ``RandomState`` 也可以传递给 :attr:`random_state` 以控制重现性。

在 :class:`NMF` 中，L1 和 L2 先验可以被添加到损失函数中以使模型正规化。 
L2 先验使用 Frobenius 范数，而L1 先验使用 L1 范数。与 :class:`ElasticNet` 一样，
我们通过 :attr:`l1_ratio` (:math:`\rho`) 参数和正则化强度参数 :attr:`alpha` (:math:`\alpha`) 来控制 L1 和 L2 的组合。那么先验项是:

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

:class:`NMF` 正则化 W 和 H . 公共函数 :func:`non_negative_factorization` 允许通过 :attr:`regularization` 属性进行更精细的控制，将 仅W ，仅H 或两者正规化。

具有 beta-divergence 的 NMF
----------------------------------

如前所述，最广泛使用的距离函数是平方 Frobenius 范数，这是欧几里得范数到矩阵的推广:

.. math::
    d_{\mathrm{Fro}}(X, Y) = \frac{1}{2} ||X - Y||_{Fro}^2 = \frac{1}{2} \sum_{i,j} (X_{ij} - {Y}_{ij})^2

其他距离函数可用于 NMF，例如（广义） Kullback-Leibler(KL) 散度，也称为 I-divergence:

.. math::
    d_{KL}(X, Y) = \sum_{i,j} (X_{ij} \log(\frac{X_{ij}}{Y_{ij}}) - X_{ij} + Y_{ij})

或者， Itakura-Saito(IS) divergence:

.. math::
    d_{IS}(X, Y) = \sum_{i,j} (\frac{X_{ij}}{Y_{ij}} - \log(\frac{X_{ij}}{Y_{ij}}) - 1)

这三个距离函数是 beta-divergence 函数族的特殊情况，其参数分别为 :math:`\beta = 2, 1, 0` [6]_ 。 beta-divergence 定义如下:

.. math::
    d_{\beta}(X, Y) = \sum_{i,j} \frac{1}{\beta(\beta - 1)}(X_{ij}^\beta + (\beta-1)Y_{ij}^\beta - \beta X_{ij} Y_{ij}^{\beta - 1})

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_beta_divergence_001.png
    :target: ../auto_examples/decomposition/plot_beta_divergence.html
    :align: center
    :scale: 75%

请注意，在 :math:`\beta \in (0; 1)` 上定义无效，仅仅在 :math:`d_{KL}` 
和 :math:`d_{IS}` 的上可以分别连续扩展。

:class:`NMF` 使用 Coordinate Descent ('cd') [5]_ 和乘法更新 ('mu') [6]_ 来实现两个求解器。 
'mu' 求解器可以优化每个 beta-divergence，包括 Frobenius 范数 (:math:`\beta=2`) ，
（广义） Kullback-Leibler divergence (:math:`\beta=1`) 和Itakura-Saito divergence（\ beta = 0） ）。
请注意，对于 :math:`\beta \in (1; 2)`，'mu' 求解器明显快于 :math:`\beta` 的其他值。
还要注意，使用负数（或0，即 'itakura-saito' ） :math:`\beta`，输入矩阵不能包含零值。

'cd' 求解器只能优化 Frobenius 范数。由于 NMF 的潜在非凸性，即使优化相同的距离函数，
不同的求解器也可能会收敛到不同的最小值。

NMF最适用于 ``fit_transform`` 方法，该方法返回矩阵W.矩阵 H 被 ``components_`` 属性中存储到拟合模型中;
方法 ``transform`` 将基于这些存储的组件分解新的矩阵 X_new::

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

隐 Dirichlet 分配（LDA）
=================================

隐 Dirichlet 分配是离散数据集（如文本语料库）的集合的生成概率模型。 
它也是一个主题模型，用于从文档集合中发现抽象主题。

LDA 的图形模型是一个三层贝叶斯模型:

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

由于后验分布难以处理，变体贝叶斯方法使用更简单的分布 :math:`q(z,\theta,\beta | \lambda, \phi, \gamma)` 近似，
并且优化了这些变体参数  :math:`\lambda`, :math:`\phi`, :math:`\gamma` 最大化Evidence Lower Bound (ELBO):

.. math::
  \log\: P(w | \alpha, \eta) \geq L(w,\phi,\gamma,\lambda) \overset{\triangle}{=}
    E_{q}[\log\:p(w,z,\theta,\beta|\alpha,\eta)] - E_{q}[\log\:q(z, \theta, \beta)]

最大化 ELBO 相当于最小化 :math:`q(z,\theta,\beta)` 和后验 :math:`p(z, \theta, \beta |w, \alpha, \eta)` 之间的 Kullback-Leibler(KL) 散度。

:class:`LatentDirichletAllocation` 实现在线变体贝叶斯算法，支持在线和批量更新方法。
批处理方法在每次完全传递数据后更新变分变量，在线方法从小批量数据点中更新变体变量。

.. note::
  虽然在线方法保证收敛到局部最优点，最优点的质量和收敛速度可能取决于与小批量大小和学习率相关的属性。

当 :class:`LatentDirichletAllocation` 应用于 "document-term" 矩阵时，矩阵将被分解为 "topic-term" 矩阵和 "document-topic" 矩阵。
虽然 "topic-term" 矩阵在模型中被存储为 :attr:`components_` ，但是可以通过 ``transform`` 方法计算 "document-topic" 矩阵。

:class:`LatentDirichletAllocation` 还实现了  ``partial_fit`` 方法。这可用于当数据被顺序提取时.

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
