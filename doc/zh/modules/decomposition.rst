.. _decompositions:


=================================================================
Decomposing signals in components (matrix factorization problems)
分解成分中的信号（矩阵分解问题）
=================================================================

.. currentmodule:: sklearn.decomposition


.. _PCA:


Principal component analysis (PCA)
主成分分析（PCA）
==================================

Exact PCA and probabilistic interpretation
准确的PCA和概率解释
------------------------------------------


PCA is used to decompose a multivariate dataset in a set of successive
orthogonal components that explain a maximum amount of the variance. In
scikit-learn, :class:`PCA` is implemented as a *transformer* object
that learns :math:`n` components in its ``fit`` method, and can be used on new
data to project it on these components.
PCA 通过引入最大数量的变量来对一组连续正交分量中的多变量数据集进行降维。
在 scikit-learn 中，:class:`PCA` 被实现为一个变换对象， 通过 ``fit`` 方法可以降维成 `n` 个成分，
并且可以将新的数据投射(project, 亦可理解为 分解)到这些成分中。

The optional parameter ``whiten=True`` makes it possible to
project the data onto the singular space while scaling each component
to unit variance. This is often useful if the models down-stream make
strong assumptions on the isotropy of the signal: this is for example
the case for Support Vector Machines with the RBF kernel and the K-Means
clustering algorithm.
可选参数 ``whiten=True`` 使得可以将数据投影到单个空间上，同时将每个成分缩放到单位方差。
如果下游模型对信号的各向同性作出强烈的假设，这通常是有用的。
例如，使用RBF内核的 SVM(Support Vector Machines) 算法和 K-Means 聚类算法。


Below is an example of the iris dataset, which is comprised of 4
features, projected on the 2 dimensions that explain most variance:
以下是iris数据集的一个示例， 该数据集包含4个特征， 通过PCA降维后投影到二维空间上：

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_pca_vs_lda_001.png
    :target: ../auto_examples/decomposition/plot_pca_vs_lda.html
    :align: center
    :scale: 75%


The :class:`PCA` object also provides a
probabilistic interpretation of the PCA that can give a likelihood of
data based on the amount of variance it explains. As such it implements a
`score` method that can be used in cross-validation:
:class:`PCA` 对象还提供了 PCA 的概率解释， 其可以基于降维后的变量数量来给出数据的可能性。
可以通过在 cross-validation 中使用 `score` 方法来实现。

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_pca_vs_fa_model_selection_001.png
    :target: ../auto_examples/decomposition/plot_pca_vs_fa_model_selection.html
    :align: center
    :scale: 75%


.. topic:: Examples例子:

    * :ref:`sphx_glr_auto_examples_decomposition_plot_pca_vs_lda.py`
    * :ref:`sphx_glr_auto_examples_decomposition_plot_pca_vs_fa_model_selection.py`


.. _IncrementalPCA:

增量PCA (Incremental PCA)
---------------

The :class:`PCA` object is very useful, but has certain limitations for
large datasets. The biggest limitation is that :class:`PCA` only supports
batch processing, which means all of the data to be processed must fit in main
memory. The :class:`IncrementalPCA` object uses a different form of
processing and allows for partial computations which almost
exactly match the results of :class:`PCA` while processing the data in a
minibatch fashion. :class:`IncrementalPCA` makes it possible to implement
out-of-core Principal Component Analysis either by:

:class:`PCA` 对象非常有用, 但对大型数据集有一定的限制。
最大的限制是 :class:`PCA` 仅支持批处理，这意味着所有要处理的数据必须适合主内存。
:class:`IncrementalPCA` 对象使用不同的处理形式，并且允许部分计算，
几乎完全匹配 :class:`PCA` 以小型批处理方式处理数据的结果。
:class:`IncrementalPCA` 可以通过以下方式实现核心主成分分析：

 * Using its ``partial_fit`` method on chunks of data fetched sequentially
   from the local hard drive or a network database.
   使用 ``partial_fit`` 方法从本地硬盘或网络数据库中以此获取数据块。

 * Calling its fit method on a memory mapped file using ``numpy.memmap``.
 通过 ``numpy.memmap`` 在一个 memory mapped file 上使用 fit 方法
。

:class:`IncrementalPCA` only stores estimates of component and noise variances,
in order update ``explained_variance_ratio_`` incrementally. This is why
memory usage depends on the number of samples per batch, rather than the
number of samples to be processed in the dataset.

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


.. topic:: Examples例子:

    * :ref:`sphx_glr_auto_examples_decomposition_plot_incremental_pca.py`


.. _RandomizedPCA:

PCA using randomized SVD
PCA 使用随机SVD
------------------------

It is often interesting to project data to a lower-dimensional
space that preserves most of the variance, by dropping the singular vector
of components associated with lower singular values.
通过丢弃具有较低奇异值的奇异向量成分，将数据降维到低维空间并保留大部分变量属性是非常有意义的。

For instance, if we work with 64x64 pixel gray-level pictures
for face recognition,
the dimensionality of the data is 4096 and it is slow to train an
RBF support vector machine on such wide data. Furthermore we know that
the intrinsic dimensionality of the data is much lower than 4096 since all
pictures of human faces look somewhat alike.
The samples lie on a manifold of much lower
dimension (say around 200 for instance). The PCA algorithm can be used
to linearly transform the data while both reducing the dimensionality
and preserve most of the explained variance at the same time.

例如，如果我们使用64x64像素的灰度级图像进行面部识别，数据的维数为4096，
并且在这样大的数据上训练含RBF内核的支持向量机是很慢的。
此外，我们知道数据的固有维度远低于4096，因为人脸的所有照片都看起来有点相似。
样本依赖于多种低维度（例如约200种）。
PCA算法可以用于线性变换数据，同时降低维数并同时保留大部分变量。

The class :class:`PCA` used with the optional parameter
``svd_solver='randomized'`` is very useful in that case: since we are going
to drop most of the singular vectors it is much more efficient to limit the
computation to an approximated estimate of the singular vectors we will keep
to actually perform the transform.

在这种情况下，具有可选参数 ``svd_solver='randomized'`` 的 :class:`PCA` 是非常有用的。
因为我们将要丢弃大部分奇异值，所以将计算限制为单个向量的近似估计更有效，我们将保持实际执行变换。

For instance, the following shows 16 sample portraits (centered around
0.0) from the Olivetti dataset. On the right hand side are the first 16
singular vectors reshaped as portraits. Since we only require the top
16 singular vectors of a dataset with size :math:`n_{samples} = 400`
and :math:`n_{features} = 64 \times 64 = 4096`, the computation time is
less than 1s:
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

Note: with the optional parameter ``svd_solver='randomized'``, we also
need to give :class:`PCA` the size of the lower-dimensional space
``n_components`` as a mandatory input parameter.
注意：使用可选参数 ``svd_solver='randomized'`` ，
在 :class:`PCA` 中我们必须输入低维度参数 ``n_components`` 。

If we note :math:`n_{\max} = \max(n_{\mathrm{samples}}, n_{\mathrm{features}})` and
:math:`n_{\min} = \min(n_{\mathrm{samples}}, n_{\mathrm{features}})`, the time complexity
of the randomized :class:`PCA` is :math:`O(n_{\max}^2 \cdot n_{\mathrm{components}})`
instead of :math:`O(n_{\max}^2 \cdot n_{\min})` for the exact method
implemented in :class:`PCA`.

如果我们注意到：:math:`n_{\max} = \max(n_{\mathrm{samples}}, n_{\mathrm{features}})` 且
:math:`n_{\min} = \min(n_{\mathrm{samples}}, n_{\mathrm{features}})`,
随机 :class:`PCA` 的时间复杂度是：:math:`O(n_{\max}^2 \cdot n_{\mathrm{components}})` ，
而不是 :math:`O(n_{\max}^2 \cdot n_{\min})` 。

The memory footprint of randomized :class:`PCA` is also proportional to
:math:`2 \cdot n_{\max} \cdot n_{\mathrm{components}}` instead of :math:`n_{\max}
\cdot n_{\min}` for the exact method.

在准确情形下，随机 :class:`PCA` 的内存占用量正比于 :math:`2 \cdot n_{\max} \cdot n_{\mathrm{components}}` ，
而不是 :math:`n_{\max}\cdot n_{\min}`

Note: the implementation of ``inverse_transform`` in :class:`PCA` with
``svd_solver='randomized'`` is not the exact inverse transform of
``transform`` even when ``whiten=False`` (default).
注意：选择参数 ``svd_solver='randomized'`` 的 :class:`PCA`，
在执行 ``inverse_transform`` 时， 并不是 ``transform`` 的确切的逆变换操作
（即使 参数设置为默认的 ``whiten=False``）

.. topic:: Examples例子:

    * :ref:`sphx_glr_auto_examples_applications_plot_face_recognition.py`
    * :ref:`sphx_glr_auto_examples_decomposition_plot_faces_decomposition.py`

.. topic:: References参考文献:

    * `"Finding structure with randomness: Stochastic algorithms for
      constructing approximate matrix decompositions"
      <http://arxiv.org/abs/0909.4061>`_
      Halko, et al., 2009


.. _kernel_PCA:

Kernel PCA
----------

:class:`KernelPCA` is an extension of PCA which achieves non-linear
dimensionality reduction through the use of kernels (see :ref:`metrics`). It
has many applications including denoising, compression and structured
prediction (kernel dependency estimation). :class:`KernelPCA` supports both
``transform`` and ``inverse_transform``.

:class:`KernelPCA` 是 PCA 的扩展， 通过使用内核（参见 :ref:`metrics` ）实现非线性维数降低。
它具有许多应用，包括去噪，压缩和结构预测（内核依赖估计）。
:class:`KernelPCA` 支持 ``transform`` 和 ``inverse_transform``。

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_kernel_pca_001.png
    :target: ../auto_examples/decomposition/plot_kernel_pca.html
    :align: center
    :scale: 75%

.. topic:: Examples例子:

    * :ref:`sphx_glr_auto_examples_decomposition_plot_kernel_pca.py`


.. _SparsePCA:

Sparse principal components analysis (SparsePCA and MiniBatchSparsePCA)
稀疏主成分分析（SparsePCA和MiniBatchSparsePCA）
-----------------------------------------------------------------------

:class:`SparsePCA` is a variant of PCA, with the goal of extracting the
set of sparse components that best reconstruct the data.
:class:`SparsePCA` 是PCA的一个变体，其目标是提取最能重建数据的稀疏成分集合。

Mini-batch sparse PCA (:class:`MiniBatchSparsePCA`) is a variant of
:class:`SparsePCA` that is faster but less accurate. The increased speed is
reached by iterating over small chunks of the set of features, for a given
number of iterations.
Mini-batch sparse PCA (:class:`MiniBatchSparsePCA`) 是 :class:`SparsePCA`
 的一种变体， 它速度更快但准确度有所降低。
 对于给定的迭代次数，通过迭代该组特征的小数据块来达到速度的增加。

Principal component analysis (:class:`PCA`) has the disadvantage that the
components extracted by this method have exclusively dense expressions, i.e.
they have non-zero coefficients when expressed as linear combinations of the
original variables. This can make interpretation difficult. In many cases,
the real underlying components can be more naturally imagined as sparse
vectors; for example in face recognition, components might naturally map to
parts of faces.
主成分分析（PCA）具有以下缺点：通过该方法提取的成分具有唯一的密集表达式。
即当以原始变量的线性组合表示时，它们具有非零系数。这可以使解释变得困难。
在许多情况下，真正的基础成分可以更自然地想象为稀疏向量;
例如在面部识别中，组件可能自然地映射到部分脸部。

Sparse principal components yields a more parsimonious, interpretable
representation, clearly emphasizing which of the original features contribute
to the differences between samples.
稀疏的主成分产生更简洁，可解释的，明确强调哪些原始特征有助于区分样本之间的差异。

The following example illustrates 16 components extracted using sparse PCA from
the Olivetti faces dataset.  It can be seen how the regularization term induces
many zeros. Furthermore, the natural structure of the data causes the non-zero
coefficients to be vertically adjacent. The model does not enforce this
mathematically: each component is a vector :math:`h \in \mathbf{R}^{4096}`, and
there is no notion of vertical adjacency except during the human-friendly
visualization as 64x64 pixel images. The fact that the components shown below
appear local is the effect of the inherent structure of the data, which makes
such local patterns minimize reconstruction error. There exist sparsity-inducing
norms that take into account adjacency and different kinds of structure; see
[Jen09]_ for a review of such methods.

以下示例说明了使用sparse PCA 从Olivetti数据集中提取的16个成分。
可以看出重整后如何引发许多零。此外，数据的自然结构导致非零系数垂直相邻。
该模型不会在数学上强制执行：每个成分都是一个向量 :math:`h \in \mathbf{R}^{4096}` ,
并且没有垂直相邻性的概念，除非人性化地可视化为64x64像素图像。
下面显示的成分出现局部的事实是数据的固有结构的影响，这使得这种局部模式使重建误差最小化。
存在考虑到邻接和不同类型结构的稀疏诱导规范; 参见 [Jen09]_ 。

For more details on how to use Sparse PCA, see the Examples section, below.
更多关于 Sparse PCA 使用的内容，参见示例部分，如下：

.. |spca_img| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_005.png
   :target: ../auto_examples/decomposition/plot_faces_decomposition.html
   :scale: 60%

.. centered:: |pca_img| |spca_img|

Note that there are many different formulations for the Sparse PCA
problem. The one implemented here is based on [Mrl09]_ . The optimization
problem solved is a PCA problem (dictionary learning) with an
:math:`\ell_1` penalty on the components:

请注意：有多种计算 Sparse PCA 问题的公式。 这里使用的是基于 [Mrl09]_ 。
优化问题的解决是一个 PCA 问题， 其针对成分使用 :math:`\ell_1` 惩罚：

.. math::
   (U^*, V^*) = \underset{U, V}{\operatorname{arg\,min\,}} & \frac{1}{2}
                ||X-UV||_2^2+\alpha||V||_1 \\
                \text{subject to\,} & ||U_k||_2 = 1 \text{ for all }
                0 \leq k < n_{components}


The sparsity-inducing :math:`\ell_1` norm also prevents learning
components from noise when few training samples are available. The degree
of penalization (and thus sparsity) can be adjusted through the
hyperparameter ``alpha``. Small values lead to a gently regularized
factorization, while larger values shrink many coefficients to zero.

当训练数据样本数量较少时，稀疏-诱导(sparsity-inducing) :math:`\ell_1` 规范同样会阻止从噪声中学习成分。
惩罚程度（因而稀疏度）可以通过超参数 ``alpha`` 进行调整。
小值导致温和的调整，而较大的值会将许多系数缩小到零。

.. note::

  While in the spirit of an online algorithm, the class
  :class:`MiniBatchSparsePCA` does not implement ``partial_fit`` because
  the algorithm is online along the features direction, not the samples
  direction.

虽然本着在线算法的精神，该类 :class:`MiniBatchSparsePCA` 不能实现 ``partial_fit`` 。
因为算法沿着特征方向在线，而不是样本方向。

.. topic:: Examples例子:

   * :ref:`sphx_glr_auto_examples_decomposition_plot_faces_decomposition.py`

.. topic:: References参考文献:

  .. [Mrl09] `"Online Dictionary Learning for Sparse Coding"
     <http://www.di.ens.fr/sierra/pdfs/icml09.pdf>`_
     J. Mairal, F. Bach, J. Ponce, G. Sapiro, 2009
  .. [Jen09] `"Structured Sparse Principal Component Analysis"
     <www.di.ens.fr/~fbach/sspca_AISTATS2010.pdf>`_
     R. Jenatton, G. Obozinski, F. Bach, 2009


.. _LSA:

Truncated singular value decomposition and latent semantic analysis
截断的奇异值分解和潜在语义分析
===================================================================

:class:`TruncatedSVD` implements a variant of singular value decomposition
(SVD) that only computes the :math:`k` largest singular values,
where :math:`k` is a user-specified parameter.
:class:`TruncatedSVD` 实现仅计算 :math:`k` 最大奇异值的奇异值分解（SVD）的变体，
其中 :math:`k` 是用户指定的参数。


When truncated SVD is applied to term-document matrices
(as returned by ``CountVectorizer`` or ``TfidfVectorizer``),
this transformation is known as
`latent semantic analysis <http://nlp.stanford.edu/IR-book/pdf/18lsi.pdf>`_
(LSA), because it transforms such matrices
to a "semantic" space of low dimensionality.
In particular, LSA is known to combat the effects of synonymy and polysemy
(both of which roughly mean there are multiple meanings per word),
which cause term-document matrices to be overly sparse
and exhibit poor similarity under measures such as cosine similarity.

当 truncated SVD 应用于 term-document matrices (由 ``CountVectorizer`` 或 ``TfidfVectorizer`` 返回)时，
该变换被成为 LSA（`latent semantic analysis <http://nlp.stanford.edu/IR-book/pdf/18lsi.pdf>`_ ，潜在语义分析）。
因为它将这样的矩阵转换为低维度的“语义”空间。
特别地，LSA 能够抵抗同义词和多义词的影响（这两者大致意味着每个单词有多重含义），
这导致 term-document matrices 过度稀疏，并且在诸如余弦相似性的度量下表现出差的相似性。

.. note::
    LSA is also known as latent semantic indexing, LSI,
    though strictly that refers to its use in persistent indexes
    for information retrieval purposes.
    LSA也被称为潜在语义索引LSI，尽管严格地说它是指在持久索引中用于信息检索的目的。

Mathematically, truncated SVD applied to training samples :math:`X`
produces a low-rank approximation :math:`X`:
在数学上， truncated SVD 用于训练样本 :math:`X`， 产生低阶逼近 :math:`X` ：

.. math::
    X \approx X_k = U_k \Sigma_k V_k^\top

After this operation, :math:`U_k \Sigma_k^\top`
is the transformed training set with :math:`k` features
(called ``n_components`` in the API).

在这个操作后，:math:`U_k \Sigma_k^\top` 具有 :math:`k` 特征的变换训练集
（在 API 中称为 ``n_components`` ）

To also transform a test set :math:`X`, we multiply it with :math:`V_k`:
用于转换测试集 :math:`X` 时， 将他与 :math:`V_k` 相乘：

.. math::
    X' = X V_k

.. note::
    Most treatments of LSA in the natural language processing (NLP)
    and information retrieval (IR) literature
    swap the axes of the matrix :math:`X` so that it has shape
    ``n_features`` × ``n_samples``.
    We present LSA in a different way that matches the scikit-learn API better,
    but the singular values found are the same.

    自然语言处理（NLP）和信息检索（IR）文献中， 大多数 LSA 的处理是交换 矩阵 :math:`X`
     的轴，使其具有形状 ``n_features`` × ``n_samples``。
    我们描述 LSA 时以不同的方式进行阐述，以与 Scikit-learn API 更好的匹配，
    ，但是找到的奇异值是相同的。

:class:`TruncatedSVD` is very similar to :class:`PCA`, but differs
in that it works on sample matrices :math:`X` directly
instead of their covariance matrices.
When the columnwise (per-feature) means of :math:`X`
are subtracted from the feature values,
truncated SVD on the resulting matrix is equivalent to PCA.
In practical terms, this means
that the :class:`TruncatedSVD` transformer accepts ``scipy.sparse``
matrices without the need to densify them,
as densifying may fill up memory even for medium-sized document collections.

:class:`TruncatedSVD` 与 :class:`PCA` 非常相似，
但不同之处在于，它直接作用于样本矩阵 :math:`X` 而不是协方差矩阵。
当 矩阵 :math:`X` 从特征值中减去列均值时，truncated SVD 得到的结果矩阵与 PCA 相同。
实际上，这意味着 :class:`TruncatedSVD` 变换接受 ``scipy.sparse`` 矩阵，
而不需要对它们进行致密化，因为即使对于中型文档集合，致密化也可能填满内存。


While the :class:`TruncatedSVD` transformer
works with any (sparse) feature matrix,
using it on tf–idf matrices is recommended over raw frequency counts
in an LSA/document processing setting.
In particular, sublinear scaling and inverse document frequency
should be turned on (``sublinear_tf=True, use_idf=True``)
to bring the feature values closer to a Gaussian distribution,
compensating for LSA's erroneous assumptions about textual data.

当 :class:`TruncatedSVD` 变换与任何（稀疏）特征矩阵一起使用时，
建议在 LSA/文档 处理设置中的原始频率计数上使用tf-idf矩阵。
特别地，应该打开子线性缩放和逆文档频率 (``sublinear_tf=True, use_idf=True``)
以使特征值更接近高斯分布，从而补偿 LSA 对文本数据的错误假设。

.. topic:: Examples例子:

   * :ref:`sphx_glr_auto_examples_text_document_clustering.py`

.. topic:: References参考文献:

  * Christopher D. Manning, Prabhakar Raghavan and Hinrich Schütze (2008),
    *Introduction to Information Retrieval*, Cambridge University Press,
    chapter 18: `Matrix decompositions & latent semantic indexing
    <http://nlp.stanford.edu/IR-book/pdf/18lsi.pdf>`_


.. _DictionaryLearning:

Dictionary Learning
词典学习
===================

.. _SparseCoder:

Sparse coding with a precomputed dictionary
用预先计算的字典稀疏编码
-------------------------------------------

The :class:`SparseCoder` object is an estimator that can be used to transform signals
into sparse linear combination of atoms from a fixed, precomputed dictionary
such as a discrete wavelet basis. This object therefore does not
implement a ``fit`` method. The transformation amounts
to a sparse coding problem: finding a representation of the data as a linear
combination of as few dictionary atoms as possible. All variations of
dictionary learning implement the following transform methods, controllable via
the ``transform_method`` initialization parameter:

:class:`SparseCoder` 对象是可从固定的、预先计算的字典
（例如离散小波基，discrete wavelet basis）中将信号转换成
原子的稀疏线性组合的估计器（estimator）。
因此，该对象不实现 ``fit`` 方法。该转换相当于稀疏编码问题：
将数据的表示为尽可能少的字典原子的线性组合。
以下所有字典学习变量实现的变换方法，可通过 ``transform_method`` 初始化参数进行控制：

* 正交匹配追踪（Orthogonal matching pursuit） (:ref:`omp`)

* 最小角度回归（Least-angle regression） (:ref:`least_angle_regression`)

* Lasso computed by least-angle regression
Lasso通过最小角度回归计算

* Lasso using coordinate descent (:ref:`lasso`)
Lasso使用坐标下降 (:ref:`lasso`)

* Thresholding 阈值

Thresholding is very fast but it does not yield accurate reconstructions.
They have been shown useful in literature for classification tasks. For image
reconstruction tasks, orthogonal matching pursuit yields the most accurate,
unbiased reconstruction.

阈值非常快，但不能产生精确的重建。它们在文本分类任务中已被证明是有用的。
对于图像重建任务，正交匹配追求产生最准确，无偏见的重建。

The dictionary learning objects offer, via the ``split_code`` parameter, the
possibility to separate the positive and negative values in the results of
sparse coding. This is useful when dictionary learning is used for extracting
features that will be used for supervised learning, because it allows the
learning algorithm to assign different weights to negative loadings of a
particular atom, from to the corresponding positive loading.

字典学习对象通过 ``split_code`` 参数提供在稀疏编码结果中分离正值和负值的可能性。
当使用字典学习来提取将用于监督学习的特征时，这是有用的，
因为它允许学习算法将不同的权重从相应的正负载中
分配给特定的负负荷（negative loading）原子。

The split code for a single sample has length ``2 * n_components``
and is constructed using the following rule: First, the regular code of length
``n_components`` is computed. Then, the first ``n_components`` entries of the
``split_code`` are
filled with the positive part of the regular code vector. The second half of
the split code is filled with the negative part of the code vector, only with
a positive sign. Therefore, the split_code is non-negative.

单个样本的分割代码具有长度 ``2 * n_components`` ， 并使用以下规则构建：
首先计算常规代码的长度 ``n_components`` 。然后，第一个 ``n_components`` 的
``split_code`` 填充了常规代码向量的正向部分。另一半分割代码的填充有负向的代码向量，只有一个正号。
因此，split_code是非负数。

.. topic:: Examples例子:

    * :ref:`sphx_glr_auto_examples_decomposition_plot_sparse_coding.py`


Generic dictionary learning
通用字典学习
---------------------------

Dictionary learning (:class:`DictionaryLearning`) is a matrix factorization
problem that amounts to finding a (usually overcomplete) dictionary that will
perform good at sparsely encoding the fitted data.
词典学习（:class:`DictionaryLearning`）是一个矩阵因式分解问题，
相当于找到一个（通常是过于完备的）字典，它将对拟合的数据进行良好的稀疏编码。

Representing data as sparse combinations of atoms from an overcomplete
dictionary is suggested to be the way the mammal primary visual cortex works.
Consequently, dictionary learning applied on image patches has been shown to
give good results in image processing tasks such as image completion,
inpainting and denoising, as well as for supervised recognition tasks.
将数据表示为来自过度完整字典的原子稀疏组合被认为是哺乳动物初级视觉皮层的工作方式。
因此，应用于图像补丁的字典学习已被证明在诸如图像完成，
修复和去噪以及监督识别任务的图像处理任务中给出良好的结果。


Dictionary learning is an optimization problem solved by alternatively updating
the sparse code, as a solution to multiple Lasso problems, considering the
dictionary fixed, and then updating the dictionary to best fit the sparse code.
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


After using such a procedure to fit the dictionary, the transform is simply a
sparse coding step that shares the same implementation with all dictionary
learning objects (see :ref:`SparseCoder`).
在使用这样一个过程来拟合字典之后，变换只是一个稀疏编码步骤，它与所有字典学习对象共享相同的实现。
（参见 :ref:`SparseCoder`）


The following image shows how a dictionary learned from 4x4 pixel image patches
extracted from part of the image of a raccoon face looks like.
以下图像显示了字典学习是如何从浣熊脸部图像中提取的4x4像素图像补丁中进行实现的。

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_image_denoising_001.png
    :target: ../auto_examples/decomposition/plot_image_denoising.html
    :align: center
    :scale: 50%


.. topic:: Examples例子:

  * :ref:`sphx_glr_auto_examples_decomposition_plot_image_denoising.py`


.. topic:: References参考文献:

  * `"Online dictionary learning for sparse coding"
    <http://www.di.ens.fr/sierra/pdfs/icml09.pdf>`_
    J. Mairal, F. Bach, J. Ponce, G. Sapiro, 2009

.. _MiniBatchDictionaryLearning:

Mini-batch dictionary learning
Mini-batch 字典学习 （Mini-batch dictionary learning）
------------------------------

:class:`MiniBatchDictionaryLearning` implements a faster, but less accurate
version of the dictionary learning algorithm that is better suited for large
datasets.
:class:`MiniBatchDictionaryLearning` 实现更适合大型数据集的字典学习算法，
其运行速度更快，但准确度有所降低。

By default, :class:`MiniBatchDictionaryLearning` divides the data into
mini-batches and optimizes in an online manner by cycling over the mini-batches
for the specified number of iterations. However, at the moment it does not
implement a stopping condition.

默认情况下，:class:`MiniBatchDictionaryLearning` 将数据分成小批量，
并通过在指定次数的迭代中循环使用小批量，以在线方式进行优化。但是，目前它没有实现停止条件。

The estimator also implements ``partial_fit``, which updates the dictionary by
iterating only once over a mini-batch. This can be used for online learning
when the data is not readily available from the start, or for when the data
does not fit into the memory.
该方法还实现了 ``partial_fit`` ， 它通过在一个 mini-batch 中迭代一次来更新字典。
当数据从一开始就不容易获得，或数据不适合内存时，这可以用于在线学习。

.. currentmodule:: sklearn.cluster

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_dict_face_patches_001.png
    :target: ../auto_examples/cluster/plot_dict_face_patches.html
    :scale: 50%
    :align: right

.. topic:: **Clustering for dictionary learning字典学习聚类**

   Note that when using dictionary learning to extract a representation
   (e.g. for sparse coding) clustering can be a good proxy to learn the
   dictionary. For instance the :class:`MiniBatchKMeans` estimator is
   computationally efficient and implements on-line learning with a
   ``partial_fit`` method.
   请注意，当使用字典学习提取文档（例如稀疏编码）时，聚类可以成为学习字典的好代理。
   例如，:class:`MiniBatchKMeans` 估计器在计算上是有效的，
   并通过一种  ``partial_fit`` 方法实现在线学习 。

    Example例子: :ref:`sphx_glr_auto_examples_cluster_plot_dict_face_patches.py`

.. currentmodule:: sklearn.decomposition

.. _FA:

Factor Analysis
因子分析
===============

In unsupervised learning we only have a dataset :math:`X = \{x_1, x_2, \dots, x_n
\}`. How can this dataset be described mathematically? A very simple
`continuous latent variable` model for :math:`X` is
在无监督的学习中，我们只有一个数据集 :math:`X = \{x_1, x_2, \dots, x_n
\}` 。这个数据集如何在数学上描述？
一个非常简单的连续潜变量 `continuous latent variable` 的模型 :math:`X` 是：

.. math:: x_i = W h_i + \mu + \epsilon

The vector :math:`h_i` is called "latent" because it is unobserved. :math:`\epsilon` is
considered a noise term distributed according to a Gaussian with mean 0 and
covariance :math:`\Psi` (i.e. :math:`\epsilon \sim \mathcal{N}(0, \Psi)`), :math:`\mu` is some
arbitrary offset vector. Such a model is called "generative" as it describes
how :math:`x_i` is generated from :math:`h_i`. If we use all the :math:`x_i`'s as columns to form
a matrix :math:`\mathbf{X}` and all the :math:`h_i`'s as columns of a matrix :math:`\mathbf{H}`
then we can write (with suitably defined :math:`\mathbf{M}` and :math:`\mathbf{E}`):

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

In other words, we *decomposed* matrix :math:`\mathbf{X}`.
换句话说，我们 *分解* 矩阵 :math:`\mathbf{X}` 。

If :math:`h_i` is given, the above equation automatically implies the following
probabilistic interpretation:

如果 :math:`h_i` 给出，上述方程自动暗示以下概率解释：

.. math:: p(x_i|h_i) = \mathcal{N}(Wh_i + \mu, \Psi)

For a complete probabilistic model we also need a prior distribution for the
latent variable :math:`h`. The most straightforward assumption (based on the nice
properties of the Gaussian distribution) is :math:`h \sim \mathcal{N}(0,
\mathbf{I})`.  This yields a Gaussian as the marginal distribution of :math:`x`:
对于一个完整的概率模型，我们还需要潜在变量 :math:`h` 的先前分布。
最直接的假设（基于高斯分布的良好属性）是 :math:`h \sim \mathcal{N}(0,
\mathbf{I})` 。这产生一个高斯作为 :math:`x` 的边际分布：


.. math:: p(x) = \mathcal{N}(\mu, WW^T + \Psi)

Now, without any further assumptions the idea of having a latent variable :math:`h`
would be superfluous -- :math:`x` can be completely modelled with a mean
and a covariance. We need to impose some more specific structure on one
of these two parameters. A simple additional assumption regards the
structure of the error covariance :math:`\Psi`:
现在，没有任何进一步的假设，具有潜在变量 :math:`h` 的想法将是多余的
 -- :math:`x` 可以用平均和协方差来完全建模。
我们需要对这两个参数之一施加一些更具体的结构。
一个简单的附加假设是误差协方差的结构 :math:`\Psi` ：


* :math:`\Psi = \sigma^2 \mathbf{I}`: This assumption leads to
  the probabilistic model of :class:`PCA`.
  :math:`\Psi = \sigma^2 \mathbf{I}`: 这个假设导致了概率模型 :class:`PCA` 。

* :math:`\Psi = \mathrm{diag}(\psi_1, \psi_2, \dots, \psi_n)`: This model is called
  :class:`FactorAnalysis`, a classical statistical model. The matrix W is
  sometimes called the "factor loading matrix".
  :math:`\Psi = \mathrm{diag}(\psi_1, \psi_2, \dots, \psi_n)`:
  这个模型叫做 :class:`FactorAnalysis` ，是一个经典的统计模型。
  矩阵 W 有时也称为“因子加载矩阵” （"factor loading matrix"）。

Both models essentially estimate a Gaussian with a low-rank covariance matrix.
Because both models are probabilistic they can be integrated in more complex
models, e.g. Mixture of Factor Analysers. One gets very different models (e.g.
:class:`FastICA`) if non-Gaussian priors on the latent variables are assumed.
两个模型基本上估计出具有低阶协方差矩阵的高斯。
因为这两个模型都是概率性的，所以它们可以集成到更复杂的模型中，例如混合因子分析。
如果假设潜在变量上的非高斯先验，则得到非常不同的模型（例如 :class:`FastICA` ）。


Factor analysis *can* produce similar components (the columns of its loading
matrix) to :class:`PCA`. However, one can not make any general statements
about these components (e.g. whether they are orthogonal):

因子分析可以产生类似于 :class:`PCA` 的成分（其加载矩阵的列）。
然而，不能对这些成分做出任何一般性的说明（例如它们是否是正交的）：

.. |pca_img3| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_002.png
    :target: ../auto_examples/decomposition/plot_faces_decomposition.html
    :scale: 60%

.. |fa_img3| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_009.png
    :target: ../auto_examples/decomposition/plot_faces_decomposition.html
    :scale: 60%

.. centered:: |pca_img3| |fa_img3|

The main advantage for Factor Analysis (over :class:`PCA` is that
it can model the variance in every direction of the input space independently
(heteroscedastic noise):
因子分析的主要优于 :class:`PCA` 的地方是
它可以独立地对输入空间的每个方向的模型进行建模（异方差噪声）：

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_008.png
    :target: ../auto_examples/decomposition/plot_faces_decomposition.html
    :align: center
    :scale: 75%

This allows better model selection than probabilistic PCA in the presence
of heteroscedastic noise:
在异方差噪声的存在下，这样可以比概率PCA更好的模型选择：

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_pca_vs_fa_model_selection_002.png
    :target: ../auto_examples/decomposition/plot_pca_vs_fa_model_selection.html
    :align: center
    :scale: 75%


.. topic:: Examples例子:

    * :ref:`sphx_glr_auto_examples_decomposition_plot_pca_vs_fa_model_selection.py`

.. _ICA:

Independent component analysis (ICA)
独立成分分析 （Independent component analysis, ICA）
====================================

Independent component analysis separates a multivariate signal into
additive subcomponents that are maximally independent. It is
implemented in scikit-learn using the :class:`Fast ICA <FastICA>`
algorithm. Typically, ICA is not used for reducing dimensionality but
for separating superimposed signals. Since the ICA model does not include
a noise term, for the model to be correct, whitening must be applied.
This can be done internally using the whiten argument or manually using one
of the PCA variants.
独立成分分析将多变量信号分解为最大独立的附加子成分。
在scikit-learn中使用 :class:`Fast ICA <FastICA>` 算法来实现。
通常，ICA不用于降低维度，而是用于分离叠加信号。
由于ICA模型不包括噪声项，因此，为了使模型正确，必须首先进行预处理。
这可以在内部使用whiten参数或手动使用其中一种PCA变体进行。


It is classically used to separate mixed signals (a problem known as
*blind source separation*), as in the example below:
通常用于分离混合信号（称为盲源分离的问题， *blind source separation* ），如下例所示：

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_ica_blind_source_separation_001.png
    :target: ../auto_examples/decomposition/plot_ica_blind_source_separation.html
    :align: center
    :scale: 60%


ICA can also be used as yet another non linear decomposition that finds
components with some sparsity:
ICA也可以用作另一种非线性分解，可以找到一些稀疏成分：

.. |pca_img4| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_002.png
    :target: ../auto_examples/decomposition/plot_faces_decomposition.html
    :scale: 60%

.. |ica_img4| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_004.png
    :target: ../auto_examples/decomposition/plot_faces_decomposition.html
    :scale: 60%

.. centered:: |pca_img4| |ica_img4|

.. topic:: Examples例子:

    * :ref:`sphx_glr_auto_examples_decomposition_plot_ica_blind_source_separation.py`
    * :ref:`sphx_glr_auto_examples_decomposition_plot_ica_vs_pca.py`
    * :ref:`sphx_glr_auto_examples_decomposition_plot_faces_decomposition.py`


.. _NMF:

Non-negative matrix factorization (NMF or NNMF)
非负矩阵分解 (Non-negative matrix factorization, NMF or NNMF)
===============================================

NMF with the Frobenius norm
具有 Frobenius 范数的 NMF
---------------------------

:class:`NMF` [1]_ is an alternative approach to decomposition that assumes that the
data and the components are non-negative. :class:`NMF` can be plugged in
instead of :class:`PCA` or its variants, in the cases where the data matrix
does not contain negative values. It finds a decomposition of samples
:math:`X` into two matrices :math:`W` and :math:`H` of non-negative elements,
by optimizing the distance :math:`d` between :math:`X` and the matrix product
:math:`WH`. The most widely used distance function is the squared Frobenius
norm, which is an obvious extension of the Euclidean norm to matrices:

:class:`NMF` [1]_ 是分解的另一种方法，它假定数据和成分是非负的。
在数据矩阵不包含负值的情况下， :class:`NMF` 可以插入，用来替代 :class:`PCA` 或其变体。
它通过优化 :math:`d` 和 :math:`X` 之间的距离以及 矩阵 :math:`WH` ，
来将样本 :math:`X `分解成为两个非负元素的矩阵 :math:`W` 和 :math:`H`。
使用最广泛的距离函数是 平方Frobenius范数 (squared Frobenius norm)，
它是是 欧几里得范数 (Euclidean norm) 到矩阵的一个明显的扩展 ：


.. math::
    d_{\mathrm{Fro}}(X, Y) = \frac{1}{2} ||X - Y||_{\mathrm{Fro}}^2 = \frac{1}{2} \sum_{i,j} (X_{ij} - {Y}_{ij})^2

Unlike :class:`PCA`, the representation of a vector is obtained in an additive
fashion, by superimposing the components, without subtracting. Such additive
models are efficient for representing images and text.
不同于 :class:`PCA` 的是，通过叠加成分而不加减去，以附加的方式获得向量的表示。
这种附加模型对于表示图像和文本是有效的。

It has been observed in [Hoyer, 2004] [2]_ that, when carefully constrained,
:class:`NMF` can produce a parts-based representation of the dataset,
resulting in interpretable models. The following example displays 16
sparse components found by :class:`NMF` from the images in the Olivetti
faces dataset, in comparison with the PCA eigenfaces.

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


The :attr:`init` attribute determines the initialization method applied, which
has a great impact on the performance of the method. :class:`NMF` implements the
method Nonnegative Double Singular Value Decomposition. NNDSVD [4]_ is based on
two SVD processes, one approximating the data matrix, the other approximating
positive sections of the resulting partial SVD factors utilizing an algebraic
property of unit rank matrices. The basic NNDSVD algorithm is better fit for
sparse factorization. Its variants NNDSVDa (in which all zeros are set equal to
the mean of all elements of the data), and NNDSVDar (in which the zeros are set
to random perturbations less than the mean of the data divided by 100) are
recommended in the dense case.
 :attr:`init` 属性决定了应用的初始化方法，对方法的性能有很大的影响。
 :class:`NMF` 实现非负双奇异值分解方法。
  NNDSVD [4]_ 基于两个 SVD 过程，一个近似于数据矩阵，
  另一个近似于 使用单位秩矩阵的代数性质，得到的部分SVD因子的正部分。
基本的 NNDSVD 算法更适合稀疏分解。
其变体 NNDSVDa（其中所有零被设置为等于数据的所有元素的平均值）
和 NNDSVDar（其中将零设置为小于数据平均值的随机扰动除以100）推荐在密集情况下使用。

Note that the Multiplicative Update ('mu') solver cannot update zeros present in
the initialization, so it leads to poorer results when used jointly with the
basic NNDSVD algorithm which introduces a lot of zeros; in this case, NNDSVDa or
NNDSVDar should be preferred.
请注意：乘法更新方法 （Multiplicative Update ('mu') solver）不能在初始化中更新零值，
因此，当其与基础的 NNDSVD 算法（包含大量的零值）一起使用时，它将导致比较差的结果。
在这种情况下，使用 NNDSVDa 或 NNDSVDar 会更好。

:class:`NMF` can also be initialized with correctly scaled random non-negative
matrices by setting :attr:`init="random"`. An integer seed or a
``RandomState`` can also be passed to :attr:`random_state` to control
reproducibility.

:class:`NMF` 也可以通过设置正确缩放的随机非负矩阵（设定，:attr:`init="random"`）进行初始化。
整数种子 或 ``RandomState`` 也可以传递 给 :attr:`random_state` ，
以保证结果可以重现。

In :class:`NMF`, L1 and L2 priors can be added to the loss function in order
to regularize the model. The L2 prior uses the Frobenius norm, while the L1
prior uses an elementwise L1 norm. As in :class:`ElasticNet`, we control the
combination of L1 and L2 with the :attr:`l1_ratio` (:math:`\rho`) parameter,
and the intensity of the regularization with the :attr:`alpha`
(:math:`\alpha`) parameter. Then the priors terms are:

在NMF L1和L2先验可以添加到损失函数，以使模型正规化。
L2之前使用Frobenius标准，而L1先验使用元素L1范数。
如ElasticNet，我们控制L1和L2的与所述组合l1_ratio（）参数，
并与正则化强度alpha （）参数。那么先修课程是：

.. math::
    \alpha \rho ||W||_1 + \alpha \rho ||H||_1
    + \frac{\alpha(1-\rho)}{2} ||W||_{\mathrm{Fro}} ^ 2
    + \frac{\alpha(1-\rho)}{2} ||H||_{\mathrm{Fro}} ^ 2

and the regularized objective function is:
正则化的目标函数是：

.. math::
    d_{\mathrm{Fro}}(X, WH)
    + \alpha \rho ||W||_1 + \alpha \rho ||H||_1
    + \frac{\alpha(1-\rho)}{2} ||W||_{\mathrm{Fro}} ^ 2
    + \frac{\alpha(1-\rho)}{2} ||H||_{\mathrm{Fro}} ^ 2

:class:`NMF` regularizes both W and H. The public function
:func:`non_negative_factorization` allows a finer control through the
:attr:`regularization` attribute, and may regularize only W, only H, or both.

:class:`NMF` 正则化 W 和 H 。公共函数 :func:`non_negative_factorization`
允许通过 :attr:`regularization` 属性进行更精细的控制 ，并且可以仅将 W ，仅 H 或两者正规化。


NMF with a beta-divergence
beta-分离的 NMF （NMF with a beta-divergence）
--------------------------

As described previously, the most widely used distance function is the squared
Frobenius norm, which is an obvious extension of the Euclidean norm to
matrices:
如前所述，最广泛使用的距离函数是Frobenius范数（the squared Frobenius norm）。
，这是欧氏距离（Euclidean norm，欧几里德范数）一个最广泛的扩展：

.. math::
    d_{\mathrm{Fro}}(X, Y) = \frac{1}{2} ||X - Y||_{Fro}^2 = \frac{1}{2} \sum_{i,j} (X_{ij} - {Y}_{ij})^2

Other distance functions can be used in NMF as, for example, the (generalized)
Kullback-Leibler (KL) divergence, also referred as I-divergence:
其他能在 NMF 中使用的距离函数，例如 the (generalized)
Kullback-Leibler (KL) divergence，也称为 I-divergence ：

.. math::
    d_{KL}(X, Y) = \sum_{i,j} (X_{ij} \log(\frac{X_{ij}}{Y_{ij}}) - X_{ij} + Y_{ij})

Or, the Itakura-Saito (IS) divergence:
或者, the Itakura-Saito (IS) divergence:

.. math::
    d_{IS}(X, Y) = \sum_{i,j} (\frac{X_{ij}}{Y_{ij}} - \log(\frac{X_{ij}}{Y_{ij}}) - 1)

These three distances are special cases of the beta-divergence family, with
:math:`\beta = 2, 1, 0` respectively [6]_. The beta-divergence are
defined by :

这三种距离函数都属于 beta-divergence 家族， 其参数分别为 :math:`\beta = 2, 1, 0` [6]_。
beta-divergence 定义为：

.. math::
    d_{\beta}(X, Y) = \sum_{i,j} \frac{1}{\beta(\beta - 1)}(X_{ij}^\beta + (\beta-1)Y_{ij}^\beta - \beta X_{ij} Y_{ij}^{\beta - 1})

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_beta_divergence_001.png
    :target: ../auto_examples/decomposition/plot_beta_divergence.html
    :align: center
    :scale: 75%

Note that this definition is not valid if :math:`\beta \in (0; 1)`, yet it can
be continously extended to the definitions of :math:`d_{KL}` and :math:`d_{IS}`
respectively.

请注意： 当 :math:`\beta \in (0; 1)` 时，这个定义是无效的， 然而，它可以是分别是 定义
:math:`d_{KL}` 和 :math:`d_{IS}` 的连续延伸。

:class:`NMF` implements two solvers, using Coordinate Descent ('cd') [5]_, and
Multiplicative Update ('mu') [6]_. The 'mu' solver can optimize every
beta-divergence, including of course the Frobenius norm (:math:`\beta=2`), the
(generalized) Kullback-Leibler divergence (:math:`\beta=1`) and the
Itakura-Saito divergence (:math:`\beta=0`). Note that for
:math:`\beta \in (1; 2)`, the 'mu' solver is significantly faster than for other
values of :math:`\beta`. Note also that with a negative (or 0, i.e.
'itakura-saito') :math:`\beta`, the input matrix cannot contain zero values.

:class:`NMF` 可以使用两种解决方案， Coordinate Descent ('cd') [5]_， 和
Multiplicative Update ('mu') [6]_。'mu' 方法可以优化每种 beta-divergence， 包括
Frobenius norm (:math:`\beta=2`), the
(generalized) Kullback-Leibler divergence (:math:`\beta=1`) 以及 the
Itakura-Saito divergence (:math:`\beta=0`)。
请注意： 当 :math:`\beta \in (1; 2)` 时,  'mu' 方法 明显快于 其他的 :math:`\beta` 值。
同样请注意， 负的 :math:`\beta` 值 （或，0， 如'itakura-saito'），
矩阵的输入值不能包含零值。


The 'cd' solver can only optimize the Frobenius norm. Due to the
underlying non-convexity of NMF, the different solvers may converge to
different minima, even when optimizing the same distance function.
'cd' 方法仅能优化 Frobenius norm。因为 NMF 可能是非凸的 （Due to the
underlying non-convexity of NMF）， 不同的方法可能得到不同的最小值，
即使在优化时使用的是相同的距离函数。


NMF is best used with the ``fit_transform`` method, which returns the matrix W.
The matrix H is stored into the fitted model in the ``components_`` attribute;
the method ``transform`` will decompose a new matrix X_new based on these
stored components::

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

.. topic:: Examples例子:

    * :ref:`sphx_glr_auto_examples_decomposition_plot_faces_decomposition.py`
    * :ref:`sphx_glr_auto_examples_applications_plot_topics_extraction_with_nmf_lda.py`
    * :ref:`sphx_glr_auto_examples_decomposition_plot_beta_divergence.py`

.. topic:: References参考文献:

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

Latent Dirichlet Allocation (LDA)

潜在 Dirichlet 分配 （Latent Dirichlet Allocation，LDA）
=================================

Latent Dirichlet Allocation is a generative probabilistic model for collections of
discrete dataset such as text corpora. It is also a topic model that is used for
discovering abstract topics from a collection of documents.
Latent Dirichlet Allocation 是离散数据集（如文本语料库）的集合的生成概率模型。
它也是一个主题模型，用于从文档集合中发现摘要内容。

The graphical model of LDA is a three-level Bayesian model:
LDA的图形模型是一个三级贝叶斯模型 （three-level Bayesian model）：

.. image:: ../images/lda_model_graph.png
   :align: center

When modeling text corpora, the model assumes the following generative process for
a corpus with :math:`D` documents and :math:`K` topics:

当建模文本语料库时，该模型对文档语料库 :math:`D` 和 主题语料库 :math:`K`
有以下假设：

  1. For each topic :math:`k`, draw :math:`\beta_k \sim \mathrm{Dirichlet}(\eta),\: k =1...K`
  针对每个主题 :math:`k`， 绘制 :math:`\beta_k \sim \mathrm{Dirichlet}(\eta),\: k =1...K`

  2. For each document :math:`d`, draw :math:`\theta_d \sim \mathrm{Dirichlet}(\alpha), \: d=1...D`
  针对每个文档 :math:`d`， 绘制 :math:`\theta_d \sim \mathrm{Dirichlet}(\alpha), \: d=1...D`

  3. For each word :math:`i` in document :math:`d`:
  针对文档 :math:`d` 中的每个单词 :math:`i` ：

    a. Draw a topic index :math:`z_{di} \sim \mathrm{Multinomial}(\theta_d)`
    绘制主题索引 :math:`z_{di} \sim \mathrm{Multinomial}(\theta_d)`
    b. Draw the observed word :math:`w_{ij} \sim \mathrm{Multinomial}(beta_{z_{di}}.)`
    绘制出观察到的单词： :math:`w_{ij} \sim \mathrm{Multinomial}(beta_{z_{di}}.)`



For parameter estimation, the posterior distribution is:
对于参数估计，后验分布为：


.. math::
  p(z, \theta, \beta |w, \alpha, \eta) =
    \frac{p(z, \theta, \beta|\alpha, \eta)}{p(w|\alpha, \eta)}

Since the posterior is intractable, variational Bayesian method
uses a simpler distribution :math:`q(z,\theta,\beta | \lambda, \phi, \gamma)`
to approximate it, and those variational parameters :math:`\lambda`, :math:`\phi`,
:math:`\gamma` are optimized to maximize the Evidence Lower Bound (ELBO):
由于后验分布难以处理，经变化的贝叶斯方法（variational Bayesian method）使用一个简单的分布
 :math:`q(z,\theta,\beta | \lambda, \phi, \gamma)` 来处理， 这些变化的参数
 :math:`\lambda`, :math:`\phi`, :math:`\gamma`
 经优化后使 Evidence Lower Bound (ELBO) 得到最大化。

.. math::
  \log\: P(w | \alpha, \eta) \geq L(w,\phi,\gamma,\lambda) \overset{\triangle}{=}
    E_{q}[\log\:p(w,z,\theta,\beta|\alpha,\eta)] - E_{q}[\log\:q(z, \theta, \beta)]

Maximizing ELBO is equivalent to minimizing the Kullback-Leibler(KL) divergence
between :math:`q(z,\theta,\beta)` and the true posterior
:math:`p(z, \theta, \beta |w, \alpha, \eta)`.
最大化 ELBO 等同于 最小化 Kullback-Leibler(KL) 的 :math:`q(z,\theta,\beta)`
与 :math:`p(z, \theta, \beta |w, \alpha, \eta)` 之间的差异。

:class:`LatentDirichletAllocation` implements online variational Bayes algorithm and supports
both online and batch update method.
While batch method updates variational variables after each full pass through the data,
online method updates variational variables from mini-batch data points.
:class:`LatentDirichletAllocation` 实现了在线 变分贝叶斯算法 （variational Bayes algorithm），
并支持在线和批量更新方法。
批处理方法在每次完全传递数据后更新变分变量，在线方法从小批量数据点更新变分变量。

.. note::

  Although online method is guaranteed to converge to a local optimum point, the quality of
  the optimum point and the speed of convergence may depend on mini-batch size and
  attributes related to learning rate setting.
  注意: 虽然在线方法保证收敛到局部最优点，
  但最优点的质量和收敛速度可能取决于小批量大小和学习率设置相关的属性。

When :class:`LatentDirichletAllocation` is applied on a "document-term" matrix, the matrix
will be decomposed into a "topic-term" matrix and a "document-topic" matrix. While
"topic-term" matrix is stored as :attr:`components_` in the model, "document-topic" matrix
can be calculated from ``transform`` method.
当 :class:`LatentDirichletAllocation` 应用于 "document-term" 矩阵时， 该矩阵将会被分解
称为 "topic-term" 矩阵和 "document-topic"  矩阵。 在模型中， "topic-term" 矩阵 以参数
:attr:`components_` 的形式保存， 而 "document-topic"  矩阵 可以从 ``transform`` 方法中
计算出来。

:class:`LatentDirichletAllocation` also implements ``partial_fit`` method. This is used
when data can be fetched sequentially.
:class:`LatentDirichletAllocation` 也实现了 ``partial_fit`` 方法。当数据可以顺序提取时使用。

.. topic:: Examples例子:

    * :ref:`sphx_glr_auto_examples_applications_plot_topics_extraction_with_nmf_lda.py`

.. topic:: References参考文献:

    * `"Latent Dirichlet Allocation"
      <https://www.cs.princeton.edu/~blei/papers/BleiNgJordan2003.pdf>`_
      D. Blei, A. Ng, M. Jordan, 2003

    * `"Online Learning for Latent Dirichlet Allocation”
      <https://www.cs.princeton.edu/~blei/papers/HoffmanBleiBach2010b.pdf>`_
      M. Hoffman, D. Blei, F. Bach, 2010

    * `"Stochastic Variational Inference"
      <http://www.columbia.edu/~jwp2128/Papers/HoffmanBleiWangPaisley2013.pdf>`_
      M. Hoffman, D. Blei, C. Wang, J. Paisley, 2013
