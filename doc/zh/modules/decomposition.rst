.. _decompositions:


=================================================================
Decomposing signals in components (matrix factorization problems)
=================================================================

.. currentmodule:: sklearn.decomposition


.. _PCA:


Principal component analysis (PCA)
==================================

Exact PCA and probabilistic interpretation
------------------------------------------

PCA is used to decompose a multivariate dataset in a set of successive
orthogonal components that explain a maximum amount of the variance. In
scikit-learn, :class:`PCA` is implemented as a *transformer* object
that learns :math:`n` components in its ``fit`` method, and can be used on new
data to project it on these components.

The optional parameter ``whiten=True`` makes it possible to
project the data onto the singular space while scaling each component
to unit variance. This is often useful if the models down-stream make
strong assumptions on the isotropy of the signal: this is for example
the case for Support Vector Machines with the RBF kernel and the K-Means
clustering algorithm.

Below is an example of the iris dataset, which is comprised of 4
features, projected on the 2 dimensions that explain most variance:

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_pca_vs_lda_001.png
    :target: ../auto_examples/decomposition/plot_pca_vs_lda.html
    :align: center
    :scale: 75%


The :class:`PCA` object also provides a
probabilistic interpretation of the PCA that can give a likelihood of
data based on the amount of variance it explains. As such it implements a
`score` method that can be used in cross-validation:

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_pca_vs_fa_model_selection_001.png
    :target: ../auto_examples/decomposition/plot_pca_vs_fa_model_selection.html
    :align: center
    :scale: 75%


.. topic:: Examples:

    * :ref:`sphx_glr_auto_examples_decomposition_plot_pca_vs_lda.py`
    * :ref:`sphx_glr_auto_examples_decomposition_plot_pca_vs_fa_model_selection.py`


.. _IncrementalPCA:

Incremental PCA
---------------

The :class:`PCA` object is very useful, but has certain limitations for
large datasets. The biggest limitation is that :class:`PCA` only supports
batch processing, which means all of the data to be processed must fit in main
memory. The :class:`IncrementalPCA` object uses a different form of
processing and allows for partial computations which almost
exactly match the results of :class:`PCA` while processing the data in a
minibatch fashion. :class:`IncrementalPCA` makes it possible to implement
out-of-core Principal Component Analysis either by:

 * Using its ``partial_fit`` method on chunks of data fetched sequentially
   from the local hard drive or a network database.

 * Calling its fit method on a memory mapped file using ``numpy.memmap``.

:class:`IncrementalPCA` only stores estimates of component and noise variances,
in order update ``explained_variance_ratio_`` incrementally. This is why
memory usage depends on the number of samples per batch, rather than the
number of samples to be processed in the dataset.

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

PCA using randomized SVD
------------------------

It is often interesting to project data to a lower-dimensional
space that preserves most of the variance, by dropping the singular vector
of components associated with lower singular values.

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

The class :class:`PCA` used with the optional parameter
``svd_solver='randomized'`` is very useful in that case: since we are going
to drop most of the singular vectors it is much more efficient to limit the
computation to an approximated estimate of the singular vectors we will keep
to actually perform the transform.

For instance, the following shows 16 sample portraits (centered around
0.0) from the Olivetti dataset. On the right hand side are the first 16
singular vectors reshaped as portraits. Since we only require the top
16 singular vectors of a dataset with size :math:`n_{samples} = 400`
and :math:`n_{features} = 64 \times 64 = 4096`, the computation time is
less than 1s:

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

If we note :math:`n_{\max} = \max(n_{\mathrm{samples}}, n_{\mathrm{features}})` and
:math:`n_{\min} = \min(n_{\mathrm{samples}}, n_{\mathrm{features}})`, the time complexity
of the randomized :class:`PCA` is :math:`O(n_{\max}^2 \cdot n_{\mathrm{components}})`
instead of :math:`O(n_{\max}^2 \cdot n_{\min})` for the exact method
implemented in :class:`PCA`.

The memory footprint of randomized :class:`PCA` is also proportional to
:math:`2 \cdot n_{\max} \cdot n_{\mathrm{components}}` instead of :math:`n_{\max}
\cdot n_{\min}` for the exact method.

Note: the implementation of ``inverse_transform`` in :class:`PCA` with
``svd_solver='randomized'`` is not the exact inverse transform of
``transform`` even when ``whiten=False`` (default).


.. topic:: Examples:

    * :ref:`sphx_glr_auto_examples_applications_plot_face_recognition.py`
    * :ref:`sphx_glr_auto_examples_decomposition_plot_faces_decomposition.py`

.. topic:: References:

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

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_kernel_pca_001.png
    :target: ../auto_examples/decomposition/plot_kernel_pca.html
    :align: center
    :scale: 75%

.. topic:: Examples:

    * :ref:`sphx_glr_auto_examples_decomposition_plot_kernel_pca.py`


.. _SparsePCA:

Sparse principal components analysis (SparsePCA and MiniBatchSparsePCA)
-----------------------------------------------------------------------

:class:`SparsePCA` is a variant of PCA, with the goal of extracting the
set of sparse components that best reconstruct the data.

Mini-batch sparse PCA (:class:`MiniBatchSparsePCA`) is a variant of
:class:`SparsePCA` that is faster but less accurate. The increased speed is
reached by iterating over small chunks of the set of features, for a given
number of iterations.


Principal component analysis (:class:`PCA`) has the disadvantage that the
components extracted by this method have exclusively dense expressions, i.e.
they have non-zero coefficients when expressed as linear combinations of the
original variables. This can make interpretation difficult. In many cases,
the real underlying components can be more naturally imagined as sparse
vectors; for example in face recognition, components might naturally map to
parts of faces.

Sparse principal components yields a more parsimonious, interpretable
representation, clearly emphasizing which of the original features contribute
to the differences between samples.

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
For more details on how to use Sparse PCA, see the Examples section, below.


.. |spca_img| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_005.png
   :target: ../auto_examples/decomposition/plot_faces_decomposition.html
   :scale: 60%

.. centered:: |pca_img| |spca_img|

Note that there are many different formulations for the Sparse PCA
problem. The one implemented here is based on [Mrl09]_ . The optimization
problem solved is a PCA problem (dictionary learning) with an
:math:`\ell_1` penalty on the components:

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

.. note::

  While in the spirit of an online algorithm, the class
  :class:`MiniBatchSparsePCA` does not implement ``partial_fit`` because
  the algorithm is online along the features direction, not the samples
  direction.

.. topic:: Examples:

   * :ref:`sphx_glr_auto_examples_decomposition_plot_faces_decomposition.py`

.. topic:: References:

  .. [Mrl09] `"Online Dictionary Learning for Sparse Coding"
     <http://www.di.ens.fr/sierra/pdfs/icml09.pdf>`_
     J. Mairal, F. Bach, J. Ponce, G. Sapiro, 2009
  .. [Jen09] `"Structured Sparse Principal Component Analysis"
     <www.di.ens.fr/~fbach/sspca_AISTATS2010.pdf>`_
     R. Jenatton, G. Obozinski, F. Bach, 2009


.. _LSA:

Truncated singular value decomposition and latent semantic analysis
===================================================================

:class:`TruncatedSVD` implements a variant of singular value decomposition
(SVD) that only computes the :math:`k` largest singular values,
where :math:`k` is a user-specified parameter.

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

.. note::
    LSA is also known as latent semantic indexing, LSI,
    though strictly that refers to its use in persistent indexes
    for information retrieval purposes.

Mathematically, truncated SVD applied to training samples :math:`X`
produces a low-rank approximation :math:`X`:

.. math::
    X \approx X_k = U_k \Sigma_k V_k^\top

After this operation, :math:`U_k \Sigma_k^\top`
is the transformed training set with :math:`k` features
(called ``n_components`` in the API).

To also transform a test set :math:`X`, we multiply it with :math:`V_k`:

.. math::
    X' = X V_k

.. note::
    Most treatments of LSA in the natural language processing (NLP)
    and information retrieval (IR) literature
    swap the axes of the matrix :math:`X` so that it has shape
    ``n_features`` × ``n_samples``.
    We present LSA in a different way that matches the scikit-learn API better,
    but the singular values found are the same.

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

While the :class:`TruncatedSVD` transformer
works with any (sparse) feature matrix,
using it on tf–idf matrices is recommended over raw frequency counts
in an LSA/document processing setting.
In particular, sublinear scaling and inverse document frequency
should be turned on (``sublinear_tf=True, use_idf=True``)
to bring the feature values closer to a Gaussian distribution,
compensating for LSA's erroneous assumptions about textual data.

.. topic:: Examples:

   * :ref:`sphx_glr_auto_examples_text_document_clustering.py`

.. topic:: References:

  * Christopher D. Manning, Prabhakar Raghavan and Hinrich Schütze (2008),
    *Introduction to Information Retrieval*, Cambridge University Press,
    chapter 18: `Matrix decompositions & latent semantic indexing
    <http://nlp.stanford.edu/IR-book/pdf/18lsi.pdf>`_


.. _DictionaryLearning:

Dictionary Learning
===================

.. _SparseCoder:

Sparse coding with a precomputed dictionary
-------------------------------------------

The :class:`SparseCoder` object is an estimator that can be used to transform signals
into sparse linear combination of atoms from a fixed, precomputed dictionary
such as a discrete wavelet basis. This object therefore does not
implement a ``fit`` method. The transformation amounts
to a sparse coding problem: finding a representation of the data as a linear
combination of as few dictionary atoms as possible. All variations of
dictionary learning implement the following transform methods, controllable via
the ``transform_method`` initialization parameter:

* Orthogonal matching pursuit (:ref:`omp`)

* Least-angle regression (:ref:`least_angle_regression`)

* Lasso computed by least-angle regression

* Lasso using coordinate descent (:ref:`lasso`)

* Thresholding

Thresholding is very fast but it does not yield accurate reconstructions.
They have been shown useful in literature for classification tasks. For image
reconstruction tasks, orthogonal matching pursuit yields the most accurate,
unbiased reconstruction.

The dictionary learning objects offer, via the ``split_code`` parameter, the
possibility to separate the positive and negative values in the results of
sparse coding. This is useful when dictionary learning is used for extracting
features that will be used for supervised learning, because it allows the
learning algorithm to assign different weights to negative loadings of a
particular atom, from to the corresponding positive loading.

The split code for a single sample has length ``2 * n_components``
and is constructed using the following rule: First, the regular code of length
``n_components`` is computed. Then, the first ``n_components`` entries of the
``split_code`` are
filled with the positive part of the regular code vector. The second half of
the split code is filled with the negative part of the code vector, only with
a positive sign. Therefore, the split_code is non-negative.


.. topic:: Examples:

    * :ref:`sphx_glr_auto_examples_decomposition_plot_sparse_coding.py`


Generic dictionary learning
---------------------------

Dictionary learning (:class:`DictionaryLearning`) is a matrix factorization
problem that amounts to finding a (usually overcomplete) dictionary that will
perform good at sparsely encoding the fitted data.

Representing data as sparse combinations of atoms from an overcomplete
dictionary is suggested to be the way the mammal primary visual cortex works.
Consequently, dictionary learning applied on image patches has been shown to
give good results in image processing tasks such as image completion,
inpainting and denoising, as well as for supervised recognition tasks.

Dictionary learning is an optimization problem solved by alternatively updating
the sparse code, as a solution to multiple Lasso problems, considering the
dictionary fixed, and then updating the dictionary to best fit the sparse code.

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

The following image shows how a dictionary learned from 4x4 pixel image patches
extracted from part of the image of a raccoon face looks like.


.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_image_denoising_001.png
    :target: ../auto_examples/decomposition/plot_image_denoising.html
    :align: center
    :scale: 50%


.. topic:: Examples:

  * :ref:`sphx_glr_auto_examples_decomposition_plot_image_denoising.py`


.. topic:: References:

  * `"Online dictionary learning for sparse coding"
    <http://www.di.ens.fr/sierra/pdfs/icml09.pdf>`_
    J. Mairal, F. Bach, J. Ponce, G. Sapiro, 2009

.. _MiniBatchDictionaryLearning:

小批量字典学习
------------------------------

:class:`MiniBatchDictionaryLearning` 实现了更快、更适合大型数据集的字典学习算法，但该版本不太准确。

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
:math:`\epsilon` 被认为是根据高斯分布的噪声项，平均值为0，协方差为 :math:`\Psi` （即 :math:`\epsilon \sim \mathcal{N}(0, \Psi)`）， 
:math:`\mu` 是一些任意的偏移向量。 这样一个模型被称为 "生成"，因为它描述了如何从 :math:`h_i` 生成 :math:`x_i` 。
如果我们使用所有的 :math:`x_i` 作为列来形成一个矩阵 :math:`\mathbf{X}`，并将所有的 :math:`h_i` 作为矩阵 :math:`\mathbf{H}` 的列，
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
====================================

独立分量分析将多变量信号分解为最大独立的加性子组件。 
它使用 :class:`Fast ICA <FastICA>` 算法在 scikit-learn 中实现。 
通常，ICA 不用于降低维度，而是用于分离叠加信号。 
由于ICA模型不包括噪声项，因此要使模型正确，必须应用美白。 
这可以在内部使用 whiten 参数或手动使用其中一种PCA变体进行。

通常用于分离混合信号（称为 *盲源分离* 的问题），如下例所示:

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_ica_blind_source_separation_001.png
    :target: ../auto_examples/decomposition/plot_ica_blind_source_separation.html
    :align: center
    :scale: 60%


ICA也可以被用作发现具有一些稀疏性的组件的另一个非线性分解:

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
===============================================

NMF 与 Frobenius 规范
---------------------------

:class:`NMF` [1]_ 是一种替代的分解方法，假设数据和分量是非负数的。 
在数据矩阵不包含负值的情况下，可以插入 :class:`NMF` 而不是 :class:`PCA` 或其变体。 
通过优化 :math:`X` 与矩阵乘积 :math:`WH` 之间的距离 :math:`d` ，可以将样本 :math:`X` 分解为非负元素的两个矩阵 :math:`W` 和 :math:`H`。 
最广泛使用的距离函数是 Frobenius 方程的平方，这是欧几里德范数到矩阵的明显延伸:

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

:class:`NMF` 正规化 W 和 H . 公共函数 :func:`non_negative_factorization` 允许通过 :attr:`regularization` 属性进行更精细的控制，并且可以仅将 W，仅 H 或两者正规化。

NMF 具有 beta-divergence
--------------------------

如前所述，最广泛使用的距离函数是平方 Frobenius 范数，这是欧几里得范数到矩阵的明显延伸:

.. math::
    d_{\mathrm{Fro}}(X, Y) = \frac{1}{2} ||X - Y||_{Fro}^2 = \frac{1}{2} \sum_{i,j} (X_{ij} - {Y}_{ij})^2

其他距离函数可用于 NMF，例如（广义） Kullback-Leibler(KL) 发散，也称为 I-divergence:

.. math::
    d_{KL}(X, Y) = \sum_{i,j} (X_{ij} \log(\frac{X_{ij}}{Y_{ij}}) - X_{ij} + Y_{ij})

或者， Itakura-Saito(IS) 分歧:

.. math::
    d_{IS}(X, Y) = \sum_{i,j} (\frac{X_{ij}}{Y_{ij}} - \log(\frac{X_{ij}}{Y_{ij}}) - 1)

这三个距离是 beta-divergence 家族的特殊情况，分别为 :math:`\beta = 2, 1, 0` [6]_ 。 beta-divergence 定义如下:

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

.. topic:: 参考:

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

LDA的图形模型是一个 three-level 贝叶斯模型:

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

由于后验是棘手的，变分贝叶斯方法使用更简单的分布 :math:`q(z,\theta,\beta | \lambda, \phi, \gamma)` 近似，
并且优化了这些变分参数  :math:`\lambda`, :math:`\phi`, :math:`\gamma` 最大化证据下限 (ELBO):

.. math::
  \log\: P(w | \alpha, \eta) \geq L(w,\phi,\gamma,\lambda) \overset{\triangle}{=}
    E_{q}[\log\:p(w,z,\theta,\beta|\alpha,\eta)] - E_{q}[\log\:q(z, \theta, \beta)]

最大化 ELBO 相当于最小化 :math:`q(z,\theta,\beta)` 和真实后 :math:`p(z, \theta, \beta |w, \alpha, \eta)` 之间的 Kullback-Leibler(KL) 发散。

:class:`LatentDirichletAllocation` 实现在线变分贝叶斯算法，支持在线和批量更新方法。
批处理方法在每次完全传递数据后更新变分变量，联机方法从小批量数据点更新变分变量。

.. note::
  虽然在线方法保证收敛到局部最优点，最优点的质量和收敛速度可能取决于小批量大小和学习率设置相关的属性。

当 :class:`LatentDirichletAllocation` 应用于 "文档术语" 矩阵时，矩阵将被分解为 "主题术语" 矩阵和 "文档主题" 矩阵。
虽然 "主题术语" 矩阵在模型中被存储为 :attr:`components_` ，但是可以通过变换方法计算 "文档主题" 矩阵。

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
