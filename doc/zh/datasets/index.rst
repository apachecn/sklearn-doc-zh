.. _datasets:

=========================
Dataset loading utilities（数据集加载工具）
=========================

.. currentmodule:: sklearn.datasets

该 ``sklearn.datasets`` 包装嵌入了 :ref:`Getting Started <loading_example_dataset>` 部分中介绍的一些小型玩具数据集。

为了在控制数据的统计特性（通常是特征的 correlation （相关性）和 informativeness （信息性））的同时评估数据集 (``n_samples`` 和 ``n_features``) 的规模的影响，也可以生成综合数据。

这个软件包还具有帮助用户获取更大的数据集的功能，这些数据集通常由机器学习社区使用，用于对来自 'real world' 的数据进行检测算法。


General dataset API（通用数据集 API）
===================

对于不同类型的数据集，有三种不同类型的数据集接口。最简单的是样品图像的界面，下面在 :ref:`sample_images` 部分中进行了描述。

数据集生成函数和 svmlight 加载器分享了一个过于简单的接口，返回一个由 ``n_samples`` * ``n_features`` 组成的 tuple ``(X, y)`` 其中的 ``X`` 是 numpy 数组 ``y`` 是包含目标值的 ``n_samples`` 长度的数组

玩具数据集以及 'real world' 数据集和从 mldata.org 获取的数据集具有更复杂的结构。这些函数返回一个类似于字典的对象包含至少两项：一个具有 ``data`` 键的 ``n_samples`` * ``n_features`` 形状的数组（除了20个新组之外）和一个具有 ``target`` 键的包含 target values （目标值）的 ``n_samples`` 长度的 numpy 数组。

数据集还包含一些描述 ``DESCR`` ，一些包含 ``feature_names`` 和 ``target_names``。有关详细信息，请参阅下面的数据集说明

Toy datasets（玩具数据集）
============

scikit-learn 有一些小型标准数据集，不需要从某个外部网站下载任何文件。

.. autosummary::

   :toctree: ../modules/generated/
   :template: function.rst

   load_boston
   load_iris
   load_diabetes
   load_digits
   load_linnerud
   load_wine
   load_breast_cancer


这些数据集有助于快速说明在 scikit 中实现的各种算法的行为。然而，它们往往太小，无法代表真实世界的机器学习任务。

.. _sample_images:

Sample images（样本图片）
=============

scikit 还嵌入了几个样本 JPEG 图片公布了通过他们的作者共同授权。这些图像对 test algorithms （测试算法）和 pipeline on 2D data （二维数据管道）是有用的。

.. autosummary::

   :toctree: ../modules/generated/
   :template: function.rst

   load_sample_images
   load_sample_image

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_color_quantization_001.png
   :target: ../auto_examples/cluster/plot_color_quantization.html
   :scale: 30
   :align: right


.. warning::

默认编码的图像是基于 ``uint8`` dtype 到空闲内存。通常，如果把输入转换为浮点数表示，机器学习算法的效果最好。另外，如果你计划使用 ``matplotlib.pyplpt.imshow`` 别忘了尺度范围 0 - 1，如下面的示例所做的。

.. topic:: 示例:

    * :ref:`sphx_glr_auto_examples_cluster_plot_color_quantization.py`


.. _sample_generators:

Sample generators（样本生成器）
=================

此外，scikit-learn 包括各种随机样本的生成器，可以用来建立控制的大小和复杂性人工数据集。

Generators for classification and clustering（分类和聚类生成器）
--------------------------------------------

这些生成器产生的特征和相应的离散矩阵目标。

Single label（单标签）
~~~~~~~~~~~~

:func:`make_blobs` 和 :func:`make_classification` 通过分配每个类的一个或多个正态分布的点的群集创建的多类数据集。 :func:`make_blobs` 提供了更大的控制对于中心和各簇的标准偏差，并用于演示聚类。 :func:`make_classification` 专门通过引入噪声相关，多余的和均匀的特点；多高斯集群每类的特征空间上的线性变换。

:func:`make_gaussian_quantiles` 分 single Gaussian cluster （单高斯簇）成近乎相等大小的同心超球面分离。 :func:`make_hastie_10_2` 产生类似的二进制、10维问题。

.. image:: ../auto_examples/datasets/images/sphx_glr_plot_random_dataset_001.png
   :target: ../auto_examples/datasets/plot_random_dataset.html
   :scale: 50
   :align: center

:func:`make_circles` and :func:`make_moons`生成二维二分类数据集时，某些算法的挑战（如质心聚类或线性分类），包括可选的高斯噪声。它们对于可视化是有用的用球面决策边界生成二值分类高斯数据。

Multilabel（多标签）
~~~~~~~~~~

:func:`make_multilabel_classification` 生成随机样本与多个标签，反映一个 bag of words （词袋）从 mixture of topics（混合的主题画）。每个文档的主题数是从泊松分布中提取的，并且主题本身是从固定的随机分布中提取的。同样地，单词的数目是从泊松图中提取的，用多项式抽取的单词，其中每个主题定义了单词的概率分布。相对于真正的简化包括 bag-of-words mixtures （单词混合包）：

* 每个主题词分布都是独立绘制的，在现实中，所有这些都会受到稀疏基分布的影响，并将相互关联。
* 对于从多个主题生成的文档，所有主题在生成单词包时都是同等权重的。
* 没有标签的随机文件，而不是从基础分布的文档

.. image:: ../auto_examples/datasets/images/sphx_glr_plot_random_multilabel_dataset_001.png
   :target: ../auto_examples/datasets/plot_random_multilabel_dataset.html
   :scale: 50
   :align: center

Biclustering（双聚类）
~~~~~~~~~~~~

.. autosummary::

   :toctree: ../modules/generated/
   :template: function.rst

   make_biclusters
   make_checkerboard


Generators for regression（回归生成器）
-------------------------

:func:`make_regression` 产生回归的目标作为一个可选择的稀疏线性组合的随机特性，噪声。它的信息特征可能是不相关的，或低秩（少数特征占大多数的方差）。

其他回归生成器产生确定性的随机特征函数。 :func:`make_sparse_uncorrelated` 产生目标为四具有固定系数的线性组合。其他编码明确的非线性关系：:func:`make_friedman1` 通过多项式和正弦相关变换； :func:`make_friedman2` 包括特征相乘与交互；和 :func:`make_friedman3` 类似目标的反正切变换。


Generators for manifold learning（流形学习生成器）
--------------------------------

.. autosummary::

   :toctree: ../modules/generated/
   :template: function.rst
   
   make_s_curve
   make_swiss_roll

Generators for decomposition（生成器分解）
----------------------------

.. autosummary::

   :toctree: ../modules/generated/
   :template: function.rst

   make_low_rank_matrix
   make_sparse_coded_signal
   make_spd_matrix
   make_sparse_spd_matrix


.. _libsvm_loader:

Datasets in svmlight / libsvm format
====================================

scikit-learn includes utility functions for loading
datasets in the svmlight / libsvm format. In this format, each line
takes the form ``<label> <feature-id>:<feature-value>
<feature-id>:<feature-value> ...``. This format is especially suitable for sparse datasets.
In this module, scipy sparse CSR matrices are used for ``X`` and numpy arrays are used for ``y``.

You may load a dataset like as follows::

  >>> from sklearn.datasets import load_svmlight_file
  >>> X_train, y_train = load_svmlight_file("/path/to/train_dataset.txt")
  ...                                                         # doctest: +SKIP

You may also load two (or more) datasets at once::

  >>> X_train, y_train, X_test, y_test = load_svmlight_files(
  ...     ("/path/to/train_dataset.txt", "/path/to/test_dataset.txt"))
  ...                                                         # doctest: +SKIP

In this case, ``X_train`` and ``X_test`` are guaranteed to have the same number
of features. Another way to achieve the same result is to fix the number of
features::

  >>> X_test, y_test = load_svmlight_file(
  ...     "/path/to/test_dataset.txt", n_features=X_train.shape[1])
  ...                                                         # doctest: +SKIP

.. topic:: Related links:

 _`Public datasets in svmlight / libsvm format`: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets

 _`Faster API-compatible implementation`: https://github.com/mblondel/svmlight-loader

.. _external_datasets:

Loading from external datasets
==============================

scikit-learn works on any numeric data stored as numpy arrays or scipy sparse
matrices. Other types that are convertible to numeric arrays such as pandas
DataFrame are also acceptable.
 
Here are some recommended ways to load standard columnar data into a 
format usable by scikit-learn: 

* `pandas.io <https://pandas.pydata.org/pandas-docs/stable/io.html>`_ 
  provides tools to read data from common formats including CSV, Excel, JSON
  and SQL. DataFrames may also be constructed from lists of tuples or dicts.
  Pandas handles heterogeneous data smoothly and provides tools for
  manipulation and conversion into a numeric array suitable for scikit-learn.
* `scipy.io <https://docs.scipy.org/doc/scipy/reference/io.html>`_ 
  specializes in binary formats often used in scientific computing 
  context such as .mat and .arff
* `numpy/routines.io <https://docs.scipy.org/doc/numpy/reference/routines.io.html>`_
  for standard loading of columnar data into numpy arrays
* scikit-learn's :func:`datasets.load_svmlight_file` for the svmlight or libSVM
  sparse format
* scikit-learn's :func:`datasets.load_files` for directories of text files where
  the name of each directory is the name of each category and each file inside
  of each directory corresponds to one sample from that category

For some miscellaneous data such as images, videos, and audio, you may wish to
refer to:

* `skimage.io <http://scikit-image.org/docs/dev/api/skimage.io.html>`_ or
  `Imageio <https://imageio.readthedocs.io/en/latest/userapi.html>`_ 
  for loading images and videos to numpy arrays
* `scipy.misc.imread <https://docs.scipy.org/doc/scipy/reference/generated/scipy.
  misc.imread.html#scipy.misc.imread>`_ (requires the `Pillow
  <https://pypi.python.org/pypi/Pillow>`_ package) to load pixel intensities
  data from various image file formats
* `scipy.io.wavfile.read 
  <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.io.wavfile.read.html>`_ 
  for reading WAV files into a numpy array

Categorical (or nominal) features stored as strings (common in pandas DataFrames) 
will need converting to integers, and integer categorical variables may be best 
exploited when encoded as one-hot variables 
(:class:`sklearn.preprocessing.OneHotEncoder`) or similar. 
See :ref:`preprocessing`.

Note: if you manage your own numerical data it is recommended to use an 
optimized file format such as HDF5 to reduce data load times. Various libraries
such as H5Py, PyTables and pandas provides a Python interface for reading and 
writing data in that format.

.. make sure everything is in a toc tree

.. toctree::
    :maxdepth: 2
    :hidden:

    olivetti_faces
    twenty_newsgroups
    mldata
    labeled_faces
    covtype
    rcv1


.. include:: olivetti_faces.rst

.. include:: twenty_newsgroups.rst

.. include:: mldata.rst

.. include:: labeled_faces.rst

.. include:: covtype.rst

.. include:: rcv1.rst

.. _boston_house_prices:

.. include:: ../sklearn/datasets/descr/boston_house_prices.rst

.. _breast_cancer:

.. include:: ../sklearn/datasets/descr/breast_cancer.rst

.. _diabetes:

.. include:: ../sklearn/datasets/descr/diabetes.rst

.. _digits:

.. include:: ../sklearn/datasets/descr/digits.rst

.. _iris:

.. include:: ../sklearn/datasets/descr/iris.rst

.. _linnerud:

.. include:: ../sklearn/datasets/descr/linnerud.rst
