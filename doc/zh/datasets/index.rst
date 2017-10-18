.. _datasets:

=========================
数据集加载工具
=========================

.. currentmodule:: sklearn.datasets

该 ``sklearn.datasets`` 包装嵌入了 :ref:`Getting Started <loading_example_dataset>` 部分中介绍的一些小型玩具数据集。

为了在控制数据的统计特性（通常是特征的 correlation （相关性）和 informativeness （信息性））的同时评估数据集 (``n_samples`` 和 ``n_features``) 的规模的影响，也可以生成综合数据。

这个软件包还具有帮助用户获取更大的数据集的功能，这些数据集通常由机器学习社区使用，用于对来自 'real world' 的数据进行检测算法。


通用数据集 API
=======================

对于不同类型的数据集，有三种不同类型的数据集接口。最简单的是样品图像的界面，下面在 :ref:`sample_images` 部分中进行了描述。

数据集生成函数和 svmlight 加载器分享了一个过于简单的接口，返回一个由 ``n_samples`` * ``n_features`` 组成的 tuple ``(X, y)`` 其中的 ``X`` 是 numpy 数组 ``y`` 是包含目标值的 ``n_samples`` 长度的数组

玩具数据集以及 'real world' 数据集和从 mldata.org 获取的数据集具有更复杂的结构。这些函数返回一个类似于字典的对象包含至少两项：一个具有 ``data`` 键的 ``n_samples`` * ``n_features`` 形状的数组（除了20个新组之外）和一个具有 ``target`` 键的包含 target values （目标值）的 ``n_samples`` 长度的 numpy 数组。

数据集还包含一些描述 ``DESCR`` ，一些包含 ``feature_names`` 和 ``target_names``。有关详细信息，请参阅下面的数据集说明

玩具数据集
=================

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

样本图片
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

样本生成器
=================

此外，scikit-learn 包括各种随机样本的生成器，可以用来建立控制的大小和复杂性人工数据集。

分类和聚类生成器
--------------------------------------------

这些生成器产生的特征和相应的离散矩阵目标。

单标签
~~~~~~~~~~~~

:func:`make_blobs` 和 :func:`make_classification` 通过分配每个类的一个或多个正态分布的点的群集创建的多类数据集。 :func:`make_blobs` 提供了更大的控制对于中心和各簇的标准偏差，并用于演示聚类。 :func:`make_classification` 专门通过引入噪声相关，多余的和均匀的特点；多高斯集群每类的特征空间上的线性变换。

:func:`make_gaussian_quantiles` 分 single Gaussian cluster （单高斯簇）成近乎相等大小的同心超球面分离。 :func:`make_hastie_10_2` 产生类似的二进制、10维问题。

.. image:: ../auto_examples/datasets/images/sphx_glr_plot_random_dataset_001.png
   :target: ../auto_examples/datasets/plot_random_dataset.html
   :scale: 50
   :align: center

:func:`make_circles` and :func:`make_moons`生成二维二分类数据集时，某些算法的挑战（如质心聚类或线性分类），包括可选的高斯噪声。它们对于可视化是有用的用球面决策边界生成二值分类高斯数据。

多标签
~~~~~~~~~~~~~~

:func:`make_multilabel_classification` 生成随机样本与多个标签，反映一个 bag of words （词袋）从 mixture of topics（混合的主题画）。每个文档的主题数是从泊松分布中提取的，并且主题本身是从固定的随机分布中提取的。同样地，单词的数目是从泊松图中提取的，用多项式抽取的单词，其中每个主题定义了单词的概率分布。相对于真正的简化包括 bag-of-words mixtures （单词混合包）：

* 每个主题词分布都是独立绘制的，在现实中，所有这些都会受到稀疏基分布的影响，并将相互关联。
* 对于从多个主题生成的文档，所有主题在生成单词包时都是同等权重的。
* 没有标签的随机文件，而不是从基础分布的文档

.. image:: ../auto_examples/datasets/images/sphx_glr_plot_random_multilabel_dataset_001.png
   :target: ../auto_examples/datasets/plot_random_multilabel_dataset.html
   :scale: 50
   :align: center

二分聚类
~~~~~~~~~~~~~~~

.. autosummary::

   :toctree: ../modules/generated/
   :template: function.rst

   make_biclusters
   make_checkerboard


回归生成器
-------------------------

:func:`make_regression` 产生回归的目标作为一个可选择的稀疏线性组合的随机特性，噪声。它的信息特征可能是不相关的，或低秩（少数特征占大多数的方差）。

其他回归生成器产生确定性的随机特征函数。 :func:`make_sparse_uncorrelated` 产生目标为四具有固定系数的线性组合。其他编码明确的非线性关系：:func:`make_friedman1` 通过多项式和正弦相关变换； :func:`make_friedman2` 包括特征相乘与交互；和 :func:`make_friedman3` 类似目标的反正切变换。


流形学习生成器
--------------------------------

.. autosummary::

   :toctree: ../modules/generated/
   :template: function.rst
   
   make_s_curve
   make_swiss_roll

生成器分解
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
===================================================

scikit-learn 中有加载svmlight / libsvm格式的数据集的功能函数。此种格式中，每行
采用如 ``<label> <feature-id>:<feature-value><feature-id>:<feature-value> ...`` 
的形式。这种格式尤其适合稀疏数据集，在该模块中，数据集 ``X`` 使用的是scipy稀疏CSR矩阵，
特征集 ``y`` 使用的是numpy数组。

你可以通过如下步骤加载数据集::

  >>> from sklearn.datasets import load_svmlight_file
  >>> X_train, y_train = load_svmlight_file("/path/to/train_dataset.txt")
  ...                                                         # doctest: +SKIP

你也可以一次加载两个或多个的数据集::

  >>> X_train, y_train, X_test, y_test = load_svmlight_files(
  ...     ("/path/to/train_dataset.txt", "/path/to/test_dataset.txt"))
  ...                                                         # doctest: +SKIP

这种情况下，保证了 ``X_train`` 和 ``X_test`` 具有相同的特征数量。
固定特征的数量也可以得到同样的结果::

  >>> X_test, y_test = load_svmlight_file(
  ...     "/path/to/test_dataset.txt", n_features=X_train.shape[1])
  ...                                                         # doctest: +SKIP

.. topic:: 相关链接:

 _`svmlight / libsvm 格式的公共数据集`: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets

 _`更快的API兼容的实现`: https://github.com/mblondel/svmlight-loader

.. _external_datasets:

从外部数据集加载
=============================================

scikit-learn使用任何存储为numpy数组或者scipy稀疏数组的数值数据。
其他可以转化成数值数组的类型也可以接受，如pandas中的DataFrame。

以下推荐一些将标准纵列形式的数据转换为scikit-learn可以使用的格式的方法:

* `pandas.io <https://pandas.pydata.org/pandas-docs/stable/io.html>`_ 
  提供了从常见格式(包括CSV,Excel,JSON,SQL等)中读取数据的工具.DateFrame 也可以从由
  元组或者字典组成的列表构建而成.Pandas能顺利的处理异构的数据，并且提供了处理和转换
  成方便scikit-learn使用的数值数据的工具。

* `scipy.io <https://docs.scipy.org/doc/scipy/reference/io.html>`_ 
  专门处理科学计算领域经常使用的二进制格式，例如.mat和.arff格式的内容。

* `numpy/routines.io <https://docs.scipy.org/doc/numpy/reference/routines.io.html>`_
  将纵列形式的数据标准的加载为numpy数组

* scikit-learn的 :func:`datasets.load_svmlight_file`处理svmlight或者libSVM稀疏矩阵

* scikit-learn的 :func:`datasets.load_files` 处理文本文件组成的目录，每个目录名是每个
  类别的名称，每个目录内的每个文件对应该类别的一个样本

对于一些杂项数据，例如图像，视屏，音频。您可以参考:

* `skimage.io <http://scikit-image.org/docs/dev/api/skimage.io.html>`_ 或
  `Imageio <https://imageio.readthedocs.io/en/latest/userapi.html>`_ 
  将图像或者视屏加载为numpy数组
* `scipy.misc.imread <https://docs.scipy.org/doc/scipy/reference/generated/scipy.
  misc.imread.html#scipy.misc.imread>`_ (requires the `Pillow
  <https://pypi.python.org/pypi/Pillow>`_ package)将各种图像文件格式加载为
  像素灰度数据

* `scipy.io.wavfile.read 
  <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.io.wavfile.read.html>`_ 
  将WAV文件读入一个numpy数组

存储为字符串的无序(或者名字)特征(在pandas的DataFrame中很常见)需要转换为整数，当整数类别变量
被编码成独热变量(:class:`sklearn.preprocessing.OneHotEncoder`)或类似数据时，它或许可以被最好的利用。
参见 :ref:`preprocessing`.

注意：如果你要管理你的数值数据，建议使用优化后的文件格式来减少数据加载时间,例如HDF5。像
H5Py, PyTables和pandas等的各种库提供了一个Python接口，来读写该格式的数据。

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
