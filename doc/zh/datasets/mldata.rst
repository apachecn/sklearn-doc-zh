..
    For doctests:

    >>> import numpy as np
    >>> import os

.. _mldata:

Downloading datasets from the mldata.org repository（从 mldata.org 上下载数据集）
===================================================

`mldata.org <http://mldata.org>`_ 是一个开放的用于机器学习数据的 public repository ,由 `PASCAL network <http://www.pascal-network.org>`_ 支持。

``sklearn.datasets`` 包可以直接从 repository 下载数据集，通过使用函数 :func:`sklearn.datasets.fetch_mldata` .

举个例子，下载 MNIST 数字识别数据集::

  >>> from sklearn.datasets import fetch_mldata
  >>> mnist = fetch_mldata('MNIST original', data_home=custom_data_home)

MNIST 数据集包含有 70000 个手写数字实例，每个实例都是带有从 0 到 9 的标签，并且像素大小为 28x28::

  >>> mnist.data.shape
  (70000, 784)
  >>> mnist.target.shape
  (70000,)
  >>> np.unique(mnist.target)
  array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.])

第一次下载之后，数据集被缓存在本地的由 ``data_home`` 关键字指定的路径中，路径默认是 ``~/scikit_learn_data/`` ::

  >>> os.listdir(os.path.join(custom_data_home, 'mldata'))
  ['mnist-original.mat']

`mldata.org <http://mldata.org>`_ 里的数据集不遵循严格的命名或格式化约定。
:func:`sklearn.datasets.fetch_mldata` 这个函数可以理解数据集的最常见的情况，并且允许对单个数据集进行缺省调整:

* `mldata.org <http://mldata.org>`_ 中数据的 data arrays (数组) 大多数都是 shape 为 ``(n_features, n_samples)`` 这样的组织形式。
  这和 ``scikit-learn`` 中的约定不一致，所以 :func:`sklearn.datasets.fetch_mldata` 这个函数默认情况下会将这个 transposes the matrix (矩阵进行转置)。这个行为由 ``transpose_data`` 这个关键字控制::

    >>> iris = fetch_mldata('iris', data_home=custom_data_home)
    >>> iris.data.shape
    (150, 4)
    >>> iris = fetch_mldata('iris', transpose_data=False,
    ...                     data_home=custom_data_home)
    >>> iris.data.shape
    (4, 150)

* 数据集有多列的时候，:func:`sklearn.datasets.fetch_mldata` 这个函数会识别目标列和数据列，
  并将它们重命名为 ``target（目标）`` 和 ``data（数据）`` 。
  这是通过在数据集中寻找名为 ``label（标签）`` 和 ``data（数据）`` 的数组来完成的，
  如果选择第一个数组是 ``target（目标）``，而第二个数组是 ``data（数据）`` ，则前边的设置会失效。
  这个行为可以通过对关键字 ``target_name`` 和 ``data_name`` 进行设置来改变，设置的值可以是具体的名字也可以是索引数字，
  数据集中列的名字和索引序号都可以在 `mldata.org <http://mldata.org>`_ 中的 "Data" 选项卡下找到::

    >>> iris2 = fetch_mldata('datasets-UCI iris', target_name=1, data_name=0,
    ...                      data_home=custom_data_home)
    >>> iris3 = fetch_mldata('datasets-UCI iris', target_name='class',
    ...                      data_name='double0', data_home=custom_data_home)
