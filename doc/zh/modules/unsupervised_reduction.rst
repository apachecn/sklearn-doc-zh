
.. _data_reduction:

=====================================
无监督降维
=====================================

如果你的特征数量很多, 在监督步骤之前, 可以通过无监督的步骤来减少特征.
很多的 :ref:`unsupervised-learning` 方法实现了一个名为 ``transform`` 的方法, 它可以用来降低维度.
下面我们将讨论大量使用这种模式的两个具体示例.

.. topic:: **Pipelining**
    无监督数据简化和监督的估计器可以链接在一个步骤中。 请参阅 :ref:`pipeline`.

.. currentmodule:: sklearn

PCA: 主成份分析
----------------------------------

:class:`decomposition.PCA` 寻找能够捕捉原始特征的差异的特征的组合.
请参阅 :ref:`decompositions`.

.. topic:: **示例**

   * :ref: 'sphx_glr_auto_examples_applications_plot_face_recognition.py'

随机投影
-------------------

模块: :mod:`random_projection` 提供了几种用于通过随机投影减少数据的工具.
请参阅文档的相关部分: :ref:`random_projection`.

.. topic:: **示例**

   * :ref:`sphx_glr_auto_examples_plot_johnson_lindenstrauss_bound.py`

特征聚集
------------------------

:class:`cluster.FeatureAgglomeration` 应用
:ref:`hierarchical_clustering` 将行为类似的特征分组在一起.

.. topic:: **示例**

   * :ref:`sphx_glr_auto_examples_cluster_plot_feature_agglomeration_vs_univariate_selection.py`
   * :ref:`sphx_glr_auto_examples_cluster_plot_digits_agglomeration.py`

.. topic:: **特征缩放**

   请注意，如果功能具有明显不同的缩放或统计属性，则 :class:`cluster.FeatureAgglomeration`
   可能无法捕获相关特征之间的关系.使用一个  :class:`preprocessing.StandardScaler` 可以在这些
   设置中使用.

