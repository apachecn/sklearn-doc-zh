.. include:: includes/big_toc_css.rst

.. _data-transforms:

数据集转换
-----------------------

scikit-learn 提供了一个用于转换数据集的库, 它也许会 clean（清理）（请参阅
:ref:`preprocessing`）, reduce（减少）（请参阅 :ref:`data_reduction`）, expand（扩展）（请参阅
:ref:`kernel_approximation`）或 generate（生成）（请参阅 :ref:`feature_extraction`）
feature representations（特征表示）.

像其它预估计一样, 它们由具有 ``fit`` 方法的类来表示, 
该方法从训练集学习模型参数（例如, 归一化的平均值和标准偏差）以及将该转换模型应用于 ``transform`` 方法到不可见数据.
同时 ``fit_transform`` 可以更方便和有效地建模与转换训练数据.

将 :ref:`combining_estimators` 中 transformers（转换）使用并行的或者串联的方式合并到一起.
:ref:`metrics` 涵盖将特征空间转换为 affinity matrices（亲和矩阵）, 
而  :ref:`preprocessing_targets` 考虑在 scikit-learn 中使用目标空间的转换（例如. 标签分类）.

.. toctree::

    modules/pipeline
    modules/feature_extraction
    modules/preprocessing
    modules/unsupervised_reduction
    modules/random_projection
    modules/kernel_approximation
    modules/metrics
    modules/preprocessing_targets
