
.. _covtype:

Forest covertypes (森林覆盖类型)
===============================

这个数据集中的样本对应美国的 30×30m 的 patches of forest(森林区域)，
收集这些数据用于预测每个 patch 的植被 cover type (覆盖类型)，即占据优势的 the dominant species of tree (植被物种)。
总共有七个植被类型，使得这是一个多类分类问题。
每个样本有 54 个特征，在 `dataset's 的主页 <http://archive.ics.uci.edu/ml/datasets/Covertype>`_ 中有具体的描述。
有些特征是布尔指标，而其他的是离散或者连续的量。

:func:`sklearn.datasets.fetch_covtype` 将加载 covertype 数据集；
它返回一个类似字典的对象，并在数据成员中使用特征矩阵以及 ``target`` 中的目标值。
如果需要，数据集可以从网上下载。
