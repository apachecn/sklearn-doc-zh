
.. _covtype:

森林覆盖类型 (Forest covertypes)
=================

这个数据集中的样本对应美国的30×30m的森林区域(patches of forest)，
收集这些数据用于预测每个patch的植被覆盖类型(cover type)，即占据优势的植被物种(the dominant species of tree)。
总共有七个植被类型，使得这是一个多类分类问题。
每个样本有54个特征，在 `dataset's 的主页 <http://archive.ics.uci.edu/ml/datasets/Covertype>`_ 中有具体的描述。
有些特征是布尔指标，而其他的是离散或者连续的量。

:func:`sklearn.datasets.fetch_covtype` 将加载covertype数据集；
它返回一个类似字典的对象，并在数据成员中使用特征矩阵以及``目标``中的目标值。
如果需要，数据集可以从网上下载。
