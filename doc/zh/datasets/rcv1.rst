
.. _rcv1:

RCV1 数据集
==========================

路透社语料库第一卷( RCV1 )是路透社为了研究目的提供的一个拥有超过 800,000 份手动分类的新闻报导的文档库。该数据集在 [1]_ 中有详细描述。

:func:`sklearn.datasets.fetch_rcv1` 将加载以下版本: RCV1-v2, vectors, full sets, topics multilabels::

    >>> from sklearn.datasets import fetch_rcv1
    >>> rcv1 = fetch_rcv1()

它返回一个类似字典的对象，具有以下属性:

``data``:
特征矩阵是一个 scipy CSR 稀疏矩阵，有 804414 个样品和 47236 个特征。
非零值包含 cosine-normalized(余弦归一化)，log TF-IDF vectors。
按照年代顺序近似划分，在 [1]_ 提出: 前 23149 个样本是训练集。后 781265 个样本是测试集。
这是官方的 LYRL2004 时间划分。数组有 0.16% 个非零值::

    >>> rcv1.data.shape
    (804414, 47236)

``target``:
目标值是存储在 scipy CSR 的稀疏矩阵，有 804414 个样本和 103 个类别。
每个样本在其所属的类别中的值为 1，在其他类别中值为 0。数组有 3.15% 个非零值::

    >>> rcv1.target.shape
    (804414, 103)

``sample_id``:
每个样本都可以通过从 2286 到 810596 不等的 ID 来标识::

    >>> rcv1.sample_id[:3]
    array([2286, 2287, 2288], dtype=uint32)

``target_names``:
目标值是每个样本的 topic (主题)。每个样本至少属于一个 topic (主题)最多 17 个 topic 。
总共有 103 个 topics ，每个 topic 用一个字符串表示。
从 `GMIL` 出现 5 次到 `CCAT` 出现 381327 次，该语料库频率跨越五个数量级::

    >>> rcv1.target_names[:3].tolist()  # doctest: +SKIP
    ['E11', 'ECAT', 'M11']

如果有需要的话，可以从 `rcv1 homepage`_ 上下载该数据集。
数据集压缩后的大小大约是 656 MB。

.. _rcv1 homepage: http://jmlr.csail.mit.edu/papers/volume5/lewis04a/


.. topic:: 参考文献

    .. [1] Lewis, D. D., Yang, Y., Rose, T. G., & Li, F. (2004). RCV1: A new benchmark collection for text categorization research. The Journal of Machine Learning Research, 5, 361-397.
