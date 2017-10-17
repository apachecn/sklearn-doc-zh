
.. _rcv1:

RCV1数据集 (RCV1 dataset)
============

路透社语料库第一卷(RCV1)是路透社为了研究目的提供的一个拥有超过800,000份手动分类的新闻报导的文档库。该数据集在 [1]_ 被描述。

:func:`sklearn.datasets.fetch_rcv1` 将加载以下版本: RCV1-v2, vectors, full sets, topics multilabels::

    >>> from sklearn.datasets import fetch_rcv1
    >>> rcv1 = fetch_rcv1()

它返回一个类似字典的对象，具有以下属性:

``data``:
特征矩阵是一个SciPy CSR稀疏矩阵，有804414个样品和47236个特征。
非零值包含余弦归一化(cosine-normalized)，log TF-IDF vectors。
按照年代顺序近似划分，在 [1]_ 提出：前23149个样本是训练集。后781265个样本是测试集。
这是官方的 LYRL2004 时间划分。数组有0.16%个非零值::

    >>> rcv1.data.shape
    (804414, 47236)

``target``:
目标值是存储在SciPy CSR的稀疏矩阵，有804414个样本和103个类别。
每个样本在其属于的类别中的值为1，在其他不属于的类别中值为0。数组有3.15%个非零值::

    >>> rcv1.target.shape
    (804414, 103)

``sample_id``:
每个样本都可以通过从2286到810596不等的ID来标识::

    >>> rcv1.sample_id[:3]
    array([2286, 2287, 2288], dtype=uint32)

``target_names``:
目标值是每个样本的的主题(topic)。每个样本至少属于一个主题(topic)，最多17个主题(topic)。
总共有103个主题(topics)，每个主题(topic)用一个字符串表示。
从`GMIL`出现5次到`CCAT`出现381327次，他们的语料库频率跨越五个数量级::

    >>> rcv1.target_names[:3].tolist()  # doctest: +SKIP
    ['E11', 'ECAT', 'M11']

如果有需要的话，可以从 `rcv1 homepage`_ 上下载数据集。
数据集压缩后的大小大约是656 MB。

.. _rcv1 homepage: http://jmlr.csail.mit.edu/papers/volume5/lewis04a/


.. topic:: References

    .. [1] Lewis, D. D., Yang, Y., Rose, T. G., & Li, F. (2004). RCV1: A new benchmark collection for text categorization research. The Journal of Machine Learning Research, 5, 361-397.
