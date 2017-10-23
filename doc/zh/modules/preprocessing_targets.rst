
.. currentmodule:: sklearn.preprocessing

.. _preprocessing_targets:

==========================================
预测目标 (``y``) 的转换 
==========================================

标签二值化
------------------

:class:`LabelBinarizer` 是一个用来从多类别列表创建标签矩阵的工具类::

    >>> from sklearn import preprocessing
    >>> lb = preprocessing.LabelBinarizer()
    >>> lb.fit([1, 2, 6, 4, 2])
    LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
    >>> lb.classes_
    array([1, 2, 4, 6])
    >>> lb.transform([1, 6])
    array([[1, 0, 0, 0],
           [0, 0, 0, 1]])

对于多类别是实例，可以使用 :class:`MultiLabelBinarizer`::

    >>> lb = preprocessing.MultiLabelBinarizer()
    >>> lb.fit_transform([(1, 2), (3,)])
    array([[1, 1, 0],
           [0, 0, 1]])
    >>> lb.classes_
    array([1, 2, 3])

标签编码
--------------

:class:`LabelEncoder` 是一个可以用来将标签规范化的工具类，它可以将标签的编码值范围限定在[0,n_classes-1]. 
这在编写高效的Cython程序时是非常有用的. :class:`LabelEncoder` 可以如下使用::

    >>> from sklearn import preprocessing
    >>> le = preprocessing.LabelEncoder()
    >>> le.fit([1, 2, 2, 6])
    LabelEncoder()
    >>> le.classes_
    array([1, 2, 6])
    >>> le.transform([1, 1, 2, 6])
    array([0, 0, 1, 2])
    >>> le.inverse_transform([0, 0, 1, 2])
    array([1, 1, 2, 6])

当然，它也可以用于非数值型标签的编码转换成数值标签（只要它们是可哈希并且可比较的）::

    >>> le = preprocessing.LabelEncoder()
    >>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
    LabelEncoder()
    >>> list(le.classes_)
    ['amsterdam', 'paris', 'tokyo']
    >>> le.transform(["tokyo", "tokyo", "paris"])
    array([2, 2, 1])
    >>> list(le.inverse_transform([2, 2, 1]))
    ['tokyo', 'tokyo', 'paris']
