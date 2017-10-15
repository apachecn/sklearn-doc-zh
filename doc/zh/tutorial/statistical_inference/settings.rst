
==========================================================================
Statistical learning: the setting and the estimator object in scikit-learn
==========================================================================

数据集
=========

Scikit-learn从可以从一个或者多个数据集中学习信息，这些数据集合必须是2维阵列，也可以认为是一个列表。列表的第一个维度代表 **样本** ，第二个维度代表 **特征** （每一行代表一个样本，每一列代表一种特征）。

.. topic:: 样例: iris 数据集（鸢尾花卉数据集）

    ::

        >>> from sklearn import datasets
        >>> iris = datasets.load_iris()
        >>> data = iris.data
        >>> data.shape
        (150, 4)

    这个数据集包含150个样本，每个样本包含4个特征：花萼长度，花萼宽度，花瓣长度，花瓣宽度，详细数据可以通过``iris.DESCR``查看。

如果原始数据并不是``(n_samples, n_features)``的形状，在使用之前必须进行预处理。

.. topic:: 数据预处理样例:digits数据集(手写数字数据集)

    .. image:: /auto_examples/datasets/images/sphx_glr_plot_digits_last_image_001.png
        :target: ../../auto_examples/datasets/plot_digits_last_image.html
        :align: right
        :scale: 60

    digits数据集包含1797个手写数字的图像，每个图像为8*8像素 ::

        >>> digits = datasets.load_digits()
        >>> digits.images.shape
        (1797, 8, 8)
        >>> import matplotlib.pyplot as plt #doctest: +SKIP
        >>> plt.imshow(digits.images[-1], cmap=plt.cm.gray_r) #doctest: +SKIP
        <matplotlib.image.AxesImage object at ...>

    为了在scikit中使用这一数据集，需要将8×8的图像转换成长度为64的一维列表 ::

        >>> data = digits.images.reshape((digits.images.shape[0], -1))


Estimators objects
===================

.. Some code to make the doctests run

   >>> from sklearn.base import BaseEstimator
   >>> class Estimator(BaseEstimator):
   ...      def __init__(self, param1=0, param2=0):
   ...          self.param1 = param1
   ...          self.param2 = param2
   ...      def fit(self, data):
   ...          pass
   >>> estimator = Estimator()

**Fitting data**: the main API implemented by scikit-learn is that of the
`estimator`. An estimator is any object that learns from data;
it may be a classification, regression or clustering algorithm or
a *transformer* that extracts/filters useful features from raw data.

All estimator objects expose a ``fit`` method that takes a dataset
(usually a 2-d array):

    >>> estimator.fit(data)

**Estimator parameters**: All the parameters of an estimator can be set
when it is instantiated or by modifying the corresponding attribute::

    >>> estimator = Estimator(param1=1, param2=2)
    >>> estimator.param1
    1

**Estimated parameters**: When data is fitted with an estimator,
parameters are estimated from the data at hand. All the estimated
parameters are attributes of the estimator object ending by an
underscore::

    >>> estimator.estimated_param_ #doctest: +SKIP
