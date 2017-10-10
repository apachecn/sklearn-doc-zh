.. _introduction:

使用 scikit-learn 的机器学习介绍
=====================================================

.. topic:: 章节内容

    在本节中，我们介绍了在 scikit-learn 中使用的 `机器学习 <https://en.wikipedia.org/wiki/Machine_learning>`_ 词汇，并给出了一个简单的学习示例。


机器学习：问题设置
-------------------------------------

一般来说，学习问题考虑了一组n 个数据 `样本 <https://en.wikipedia.org/wiki/Sample_(statistics)>`_ ，然后尝试预测未知数据的属性。
如果每个样本多于单个数字，并且例如多维条目（也称为 `多变量 <https://en.wikipedia.org/wiki/Multivariate_random_variable>`_ 数据），则称其具有多个属性或 **特征**.

我们可以在几个大类上分解学习问题:

 * `监督学习 <https://en.wikipedia.org/wiki/Supervised_learning>`_,
   其中数据带有我们想要预测的附加属性（:ref:`点击此处 <supervised-learning>` 转到 scikit-learn 监督学习页面）。这个问题可以是:

    * `分类 <https://en.wikipedia.org/wiki/Classification_in_machine_learning>`_:
      样本属于两个或更多个类，我们想从已经标记的数据中学习如何预测未标记数据的类别。
      分类问题的一个例子是手写数字识别示例，其目的是将每个输入向量分配给有限数目的离散类别之一。
      考虑分类的另一种方法是作为监督学习的离散形式（而不是连续的），其中有一个 categories(类型)数量有限，
      并且是针对于所提供的 n 个样本中的每一个样本，一个是尝试用正确的 category（范畴）或 class （类别）来 label （标记）它们。

    * `回归 <https://en.wikipedia.org/wiki/Regression_analysis>`_: 
      如果期望的输出由一个或多个连续变量组成，则该任务称为 *回归*.
      回归问题的一个示例是预测鲑鱼的长度是其年龄和体重的函数。

 * `无监督学习 <https://en.wikipedia.org/wiki/Unsupervised_learning>`_,
   其中训练数据由没有任何相应目标值的一组输入向量x组成。这种问题的目标可能是在数据中发现类似示例的组，称为 `聚类 <https://en.wikipedia.org/wiki/Cluster_analysis>`_,
   或者确定输入空间内的数据分布，称为 `密度估计 <https://en.wikipedia.org/wiki/Density_estimation>`_，或从高维数据投影数据空间缩小到二维或三维以进行 *可视化* （:ref:`点击此处 <unsupervised-learning>` 转到 scikit-learn 无监督学习页面）。

.. topic:: 训练集和测试集

    机器学习是关于学习数据集的某些属性并将其应用于新数据。
    这就是为什么在机器的普遍做法学习评价的算法是手头上的数据分成两组，
    一个是我们所说的 **训练集** 上，我们了解到，我们称之为数据属性和一个 **测试集** 上，我们测试这些属性。

.. _loading_example_dataset:

加载示例数据集
--------------------------

`scikit-learn` 提供了一些标准数据集，例如 用于分类的 `iris <https://en.wikipedia.org/wiki/Iris_flower_data_set>`_ 
和 `digits <http://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits>`_ 数据集
和 `波士顿房价回归数据集 <http://archive.ics.uci.edu/ml/datasets/Housing>`_ .

在下文中，我们从我们的 shell 启动一个 Python 解释器，然后加载 ``iris`` 和 ``digits`` 数据集。我们的符号约定是 ``$`` 表示shell提示符，而 ``>>>`` 表示 Python 解释器提示符::

  $ python
  >>> from sklearn import datasets
  >>> iris = datasets.load_iris()
  >>> digits = datasets.load_digits()

数据集是一个类似字典的对象，它保存有关数据的所有数据和一些元数据。 该数据存储在 ``.data`` 成员中，它是 ``n_samples, n_features`` 数组。 
在监督问题的情况下，一个或多个响应变量存储在 ``.target`` 成员中。 有关不同数据集的更多详细信息，请参见 :ref:`专用数据集部分 <datasets>`.

例如，在数字数据集的情况下，``digits.data`` 可以访问可用于对数字样本进行分类的功能::

  >>> print(digits.data)  # doctest: +NORMALIZE_WHITESPACE
  [[  0.   0.   5. ...,   0.   0.   0.]
   [  0.   0.   0. ...,  10.   0.   0.]
   [  0.   0.   0. ...,  16.   9.   0.]
   ...,
   [  0.   0.   1. ...,   6.   0.   0.]
   [  0.   0.   2. ...,  12.   0.   0.]
   [  0.   0.  10. ...,  12.   1.   0.]]

并且 ``digits.target`` 给出数字数据集的基本真值，即我们正在尝试学习的每个数字图像对应的数字::

  >>> digits.target
  array([0, 1, 2, ..., 8, 9, 8])

.. topic:: 数据数组的形状

    数据总是2D数组，形状 ``(n_samples, n_features)``，尽管原始数据可能具有不同的形状。 
    在数字的情况下，每个原始样本是形状 ``(8, 8)`` 的图像，可以使用以下方式访问::

      >>> digits.images[0]
      array([[  0.,   0.,   5.,  13.,   9.,   1.,   0.,   0.],
             [  0.,   0.,  13.,  15.,  10.,  15.,   5.,   0.],
             [  0.,   3.,  15.,   2.,   0.,  11.,   8.,   0.],
             [  0.,   4.,  12.,   0.,   0.,   8.,   8.,   0.],
             [  0.,   5.,   8.,   0.,   0.,   9.,   8.,   0.],
             [  0.,   4.,  11.,   0.,   1.,  12.,   7.,   0.],
             [  0.,   2.,  14.,   5.,  10.,  12.,   0.,   0.],
             [  0.,   0.,   6.,  13.,  10.,   0.,   0.,   0.]])
    
    该  :ref:`数据集上的简单示例 <sphx_glr_auto_examples_classification_plot_digits_classification.py>` 说明了如何从原始问题开始，可以在 scikit-learn 中形成消费数据。
    
.. topic:: 从外部数据集加载

    要从外部数据集加载，请参阅 :ref:`加载外部数据集 <external_datasets>`.

学习和预测
------------------------

在数字数据集的情况下，任务是给出图像来预测其表示的数字。 
我们给出了10个可能类（数字0到9）中的每一个的样本，我们在这些类上给出了一个 `估计量 <https://en.wikipedia.org/wiki/Estimator>`_ ，以便能够*预测*看不见的样本所属的类。

在 scikit-learn 中，分类的估计是一个 Python 对象，它实现了 ``fit(X, y)`` 和 ``predict(T)`` 的方法。

估计器的一个例子是实现 `支持向量分类 <https://en.wikipedia.org/wiki/Support_vector_machine>`_ 的类 ``sklearn.svm.SVC``. 估计器的构造函数以模型的参数为参数，但目前我们将把估计器视为黑盒子::

  >>> from sklearn import svm
  >>> clf = svm.SVC(gamma=0.001, C=100.)

.. topic:: 选择模型的参数

  在这个例子中，我们设置 ``gamma`` 手动的值。通过使用 :ref:`网格搜索  <grid_search>` 和 :ref:`交叉验证 <cross_validation>` 等工具，可以自动找到参数的良好值。

们称之为我们的估计器实例 ``clf``，因为它是一个分类器。它现在必须适应模型，也就是说，它必须从模型中*学习*。
这是通过将我们的训练集传递给该 ``fit`` 方法来完成的。作为一个训练集，让我们使用除最后一个数据集的所有图像。
我们用 ``[:-1]`` Python 语法选择这个训练集，它产生一个包含除最后一个条目之外的所有数组的新数组 ``digits.data``::

  >>> clf.fit(digits.data[:-1], digits.target[:-1])  # doctest: +NORMALIZE_WHITESPACE
  SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

现在你可以预测新的值，特别是我们可以向还没有用来训练分类器询问 ``digits`` 数据集中最后一个图像的数字是什么::

  >>> clf.predict(digits.data[-1:])
  array([8])

相应的图像如下:

.. image:: /auto_examples/datasets/images/sphx_glr_plot_digits_last_image_001.png
    :target: ../../auto_examples/datasets/plot_digits_last_image.html
    :align: center
    :scale: 50

正如你所看到的，这是一项具有挑战性的任务：图像分辨率差。你同意分类器吗？

这个分类问题的一个完整例子可以作为一个例子来运行和学习： 识别手写数字。
:ref:`sphx_glr_auto_examples_classification_plot_digits_classification.py`.


模型持久性
-----------------

可以通过使用Python的内置持久性模型（即 `pickle <https://docs.python.org/2/library/pickle.html>`_ ）将模型保存在scikit中::

  >>> from sklearn import svm
  >>> from sklearn import datasets
  >>> clf = svm.SVC()
  >>> iris = datasets.load_iris()
  >>> X, y = iris.data, iris.target
  >>> clf.fit(X, y)  # doctest: +NORMALIZE_WHITESPACE
  SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

  >>> import pickle
  >>> s = pickle.dumps(clf)
  >>> clf2 = pickle.loads(s)
  >>> clf2.predict(X[0:1])
  array([0])
  >>> y[0]
  0

在scikit的具体情况下，使用 joblib 替换 pickle（``joblib.dump`` & ``joblib.load``）可能会更有趣，这对大数据更有效，但只能 pickle 到磁盘而不是字符串::

  >>> from sklearn.externals import joblib
  >>> joblib.dump(clf, 'filename.pkl') # doctest: +SKIP

之后，您可以加载 pickle 模型（可能在另一个 Python 进程中）::

  >>> clf = joblib.load('filename.pkl') # doctest:+SKIP

.. 注意::

    ``joblib.dump`` 并且 ``joblib.load`` 函数也接受 file-like 对象而不是文件名。有关 Joblib 的数据持久性的更多信息，请 `点击此处 <https://pythonhosted.org/joblib/persistence.html>`_。

请注意，pickle 有一些安全性和可维护性问题。有关使用 scikit-learn 的模型持久性的更多详细信息，请参阅 :ref:`模型持久性` 部分。


规定
-----------

scikit-learn 估计器遵循某些规则，使其行为更具预测性。


类型转换
~~~~~~~~~~~~

除非另有规定，输入将被转换为 ``float64``::

  >>> import numpy as np
  >>> from sklearn import random_projection

  >>> rng = np.random.RandomState(0)
  >>> X = rng.rand(10, 2000)
  >>> X = np.array(X, dtype='float32')
  >>> X.dtype
  dtype('float32')

  >>> transformer = random_projection.GaussianRandomProjection()
  >>> X_new = transformer.fit_transform(X)
  >>> X_new.dtype
  dtype('float64')

在这个例子中，``X`` 是 ``float32``，被转换 ``float64`` 的 ``fit_transform(X)``。
回归目标被归结为 ``float64``，维护分类目标::

    >>> from sklearn import datasets
    >>> from sklearn.svm import SVC
    >>> iris = datasets.load_iris()
    >>> clf = SVC()
    >>> clf.fit(iris.data, iris.target)  # doctest: +NORMALIZE_WHITESPACE
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)

    >>> list(clf.predict(iris.data[:3]))
    [0, 0, 0]

    >>> clf.fit(iris.data, iris.target_names[iris.target])  # doctest: +NORMALIZE_WHITESPACE
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)

    >>> list(clf.predict(iris.data[:3]))  # doctest: +NORMALIZE_WHITESPACE
    ['setosa', 'setosa', 'setosa']

这里，第一个 ``predict()`` 返回一个整数数组，因为在 ``fit`` 中使用了 ``iris.target`` （一个整数数组）。 
第二个 ``predict()`` 返回一个字符串数组，因为 ``iris.target_names`` 是用于拟合的。

修改和更新参数
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

估计器的超参数可以在通过 :func:`sklearn.pipeline.Pipeline.set_params` 方法构建之后进行更新。 
调用 ``fit()`` 多次将覆盖以前的 ``fit()`` 中学到的内容::

  >>> import numpy as np
  >>> from sklearn.svm import SVC

  >>> rng = np.random.RandomState(0)
  >>> X = rng.rand(100, 10)
  >>> y = rng.binomial(1, 0.5, 100)
  >>> X_test = rng.rand(5, 10)

  >>> clf = SVC()
  >>> clf.set_params(kernel='linear').fit(X, y)  # doctest: +NORMALIZE_WHITESPACE
  SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
  >>> clf.predict(X_test)
  array([1, 0, 1, 1, 0])

  >>> clf.set_params(kernel='rbf').fit(X, y)  # doctest: +NORMALIZE_WHITESPACE
  SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
  >>> clf.predict(X_test)
  array([0, 0, 0, 1, 0])

在这里，默认内核 ``rbf`` 首先被改变到 ``linear`` 估计器被构造之后 ``SVC()``，并且改回到 ``rbf`` 重新设计估计器并进行第二预测。

多分类与多标签拟合
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

当使用 :class:`多类分类器 <sklearn.multiclass>` 时，执行的学习和预测任务取决于适合的目标数据的格式::

    >>> from sklearn.svm import SVC
    >>> from sklearn.multiclass import OneVsRestClassifier
    >>> from sklearn.preprocessing import LabelBinarizer

    >>> X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
    >>> y = [0, 0, 1, 1, 2]

    >>> classif = OneVsRestClassifier(estimator=SVC(random_state=0))
    >>> classif.fit(X, y).predict(X)
    array([0, 0, 1, 1, 2])

在上述情况下，分类器适合于多分类标签的 1d 矩阵，``predict()`` 因此该方法提供了相应的多类预测。还可以使用二维标签二维矩阵::

    >>> y = LabelBinarizer().fit_transform(y)
    >>> classif.fit(X, y).predict(X)
    array([[1, 0, 0],
           [1, 0, 0],
           [0, 1, 0],
           [0, 0, 0],
           [0, 0, 0]])

这里，分类器是 ``fit()`` 上的 2D 二进制标记表示 y，使用 :class:`LabelBinarizer <sklearn.preprocessing.LabelBinarizer>`。在这种情况下，``predict()`` 返回一个表示相应多重标签预测的 2d 矩阵。

请注意，第四个和第五个实例返回所有零，表示它们没有匹配三个标签 ``fit``。使用多分类输出，类似地可以为一个实例分配多个标签::

  >> from sklearn.preprocessing import MultiLabelBinarizer
  >> y = [[0, 1], [0, 2], [1, 3], [0, 2, 3], [2, 4]]
  >> y = MultiLabelBinarizer().fit_transform(y)
  >> classif.fit(X, y).predict(X)
  array([[1, 1, 0, 0, 0],
         [1, 0, 1, 0, 0],
         [0, 1, 0, 1, 0],
         [1, 0, 1, 1, 0],
         [0, 0, 1, 0, 1]])

在这种情况下，分类器适合每个分配多个标签的实例。
所述 :class:`MultiLabelBinarizer <sklearn.preprocessing.MultiLabelBinarizer>` 用于多分类的 2D 矩阵以二进制化 ``fit`` 时。
因此，``predict()`` 返回具有每个实例的多个预测标签的 2d 矩阵。