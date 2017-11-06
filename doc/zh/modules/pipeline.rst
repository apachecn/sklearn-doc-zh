.. _combining_estimators:

=========================================================
Pipeline（管道）和 FeatureUnion（特征联合）: 合并的评估器
=========================================================

.. _pipeline:

Pipeline: 链式评估器
=============================

.. currentmodule:: sklearn.pipeline

:class:`Pipeline` 可以把多个评估器链接成一个。这个是很有用的，因为处理数据的步骤一般都是固定的，例如特征选择、标准化和分类。
 :class:`Pipeline` 主要有两个目的:

便捷性和封装性
    你只要对数据调用 ``fit``和 ``predict``一次来适配所有的一系列评估器。
联合的参数选择
    你可以一次 :ref:`grid search <grid_search>`管道中所有评估器的参数。
安全性
    训练转换器和预测器使用的是相同样本，管道有助于防止来自测试数据的统计数据泄露到交叉验证的训练模型中。

管道中的所有评估器，除了最后一个评估器，管道的所有评估器必须是转换器。
(例如，必须有 ``transform`` 方法).
最后一个评估器的类型不限（转换器、分类器等等）


用法
----

 :class:`Pipeline` 使用一系列 ``(key, value)`` 键值对来构建,其中 ``key`` 是你给这个步骤起的名字， ``value`` 是一个评估器对象::

    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.svm import SVC
    >>> from sklearn.decomposition import PCA
    >>> estimators = [('reduce_dim', PCA()), ('clf', SVC())]
    >>> pipe = Pipeline(estimators)
    >>> pipe # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    Pipeline(memory=None,
             steps=[('reduce_dim', PCA(copy=True,...)),
                    ('clf', SVC(C=1.0,...))])

功能函数 :func:`make_pipeline` 是构建管道的缩写;
它接收多个评估器并返回一个管道，自动填充评估器名::

    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.naive_bayes import MultinomialNB
    >>> from sklearn.preprocessing import Binarizer
    >>> make_pipeline(Binarizer(), MultinomialNB()) # doctest: +NORMALIZE_WHITESPACE
    Pipeline(memory=None,
             steps=[('binarizer', Binarizer(copy=True, threshold=0.0)),
                    ('multinomialnb', MultinomialNB(alpha=1.0,
                                                    class_prior=None,
                                                    fit_prior=True))])

管道中的评估器作为一个列表保存在 ``steps`` 属性内::

    >>> pipe.steps[0]
    ('reduce_dim', PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False))

并作为 ``dict`` 保存在 ``named_steps``::

    >>> pipe.named_steps['reduce_dim']
    PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
      svd_solver='auto', tol=0.0, whiten=False)

管道中的评估器参数可以通过 ``<estimator>__<parameter>`` 语义来访问::

    >>> pipe.set_params(clf__C=10) # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    Pipeline(memory=None,
             steps=[('reduce_dim', PCA(copy=True, iterated_power='auto',...)),
                    ('clf', SVC(C=10, cache_size=200, class_weight=None,...))])

named_steps 的属性映射到多个值,在交互环境支持 tab 补全::

    >>> pipe.named_steps.reduce_dim is pipe.named_steps['reduce_dim']
    True

这对网格搜索尤其重要::

    >>> from sklearn.model_selection import GridSearchCV
    >>> param_grid = dict(reduce_dim__n_components=[2, 5, 10],
    ...                   clf__C=[0.1, 10, 100])
    >>> grid_search = GridSearchCV(pipe, param_grid=param_grid)

单独的步骤可以用多个参数替换，除了最后步骤，其他步骤都可以设置为 ``None`` 来跳过 ::

    >>> from sklearn.linear_model import LogisticRegression
    >>> param_grid = dict(reduce_dim=[None, PCA(5), PCA(10)],
    ...                   clf=[SVC(), LogisticRegression()],
    ...                   clf__C=[0.1, 10, 100])
    >>> grid_search = GridSearchCV(pipe, param_grid=param_grid)

.. topic:: 例子:

 * :ref:`sphx_glr_auto_examples_feature_selection_plot_feature_selection_pipeline.py`
 * :ref:`sphx_glr_auto_examples_model_selection_grid_search_text_feature_extraction.py`
 * :ref:`sphx_glr_auto_examples_plot_digits_pipe.py`
 * :ref:`sphx_glr_auto_examples_plot_kernel_approximation.py`
 * :ref:`sphx_glr_auto_examples_svm_plot_svm_anova.py`
 * :ref:`sphx_glr_auto_examples_plot_compare_reduction.py`

.. topic:: 也可以参阅:

 * :ref:`grid_search`


注意点
------

对管道调用 ``fit`` 方法的效果跟依次对每个评估器调用 ``fit`` 方法一样, 都是``transform`` 输入并传递给下个步骤。
管道中最后一个评估器的所有方法，管道都有,例如，如果最后的评估器是一个分类器， :class:`Pipeline` 可以当做分类器来用。如果最后一个评估器是转换器，管道也一样可以。

.. _pipeline_cache:

缓存转换器：避免重复计算
-------------------------------------------------

.. currentmodule:: sklearn.pipeline

适配转换器是很耗费计算资源的。设置了``memory`` 参数， :class:`Pipeline` 将会在调用``fit``方法后缓存每个转换器。
如果参数和输入数据相同，这个特征用于避免重复计算适配的转换器。典型的例子是网格搜索转换器，该转化器只要适配一次就可以多次使用。

 ``memory`` 参数用于缓存转换器。
``memory`` 可以是包含要缓存的转换器的目录的字符串或一个 `joblib.Memory <https://pythonhosted.org/joblib/memory.html>`_
对象::

    >>> from tempfile import mkdtemp
    >>> from shutil import rmtree
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.svm import SVC
    >>> from sklearn.pipeline import Pipeline
    >>> estimators = [('reduce_dim', PCA()), ('clf', SVC())]
    >>> cachedir = mkdtemp()
    >>> pipe = Pipeline(estimators, memory=cachedir)
    >>> pipe # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    Pipeline(...,
             steps=[('reduce_dim', PCA(copy=True,...)),
                    ('clf', SVC(C=1.0,...))])
    >>> # Clear the cache directory when you don't need it anymore
    >>> rmtree(cachedir)

.. warning:: **Side effect of caching transfomers**

   使用 :class:`Pipeline` 而不开启缓存功能,还是可以通过查看原始实例的，例如::

     >>> from sklearn.datasets import load_digits
     >>> digits = load_digits()
     >>> pca1 = PCA()
     >>> svm1 = SVC()
     >>> pipe = Pipeline([('reduce_dim', pca1), ('clf', svm1)])
     >>> pipe.fit(digits.data, digits.target)
     ... # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
     Pipeline(memory=None,
              steps=[('reduce_dim', PCA(...)), ('clf', SVC(...))])
     >>> # The pca instance can be inspected directly
     >>> print(pca1.components_) # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
         [[ -1.77484909e-19  ... 4.07058917e-18]]

   开启缓存会在适配前触发转换器的克隆。因此，管道的转换器实例不能被直接查看。
   在下面例子中， 访问 :class:`PCA` 实例 ``pca2``
   将会引发 ``AttributeError`` 因为 ``pca2`` 是一个未适配的转换器。
   这时应该使用属性 ``named_steps`` 来检查管道的评估器::

     >>> cachedir = mkdtemp()
     >>> pca2 = PCA()
     >>> svm2 = SVC()
     >>> cached_pipe = Pipeline([('reduce_dim', pca2), ('clf', svm2)],
     ...                        memory=cachedir)
     >>> cached_pipe.fit(digits.data, digits.target)
     ... # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
      Pipeline(memory=...,
               steps=[('reduce_dim', PCA(...)), ('clf', SVC(...))])
     >>> print(cached_pipe.named_steps['reduce_dim'].components_)
     ... # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
         [[ -1.77484909e-19  ... 4.07058917e-18]]
     >>> # Remove the cache directory
     >>> rmtree(cachedir)

.. topic:: 例子:

 * :ref:`sphx_glr_auto_examples_plot_compare_reduction.py`

.. _feature_union:

FeatureUnion（特征联合）: 个特征层面
======================================

.. currentmodule:: sklearn.pipeline

:class:`FeatureUnion` 合并了多个转换器对象形成一个新的转换器，该转换器合并了他们的输出。一个 :class:`FeatureUnion` 可以接收多个转换器对象。在适配期间，每个转换器都单独的和数据适配。
对于转换数据，转换器可以并发使用，且输出的样本向量被连接成更大的向量。

:class:`FeatureUnion` 功能与 :class:`Pipeline` 一样-
便捷性和联合参数的估计和验证。

可以结合:class:`FeatureUnion` 和 :class:`Pipeline` 来创造出复杂模型。

(一个 :class:`FeatureUnion` 没办法检查两个转换器是否会产出相同的特征。它仅仅在特征集合不相关时产生联合并确认是调用者的职责。)


用法
----

一个 :class:`FeatureUnion` 是通过一系列 ``(key, value)`` 键值对来构建的,其中的 ``key`` 给转换器指定的名字
(一个绝对的字符串; 他只是一个代号)， ``value`` 是一个评估器对象::

    >>> from sklearn.pipeline import FeatureUnion
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.decomposition import KernelPCA
    >>> estimators = [('linear_pca', PCA()), ('kernel_pca', KernelPCA())]
    >>> combined = FeatureUnion(estimators)
    >>> combined # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    FeatureUnion(n_jobs=1,
                 transformer_list=[('linear_pca', PCA(copy=True,...)),
                                   ('kernel_pca', KernelPCA(alpha=1.0,...))],
                 transformer_weights=None)


跟管道一样，特征联合有一个精简版的构造器叫做:func:`make_union` ，该构造器不需要显式给每个组价起名字。


正如 ``Pipeline``, 单独的步骤可能用``set_params``替换 ,并设置为 ``None``来跳过::

    >>> combined.set_params(kernel_pca=None)
    ... # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
    FeatureUnion(n_jobs=1,
                 transformer_list=[('linear_pca', PCA(copy=True,...)),
                                   ('kernel_pca', None)],
                 transformer_weights=None)

.. topic:: 例子:

 * :ref:`sphx_glr_auto_examples_plot_feature_stacker.py`
 * :ref:`sphx_glr_auto_examples_hetero_feature_union.py`
