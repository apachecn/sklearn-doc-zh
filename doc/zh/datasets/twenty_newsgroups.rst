.. _20newsgroups:

20个新闻组文本数据集
=====================================

20个新闻组文本数据集包含有关20个主题的大约18000个新闻组，被分为两个子集：一个用于
训练(或者开发)，另一个用于测试(或者用于性能评估)。训练和测试集的划分是基于某个特定日期
前后发布的消息。

这个模块包含两个加载器。第一个是 :func:`sklearn.datasets.fetch_20newsgroups`，
返回一个能够被文本特征提取器接受的原始文本列表，例如 :class:`sklearn.feature_extraction.text.CountVectorizer`
使用自定义的参数来提取特征向量。第二个是 :func:`sklearn.datasets.fetch_20newsgroups_vectorized`，
返回即用特征，换句话说就是，这样就没必要使用特征提取器了。

用法
--------

 :func:`sklearn.datasets.fetch_20newsgroups`  是一个用于从原始的20个新闻组网址( `20 newsgroups website`_)
下载数据归档的数据获取/缓存函数，提取 ``~/scikit_learn_data/20news_home`` 文件夹中的
归档内容。并且在训练集或测试集文件夹，或者两者上调用函数 :func:`sklearn.datasets.load_files`::

  >>> from sklearn.datasets import fetch_20newsgroups
  >>> newsgroups_train = fetch_20newsgroups(subset='train')

  >>> from pprint import pprint
  >>> pprint(list(newsgroups_train.target_names))
  ['alt.atheism',
   'comp.graphics',
   'comp.os.ms-windows.misc',
   'comp.sys.ibm.pc.hardware',
   'comp.sys.mac.hardware',
   'comp.windows.x',
   'misc.forsale',
   'rec.autos',
   'rec.motorcycles',
   'rec.sport.baseball',
   'rec.sport.hockey',
   'sci.crypt',
   'sci.electronics',
   'sci.med',
   'sci.space',
   'soc.religion.christian',
   'talk.politics.guns',
   'talk.politics.mideast',
   'talk.politics.misc',
   'talk.religion.misc']

真实数据在属性 ``filenames`` 和 ``target`` 中，target属性就是类别的整数索引::

  >>> newsgroups_train.filenames.shape
  (11314,)
  >>> newsgroups_train.target.shape
  (11314,)
  >>> newsgroups_train.target[:10]
  array([12,  6,  9,  8,  6,  7,  9,  2, 13, 19])

可以通过将类别列表传给 :func:`sklearn.datasets.fetch_20newsgroups` 函数来实现只加载一部分的类别::

  >>> cats = ['alt.atheism', 'sci.space']
  >>> newsgroups_train = fetch_20newsgroups(subset='train', categories=cats)
  >>> list(newsgroups_train.target_names)
  ['alt.atheism', 'sci.space']
  >>> newsgroups_train.filenames.shape
  (1073,)
  >>> newsgroups_train.target.shape
  (1073,)
  >>> newsgroups_train.target[:10]
  array([1, 1, 1, 0, 1, 0, 0, 1, 1, 1])

将文本转换成向量
-------------------------------------

为了用文本数据训练预测或者聚类模型，首先需要做的是将文本转换成适合统计分析的数值
向量。这能使用 ``sklearn.feature_extraction.text`` 的功能来实现，正如下面展示的
从一个20个新闻的子集中提取单个词的 `TF-IDF`_ 向量的例子

  >>> from sklearn.feature_extraction.text import TfidfVectorizer
  >>> categories = ['alt.atheism', 'talk.religion.misc',
  ...               'comp.graphics', 'sci.space']
  >>> newsgroups_train = fetch_20newsgroups(subset='train',
  ...                                       categories=categories)
  >>> vectorizer = TfidfVectorizer()
  >>> vectors = vectorizer.fit_transform(newsgroups_train.data)
  >>> vectors.shape
  (2034, 34118)

提取的TF-IDF向量非常稀疏，在一个超过30000维的空间中采样，
平均只有159个非零成分(少于.5%的非零成分)::

  >>> vectors.nnz / float(vectors.shape[0])
  159.01327433628319

:func:`sklearn.datasets.fetch_20newsgroups_vectorized` 是一个返回即用的tfidf特征的函数
，而不是返回文件名。

.. _`20 newsgroups website`: http://people.csail.mit.edu/jrennie/20Newsgroups/
.. _`TF-IDF`: https://en.wikipedia.org/wiki/Tf-idf

过滤文本进行更加逼真的训练
-----------------------------------------------------

分类器很容易过拟合一个出现在20个新闻组数据中的特定事物，例如新闻组标头。许多分类器有
很好的F分数，但是他们的结果不能泛化到不在这个时间窗的其他文档。

例如，我们来看一下多项式贝叶斯分类器，它训练速度快并且能获得很好的F分数。

  >>> from sklearn.naive_bayes import MultinomialNB
  >>> from sklearn import metrics
  >>> newsgroups_test = fetch_20newsgroups(subset='test',
  ...                                      categories=categories)
  >>> vectors_test = vectorizer.transform(newsgroups_test.data)
  >>> clf = MultinomialNB(alpha=.01)
  >>> clf.fit(vectors, newsgroups_train.target)
  >>> pred = clf.predict(vectors_test)
  >>> metrics.f1_score(newsgroups_test.target, pred, average='macro')
  0.88213592402729568

(:ref:`sphx_glr_auto_examples_text_document_classification_20newsgroups.py` 的例子将训练和测试数据混合，
而不是按时间划分，这种情况下，多项式贝叶斯能得到更高的0.88的F分数.你是否还不信任这个分类器的内部实现？)

让我们看看信息量最大一些特征是:

  >>> import numpy as np
  >>> def show_top10(classifier, vectorizer, categories):
  ...     feature_names = np.asarray(vectorizer.get_feature_names())
  ...     for i, category in enumerate(categories):
  ...         top10 = np.argsort(classifier.coef_[i])[-10:]
  ...         print("%s: %s" % (category, " ".join(feature_names[top10])))
  ...
  >>> show_top10(clf, vectorizer, newsgroups_train.target_names)
  alt.atheism: sgi livesey atheists writes people caltech com god keith edu
  comp.graphics: organization thanks files subject com image lines university edu graphics
  sci.space: toronto moon gov com alaska access henry nasa edu space
  talk.religion.misc: article writes kent people christian jesus sandvik edu com god

你现在可以看到这些特征过拟合了许多东西:

- 几乎所有的组都通过标题是出现更多还是更少来区分，例如 ``NNTP-Posting-Host:`` 和 ``Distribution:`` 标题
- 正如他的标头或者签名所表示，另外重要的特征有关发送者是否隶属于一个大学。
- "article"这个单词是一个重要的特征，它基于人们像 "In article [article ID], [name] <[e-mail address]>
  wrote:" 的方式引用原先的帖子频率。
- 其他特征和当时发布的特定的人的名字和e-mail相匹配。

有如此大量的线索来区分新闻组，分类器根本不需要从文本中识别主题，而且他们的性能都一样好。

由于这个原因，加载20个新闻组数据的函数提供了一个叫做 **remove** 的参数，来告诉函数需要从文件
中去除什么类别的信息。 **remove** 应该是一个来自集合 ``('headers', 'footers', 'quotes')`` 的子集
的元组，来告诉函数分别移除标头标题，签名块还有引用块。

  >>> newsgroups_test = fetch_20newsgroups(subset='test',
  ...                                      remove=('headers', 'footers', 'quotes'),
  ...                                      categories=categories)
  >>> vectors_test = vectorizer.transform(newsgroups_test.data)
  >>> pred = clf.predict(vectors_test)
  >>> metrics.f1_score(pred, newsgroups_test.target, average='macro')
  0.77310350681274775

由于我们移除了跟主题分类几乎没有关系的元数据，分类器的F分数降低了很多。
如果我们从训练数据中也移除这个元数据，F分数将会更低:

  >>> newsgroups_train = fetch_20newsgroups(subset='train',
  ...                                       remove=('headers', 'footers', 'quotes'),
  ...                                       categories=categories)
  >>> vectors = vectorizer.fit_transform(newsgroups_train.data)
  >>> clf = MultinomialNB(alpha=.01)
  >>> clf.fit(vectors, newsgroups_train.target)
  >>> vectors_test = vectorizer.transform(newsgroups_test.data)
  >>> pred = clf.predict(vectors_test)
  >>> metrics.f1_score(newsgroups_test.target, pred, average='macro')
  0.76995175184521725

其他的一些分类器能够更好的处理这个更难版本的任务。试着带 ``--filter`` 选项和不带 ``--filter`` 选项运行
 :ref:`sphx_glr_auto_examples_model_selection_grid_search_text_feature_extraction.py` 来比较结果间的差异。
.. topic:: 推荐

  当使用20个新闻组数据中评估文本分类器时，你应该移除与新闻组相关的元数据。你可以通过设置
   ``remove=('headers', 'footers', 'quotes')`` 来实现。F分数将更加低因为这更符合实际
.. topic:: 例子

   * :ref:`sphx_glr_auto_examples_model_selection_grid_search_text_feature_extraction.py`

   * :ref:`sphx_glr_auto_examples_text_document_classification_20newsgroups.py`
