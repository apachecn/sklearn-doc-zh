.. _text_data_tutorial:

======================
使用文本数据
======================

这部分教程详细介绍了 ``scikit-learn`` 中在单独实际中的应用的工具: 在 20 个不同的主题中文本文档聚集分析（newsgroups posts (新闻内容)）。

在这个教程中，我们将会学习如何:

  - 读取文件内容以及所属的类别

  - 提取特征向量为机器学习算法提供数据

  - 训练一个线性模型来分类

  - 使用 grid search strategy（网格搜索策略）来发现 feature extraction components and the classifier （特征提取组件以及分类器的良好配置）



教程设置
--------------

开始这篇教程之前，必须首先安装 *scikit-learn* 以及依赖的所有库。

请参考 :ref:`installation instructions <installation-instructions>` 获取更多信息以及安装指导。

这篇入门教程的源代码在您的 scikit-learn 文件夹下面::

    scikit-learn/doc/tutorial/text_analytics/

在这个入门教程中，应该包含以下的子文件夹:

  * ``*.rst files`` - sphinx 格式的入门教程

  * ``data`` - 入门教程所用到的数据集

  * ``skeletons`` - 一些练习所用到的待完成的脚本

  * ``solutions`` - 练习的答案


你也可以将这个文件夹拷贝到您的电脑的硬盘里命名为 ``sklearn_tut_workspace`` 来完成练习而不会破坏原始的代码结构::

    % cp -r skeletons work_directory/sklearn_tut_workspace

机器学习算法需要数据. 进入每一个 ``$TUTORIAL_HOME/data`` 子文件夹，并且运行 ``fetch_data.py`` 脚本 (首先确保您已经读取完成).

例如::

    % cd $TUTORIAL_HOME/data/languages
    % less fetch_data.py
    % python fetch_data.py


加载20个新闻组数据集
---------------------------------

这个数据集被称为 "Twenty Newsgroups". 下面就是这个数据集的介绍, 来自于网站 `website <http://people.csail.mit.edu/jrennie/20Newsgroups/>`_:

  20 个新闻组数据集是大约 20,000 的集合新闻组文件，分区（几乎）平均 20 个不同新闻组。 据我们所知，这是最初收集的由 Ken Lang ，可能是为了他的论文 "Newsweeder: Learning to filter netnews," 虽然他没有明确提及这个集合。现在 20 个新闻组集已成为一个流行的数据集，广泛应用于机器学习中的文本应用，如文本分类和文本聚类。

接下来我们会使用scikit-learn中的内置数据集读取函数. 当然, 也可以手动从网站上下载数据集，然后使用函数 :func:`sklearn.datasets.load_files` 以及指明 ``20news-bydate-train`` 子文件夹，该子文件夹包含了未压缩的数据集。

为了节约时间，在第一个示例中我们先简单测试 20 类别中 4 个类别::

  >>> categories = ['alt.atheism', 'soc.religion.christian',
  ...               'comp.graphics', 'sci.med']

接下来我们能够读取列表中对应的类别的文件::

  >>> from sklearn.datasets import fetch_20newsgroups
  >>> twenty_train = fetch_20newsgroups(subset='train',
  ...     categories=categories, shuffle=True, random_state=42)

返回的数据类型是 ``scikit-learn`` "bunch": 一个简单的数据类型能够与 python 中的 ``dict`` keys 或 ``object`` 中属性来读取, 比如 ``target_names`` 包含了所有类别的名称::

  >>> twenty_train.target_names
  ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']

这些文件本身被读进内存 ``data`` 属性. 这些文件名称也可以容易获取到::

  >>> len(twenty_train.data)
  2257
  >>> len(twenty_train.filenames)
  2257

让我们打印出读进的内容的第一行::

  >>> print("\n".join(twenty_train.data[0].split("\n")[:3]))
  From: sd345@city.ac.uk (Michael Collier)
  Subject: Converting images to HP LaserJet III?
  Nntp-Posting-Host: hampton

  >>> print(twenty_train.target_names[twenty_train.target[0]])
  comp.graphics

监督学习需要每个类别标签以及所对应的文件在测试集中. 在这个例子中，类别是每个新闻组的名称，同时也是每个文件夹的名称，来区分不同的文本文件。.

考虑到速度以及空间 ``scikit-learn`` 读取目标属性使用数字来表示类别 ``target_names`` 列表中的索引. 每个样本的类别 id 存放在 ``target`` 属性中::

  >>> twenty_train.target[:10]
  array([1, 1, 3, 3, 3, 3, 3, 2, 2, 2])

当然也可以读取对应的类别属于的类型::

  >>> for t in twenty_train.target[:10]:
  ...     print(twenty_train.target_names[t])
  ...
  comp.graphics
  comp.graphics
  soc.religion.christian
  soc.religion.christian
  soc.religion.christian
  soc.religion.christian
  soc.religion.christian
  sci.med
  sci.med
  sci.med

你可以发现所有的样本都被随机打乱 (使用了修正的 RNG 种子): 这是非常有用的，在进行整个数据集训练之前，需要快速训练一个模型的时候以及验证想法。


从文本文件中提取特征
-----------------------------------

为了在文本文件中应用机器学习算法, 我们首先要做的就是讲文本内容转化成数字形式的特征向量。

.. currentmodule:: sklearn.feature_extraction.text


词袋模型
~~~~~~~~~~~~~

最直接的方法就是用词袋来表示:

  1. 在训练集中使用合适的数值来表示每一个单词的出现次数 (比如建立一个从单词到数值的字典)。

  2. 对于每个文档 ``#i``，计算每个单词 ``w`` 的出现次数并将其存储在 ``X[i, j]`` 中作为特征 ``#j`` 的值，其中 ``j`` 是词典中词 ``w`` 的索引

词袋模型中 ``n_features`` 是整个不同单词的数量: 这个值一般来说超过 100,000.

如果 ``n_samples == 10000``, 使用 numpy 数组来存储 ``X`` 将会需要 10000 x 100000 x 4 bytes = **4GB内存** ，在当前的计算机中不太可能的。

幸运的是, **X 数组中大多是 0** ，是因为文档中使用的单词数量远远少于总体的词袋单词个数. 因此我们可以称词袋模型是典型的
**high-dimensional sparse datasets（高维稀疏数据集）**. 我们能够通过只保存那些非0的部分表示的特征向量来节约内存。

``scipy.sparse`` 数据结构就是这样的功能,同时 ``scikit-learn`` 有内置的模块支持这样的数据结构


使用 ``scikit-learn`` 来分词
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

文本处理, 分词和过滤停词在建立特征字典和将文本转换成特征向量中都是用到的::

  >>> from sklearn.feature_extraction.text import CountVectorizer
  >>> count_vect = CountVectorizer()
  >>> X_train_counts = count_vect.fit_transform(twenty_train.data)
  >>> X_train_counts.shape
  (2257, 35788)

:class:`CountVectorizer` 提供了 N-gram 模型以及连续词模型.
一旦使用, 向量化就会建立特征索引字典::

  >>> count_vect.vocabulary_.get(u'algorithm')
  4690

在字典中一个单词的索引值代表了该单词在整个词袋中出现的频率.

.. note:

  方法 ``count_vect.fit_transform`` 表示了两种操作: 学习词汇并将其转换成统计向量。当然也可以将步骤分解开来：首先 ``X_train_counts = count_vect.transform(twenty_train.data)`` ，其次 ``count_vect.fit(twenty_train.data)`` ，但是这样做的话需要进行两次操作.


从出现次数到词频
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

出现次数的统计是非常好的想法，但是有一些问题：长的文本相对于短的文本有更高的平均单词出现次数，尽管都是描述的同一个主题。

为了防止这种潜在的缺点，使用每个单词出现的次数除以总共单词出现的次数：这个特征称之为词频(tf)。

另一种情况就是，有些单词在很多文档中出现，但是有更小的信息量相对于其他的仅仅在一些文档中出现的次数。

这种下 downscaling （缩减比例）称为 `tf–idf`_ ，用于 "Term Frequency times Inverse Document Frequency".

.. _`tf–idf`: https://en.wikipedia.org/wiki/Tf–idf


**tf** 和 **tf–idf** 都可以按照下面的方式计算::

  >>> from sklearn.feature_extraction.text import TfidfTransformer
  >>> tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
  >>> X_train_tf = tf_transformer.transform(X_train_counts)
  >>> X_train_tf.shape
  (2257, 35788)

在上面的例子中，我们首先使用了 ``fit(..)`` 方法来拟合数据，接下来使用 ``transform(..)`` 方法来构建 `tf-idf` 矩阵.这两步可以直接使用一步. 使用 ``fit_transform(..)`` 方法::

  >>> tfidf_transformer = TfidfTransformer()
  >>> X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
  >>> X_train_tfidf.shape
  (2257, 35788)


训练分类器
---------------------

现在我们有了特征，我们来训练一个分类器来预测文章所属的类别. 我们来使用 :ref:`naïve Bayes <naive_bayes>` 分类器,该分类器在该任务上表现是特别好的. ``scikit-learn`` 包含了这种分类器的各种变种;一个在该问题上最好的分类器就是多项式分类器::

  >>> from sklearn.naive_bayes import MultinomialNB
  >>> clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

为了预测新文档所属的类别我们和之前一样需要抽取新文档的特征值。不同的地方在于使用 ``transform`` 而不是 ``fit_transform`` 因为文档已经在训练集的时候处理过了::

  >>> docs_new = ['God is love', 'OpenGL on the GPU is fast']
  >>> X_new_counts = count_vect.transform(docs_new)
  >>> X_new_tfidf = tfidf_transformer.transform(X_new_counts)

  >>> predicted = clf.predict(X_new_tfidf)

  >>> for doc, category in zip(docs_new, predicted):
  ...     print('%r => %s' % (doc, twenty_train.target_names[category]))
  ...
  'God is love' => soc.religion.christian
  'OpenGL on the GPU is fast' => comp.graphics


建一条 Pipeline（管道）
-----------------------

为了使得向量化 => 预处理 => 分类器 过程更加简单,``scikit-learn`` 提供了 一个 ``管道`` 来融合成一个混合模型::

  >>> from sklearn.pipeline import Pipeline
  >>> text_clf = Pipeline([('vect', CountVectorizer()),
  ...                      ('tfidf', TfidfTransformer()),
  ...                      ('clf', MultinomialNB()),
  ... ])

名称 ``vect``, ``tfidf`` and ``clf`` (分类器)都是固定的。我们将会在下面看到如何使用它们进行网格搜索。接下来我们来训练数据::

  >>> text_clf.fit(twenty_train.data, twenty_train.target)  # doctest: +ELLIPSIS
  Pipeline(...)


评估测试集上的性能
---------------------------------------------

评估模型的精度同样简单::

  >>> import numpy as np
  >>> twenty_test = fetch_20newsgroups(subset='test',
  ...     categories=categories, shuffle=True, random_state=42)
  >>> docs_test = twenty_test.data
  >>> predicted = text_clf.predict(docs_test)
  >>> np.mean(predicted == twenty_test.target)            # doctest: +ELLIPSIS
  0.834...

如上, 我们模型的精度为 83.4%。我们使用线性分类模型 :ref:`support vector machine (SVM) <svm>`, 一种公认的最好的文本分类算法 (尽管训练速度没有朴素贝叶斯快). 我们改变分类算法只需要在管道中添加不同的算法即可。::

  >>> from sklearn.linear_model import SGDClassifier
  >>> text_clf = Pipeline([('vect', CountVectorizer()),
  ...                      ('tfidf', TfidfTransformer()),
  ...                      ('clf', SGDClassifier(loss='hinge', penalty='l2',
  ...                                            alpha=1e-3, random_state=42,
  ...                                            max_iter=5, tol=None)),
  ... ])
  >>> text_clf.fit(twenty_train.data, twenty_train.target)  # doctest: +ELLIPSIS
  Pipeline(...)
  >>> predicted = text_clf.predict(docs_test)
  >>> np.mean(predicted == twenty_test.target)            # doctest: +ELLIPSIS
  0.912...

``scikit-learn`` 同样提供了更加细节化的模型评估工具::

  >>> from sklearn import metrics
  >>> print(metrics.classification_report(twenty_test.target, predicted,
  ...     target_names=twenty_test.target_names))
  ...                                         # doctest: +NORMALIZE_WHITESPACE
                          precision    recall  f1-score   support
  <BLANKLINE>
             alt.atheism       0.95      0.81      0.87       319
           comp.graphics       0.88      0.97      0.92       389
                 sci.med       0.94      0.90      0.92       396
  soc.religion.christian       0.90      0.95      0.93       398
  <BLANKLINE>
             avg / total       0.92      0.91      0.91      1502
  <BLANKLINE>

  >>> metrics.confusion_matrix(twenty_test.target, predicted)
  array([[258,  11,  15,  35],
         [  4, 379,   3,   3],
         [  5,  33, 355,   3],
         [  5,  10,   4, 379]])


从混淆矩阵中可以看出，新闻所属类别在无神论以及基督教两者之间是非常令人困惑的，而与计算机图形学差别是特别大的.

.. note:

  SGD 表示 Stochastic Gradient Descent（随机梯度下降算法）。这是一个非常简单的优化算法，尤其是在大规模数据集下显得特别有效.

  通过设置 ``loss="hinge"`` and ``penalty="l2"`` 我们可以通过微调分类算法的参数来设置支持向量机的损失函数.

  当然我们也可以使用 ``sklearn.svm.LinearSVC`` (线性支持向量机分类器)，提供了一种基于C语言编写的liblinear库的损失函数优化器。

.. _liblinear: http://www.csie.ntu.edu.tw/~cjlin/liblinear/


使用网格搜索的参数调优
----------------------------------

我们已经接触了类似于 ``TfidfTransformer`` 中 ``use_idf`` 这样的参数 ，分类器有多种这样的参数;比如, ``MultinomialNB`` 包含了平滑参数 ``alpha`` 以及 ``SGDClassifier`` 有惩罚参数 ``alpha`` 和设置损失以及惩罚因子 (更多信息请使用 python 的 ``help`` 文档).

除了通过随机（chain）来寻找参数, 通过构建巨大的网格搜索来寻找可行的参数也是可以的。 我们可以对线性支持向量机使用每个单词或者使用 n-gram, 是否使用 idf, 以及设置惩罚参数从 0.01 到 0.001::

  >>> from sklearn.model_selection import GridSearchCV
  >>> parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
  ...               'tfidf__use_idf': (True, False),
  ...               'clf__alpha': (1e-2, 1e-3),
  ... }

很明显的可以发现, 如此的搜索是非常耗时的. 如果我们有多个 cpu 核心可以使用, 通过设置 ``n_jobs`` 参数能够进行并行搜索. 如果我们将该参数设置为 ``-1``, 该方法会使用机器的所有 cpu 核心::

  >>> gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)

网格搜索在 ``scikit-learn`` 中是非常常见的。 让我们来选择一部分训练集来加速训练::

  >>> gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])

在 ``GridSearchCV`` 中使用 ``fit``，因此我们也能够使 ``predict``::

  >>> twenty_train.target_names[gs_clf.predict(['God is love'])[0]]
  'soc.religion.christian'

对象 ``best_score_`` 和 ``best_params_`` 存放了最佳的平均分数以及所对应的参数::

  >>> gs_clf.best_score_                                  # doctest: +ELLIPSIS
  0.900...
  >>> for param_name in sorted(parameters.keys()):
  ...     print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
  ...
  clf__alpha: 0.001
  tfidf__use_idf: True
  vect__ngram_range: (1, 1)

更多的详细信息可以在 ``gs_clf.cv_results_`` 中发现.

``cv_results_`` 参数也能够被导入进 ``DataFrame``，供后期使用.

.. note:

  ``GridSearchCV`` 对象也保存了最好的分类器保存在 ``best_estimator_`` 属性中。 在这个例子中, 训练一个仅仅有 400 个文档的子数据集没有什么效果的。


练习
~~~~~~~~~

为了做这个练习, 请拷贝 'skeletons' 文件夹到新的文件夹并命名为 'workspace'::

  % cp -r skeletons workspace

这时候可以任意更改练习的代码而不会破坏原始的代码结构。

然后启动 ipython 交互环境，并键入以下代码::

  [1] %run workspace/exercise_XX_script.py arg1 arg2 arg3

如果出现错误, 请使用 ``%debug`` 来启动 ipdb 调试环境.

迭代更改答案直到练习完成.

**在每个练习中, skeleton 文件包含了做练习使用的一切数据及代码。**


练习1：语言识别
-----------------------------------

- 请使用自己定制的处理的 ``CharNGramAnalyzer`` 来编写一个语言分类器，使用维基百科的预料作为训练集.

- 使用测试集来测试分类性能。

ipython command line::

  %run workspace/exercise_01_language_train_model.py data/languages/paragraphs/


练习2：情绪分析电影评论
-----------------------------------------------

- 编写一个电影评论分类器判断评论正面还是负面。

- 使用网格搜索来找到最好的参数集。

- 使用测试集来测试分类性能.

ipython 命令行::

  %run workspace/exercise_02_sentiment.py data/movie_reviews/txt_sentoken/


练习3：CLI文本分类实用程序
-------------------------------------------

使用刚刚的练习的结果以及标准库 ``cPickle`` , 编写一个命令行工具来检测输入的英文的文本是正面信息还是负面的.

如果实用程序能够给出其预测的置信水平，就可以得到奖励.


接下来要去哪里
------------------

当你完成这个章节时，以下是几个建议:


* 尝试使用 ``analyzer`` 以及 ``token normalisation`` 在该 :class:`CountVectorizer` 类下

* 如果你没有标签, 使用:ref:`Clustering <sphx_glr_auto_examples_text_document_clustering.py>` 来解决你的问题。

* 如果对每一篇文章有多个标签，请参考 :ref:`Multiclass and multilabel section <multiclass>`

* 使用 :ref:`Truncated SVD <LSA>` 解决 `latent semantic analysis <https://en.wikipedia.org/wiki/Latent_semantic_analysis>`_.

* 使用 :ref:`Out-of-core Classification <sphx_glr_auto_examples_applications_plot_out_of_core_classification.py>` 来在没有全部读入数据来进行机器学习。

* 使用 :ref:`Hashing Vectorizer <hashing_vectorizer>` 另一种方案来节省内存 :class:`CountVectorizer` 。
