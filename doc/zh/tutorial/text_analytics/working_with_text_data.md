.. _text_data_tutorial:

======================
使用文本数据
======================

这部分教程详细介绍了 ``scikit-learn``中在单独实际中的应用的工具：在20个不同的主题中文本文档聚集分析（新闻内容）。

在这个教程中，我们将会学习如何:

  - 读取文件内容以及所属的类别

  - 提取特征向量为机器学习算法提供数据

  - 训练一个线性模型来分类

  - 使用网格搜索来发现较好的特征以及分类器



教程设置
--------------

开始这篇教程之前，必须首先安装*scikit-learn*以及依赖的所有库。 

请参考:ref:`installation instructions <installation-instructions>`获取更多信息以及安装指导。

这篇入门教程的源代码在您的scikit-learn文件夹下面::

    scikit-learn/doc/tutorial/text_analytics/

在这个入门教程中，应该包含以下的子文件夹:

  * ``*.rst files`` - sphinx格式的入门教程

  * ``data`` - 入门教程所用到的数据集

  * ``skeletons`` - 一些练习所用到的待完成的脚本

  * ``solutions`` - 练习的答案

你也可以将这个文件夹拷贝到您的电脑的硬盘里命名为
 ``sklearn_tut_workspace`` 来完成练习而不会破坏原始的代码结构::

    % cp -r skeletons work_directory/sklearn_tut_workspace

机器学习算法需要数据. 进入每一个 ``$TUTORIAL_HOME/data``
子文件夹 并且运行 ``fetch_data.py`` 脚本 (首先确保您已经看完).

比如::

    % cd $TUTORIAL_HOME/data/languages
    % less fetch_data.py
    % python fetch_data.py


加载20个新闻组数据集
---------------------------------

这个数据集被称为 "Twenty Newsgroups". 下面就是这个数据集的介绍, 来自于网站 `website
<http://people.csail.mit.edu/jrennie/20Newsgroups/>`_:

20个新闻组数据集是大约20,000的集合新闻组文件，分区（几乎）平均20个不同
新闻组。 据我们所知，这是最初收集的由Ken Lang，可能是为了他的论文“Newsweeder: Learning to filter netnews“，虽然他没有明确提及这个集合。现在20个新闻组集已成为一个流行的数据集，广泛应用于机器学习中的文本应用，如文本分类和文本聚类。

接下来我们会使用scikit-learn中的内置数据集读取函数. 当然, 也可以手动从网站上下载数据集，然后使用函数:func:`sklearn.datasets.load_files`
以及指明 ``20news-bydate-train`` 子文件夹，该子文件夹包含了未压缩的数据集。

为了节约时间，在第一个示例中我们先简单测试20类别中4个类别::

  >>> categories = ['alt.atheism', 'soc.religion.christian',
  ...               'comp.graphics', 'sci.med']

接下来我们能够读取列表中对应的类别的文件::

  >>> from sklearn.datasets import fetch_20newsgroups
  >>> twenty_train = fetch_20newsgroups(subset='train',
  ...     categories=categories, shuffle=True, random_state=42)

返回的数据类型是 ``scikit-learn`` "簇": 一个简单的数据类型能够与python中的 ``dict``键 或 ``object`` 中属性来读取, 比如``target_names``包含了所有类别的名称 ::

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

监督学习需要每个类别标签以及所对应的文件在测试集中. 在这个例子中，类别是每个新闻组的名称，同时也是每个文件夹的名称，来区分不同的文本文件。

考虑到速度以及空间 ``scikit-learn`` 读取目标属性使用数字来表示类别 ``target_names`` 列表中的索引. 每个样本的类别id存放在 ``target`` 属性中::

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

你可以发现所有的样本都被随机打乱 (使用了修正的RNG种子): 这是非常有用的，在进行整个数据集训练之前，需要快速训练一个模型的时候以及验证想法。


从文本文件中提取特征
-----------------------------------

In order to perform machine learning on text documents, we first need to
turn the text content into numerical feature vectors.

.. currentmodule:: sklearn.feature_extraction.text


Bags of words
~~~~~~~~~~~~~

The most intuitive way to do so is the bags of words representation:

  1. assign a fixed integer id to each word occurring in any document
     of the training set (for instance by building a dictionary
     from words to integer indices).

  2. for each document ``#i``, count the number of occurrences of each
     word ``w`` and store it in ``X[i, j]`` as the value of feature
     ``#j`` where ``j`` is the index of word ``w`` in the dictionary

The bags of words representation implies that ``n_features`` is
the number of distinct words in the corpus: this number is typically
larger than 100,000.

If ``n_samples == 10000``, storing ``X`` as a numpy array of type
float32 would require 10000 x 100000 x 4 bytes = **4GB in RAM** which
is barely manageable on today's computers.

Fortunately, **most values in X will be zeros** since for a given
document less than a couple thousands of distinct words will be
used. For this reason we say that bags of words are typically
**high-dimensional sparse datasets**. We can save a lot of memory by
only storing the non-zero parts of the feature vectors in memory.

``scipy.sparse`` matrices are data structures that do exactly this,
and ``scikit-learn`` has built-in support for these structures.


Tokenizing text with ``scikit-learn``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Text preprocessing, tokenizing and filtering of stopwords are included in a high level component that is able to build a
dictionary of features and transform documents to feature vectors::

  >>> from sklearn.feature_extraction.text import CountVectorizer
  >>> count_vect = CountVectorizer()
  >>> X_train_counts = count_vect.fit_transform(twenty_train.data)
  >>> X_train_counts.shape
  (2257, 35788)

:class:`CountVectorizer` supports counts of N-grams of words or consecutive characters.
Once fitted, the vectorizer has built a dictionary of feature indices::

  >>> count_vect.vocabulary_.get(u'algorithm')
  4690

The index value of a word in the vocabulary is linked to its frequency
in the whole training corpus.

.. note:

  The method ``count_vect.fit_transform`` performs two actions:
  it learns the vocabulary and transforms the documents into count vectors.
  It's possible to separate these steps by calling
  ``count_vect.fit(twenty_train.data)`` followed by
  ``X_train_counts = count_vect.transform(twenty_train.data)``,
  but doing so would tokenize and vectorize each text file twice.


From occurrences to frequencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Occurrence count is a good start but there is an issue: longer
documents will have higher average count values than shorter documents,
even though they might talk about the same topics.

To avoid these potential discrepancies it suffices to divide the
number of occurrences of each word in a document by the total number
of words in the document: these new features are called ``tf`` for Term
Frequencies.

Another refinement on top of tf is to downscale weights for words
that occur in many documents in the corpus and are therefore less
informative than those that occur only in a smaller portion of the
corpus.

This downscaling is called `tf–idf`_ for "Term Frequency times
Inverse Document Frequency".

.. _`tf–idf`: https://en.wikipedia.org/wiki/Tf–idf


Both **tf** and **tf–idf** can be computed as follows::

  >>> from sklearn.feature_extraction.text import TfidfTransformer
  >>> tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
  >>> X_train_tf = tf_transformer.transform(X_train_counts)
  >>> X_train_tf.shape
  (2257, 35788)

In the above example-code, we firstly use the ``fit(..)`` method to fit our
estimator to the data and secondly the ``transform(..)`` method to transform
our count-matrix to a tf-idf representation.
These two steps can be combined to achieve the same end result faster
by skipping redundant processing. This is done through using the
``fit_transform(..)`` method as shown below, and as mentioned in the note
in the previous section::

  >>> tfidf_transformer = TfidfTransformer()
  >>> X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
  >>> X_train_tfidf.shape
  (2257, 35788)


训练分类器
---------------------

Now that we have our features, we can train a classifier to try to predict
the category of a post. Let's start with a :ref:`naïve Bayes <naive_bayes>`
classifier, which
provides a nice baseline for this task. ``scikit-learn`` includes several
variants of this classifier; the one most suitable for word counts is the
multinomial variant::

  >>> from sklearn.naive_bayes import MultinomialNB
  >>> clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

To try to predict the outcome on a new document we need to extract
the features using almost the same feature extracting chain as before.
The difference is that we call ``transform`` instead of ``fit_transform``
on the transformers, since they have already been fit to the training set::

  >>> docs_new = ['God is love', 'OpenGL on the GPU is fast']
  >>> X_new_counts = count_vect.transform(docs_new)
  >>> X_new_tfidf = tfidf_transformer.transform(X_new_counts)

  >>> predicted = clf.predict(X_new_tfidf)

  >>> for doc, category in zip(docs_new, predicted):
  ...     print('%r => %s' % (doc, twenty_train.target_names[category]))
  ...
  'God is love' => soc.religion.christian
  'OpenGL on the GPU is fast' => comp.graphics


建一条管道
-------------------

In order to make the vectorizer => transformer => classifier easier
to work with, ``scikit-learn`` provides a ``Pipeline`` class that behaves
like a compound classifier::

  >>> from sklearn.pipeline import Pipeline
  >>> text_clf = Pipeline([('vect', CountVectorizer()),
  ...                      ('tfidf', TfidfTransformer()),
  ...                      ('clf', MultinomialNB()),
  ... ])

The names ``vect``, ``tfidf`` and ``clf`` (classifier) are arbitrary.
We shall see their use in the section on grid search, below.
We can now train the model with a single command::

  >>> text_clf.fit(twenty_train.data, twenty_train.target)  # doctest: +ELLIPSIS
  Pipeline(...)


评估测试集上的性能
---------------------------------------------

Evaluating the predictive accuracy of the model is equally easy::

  >>> import numpy as np
  >>> twenty_test = fetch_20newsgroups(subset='test',
  ...     categories=categories, shuffle=True, random_state=42)
  >>> docs_test = twenty_test.data
  >>> predicted = text_clf.predict(docs_test)
  >>> np.mean(predicted == twenty_test.target)            # doctest: +ELLIPSIS
  0.834...

I.e., we achieved 83.4% accuracy. Let's see if we can do better with a
linear :ref:`support vector machine (SVM) <svm>`,
which is widely regarded as one of
the best text classification algorithms (although it's also a bit slower
than naïve Bayes). We can change the learner by just plugging a different
classifier object into our pipeline::

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

``scikit-learn`` further provides utilities for more detailed performance
analysis of the results::

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


As expected the confusion matrix shows that posts from the newsgroups
on atheism and christian are more often confused for one another than
with computer graphics.

.. note:

  SGD stands for Stochastic Gradient Descent. This is a simple
  optimization algorithms that is known to be scalable when the dataset
  has many samples.

  By setting ``loss="hinge"`` and ``penalty="l2"`` we are configuring
  the classifier model to tune its parameters for the linear Support
  Vector Machine cost function.

  Alternatively we could have used ``sklearn.svm.LinearSVC`` (Linear
  Support Vector Machine Classifier) that provides an alternative
  optimizer for the same cost function based on the liblinear_ C++
  library.

.. _liblinear: http://www.csie.ntu.edu.tw/~cjlin/liblinear/


使用网格搜索的参数调优
----------------------------------

We've already encountered some parameters such as ``use_idf`` in the
``TfidfTransformer``. Classifiers tend to have many parameters as well;
e.g., ``MultinomialNB`` includes a smoothing parameter ``alpha`` and
``SGDClassifier`` has a penalty parameter ``alpha`` and configurable loss
and penalty terms in the objective function (see the module documentation,
or use the Python ``help`` function, to get a description of these).

Instead of tweaking the parameters of the various components of the
chain, it is possible to run an exhaustive search of the best
parameters on a grid of possible values. We try out all classifiers
on either words or bigrams, with or without idf, and with a penalty
parameter of either 0.01 or 0.001 for the linear SVM::

  >>> from sklearn.model_selection import GridSearchCV
  >>> parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
  ...               'tfidf__use_idf': (True, False),
  ...               'clf__alpha': (1e-2, 1e-3),
  ... }

Obviously, such an exhaustive search can be expensive. If we have multiple
CPU cores at our disposal, we can tell the grid searcher to try these eight
parameter combinations in parallel with the ``n_jobs`` parameter. If we give
this parameter a value of ``-1``, grid search will detect how many cores
are installed and uses them all::

  >>> gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)

The grid search instance behaves like a normal ``scikit-learn``
model. Let's perform the search on a smaller subset of the training data
to speed up the computation::

  >>> gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])

The result of calling ``fit`` on a ``GridSearchCV`` object is a classifier
that we can use to ``predict``::

  >>> twenty_train.target_names[gs_clf.predict(['God is love'])[0]]
  'soc.religion.christian'

The object's ``best_score_`` and ``best_params_`` attributes store the best
mean score and the parameters setting corresponding to that score::

  >>> gs_clf.best_score_                                  # doctest: +ELLIPSIS
  0.900...
  >>> for param_name in sorted(parameters.keys()):
  ...     print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
  ...
  clf__alpha: 0.001
  tfidf__use_idf: True
  vect__ngram_range: (1, 1)

A more detailed summary of the search is available at ``gs_clf.cv_results_``.

The ``cv_results_`` parameter can be easily imported into pandas as a
``DataFrame`` for further inspection.

.. note:

  A ``GridSearchCV`` object also stores the best classifier that it trained
  as its ``best_estimator_`` attribute. In this case, that isn't much use as
  we trained on a small, 400-document subset of our full training set.


Exercises
~~~~~~~~~

To do the exercises, copy the content of the 'skeletons' folder as
a new folder named 'workspace'::

  % cp -r skeletons workspace

You can then edit the content of the workspace without fear of loosing
the original exercise instructions.

Then fire an ipython shell and run the work-in-progress script with::

  [1] %run workspace/exercise_XX_script.py arg1 arg2 arg3

If an exception is triggered, use ``%debug`` to fire-up a post
mortem ipdb session.

Refine the implementation and iterate until the exercise is solved.

**For each exercise, the skeleton file provides all the necessary import
statements, boilerplate code to load the data and sample code to evaluate
the predictive accurracy of the model.**


练习1：语言识别
-----------------------------------

- Write a text classification pipeline using a custom preprocessor and
  ``CharNGramAnalyzer`` using data from Wikipedia articles as training set.

- Evaluate the performance on some held out test set.

ipython command line::

  %run workspace/exercise_01_language_train_model.py data/languages/paragraphs/


练习2：情绪分析电影评论
-----------------------------------------------

- Write a text classification pipeline to classify movie reviews as either
  positive or negative.

- Find a good set of parameters using grid search.

- Evaluate the performance on a held out test set.

ipython command line::

  %run workspace/exercise_02_sentiment.py data/movie_reviews/txt_sentoken/


练习3：CLI文本分类实用程序
-------------------------------------------

Using the results of the previous exercises and the ``cPickle``
module of the standard library, write a command line utility that
detects the language of some text provided on ``stdin`` and estimate
the polarity (positive or negative) if the text is written in
English.

Bonus point if the utility is able to give a confidence level for its
predictions.


接下来要去哪里
------------------

Here are a few suggestions to help further your scikit-learn intuition
upon the completion of this tutorial:


* Try playing around with the ``analyzer`` and ``token normalisation`` under
  :class:`CountVectorizer`

* If you don't have labels, try using
  :ref:`Clustering <sphx_glr_auto_examples_text_document_clustering.py>`
  on your problem.

* If you have multiple labels per document, e.g categories, have a look
  at the :ref:`Multiclass and multilabel section <multiclass>`

* Try using :ref:`Truncated SVD <LSA>` for
  `latent semantic analysis <https://en.wikipedia.org/wiki/Latent_semantic_analysis>`_.

* Have a look at using
  :ref:`Out-of-core Classification
  <sphx_glr_auto_examples_applications_plot_out_of_core_classification.py>` to
  learn from data that would not fit into the computer main memory.

* Have a look at the :ref:`Hashing Vectorizer <hashing_vectorizer>`
  as a memory efficient alternative to :class:`CountVectorizer`.
