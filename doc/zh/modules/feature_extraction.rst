.. _feature_extraction:

==================
特征提取
==================

.. currentmodule:: sklearn.feature_extraction

The :mod:`sklearn.feature_extraction` 模块可用于从包括文本和图像等格式的数据集中提取机器学习算法支持的格式的特征.

.. note::

   意特征提取与 :ref:`特征选择` 非常不同：前者包括将任意数据（如文本或图像）转换为可用于机器学习的数值特征。后者是应用于这些特征的机器学习技术。

.. _dict_feature_extraction:

从字典类型加载特征
===========================

该类 :class:`DictVectorizer` 可用于将表示为标准 Python ``dict`` 对象列表的要素数组转换为 scikit-learn 估计器使用的 NumPy/SciPy 表示形式。

虽然 Python 的处理速度不是特别快，但 Python 的 ``dict`` 优点是使用方便，稀疏（不需要存储的特征），并且除了值之外还存储特征名称。

:class:`DictVectorizer` 实现了所谓的 "one-of-K" 或 "one-hot" 编码，用于分类（也称为标称，离散）特征。分类功能是 "属性值" 对，其中该值被限制为不排序的可能性的离散列表（例如主题标识符，对象类型，标签，名称...）。

在下文中，"城市" 是一个分类属性，而 "温度" 是传统的数字特征::

  >>> measurements = [
  ...     {'city': 'Dubai', 'temperature': 33.},
  ...     {'city': 'London', 'temperature': 12.},
  ...     {'city': 'San Francisco', 'temperature': 18.},
  ... ]

  >>> from sklearn.feature_extraction import DictVectorizer
  >>> vec = DictVectorizer()

  >>> vec.fit_transform(measurements).toarray()
  array([[  1.,   0.,   0.,  33.],
         [  0.,   1.,   0.,  12.],
         [  0.,   0.,   1.,  18.]])

  >>> vec.get_feature_names()
  ['city=Dubai', 'city=London', 'city=San Francisco', 'temperature']

:class:`DictVectorizer` 也是对自然语言处理模型中训练序列分类器的有用的表示变换，通常通过提取特定的感兴趣词的特征窗口来工作。

例如，假设我们有第一个算法来提取我们想用作训练序列分类器（例如一个块）的补码的语音（PoS）标签。以下 dict 可以是在 "坐在垫子上的猫" 的句子中围绕 "sat" 一词提取的这样一个特征窗口::

  >>> pos_window = [
  ...     {
  ...         'word-2': 'the',
  ...         'pos-2': 'DT',
  ...         'word-1': 'cat',
  ...         'pos-1': 'NN',
  ...         'word+1': 'on',
  ...         'pos+1': 'PP',
  ...     },
  ...     # in a real application one would extract many such dictionaries
  ... ]

该描述可以被矢量化为适合于呈递分类器的稀疏二维矩阵（可能在被管道 :class:`text.TfidfTransformer` 进行归一化之后）::

  >>> vec = DictVectorizer()
  >>> pos_vectorized = vec.fit_transform(pos_window)
  >>> pos_vectorized                # doctest: +NORMALIZE_WHITESPACE  +ELLIPSIS
  <1x6 sparse matrix of type '<... 'numpy.float64'>'
      with 6 stored elements in Compressed Sparse ... format>
  >>> pos_vectorized.toarray()
  array([[ 1.,  1.,  1.,  1.,  1.,  1.]])
  >>> vec.get_feature_names()
  ['pos+1=PP', 'pos-1=NN', 'pos-2=DT', 'word+1=on', 'word-1=cat', 'word-2=the']

你可以想象，如果一个文本语料库的每一个单词都提取了这样一个上下文，那么所得的矩阵将会非常宽（许多 one-hot-features），其中绝大部分的大部分时间被重视为零。
为了使结果数据结构能够适应内存，该类``DictVectorizer`` 的 ``scipy.sparse`` 默认使用一个矩阵而不是一个 ``numpy.ndarray``。


.. _feature_hashing:

特征散列
===============

.. currentmodule:: sklearn.feature_extraction

该类 :class:`FeatureHasher` 是一种高速，低内存向量化器，它使用称为`特征散列 feature hashing <https://en.wikipedia.org/wiki/Feature_hashing>`_ 的技术 ，或 "散列技巧"。
不像在向量化器中那样构建训练中遇到的特征的哈希表，而是 :class:`FeatureHasher` 将哈希函数应用于特征，以便直接在样本矩阵中确定它们的列索引。
结果是增加速度和减少内存使用，牺牲可视性; 哈希表不记得输入特性是什么样的，没有 ``inverse_transform`` 办法。

由于散列函数可能导致（不相关）特征之间的冲突，因此使用带符号散列函数，并且散列值的符号确定存储在特征的输出矩阵中的值的符号。
这样，碰撞可能会抵消而不是累积错误，并且任何输出要素的值的预期平均值为零。默认情况下，此机制将使用 ``alternate_sign=True`` 启用，对于小型哈希表大小（``n_features < 10000``）特别有用。
对于大的哈希表大小，可以禁用它，以便将输出传递给估计器，如 :class:`sklearn.naive_bayes.MultinomialNB` 或 :class:`sklearn.feature_selection.chi2` 特征选择器，这些特征选项器可以使用非负输入。

:class:`FeatureHasher` 接受映射（如 Python 的 ``dict`` 及其在 ``collections`` 模块中的变体），``(feature, value)``对或字符串，具体取决于构造函数参数 ``input_type``。
映射被视为 ``(feature, value)`` 对的列表，而单个字符串的隐含值为1，因此 ``['feat1', 'feat2', 'feat3']`` 被解释为 ``[('feat1', 1), ('feat2', 1), ('feat3', 1)]``。
如果单个特征在样本中多次出现，相关值将被求和（所以 ``('feat', 2)`` 和 ``('feat', 3.5)`` 变为 ``('feat', 5.5)``）。 :class:`FeatureHasher` 的输出始终是 CSR 格式的 ``scipy.sparse`` 矩阵。

在文档分类中可以使用特征散列，但与 :class:`text.CountVectorizer` 不同，:class:`FeatureHasher` 不执行除 Unicode 或 UTF-8 编码之外的任何其他预处理;
请参阅下面的哈希技巧向量化大文本语料库，用于组合的 tokenizer/hasher。

例如，考虑需要从 ``(token, part_of_speech)`` 对提取的特征的单词级自然语言处理任务。可以使用 Python 生成器函数来提取功能::

  def token_features(token, part_of_speech):
      if token.isdigit():
          yield "numeric"
      else:
          yield "token={}".format(token.lower())
          yield "token,pos={},{}".format(token, part_of_speech)
      if token[0].isupper():
          yield "uppercase_initial"
      if token.isupper():
          yield "all_uppercase"
      yield "pos={}".format(part_of_speech)

然后，将 ``raw_X``  要被馈送到 ``FeatureHasher.transform`` 可使用被构造::

  raw_X = (token_features(tok, pos_tagger(tok)) for tok in corpus)

并创建一个 hasher::

  hasher = FeatureHasher(input_type='string')
  X = hasher.transform(raw_X)

得到一个 ``scipy.sparse`` 矩阵 ``X``。

注意使用发生器的理解，它将懒惰引入到特征提取中：令牌只能根据需要从哈希值进行处理。

实现细节
----------------------

:class:`FeatureHasher` 使用签名的 32-bit 变体的 MurmurHash3。
结果（并且由于限制 ``scipy.sparse``），当前支持的功能的最大数量 :math:`2^{31} - 1`.

Weinberger等人的散文技巧的原始方法 使用两个单独的哈希函数，:math:`h` 和 :math:`\xi` 分别确定特征的列索引和符号。
本实现在假设 MurmurHash3 的符号位与其他 bits 无关的情况下工作。

由于使用简单的模数将哈希函数转换为列索引，建议使用2的幂作为 ``n_features`` 参数; 否则功能将不会均匀地映射到列。

参考文献：
Kilian Weinberger，Anirban Dasgupta，John Langford，Alex Smola和Josh Attenberg（2009）。用于大规模多任务学习的特征散列。PROC。ICML。
MurmurHash3。

.. topic:: 参考文献:

 * Kilian Weinberger, Anirban Dasgupta, John Langford, Alex Smola and
   Josh Attenberg (2009). `用于大规模多任务学习的特征散列 <http://alex.smola.org/papers/2009/Weinbergeretal09.pdf>`_. Proc. ICML.

 * `MurmurHash3 <https://github.com/aappleby/smhasher>`_.


.. _text_feature_extraction:

文本特征提取
=======================

.. currentmodule:: sklearn.feature_extraction.text


话语表示
-------------------------------

文本分析是机器学习算法的主要应用领域。
然而，原始数据，一系列符号不能直接馈送到算法本身，因为它们大多数期望具有固定大小的数字特征向量，而不是具有可变长度的原始文本文档。

为了解决这个问题，scikit-learn提供了从文本内容中提取数字特征的最常见方法的实用程序，即：

- **令牌化** 字符串，并为每个可能的令牌提供整数，例如通过使用空格和标点符号作为令牌分隔符。

- **统计** 每个文件中的令牌的发生。

- **标准化** 和加权，在大多数样品/文件中发生的重要性减弱。

在该方案中，特征和样本定义如下：

- 每个**单独的令牌发生频率**（归一化或不归零）被视为一个**特征**。

- 给定**文档**的所有令牌频率的向量被认为是多变量**样本**。

因此，文档语料库可以由每个文档具有一行的矩阵和在语料库中出现的每个令牌（例如，单词）的一列表示。

我们称**向量化**将文本文档集合转换为数字特征向量的一般过程。
这种具体的策略（令牌化，计数和归一化）被称为 **Bag of Words** 或 "Bag of n-grams" 表示。
文档由单词出现来描述，同时完全忽略文档中单词的相对位置信息。


稀疏
--------

由于大多数文档通常会使用语料库中使用的单词的非常小的子集，所以得到的矩阵将具有许多特征值，它们是零（通常大于99％）。

例如，10,000 个短文本文档（如电子邮件）的集合将使用总共100,000个独特词的大小的词汇，而每个文档将单独使用100到1000个独特的单词。

为了能够将这样的矩阵存储在存储器中，并且还可以加速代数运算 矩阵/向量，实现通常将使用诸如 ``scipy.sparse`` 包中可用的实现的稀疏表示 。


常用 Vectorizer 使用
-----------------------

:class:`CountVectorizer`  在单个类中实现标记化和事件计数::

  >>> from sklearn.feature_extraction.text import CountVectorizer

这个模型有很多参数，但默认值是相当合理的（请参阅 :ref:`参考文档<text_feature_extraction_ref>` 了解详细信息）::

  >>> vectorizer = CountVectorizer()
  >>> vectorizer                     # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
  CountVectorizer(analyzer=...'word', binary=False, decode_error=...'strict',
          dtype=<... 'numpy.int64'>, encoding=...'utf-8', input=...'content',
          lowercase=True, max_df=1.0, max_features=None, min_df=1,
          ngram_range=(1, 1), preprocessor=None, stop_words=None,
          strip_accents=None, token_pattern=...'(?u)\\b\\w\\w+\\b',
          tokenizer=None, vocabulary=None)

让我们用它来标记和计数文本文档的简约语料库的出现::

  >>> corpus = [
  ...     'This is the first document.',
  ...     'This is the second second document.',
  ...     'And the third one.',
  ...     'Is this the first document?',
  ... ]
  >>> X = vectorizer.fit_transform(corpus)
  >>> X                              # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
  <4x9 sparse matrix of type '<... 'numpy.int64'>'
      with 19 stored elements in Compressed Sparse ... format>

默认配置通过提取至少2个字母的字符来标记字符串。可以明确地请求执行此步骤的具体功能::

  >>> analyze = vectorizer.build_analyzer()
  >>> analyze("This is a text document to analyze.") == (
  ...     ['this', 'is', 'text', 'document', 'to', 'analyze'])
  True

analyzer 在拟合期间发现的每个项都被分配一个与所得矩阵中的列对应的唯一整数索引。列的这种解释可以检索如下::

  >>> vectorizer.get_feature_names() == (
  ...     ['and', 'document', 'first', 'is', 'one',
  ...      'second', 'the', 'third', 'this'])
  True

  >>> X.toarray()           # doctest: +ELLIPSIS
  array([[0, 1, 1, 1, 0, 0, 1, 0, 1],
         [0, 1, 0, 1, 0, 2, 1, 0, 1],
         [1, 0, 0, 0, 1, 0, 1, 1, 0],
         [0, 1, 1, 1, 0, 0, 1, 0, 1]]...)

从功能名称到列索引的相反映射存储在 ``vocabulary_`` 矢量化器的 属性中::

  >>> vectorizer.vocabulary_.get('document')
  1

因此，在将来的调用转换方法中，在训练语料库中看不到的单词将被完全忽略::

  >>> vectorizer.transform(['Something completely new.']).toarray()
  ...                           # doctest: +ELLIPSIS
  array([[0, 0, 0, 0, 0, 0, 0, 0, 0]]...)

请注意，在上一个语料库中，第一个和最后一个文档具有完全相同的单词，因此以相等的向量编码。
特别是我们失去了最后一个文件是一个疑问形式的信息。为了保留一些本地的订购信息，除了 1-grams（个别词）之外，我们还可以提取 2-grams 的单词::

  >>> bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),
  ...                                     token_pattern=r'\b\w+\b', min_df=1)
  >>> analyze = bigram_vectorizer.build_analyzer()
  >>> analyze('Bi-grams are cool!') == (
  ...     ['bi', 'grams', 'are', 'cool', 'bi grams', 'grams are', 'are cool'])
  True

因此，此向量化器提取的词汇量大得多，现在可以解决以本地定位模式编码的模糊::

  >>> X_2 = bigram_vectorizer.fit_transform(corpus).toarray()
  >>> X_2
  ...                           # doctest: +ELLIPSIS
  array([[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
         [0, 0, 1, 0, 0, 1, 1, 0, 0, 2, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0],
         [1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1]]...)

特别是 "Is this" 的疑问形式只存在于最后一份文件中::

  >>> feature_index = bigram_vectorizer.vocabulary_.get('is this')
  >>> X_2[:, feature_index]     # doctest: +ELLIPSIS
  array([0, 0, 0, 1]...)


.. _tfidf:

Tf–idf 项加权
---------------------

在一个大的文本语料库中，一些单词将会非常的出现（例如 "the", "a", "is" 是英文），因此对文档的实际内容没有什么有意义的信息。
如果我们将直接计数数据直接提供给分类器，那么这些非常频繁的术语将会影响更少更有趣的术语的频率。

为了将计数特征重新加载为适合分类器使用的浮点值，使用 tf-idf 变换是非常常见的。

Tf表示**术语频率**，而 tf-idf 表示术语频率乘以**反文档频率**: 
:math:`\text{tf-idf(t,d)}=\text{tf(t,d)} \times \text{idf(t)}`.

使用 ``TfidfTransformer`` 的默认设置，``TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)`` 术语频率，一个术语在给定文档中出现的次数乘以 idf 组件， 计算为

:math:`\text{idf}(t) = log{\frac{1 + n_d}{1+\text{df}(d,t)}} + 1`,

其中 :math:`n_d` 是文档的总数，:math:`\text{df}(d,t)` 是包含术语 :math:`t` 的文档数。
然后，所得到的 tf-idf 向量通过欧几里得范数归一化：

:math:`v_{norm} = \frac{v}{||v||_2} = \frac{v}{\sqrt{v{_1}^2 + v{_2}^2 + \dots + v{_n}^2}}`.

这最初是用于信息检索开发的术语加权方案（作为搜索引擎结果的排名功能），也在文档分类和聚类中得到很好的利用。

以下部分包含进一步说明和示例，说明如何精确计算 tf-idfs 以及如何在 scikit-learn 中计算 tf-idfs， :class:`TfidfTransformer` 并 :class:`TfidfVectorizer` 与定义 idf 的标准教科书符号略有不同

:math:`\text{idf}(t) = log{\frac{n_d}{1+\text{df}(d,t)}}.`

在 :class:`TfidfTransformer` 和 :class:`TfidfVectorizer` 中 ``smooth_idf=False``，将 "1" 计数添加到 idf 而不是 idf 的分母:

:math:`\text{idf}(t) = log{\frac{n_d}{\text{df}(d,t)}} + 1`

该规范化由 :class:`TfidfTransformer` 类实现::

  >>> from sklearn.feature_extraction.text import TfidfTransformer
  >>> transformer = TfidfTransformer(smooth_idf=False)
  >>> transformer   # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
  TfidfTransformer(norm=...'l2', smooth_idf=False, sublinear_tf=False,
                   use_idf=True)

有关所有参数的详细信息，请参阅 :ref:`参考文档<text_feature_extraction_ref>`。

让我们举个例子来看看下面的几个例子。第一个术语是100％的时间，因此不是很有趣。另外两个功能只有不到50％的时间，因此可能更具代表性的文件内容::

  >>> counts = [[3, 0, 1],
  ...           [2, 0, 0],
  ...           [3, 0, 0],
  ...           [4, 0, 0],
  ...           [3, 2, 0],
  ...           [3, 0, 2]]
  ...
  >>> tfidf = transformer.fit_transform(counts)
  >>> tfidf                         # doctest: +NORMALIZE_WHITESPACE  +ELLIPSIS
  <6x3 sparse matrix of type '<... 'numpy.float64'>'
      with 9 stored elements in Compressed Sparse ... format>

  >>> tfidf.toarray()                        # doctest: +ELLIPSIS
  array([[ 0.81940995,  0.        ,  0.57320793],
         [ 1.        ,  0.        ,  0.        ],
         [ 1.        ,  0.        ,  0.        ],
         [ 1.        ,  0.        ,  0.        ],
         [ 0.47330339,  0.88089948,  0.        ],
         [ 0.58149261,  0.        ,  0.81355169]])

每行标准化为具有单位欧几里得规范:

:math:`v_{norm} = \frac{v}{||v||_2} = \frac{v}{\sqrt{v{_1}^2 + v{_2}^2 + \dots + v{_n}^2}}`

例如，我们可以计算`计数`数组中第一个文档中第一个项的 tf-idf ，如下所示:

:math:`n_{d, {\text{term1}}} = 6`

:math:`\text{df}(d, t)_{\text{term1}} = 6`

:math:`\text{idf}(d, t)_{\text{term1}} = log \frac{n_d}{\text{df}(d, t)} + 1 = log(1)+1 = 1`

:math:`\text{tf-idf}_{\text{term1}} = \text{tf} \times \text{idf} = 3 \times 1 = 3`

现在，如果我们对文档中剩下的2个术语重复这个计算，我们得到

:math:`\text{tf-idf}_{\text{term2}} = 0 \times log(6/1)+1 = 0`

:math:`\text{tf-idf}_{\text{term3}} = 1 \times log(6/2)+1 \approx 2.0986`

和原始 tf-idfs 的向量:

:math:`\text{tf-idf}_raw = [3, 0, 2.0986].`

然后，应用欧几里德（L2）规范，我们获得文档1的以下 tf-idfs:

:math:`\frac{[3, 0, 2.0986]}{\sqrt{\big(3^2 + 0^2 + 2.0986^2\big)}} = [ 0.819,  0,  0.573].`

此外，默认参数 ``smooth_idf=True`` 将 "1" 添加到分子和分母，就好像一个额外的文档被看到一样包含集合中的每个术语，这样可以避免零分割:

:math:`\text{idf}(t) = log{\frac{1 + n_d}{1+\text{df}(d,t)}} + 1`

使用此修改，文档1中第三项的 tf-idf 更改为 1.8473:

:math:`\text{tf-idf}_{\text{term3}} = 1 \times log(7/3)+1 \approx 1.8473`

而 L2 标准化的 tf-idf 变为

:math:`\frac{[3, 0, 1.8473]}{\sqrt{\big(3^2 + 0^2 + 1.8473^2\big)}} = [0.8515, 0, 0.5243]`::

  >>> transformer = TfidfTransformer()
  >>> transformer.fit_transform(counts).toarray()
  array([[ 0.85151335,  0.        ,  0.52433293],
         [ 1.        ,  0.        ,  0.        ],
         [ 1.        ,  0.        ,  0.        ],
         [ 1.        ,  0.        ,  0.        ],
         [ 0.55422893,  0.83236428,  0.        ],
         [ 0.63035731,  0.        ,  0.77630514]])

通过 ``拟合`` 方法调用计算的每个特征的权重存储在模型属性中::

  >>> transformer.idf_                       # doctest: +ELLIPSIS
  array([ 1. ...,  2.25...,  1.84...])

由于 tf-idf 经常用于文本特征，还有一个名为 :class:`TfidfVectorizer` 的类，它将 :class:`CountVectorizer` 
和 :class:`TfidfTransformer` 的所有选项组合在一个单例模型中::

  >>> from sklearn.feature_extraction.text import TfidfVectorizer
  >>> vectorizer = TfidfVectorizer()
  >>> vectorizer.fit_transform(corpus)
  ...                                # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
  <4x9 sparse matrix of type '<... 'numpy.float64'>'
      with 19 stored elements in Compressed Sparse ... format>

虽然tf-idf标准化通常非常有用，但是可能存在二进制发生标记可能提供更好的特征的情况。
这可以通过使用 :class:`CountVectorizer` 的 ``二进制`` 参数来实现。 
特别地，诸如 :ref:`bernoulli_naive_bayes` 的一些估计器明确地模拟离散布尔随机变量。 
而且，非常短的文本很可能具有嘈杂的 tf-idf 值，而二进制出现信息更稳定。

像往常一样，调整特征提取参数的最佳方法是使用交叉验证的网格搜索，例如通过将特征提取器与分类器进行流水线化:

 * 用于文本特征提取和评估的样本管道 :ref:`sphx_glr_auto_examples_model_selection_grid_search_text_feature_extraction.py`


解码文本文件
-------------------

文本由字符组成，但文件由字节组成。这些字节表示根据某些 *编码* 的字符。
要使用Python中的文本文件，它们的字节必须被 *解码* 为一个称为 Unicode 的字符集。
常用的编码方式有 ASCII，Latin-1（西欧），KOI8-R（俄语）和通用编码 UTF-8 和 UTF-16。还有许多其他的编码存在

.. note::
    编码也可以称为 '字符集', 但是这个术语不太准确: 单个字符集可能存在多个编码。

scikit-learn 中的文本功能提取器知道如何解码文本文件，
但只有当您告诉他们文件的编码时， :class:`CountVectorizer` 才需要一个 ``encoding`` 参数。
对于现代文本文件，正确的编码可能是 UTF-8，因此它是 default (``encoding="utf-8"``).

如果正在加载的文本实际上并没有使用UTF-8进行编码，则会得到 ``UnicodeDecodeError``. 
通过将 ``decode_error`` 参数设置为  ``"ignore"`` 或 ``"replace"``, 向量化器可以被解释为解码错误。 
有关详细信息，请参阅Python函数 ``bytes.decode`` 的文档（在Python提示符下键入 ``help(bytes.decode)`` ）。

如果您在解码文本时遇到问题，请尝试以下操作:

- 了解文本的实际编码方式。该文件可能带有标题或 README，告诉您编码，或者可能有一些标准编码，您可以根据文本来自哪里。

- 您可能可以使用 UNIX 命令 ``file`` 找出它一般使用什么样的编码。 Python ``chardet`` 模块附带一个名为 ``chardetect.py`` 的脚本，它会猜测具体的编码，尽管你不能依靠它的猜测是正确的。

- 你可以尝试 UTF-8 并忽略错误。您可以使用 ``bytes.decode(errors='replace')`` 对字节字符串进行解码，以用无意义字符替换所有解码错误，或在向量化器中设置 ``decode_error='replace'``. 这可能会损坏您的功能的有用性。

- 真实文本可能来自各种使用不同编码的来源，或者甚至以与编码的编码不同的编码进行粗略解码。这在从 Web 检索的文本中是常见的。Python 包 `ftfy`_ 可以自动排序一些解码错误类，所以您可以尝试将未知文本解码为 ``latin-1``，然后使用 ``ftfy`` 修复错误。

- 如果文本是一个简单的难以整理的编码的混合（20个新闻组数据集的情况），您可以回到简单的单字节编码，如 ``latin-1``。某些文本可能显示不正确，但至少相同的字节序列将始终代表相同的功能。

例如，以下代码段使用 ``chardet`` （未附带 scikit-learn，必须单独安装），以找出三个文本的编码。然后，它将文本向量化并打印学习的词汇。此处未显示输出。

  >>> import chardet    # doctest: +SKIP
  >>> text1 = b"Sei mir gegr\xc3\xbc\xc3\x9ft mein Sauerkraut"
  >>> text2 = b"holdselig sind deine Ger\xfcche"
  >>> text3 = b"\xff\xfeA\x00u\x00f\x00 \x00F\x00l\x00\xfc\x00g\x00e\x00l\x00n\x00 \x00d\x00e\x00s\x00 \x00G\x00e\x00s\x00a\x00n\x00g\x00e\x00s\x00,\x00 \x00H\x00e\x00r\x00z\x00l\x00i\x00e\x00b\x00c\x00h\x00e\x00n\x00,\x00 \x00t\x00r\x00a\x00g\x00 \x00i\x00c\x00h\x00 \x00d\x00i\x00c\x00h\x00 \x00f\x00o\x00r\x00t\x00"
  >>> decoded = [x.decode(chardet.detect(x)['encoding'])
  ...            for x in (text1, text2, text3)]        # doctest: +SKIP
  >>> v = CountVectorizer().fit(decoded).vocabulary_    # doctest: +SKIP
  >>> for term in v: print(v)                           # doctest: +SKIP

（根据 ``chardet`` 的版本，可能会遇到第一个错误。）
有关 Unicode 和字符编码的一般介绍，请参阅Joel Spolsky的 `绝对最小每个软件开发人员必须了解 Unicode <http://www.joelonsoftware.com/articles/Unicode.html>`_.

.. _`ftfy`: https://github.com/LuminosoInsight/python-ftfy


应用和实例
-------------------------

词汇表达方式相当简单，但在实践中却非常有用。

特别是在 **受监督的设置** 中，它可以成功地与快速和可扩展的线性模型组合来训练 **文档分类器**, 例如:

 * 使用稀疏特征对文本文档进行分类 :ref:`sphx_glr_auto_examples_text_document_classification_20newsgroups.py`

在 **无监督的设置** 中，可以通过应用诸如 :ref:`k_means` 的聚类算法来将相似文档分组在一起：

  * 使用k-means聚类文本文档 :ref:`sphx_glr_auto_examples_text_document_clustering.py`

最后，通过放松聚类的硬分配约束，可以通过使用非负矩阵分解（ :ref:`NMF` 或NNMF）来发现语料库的主要主题：

  * 主题提取与非负矩阵分解和潜在Dirichlet分配 :ref:`sphx_glr_auto_examples_applications_plot_topics_extraction_with_nmf_lda.py`

词语表示的限制
----------------------------------------------

一组单词（什么是单词）无法捕获短语和多字表达，有效地忽略任何单词顺序依赖。另外，这个单词模型不包含潜在的拼写错误或词汇导出。

N克抢救！而不是构建一个简单的unigrams集合 (n=1)，可能更喜欢一组二进制 (n=2)，其中计算连续字对。

还可以考虑一个字符 n-gram 的集合，这是一种对拼写错误和派生有弹性的表示。

例如，假设我们正在处理两个文档的语料库： ``['words', 'wprds']``. 第二个文件包含 'words' 一词的拼写错误。
一个简单的单词表示将把这两个视为非常不同的文档，两个可能的特征都是不同的。
然而，一个字符 2-gram 的表示可以找到匹配的文档中的8个特征中的4个，这可能有助于优选的分类器更好地决定::

  >>> ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 2))
  >>> counts = ngram_vectorizer.fit_transform(['words', 'wprds'])
  >>> ngram_vectorizer.get_feature_names() == (
  ...     [' w', 'ds', 'or', 'pr', 'rd', 's ', 'wo', 'wp'])
  True
  >>> counts.toarray().astype(int)
  array([[1, 1, 1, 0, 1, 1, 1, 0],
         [1, 1, 0, 1, 1, 1, 0, 1]])

在上面的例子中，使用 ``'char_wb`` 分析器'，它只能从字边界内的字符（每侧填充空格）创建 n-gram。 ``'char'`` 分析器可以创建跨越单词的 n-gram::

  >>> ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(5, 5))
  >>> ngram_vectorizer.fit_transform(['jumpy fox'])
  ...                                # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
  <1x4 sparse matrix of type '<... 'numpy.int64'>'
     with 4 stored elements in Compressed Sparse ... format>
  >>> ngram_vectorizer.get_feature_names() == (
  ...     [' fox ', ' jump', 'jumpy', 'umpy '])
  True

  >>> ngram_vectorizer = CountVectorizer(analyzer='char', ngram_range=(5, 5))
  >>> ngram_vectorizer.fit_transform(['jumpy fox'])
  ...                                # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
  <1x5 sparse matrix of type '<... 'numpy.int64'>'
      with 5 stored elements in Compressed Sparse ... format>
  >>> ngram_vectorizer.get_feature_names() == (
  ...     ['jumpy', 'mpy f', 'py fo', 'umpy ', 'y fox'])
  True

对于使用白色空格进行单词分离的语言，对于语言边界感知变体 ``char_wb`` 尤其有趣，因为在这种情况下，它会产生比原始 ``char`` 变体显着更少的噪音特征。 
对于这样的语言，它可以增加使用这些特征训练的分类器的预测精度和收敛速度，同时保持关于拼写错误和词导出的稳健性。

虽然可以通过提取 n-gram 而不是单独的单词来保存一些本地定位信息，但是包含 n-gram 的单词和袋子可以破坏文档的大部分内部结构，因此破坏了该内部结构的大部分含义。

为了处理自然语言理解的更广泛的任务，因此应考虑到句子和段落的地方结构。因此，许多这样的模型将被称为 "结构化输出" 问题，这些问题目前不在 scikit-learn 的范围之内。


.. _hashing_vectorizer:

用哈希技巧矢量化大文本语料库
------------------------------------------------------

上述向量化方案是简单的，但是它存在 **从字符串令牌到整数特征索引的内存映射** （ ``vocabulary_`` 属性），在处理 **大型数据集时会引起几个问题** :

- 语料库越大，词汇量越大，记忆体的使用也越大.

- 拟合需要分配大小与原始数据集成正比的中间数据结构.

- 构建单词映射需要完全传递数据集，因此不可能以严格的在线方式来适应文本分类器.

- 具有较大词汇量的 pickling和 un-pickling 矢量化器可能非常慢（通常比 pickling / un-pickling 平坦数据结构慢，比如相同尺寸的 NumPy 阵列）.

- 由于 ``vocabulary_`` 属性必须是具有细粒度同步屏障的共享状态，所以将向量化工作分解为并发子任务是不容易的：
  从令牌字符串到特征索引的映射取决于每个标记的第一次出现的顺序，因此必须被共享，从而潜在地损害并发工作者的性能，使得它们比顺序变体慢。

通过组合由 :class:`sklearn.feature_extraction.FeatureHasher` 类实现的 "散列技巧" (:ref:`Feature_hashing`) 和 :class:`CountVectorizer` 的文本预处理和标记化功能，可以克服这些限制。

这种组合是在 :class:`HashingVectorizer` 中实现的，该类是与 :class:`CountVectorizer` 大部分 API 兼容的变压器类。 :class:`HashingVectorizer` 是无状态的，这意味着您不需要 ``fit`` 它::

  >>> from sklearn.feature_extraction.text import HashingVectorizer
  >>> hv = HashingVectorizer(n_features=10)
  >>> hv.transform(corpus)
  ...                                # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
  <4x10 sparse matrix of type '<... 'numpy.float64'>'
      with 16 stored elements in Compressed Sparse ... format>

您可以看到在矢量输出中提取了16个非零特征令牌：这小于 :class:`CountVectorizer` 先前在同一玩具语料库上提取的19个非零值。
由于 ``n_features`` 参数的值较低，差异来自哈希函数冲突。

在现实世界的设置中, ``n_features`` 参数可以保留为默认值 ``2 ** 20``（大约一百万个可能的功能）。
如果内存或下游模型大小是选择较低值（例如 ``2 ** 18`` ）的问题，则可能有助于在典型文本分类任务上引入太多额外的冲突。

请注意，维度不影响在 CSR 矩阵 (``LinearSVC(dual=True)``, ``Perceptron``, ``SGDClassifier``, ``PassiveAggressive``) 上运行的算法的CPU训练时间，
但对于使用CSC矩阵的算法(``LinearSVC(dual=False)``, ``Lasso()``, 等).

让我们再次尝试使用默认设置::

  >>> hv = HashingVectorizer()
  >>> hv.transform(corpus)
  ...                               # doctest: +NORMALIZE_WHITESPACE  +ELLIPSIS
  <4x1048576 sparse matrix of type '<... 'numpy.float64'>'
      with 19 stored elements in Compressed Sparse ... format>

我们不再得到碰撞，但这是牺牲了更大尺寸的输出空间。当然，这里使用的其他术语可能仍然会相互冲突。

:class:`HashingVectorizer` 还具有以下限制：

- 由于执行映射的哈希函数的单向特性，不可能反转模型（无 ``逆转换`` 方法），也不能访问特征的原始字符串表示。

- 它不提供 IDF 权重，因为这将引入模型中的状态。 如果需要， :class:`TfidfTransformer` 可以在管道中附加到它。

使用 HashingVectorizer 执行外核缩放
------------------------------------------------------

使用 :class:`HashingVectorizer` 的一个有趣的开发是执行外核 `out-of-core`_ 缩放的能力。 这意味着我们可以从不符合计算机主内存的数据中学习。

.. _out-of-core: https://en.wikipedia.org/wiki/Out-of-core_algorithm

实现核心外扩展的策略是将数据以小批量流式传输到估计器。每个小批量使用 :class:`HashingVectorizer` 进行向量化，以保证估计器的输入空间总是具有相同的维度。 
因此，随时使用的内存量由小批量的大小限制。 虽然使用这种方法可以摄取的数据量没有限制，但从实际的角度来看，学习时间通常受到CPU所花费的时间的限制。

对于文本分类任务中的外核缩放的完整示例，请参阅文本文档的外核分类 :ref:`sphx_glr_auto_examples_applications_plot_out_of_core_classification.py`.

自定义矢量化器类
----------------------------------

通过将可调用传递给向量化程序构造函数可以定制行为::

  >>> def my_tokenizer(s):
  ...     return s.split()
  ...
  >>> vectorizer = CountVectorizer(tokenizer=my_tokenizer)
  >>> vectorizer.build_analyzer()(u"Some... punctuation!") == (
  ...     ['some...', 'punctuation!'])
  True

特别是我们命名：

  * ``预处理器``: 可以将整个文档作为输入（作为单个字符串）的可调用，并返回文档的可能转换的版本，仍然是整个字符串。这可以用于删除HTML标签，小写整个文档等。

  * ``tokenizer``: 一个可从预处理器接收输出并将其分成标记的可调用函数，然后返回这些列表。

  * ``分析器``: 一个可替代预处理程序和标记器的可调用程序。默认分析仪都会调用预处理器和刻录机，但是自定义分析仪将会跳过这个。 
    N-gram提取和停止字过滤在分析器级进行，因此定制分析器可能必须重现这些步骤。

（Lucene 用户可能会识别这些名称，但请注意，scikit-learn 概念可能无法一对一映射到 Lucene 概念上。）

为了使预处理器，标记器和分析器了解模型参数，可以从类派生并覆盖 ``build_preprocessor``, ``build_tokenizer``` 和 ``build_analyzer`` 工厂方法，而不是传递自定义函数。

一些提示和技巧:

  * 如果文档由外部包进行预先标记，则将它们存储在文件（或字符串）中，令牌由空格分隔，并通过 ``analyzer=str.split``

  * Fancy 令牌级分析，如词干，词法，复合分割，基于词性的过滤等不包括在 scikit-learn 代码库中，但可以通过定制分词器或分析器来添加。 
  这是一个 ``CountVectorizer``, 使用 `NLTK <http://www.nltk.org>`_ 的 tokenizer 和 lemmatizer::

        >>> from nltk import word_tokenize          # doctest: +SKIP
        >>> from nltk.stem import WordNetLemmatizer # doctest: +SKIP
        >>> class LemmaTokenizer(object):
        ...     def __init__(self):
        ...         self.wnl = WordNetLemmatizer()
        ...     def __call__(self, doc):
        ...         return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
        ...
        >>> vect = CountVectorizer(tokenizer=LemmaTokenizer())  # doctest: +SKIP

    （请注意，这不会过滤标点符号。）
    例如，以下例子将英国的一些拼写变成美国拼写::

        >>> import re
        >>> def to_british(tokens):
        ...     for t in tokens:
        ...         t = re.sub(r"(...)our$", r"\1or", t)
        ...         t = re.sub(r"([bt])re$", r"\1er", t)
        ...         t = re.sub(r"([iy])s(e$|ing|ation)", r"\1z\2", t)
        ...         t = re.sub(r"ogue$", "og", t)
        ...         yield t
        ...
        >>> class CustomVectorizer(CountVectorizer):
        ...     def build_tokenizer(self):
        ...         tokenize = super(CustomVectorizer, self).build_tokenizer()
        ...         return lambda doc: list(to_british(tokenize(doc)))
        ...
        >>> print(CustomVectorizer().build_analyzer()(u"color colour")) # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        [...'color', ...'color']

    用于其他样式的预处理; 例子包括 stemming, lemmatization, 或 normalizing numerical tokens, 后者说明如下:

     * :ref:`sphx_glr_auto_examples_bicluster_plot_bicluster_newsgroups.py`

在处理不使用显式字分隔符（例如空格）的亚洲语言时，自定义向量化器也是有用的。

.. _image_feature_extraction:

图像特征提取
========================

.. currentmodule:: sklearn.feature_extraction.image

补丁提取
----------------

:func:`extract_patches_2d` 函数从存储为二维数组的图像或沿着第三轴的颜色信息三维提取修补程序。
要从其所有补丁重建图像，请使用 :func:`reconstruct_from_patches_2d`. 例如让我们使用3个彩色通道（例如 RGB 格式）生成一个 4x4 像素的图像::

    >>> import numpy as np
    >>> from sklearn.feature_extraction import image

    >>> one_image = np.arange(4 * 4 * 3).reshape((4, 4, 3))
    >>> one_image[:, :, 0]  # R channel of a fake RGB picture
    array([[ 0,  3,  6,  9],
           [12, 15, 18, 21],
           [24, 27, 30, 33],
           [36, 39, 42, 45]])

    >>> patches = image.extract_patches_2d(one_image, (2, 2), max_patches=2,
    ...     random_state=0)
    >>> patches.shape
    (2, 2, 2, 3)
    >>> patches[:, :, :, 0]
    array([[[ 0,  3],
            [12, 15]],
    <BLANKLINE>
           [[15, 18],
            [27, 30]]])
    >>> patches = image.extract_patches_2d(one_image, (2, 2))
    >>> patches.shape
    (9, 2, 2, 3)
    >>> patches[4, :, :, 0]
    array([[15, 18],
           [27, 30]])

现在让我们尝试通过在重叠区域进行平均来从补丁重建原始图像::

    >>> reconstructed = image.reconstruct_from_patches_2d(patches, (4, 4, 3))
    >>> np.testing.assert_array_equal(one_image, reconstructed)

在 :class:`PatchExtractor` 以同样的方式类作品 :func:`extract_patches_2d`, 只是它支持多种图像作为输入。它被实现为一个估计器，因此它可以在管道中使用。看到::

    >>> five_images = np.arange(5 * 4 * 4 * 3).reshape(5, 4, 4, 3)
    >>> patches = image.PatchExtractor((2, 2)).transform(five_images)
    >>> patches.shape
    (45, 2, 2, 3)

图像的连接图
-------------------------------

scikit-learn 中的几个估计可以使用特征或样本之间的连接信息。
例如，Ward聚类（层次聚类 :ref:`hierarchical_clustering` ）可以聚集在一起，只有图像的相邻像素，从而形成连续的斑块:

.. figure:: ../auto_examples/cluster/images/sphx_glr_plot_face_ward_segmentation_001.png
   :target: ../auto_examples/cluster/plot_face_ward_segmentation.html
   :align: center
   :scale: 40

为此，估计器使用 '连接性' 矩阵，给出连接的样本。

该函数 :func:`img_to_graph` 从2D或3D图像返回这样一个矩阵。类似地，:func:`grid_to_graph` 为给定这些图像的形状的图像构建连接矩阵。

这些矩阵可用于在使用连接信息的估计器中强加连接，如 Ward 聚类（层次聚类 :ref:`hierarchical_clustering` ），而且还要构建预计算的内核或相似矩阵。

.. note:: **示例**

   * :ref:`sphx_glr_auto_examples_cluster_plot_face_ward_segmentation.py`

   * :ref:`sphx_glr_auto_examples_cluster_plot_segmentation_toy.py`

   * :ref:`sphx_glr_auto_examples_cluster_plot_feature_agglomeration_vs_univariate_selection.py`
