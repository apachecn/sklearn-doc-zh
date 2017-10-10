.. _faq:

===========================
常见问题
===========================

在这里，我们试着给出一些经常出现在邮件列表上的问题的答案。

项目名称是什么 (很多人弄错)?
--------------------------------------------------------
scikit-learn, 但既不是scikit或是SciKit也不是sci-kit learn。
也不是我们过去用过的scikits.learn和scikits-learn。

你怎么念这个项目的名称呢?
------------------------------------------
sy-kit learn。sci代表着科学!

为什么是scikit?
------------
拥有很多围绕Scipy构建的科学工具箱。
你可以在这里看见列表 `<https://scikits.appspot.com/scikits>`_.
除了scikit-learn,另一个受欢迎的是`scikit-image <http://scikit-image.org/>`_.

我怎么样才能为scikit-learn做出贡献?
-----------------------------------------
见:ref:`contributing`. 在添加一个通常是重要冗长的新算法前, 推荐你观看
:ref:`known issues <new_contributors>`。
关于scikit-learn贡献，请不要直接和scikit-learn的贡献者联系。

如何得到scikit-learn最好的用法帮助?
--------------------------------------------------------------
**对于一般的机器学习问题**, 请使用
`交叉验证 <http://stats.stackexchange.com>`_ 和 ``[machine-learning]`` 。

**对于scikit-learn使用的问题**, 请点击 `Stack Overflow <http://stackoverflow.com/questions/tagged/scikit-learn>`_
with the ``[scikit-learn]`` 和 ``[python]`` 。 你也可以使用`联系列表
<https://mail.python.org/mailman/listinfo/scikit-learn>`_.

请确保包含一个最小的复制代码片段(最好少于10行)，突出显示玩具数据集上的问题 (例如从
``sklearn.datasets``或者是用固定随机数种子利用``numpy.random``函数生成). 请删除任何不需要重现您的问题的代码行

该问题应该可以在安装了scikit-learn的python命令行中简单地复制粘贴您的代码重现. 并且不要忘了import语句.

编写能够重现的代码的更多指南可以在以下网址找到:

http://stackoverflow.com/help/mcve

即使在谷歌搜索之后,你的代码依旧引起了你不明白的异常，
请确保在运行复制脚本时包含完整的回溯。

有关错误报告或者功能请求，请使用
`issue tracker on Github <https://github.com/scikit-learn/scikit-learn/issues>`_.

你还可以在 `scikit-learn Gitter channel
<https://gitter.im/scikit-learn/scikit-learn>`_ 找到一些用户与开发人员。

**请不要直接给任何作者发邮件请求帮助，报告bug或其他与scikit-learn相关的问题。**

我怎么才可以创建一个bunch对象?
------------------------------------------------

不要创建bunch对象! 它们不是scikit-learn API的一部分. Bunch
对象只是打包一些numpy数组的一种方式. 作为一个scikit-learn用户你仅需要
numpy数组来给你的模型提供数据。

例如，训练一个分类器, 你需要的是用于输入变量的一个2D数组 ``X``  
和目标变量的1D数组``y``。``X``数组将特征作为列，样本保存为行。
``y``数组包含用于对每个样本的类成员资格编码的整型数值``X``.

如何将我自己的数据集加载到scikit-learn可用的格式?
--------------------------------------------------------------------

一般来说,scikit-learn可以在诸如numpy数组或者scipy稀疏矩阵这样的数字数据上运行。其他格式的如
pandas Dataframe的数组也是可以的。

有关将数据文件加载到可用数据结构中的更多信息，参阅:ref:`加载外部数据集<external_datasets>`.

新算法的纳入标准是什么 ?
----------------------------------------------------

我们仅考虑添加已经完善的算法。录入的一个标准是，自发布时间已过3年，被引用超过200次，广泛使用。
对广泛使用的方法提供了明确改进的技术（如增强型数据结构或更有效的近似技术）也将被考虑纳入。

在满足上述标准的算法或技术中,只有这些能够适合现在scikit-learn API的, 这是一个``适合``定义,
``预测/转换``接口通常具有一个numpy阵列或稀疏矩阵的输入/输出。

贡献者应该支持通过研究论文和/或其他类似软件包中的实现来提出增加的重要性，通过常见的用例/应用程序
证明其有用性，并通过基准和/或图证实性能改进（如果有的话）。预计所提出的算法应该在某些领域
优于已经在scikit-learn中的方法.

还要注意，您的实现不需要在scikit-learn中以和scikit-learn工具一起使用。您可以以scikit-learn兼容的
方式实现您最喜欢的算法，将其上传到github并让我们知道。我们将在:ref:`related_projects`列出。

.. _selectiveness:

为什么你对scikit-learn中的算法如此讲究?
------------------------------------------------------------------------
代码是维护成本, 我们需要平衡我们与团队规模的代码量
(再加上这个事实：复杂性与功能的数量成线性关系).
该软件包依赖于核心开发人员利用他们的空闲时间修复错误，维护代码和审查贡献。
添加的任何算法都需要开发人员的关注，此时原作者可能已经长久失去兴趣。
也可以在`邮件列表上的这个线程
<https://sourceforge.net/p/scikit-learn/mailman/scikit-learn-general/thread/CAAkaFLWcBG+gtsFQzpTLfZoCsHMDv9UG5WaqT0LwUApte0TVzg@mail.gmail.com/#msg33104380>`_.

为什么从scikit-learn中删除HMMS?
--------------------------------------------
见:ref:`adding_graphical_models`.

.. _adding_graphical_models:

你会在scikit-learn中添加图形模型或序列预测吗?
---------------------------------------------------------------------

不可预见的未来。
scikit-learn尝试为机器学习中的基本任务提供统一的API，
使用管道和元算法（如网格搜索）将所有内容都集中在一起。
结构化学习所需的概念，API，算法和专业知识与scikit学习所提供的不同。
果我们开始进行任意的结构化学习，那么我们需要重新设计整个软件包，
这个项目可能在自身的负担下崩溃。

这里有两个类似于scikit-learn的做结构化预测的API:

* `pystruct <http://pystruct.github.io/>`_ 处理一般结构化学习
(关注具有近似推理的任意图形结构上的SSVMs;将样本的概念定义为图形结构的一个实例)

* `seqlearn <http://larsmans.github.io/seqlearn/>`_ 仅处理序列（专注于精确推断;
主要是为了完整性附带了HMMs;将特征向量作为样本，并对特征向量之间的依赖使用偏移编码）

你会添加GPU支持吗?
-------------------------

不，或者至少在最近不会。
主要原因是GPU支持将引入许多软件依赖关系并引入平台特定的问题。
scikit-learn旨在轻松安装在各种平台上。
除了神经网络，GPU在当今的机器学习中不起重要作用，
通常我们可以通过仔细选择算法来获得更大的速度增益。

你支持PyPy吗?
--------------------

防止您不知道`PyPy <http://pypy.org/>`_ 是新的，快速，及时的编译Python实现。
我们不支持它。
当PyPy中的`NumPy support <http://buildbot.pypy.org/numpy-status/latest.html>`_
完成或接近完成，并且SciPy也被移植时，我们可以开始考虑移植。
我们使用了太多的NumPy而不能完成部分实现。

如何处理字符串数据（或树，图...）？
-----------------------------------------------------

scikit-learn估计器假设您将为他们提供实值特征向量。
这个假设在几乎所有的库都是硬编码的。
但是，您可以通过多种方式将非数字输入馈送到估计器。

如果您有文本文档，可以使用术语频率特征; 参阅内置*文本向量化器*的
:ref:`text_feature_extraction`。
对于从任何类型的数据更一般的特征提取，见
:ref:`dict_feature_extraction` 和 :ref:`feature_hashing`。

另一个常见的情况是当您对这些数据有非数字数据和自定义距离（或相似度）指标时。
示例包括具有编辑距离的字符串（也称为Levenshtein距离;例如DNA或RNA序列）。
这些可以编码为数字，但这样做是令人不快和容易出错的。
使用任意数据的距离度量可以通过以下两种方式完成。

首先，许多估计器采用预计算的距离/相似矩阵，
因此如果数据集不太大，可以计算所有输入对的距离。
如果数据集很大，您可以使用仅具有一个“特征”的特征向量，
该特征是单独数据结构的索引，
并提供在该数据结构中查找实际数据的自定义度量函数。
例如，使用DBSCAN与Levenshtein距离::

    >>> from leven import levenshtein       # doctest: +SKIP
    >>> import numpy as np
    >>> from sklearn.cluster import dbscan
    >>> data = ["ACCTCCTAGAAG", "ACCTACTAGAAGTT", "GAATATTAGGCCGA"]
    >>> def lev_metric(x, y):
    ...     i, j = int(x[0]), int(y[0])     # extract indices
    ...     return levenshtein(data[i], data[j])
    ...
    >>> X = np.arange(len(data)).reshape(-1, 1)
    >>> X
    array([[0],
           [1],
           [2]])
    >>> dbscan(X, metric=lev_metric, eps=5, min_samples=2)  # doctest: +SKIP
    ([0, 1], array([ 0,  0, -1]))

(这里使用了第三方编辑距离包 ``leven``)

类似的技巧也可以用在树形内核、图形内核等上

为什么我有时会在OSX或Linux下遇到n_jobs> 1崩溃/冻结?
------------------------------------------------------------------------

一些例如``GridSearchCV``和``cross_val_score``的scikit-learn工具，
它们可以依靠Python的内置`多重处理`模块，通过“n_jobs > 1”作为参数，将执行并行化到多个Python进程。

问题是Python由于性能原因``多重处理``会执行``fork``系统调用
而不是``exec``系统调用。
许多库如OSX下的（某些版本的）Accelerate / vecLib, (默写版本的)MKL,GCC的OpenMP运行时,
nvidia的Cuda(可能还有很多),
都是自行管理自己的内部线程池。在调用`fork`时，子进程中的线程池状态已损坏：
线程池认为它有许多线程，而只有主线程状态已被fork。
有可能更改库，使它们在发生fork时检测，并在该情况下重新初始化线程池：
我们对OpenBLAS执行了此操作（从0.2.10开始在master中合并），
并且我们向GCC的OpenMP运行时提供了一个`补丁
<https://gcc.gnu.org/bugzilla/show_bug.cgi?id=60035>`_ 
(尚未审查)。

但最终，真正的罪魁祸首是Python的``多重处理``，执行
``fork``而不``exec``来减少开始和使用新的并行计算的Python进程的开销。
不幸的是，这违反了POSIX标准。
因此一些如苹果的软件编辑器拒绝认为在Accelerate / vecLib中缺乏fork安全是一个bug。

在Python 3.4或以上版本中，现在可以配置``多重处理``决定使用
'forkserver'或者'spawn'启动方法(而不是默认的
'fork')来管理进程池。要使用scikit-learn来解决此问题，
你可以将JOBLIB_START_METHOD的环境变量设为'forkserver'。
但是用户应该意识到使用'forkserver'方法会阻止joblib.Parallel调用在shell会话中交互定义的函数。

如果你有直接使用``多重处理``的自定义代码而非通过joblib使用，你可以为你的程序全局启用'forkserver'模式：
在主脚本中插入以下说明::

    import multiprocessing

    # other imports, custom code, load data, define model...

    if __name__ == '__main__':
        multiprocessing.set_start_method('forkserver')

        # call scikit-learn utils with n_jobs > 1 here

你可以在`多重处理文档 <https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods>`_上
找到更多新启动方法的默认值。

为什么不支持深度学习或强化学习/scikit-learn中将会支持深度学习或强化学习吗?
--------------------------------------------------------------------------------------------------------------------------------------

深度学习和强化学习需要丰富的词汇来定义一个架构，
深度学习还需要GPU来进行有效的计算。
然而，这些都不符合scikit-learn的设计限制。
因此，深度学习和强化学习目前已经超出了scikit-learn寻求实现的范围。

你可以找到更多关于gpu支持的信息
`Will you add GPU support?`_.

为什么我的拉请求没有得到注意?
-------------------------------------------------

scikit-learn审查过程需要大量的时间，因此
贡献者不应该因为拉请求缺乏活动或没有被审查而沮丧。
我们非常关心第一次正确的使用，因为维护和以后的更改带来了高成本。
我们不会发布"实验性"代码, 
所以我们所有的贡献将会立即得到大量使用，并且在最初的时候就应该是最高的质量。

除此之外，scikit-learn在审查带宽方面是有限的; 
许多审稿人和核心开发人员都是利用自己的时间在scikit-learn工作。
如果您的拉动请求的检查缓慢，可能是因为审阅者很忙。
我们要求您的理解，并要求您不要因为这个原因而关闭您的拉取请求或停止您的工作。

如何为整个执行设置一个统一的``random_state`` ?
---------------------------------------------------------

对于测试和复制，通常重要的是让整个执行由具有随机组件
的算法中使用的伪随机数生成器的单个种子进行控制。
Scikit-learn不使用自己的全局随机状态;
每当RandomState实例或整数随机种子不作为参数提供时，
它依赖于可以这样使用的:func:`numpy.random.seed`numpy全局随机数种子。
例如，要将执行的numpy全局随机状态设置为42，可以在他或她的脚本中执行以下操作::

    import numpy as np
    np.random.seed(42)

然而，全局随机状态在执行期间容易被其他代码修改。
因此，确保可复制性的唯一方法是在每个地方传递``RandomState``实例，
并确保估算器和交叉验证分隔符都具有其``random_state``参数集。
