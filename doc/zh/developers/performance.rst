.. _performance-howto:

=========================
如何优化速度
=========================

这是一篇帮助你写出高效的 scikit-learn 项目代码的实用指南。

.. note::
  
  尽管分析代码进而 **检查代码的性能** 往往是有价值的，但在你投入代价高昂的对算法实现的优化工作之前，我们仍然强力推荐你先 **查阅相关文献**，以保证对于当前的任务来说，已经实现（并且你打算优化它的实现）的算法是当前为止最为先进的。
  
  有多少次，不论后来发现了怎样简单的 **算法技巧**，或者对于问题有怎样更合适的替代算法可以全部使用，无数工程人员还是前仆后继地将无数的努力投入在了复杂的算法实现细节的优化中。

  

   :ref:`热重启` 部分给出了一个这样的技巧的例子


Python, Cython 还是 C/C++?
==========================================


.. currentmodule:: sklearn

通常来说，scikit-learn 项目强调源码的 **可读性**，以保证项目的使用者们能轻松地深入源码，理解算法是如何在他们的数据上运作的，同时也让项目的（对于开发者的）可维护性更佳。

在用 Python 实现一个新算法时，我们建议：**尽量借助 Numpy 和 Scipy 实现它**，注意在代码中避免循环，而改用这些库向量化的方式去完成对应的任务。在实践中，这就意味着尽量 **把任何嵌套的 for 循环替换为等价的 Numpy 数组方法的调用**。我们的目标应该包括避免（或减少） Python 解释器中的 CPU 时间浪费，而不仅仅只是处理一堆数字，让他们拟合你的统计模型（译者注：不知道这样翻译对不对，因为我感觉 avoid CPU wasting time 和 crunch numbers to fit your statistical model 并不是对立的关系，就翻译成递进的关系）。 通常来说最好是参考一下 NumPy 和 SciPy 的性能技巧: http://scipy.github.io/old-wiki/pages/PerformanceTips

但有时，仅以简单的向量化 Numpy 代码的形式无法有效地表达一个算法。在这种情况下，我们推荐的方法是这样的：

  1. **分析** 算法的 Python 实现，找到影响性能的主要瓶颈，将它分离出来成为一个 **专用的模块级函数**。这个函数将会以一个编译过的扩展模块的形式被重新实现。

  2. 如果对于同样的算法，已经存在一个维护良好的 BSD 或 MIT 许可的 **C/C++** 实现，且规模不大，你可以为它编写一个 **Cython 包装器**， 并在scikit-learn 的源码树中包含一份这个库的源码的拷贝 —— :class:`svm.LinearSVC`, :class:`svm.SVC` 和 :class:`linear_model.LogisticRegression`类采用的就是这种方法（liblinear 和 libsvm 的包装器）。

  3. 你也可以选择直接用 **Cython** 编写一个你的 Python 函数的优化版本。一些类，像 :class:`linear_model.ElasticNet` 和
:class:`linear_model.SGDClassifier`，就是采用这种方法编写的。
     
  4. **将函数原始的 Python 版本用于测试** ，来检验编译扩展产生的结果是否与高度标准的，易调试的 Python 版本一致。
     
  5. 一旦确定代码是已经优化过的（通过代码分析无法找到一般的性能瓶颈），就应该确定能否用 ``joblib.Parallel`` 类实现具有 **粗粒度并行性** 的 **多进程处理**。

在使用 Cython 时，可以用以下指令之一

   $ python setup.py build_ext -i
   
   $ python setup.py install

来生成 C 文件。你需要自行在各个子模块的 ``setup.py`` 中添加 .c/.cpp 扩展名和构建参数。

在我们发布的稳定版本中（译者注：不确定这里是发布的版本还是分布式版本）都嵌入有 C/C++ 生成文件。这样做的目标是使 scikit-learn 的稳定版本在任何带有 Python, Numpy, Scipy 和 C/C++ 编译器的机器上都可以进行安装。

.. _profiling-python-code:

分析 Python 代码
==============================

为了分析 Python 代码，我们建议先写好一个加载和准备所需要数据的脚本，然后用 IPython 集成的分析器来交互式地对代码相关的部分进行探索。

假设我们想要分析 scikit-learn 中的 Non Negative Matrix Factorization (非负矩阵分解) 模块，那么现在我们先开启一个新的 IPython 会话，然后载入 scikit-learn 提供的数字数据集，就和 :ref:`sphx_glr_auto_examples_classification_plot_digits_classification.py` 中给出的例子中的一样，我们得到::

  In [1]: from sklearn.decomposition import NMF

  In [2]: from sklearn.datasets import load_digits

  In [3]: X = load_digits().data


在开始分析会话和投入实验性的优化迭代过程之前，我们先测量想要优化的函数不包含任何分析器开销的总运行时间，然后将其保存起来供以后参考，这一步很重要::

  In [4]: %timeit NMF(n_components=16, tol=1e-2).fit(X)
  1 loops, best of 3: 1.7 s per loop

用 ``%prun`` 魔术命令查看总体的性能分析::

  In [5]: %prun -l nmf.py NMF(n_components=16, tol=1e-2).fit(X)
           14496 function calls in 1.682 CPU seconds

     Ordered by: internal time
     List reduced from 90 to 9 due to restriction <'nmf.py'>

     ncalls  tottime  percall  cumtime  percall filename:lineno(function)
         36    0.609    0.017    1.499    0.042 nmf.py:151(_nls_subproblem)
       1263    0.157    0.000    0.157    0.000 nmf.py:18(_pos)
          1    0.053    0.053    1.681    1.681 nmf.py:352(fit_transform)
        673    0.008    0.000    0.057    0.000 nmf.py:28(norm)
          1    0.006    0.006    0.047    0.047 nmf.py:42(_initialize_nmf)
         36    0.001    0.000    0.010    0.000 nmf.py:36(_sparseness)
         30    0.001    0.000    0.001    0.000 nmf.py:23(_neg)
          1    0.000    0.000    0.000    0.000 nmf.py:337(__init__)
          1    0.000    0.000    1.681    1.681 nmf.py:461(fit)

``tottime`` 这一列最为有趣：它给出的是一个函数忽略它的子函数执行后的总执行时间。真实的总时间（局部代码+子函数调用）是在 ``cumtime`` 列中给出的。

注意这里我们用 ``-l nmf.py`` 将输出的函数限制在包含 "nmf.py" 这个字符串的行的范围。这对于快速查看 nmf Python 模块自身，而忽略掉其余部分来说非常有用。

同样的命令，去掉 ``-l nmf.py`` 的筛选之后，输出的开头部分是这样的::

  In [5] %prun NMF(n_components=16, tol=1e-2).fit(X)
           16159 function calls in 1.840 CPU seconds

     Ordered by: internal time

     ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       2833    0.653    0.000    0.653    0.000 {numpy.core._dotblas.dot}
         46    0.651    0.014    1.636    0.036 nmf.py:151(_nls_subproblem)
       1397    0.171    0.000    0.171    0.000 nmf.py:18(_pos)
       2780    0.167    0.000    0.167    0.000 {method 'sum' of 'numpy.ndarray' objects}
          1    0.064    0.064    1.840    1.840 nmf.py:352(fit_transform)
       1542    0.043    0.000    0.043    0.000 {method 'flatten' of 'numpy.ndarray' objects}
        337    0.019    0.000    0.019    0.000 {method 'all' of 'numpy.ndarray' objects}
       2734    0.011    0.000    0.181    0.000 fromnumeric.py:1185(sum)
          2    0.010    0.005    0.010    0.005 {numpy.linalg.lapack_lite.dgesdd}
        748    0.009    0.000    0.065    0.000 nmf.py:28(norm)
  ...

上面的结果告诉我们，代码的执行时间很大程度上来自点乘运算（代由 blas 执行）。因此我们不能指望通过以 Cython 或 C/C++ 重写代码来显著地提高效率：在这个例子中，1.7s (译者注：命令行输出中 fit_transform 的 cumtime 是 1.840s，没搞懂是怎么回事) 的总执行时间中，接近 0.7s 是来自这段我们可以认为已经达到最优的编译过的代码。如果重写 Python 代码中的剩余部分，即使假设我们能在这部分达到 1000% 的效率提升（由于实际代码中的 Python 循环层次较浅，这样的提升不太现实），最终得到的全局速度提升也不会超过 2.4 倍。

因此在这个例子中，显著的性能提升只能通过 **算法上的改进** 实现（例如，尝试寻找开销高而无用的运算，然后避免它们，而非尝试对这个既定算法做实现上的优化）。

尽管如此，查看 ``_nls_subproblem`` 函数的内部究竟发生了些什么仍然是一件有趣的事；如果只考虑 Python 代码，这个函数是程序的热区：它的累计（真实）运行时间几乎占据了整个模块的 100%。为了更好的理解这个具体的函数的运行信息，让我们安装 ``line_profiler`` 并且将其连接到 IPython::

  $ pip install line_profiler

- **在 IPython 0.13+ 下**, 先创建一个配置参数文件::

    $ ipython profile create

  然后在 ``~/.ipython/profile_default/ipython_config.py`` 中注册 line_profiler 扩展::

    c.TerminalIPythonApp.extensions.append('line_profiler')
    c.InteractiveShellApp.extensions.append('line_profiler')

  这将在 IPython 终端应用和其他诸如 qtconsole 和 notebook 的前端应用中注册 ``%lprun`` 魔术命令。

现在让我们重启 IPython ，然后玩玩这个新玩具::

  In [1]: from sklearn.datasets import load_digits

  In [2]: from sklearn.decomposition.nmf import _nls_subproblem, NMF

  In [3]: X = load_digits().data

  In [4]: %lprun -f _nls_subproblem NMF(n_components=16, tol=1e-2).fit(X)
  Timer unit: 1e-06 s

  File: sklearn/decomposition/nmf.py
  Function: _nls_subproblem at line 137
  Total time: 1.73153 s

  Line #      Hits         Time  Per Hit   % Time  Line Contents
  ==============================================================
     137                                           def _nls_subproblem(V, W, H_init, tol, max_iter):
     138                                               """Non-negative least square solver
     ...
     170                                               """
     171        48         5863    122.1      0.3      if (H_init < 0).any():
     172                                                   raise ValueError("Negative values in H_init passed to NLS solver.")
     173
     174        48          139      2.9      0.0      H = H_init
     175        48       112141   2336.3      5.8      WtV = np.dot(W.T, V)
     176        48        16144    336.3      0.8      WtW = np.dot(W.T, W)
     177
     178                                               # values justified in the paper
     179        48          144      3.0      0.0      alpha = 1
     180        48          113      2.4      0.0      beta = 0.1
     181       638         1880      2.9      0.1      for n_iter in xrange(1, max_iter + 1):
     182       638       195133    305.9     10.2          grad = np.dot(WtW, H) - WtV
     183       638       495761    777.1     25.9          proj_gradient = norm(grad[np.logical_or(grad < 0, H > 0)])
     184       638         2449      3.8      0.1          if proj_gradient < tol:
     185        48          130      2.7      0.0              break
     186
     187      1474         4474      3.0      0.2          for inner_iter in xrange(1, 20):
     188      1474        83833     56.9      4.4              Hn = H - alpha * grad
     189                                                       # Hn = np.where(Hn > 0, Hn, 0)
     190      1474       194239    131.8     10.1              Hn = _pos(Hn)
     191      1474        48858     33.1      2.5              d = Hn - H
     192      1474       150407    102.0      7.8              gradd = np.sum(grad * d)
     193      1474       515390    349.7     26.9              dQd = np.sum(np.dot(WtW, d) * d)
     ...

通过查看 ``% Time`` 一列中最高的值，我们很容易就能准确地找到那些值得多加注意的代价最高昂的语句。


内存使用情况分析
======================

在 `memory_profiler <https://pypi.python.org/pypi/memory_profiler>`_ 的帮助下你可以详尽地分析任何 Python 代码的内存使用状况。首先，安装这个工具的最新版本::

    $ pip install -U memory_profiler

然后，用与 ``line_profiler`` 类似的方式配置好魔术命令.

- **在 IPython 0.11+ 下**, 先创建一个配置参数文件::

    $ ipython profile create

  再在 ``~/.ipython/profile_default/ipython_config.py`` 中紧随 line profiler 之后注册这个扩展::

    c.TerminalIPythonApp.extensions.append('memory_profiler')
    c.InteractiveShellApp.extensions.append('memory_profiler')

  这将在 IPython 终端应用和其他诸如 qtconsole 和 notebook 的前端应用中注册 ``%memit`` 和 ``%mprun`` 魔术命令。

``%mprun`` 对于逐行检查程序中关键函数的内存使用情况十分有用。 它与我们在上一部分中讨论过的 ``%lprun`` 很相似。 我们来看一个来自 ``memory_profiler`` ``examples`` 目录中的例子::

    In [1] from example import my_func

    In [2] %mprun -f my_func my_func()
    Filename: example.py

    Line #    Mem usage  Increment   Line Contents
    ==============================================
         3                           @profile
         4      5.97 MB    0.00 MB   def my_func():
         5     13.61 MB    7.64 MB       a = [1] * (10 ** 6)
         6    166.20 MB  152.59 MB       b = [2] * (2 * 10 ** 7)
         7     13.61 MB -152.59 MB       del b
         8     13.61 MB    0.00 MB       return a

另外一个 ``memory_profiler`` 定义的实用的魔术命令就是 ``%memit``，它可以看作是 ``%timeit`` 的类比。你可以这样使用它::

    In [1]: import numpy as np

    In [2]: %memit np.zeros(1e7)
    maximum of 3: 76.402344 MB per loop

想知道更多细节，你可以用 ``%memit?`` 和 ``%mprun?`` 指令查看这些魔术命令的文档字符串。


给 Cython 开发者的性能建议
=========================================

如果 Python 代码分析显示， Python 解释器的开销比实际数值计算的开销要高上一个或更多数量级（例如，向量组件上的 ``for`` 循环，条件表达式的嵌套求值，标量运算...），那么提取代码的热点部分到一个 ``.pyx`` 文件中作为独立的函数，增加静态类型声明，然后用 Cython 生成适合编译成 Python 扩展模块的 C 程序可能是不错的做法。

在 http://docs.cython.org/ 上可以找到的官方文档中有一份教程和一份开发这样一个模块的参考指南。下面我们只重点展示在 scikit-learn 项目现存的 cython 代码库上我们发现在实践中比较重要的几个技巧。

TODO: 网页报表（译者注：好像混入了什么奇怪的东西），类型声明，边界检验，处理零检查，内存对齐，blas 的直接调用...

- https://www.youtube.com/watch?v=gMvkiQ-gOW8
- http://conference.scipy.org/proceedings/SciPy2009/paper_1/
- http://conference.scipy.org/proceedings/SciPy2009/paper_2/


.. _profiling-compiled-extension:

分析编译过的扩展
=============================

面对编译过的扩展（包装器包装下的 C/C++ 程序，或者直接由 Cython 编写的扩展），默认的 Python 分析器是无用武之地的：我们需要一个专用的用于深入检视编译过的扩展内部所进行的活动的工具。

使用 yep 和 google-perftools
--------------------------------

用 yep 对不带特殊编译选项的编译扩展进行简单分析（译者注：并不确定 without special compilation 是指 extension 还是指 yep）：

- https://pypi.python.org/pypi/yep
- http://fa.bianp.net/blog/2011/a-profiler-for-python-extensions

.. note::
  
  google-perftools 提供了一个很好的 “逐行” 报告模式，可以用 ``--lines`` 触发。但是其运作似乎在写入的时候有些问题。这个问题在 `project issue tracker <https://github.com/gperftools/gperftools>`_ 上可以追踪。



使用 gprof
-------------

在用 ``gcc -pg`` 和在 debian / ubuntu 上通过 ``python-dbg`` 生成的解释器变种重新编译了项目后，你可以用 ``gporf`` 来分析编译过的 Python 扩展：但是这种方法还要求有用 ``-pg`` 重新编译过的 ``numpy`` 和 ``scipy``，而这一步从某种程度上来说是很复杂的。

幸运的是，现在已经有两款其他分析器可供替代，无需重新编译一切就能进行代码分析。


使用 valgrind / callgrind / kcachegrind
----------------------------------------

TODO


用 ``joblib.Parallel`` 进行多核并行
================================================

TODO: 在这举一个简单有意思的例子。

查阅 joblib 的官方文档：

- https://pythonhosted.org/joblib


.. _warm-restarts:

一个算法技巧的例子：交叉验证的热重启
==============================================================

TODO: 展示坐标下降线性回归的交叉验证的热重启技巧
