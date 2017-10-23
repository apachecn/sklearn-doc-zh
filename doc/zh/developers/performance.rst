.. _performance-howto:

=========================
如何优化速度
=========================

以下给出了一些实用的指导,以帮助您编写高效scikit-learn项目的代码.

.. note::

  配置您的代码 **check
  performance assumptions**通常是有帮助的, 在实施优化之前.同样强烈建议 **查看文献**以确保任务使用实现的算法是最先进的.
  投入时间努力优化被认为是无关的复杂实现细节,发现更简单的 **算法技巧**, 或者通过使用更适合这类问题的其他算法.

  The section :ref:`warm-restarts` 提供了一个这样的技巧的例子.


Python, Cython or C/C++?
========================

.. currentmodule:: sklearn

一般来说,scikit-learn 项目强调源码的 **可读性** 使项目用户轻松使用源代码,以便了解算法是如何操作数据的而且易于维护.

推荐实现新算法时**通过使用Numpy和Scipy**在Python中实现它,通过使用这些库的向量化模块来避免循环代码.在实践中这意味着尝试通过调用**Numpy数组方法等效替换任何形式的嵌套for循环**
 The goal is to avoid the CPU wasting time in the
目的是避免CPU浪费时间在Python解释器上,是不断的计算数字以适应您的统计
模型. 考虑NumPy和SciPy的性能提示通常是个好主意:
http://scipy.github.io/old-wiki/pages/PerformanceTips

然而,有时候,算法不能简单地表达得很有效矢量化的Numpy代码. 在这种情况下,推荐的策略是
以下:

  1. **Profile** 纯Python实现的性能分析模块,将其隔离在**专用模块级功能**中.该函数将作
  	  为编译扩展模块实现.

  2. 如果存在维护良好的BSD或MIT **C/C++**实现同样的算法不算太大,可以写一个
     **Cython wrapper** 并且 包含library源代码的副本在scikit-learn的source tree:
     这个策略应用在 classes :class:`svm.LinearSVC`, :class:`svm.SVC` and
     :class:`linear_model.LogisticRegression` (wrappers for liblinear
     and libsvm).

  3. 否则,直接使用**Cython**编写一个优化版本的Python函数.使用这个策略的
     class:`linear_model.ElasticNet`和
     :class:`linear_model.SGDClassifier`

  4. **将Python版本的函数移动到tests模块中** 使用
      它检查编译扩展的结果是否一致
      具有黄金标准,且易于调试.

  5. 一旦代码优化完成 (不是简单的性能瓶颈分析), 检查是适合于 **多处理**的**粗粒度并行**
     通过使用``joblib.Parallel`` class.

使用Cython时,请使用

   $ python setup.py build_ext -i
   $ python setup.py install

生成C文件.您负责在每个子模块“setup.py”中使用构建参数添加.c/.cpp扩展名.

C/C++生成的文件嵌入到分布式的稳定包中. 目标是
使其可以在任何带有Python,Numpy,Scipy和C/C ++编译器的机器上安装scikit-learn稳定版本.

.. _profiling-python-code:

分析Python代码
=====================

为了分析Python代码,我们建议您编写一个脚本
加载并准备好数据,然后使用IPython集成分析器
用于交互式探索代码的相关部分.

假设我们要分析scikit的Non Negative Matrix Factorization 模块. 让我们设置一个新的IPython会话并加载数字数据集和:ref:`sphx_glr_auto_examples_classification_plot_digits_classification.py` example::

  In [1]: from sklearn.decomposition import NMF

  In [2]: from sklearn.datasets import load_digits

  In [3]: X = load_digits().data

在开始分析会话进行
优化迭代之前,测量函数的总运行时间是非常重要的,我们需要没有任何类型的分析器优化
开销并将其保存在某个地方供以后参考::

  In [4]: %timeit NMF(n_components=16, tol=1e-2).fit(X)
  1 loops, best of 3: 1.7 s per loop

使用magic命令``％prun``来看整体性能配置文件::

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

``tottime`` 列是最有趣的: 它给出的时间是所给定函数不包括其子函数的花费时间. 真实的花费总时间是``cumtime`` 列给出的(local code + sub-function calls).

请注意使用“-l nmf.py”,将输出限制为行包含“nmf.py”字符串. 这有助于快速查看热点的nmf Python模块,它自己忽略其他任何东西.

以下是相同命令的输出的开头,没有``-l nmf.py``
filter::

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

上述结果表明,执行情况主要由dot操作主导(委托给blas)时,所以有可能在Cython或C/C++中重写此代码没有什么大的收获: 在这个案例中,在总执行时间为1.7s的时间里几乎花费0.7s在编译代码中,我们可以考虑优化. 通过重写其余的Python代码,假设我们可以在这个部分达到1000％的提升
（由于Python循环的浅层,这是非常不可能的),我们也不会获得全局超过2.4倍的速度优化.

因此,在这个具体例子中主要的改进只能通过 **算法
改进** (e.g. 试图找到操作这是昂贵的和无用的,以避免计算,而不是试图优化其实现).

然而,检查``_nls_subproblem``函数内部发生的情况仍然很有意思,如果我们只考虑Python代码,那这个就是hotspot:大约占模块累计时间的100％. 为了更好地了解分析这个具体函数的,让我们安装``line_profiler``并将其连接到IPython看看::

  $ pip install line_profiler

- **Under IPython 0.13+**, first create a configuration profile::

    $ ipython profile create

 然后在下面文件中注册 line_profiler 拓展
  ``~/.ipython/profile_default/ipython_config.py``::

    c.TerminalIPythonApp.extensions.append('line_profiler')
    c.InteractiveShellApp.extensions.append('line_profiler')

  在IPython terminal和其他应用程序如qtconsole和notebook中注册 ``%lprun`` magic命令.
  重新启动IPython,让我们使用这个新工具::

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

通过查看``% Time``列的顶部值是非常容易的指出最昂贵的表现,值得额外注意.


内存使用分析
======================

您可以在此帮助下详细分析任何Python代码的内存使用情况
`memory_profiler <https://pypi.python.org/pypi/memory_profiler>`_. 首先,
安装最新版本::

    $ pip install -U memory_profiler

然后, 在``line_profiler``以类似的方式设置magics.

- **在IPython 0.11+**, 首先创建配置文件::

    $ ipython profile create

  然后在注册拓展插件
  ``~/.ipython/profile_default/ipython_config.py``
  同样在line profiler中::

    c.TerminalIPythonApp.extensions.append('memory_profiler')
    c.InteractiveShellApp.extensions.append('memory_profiler')

  然后在IPython terminal 或其他应用工具例如 qtconsole和notebook注册 ``%memit`` 和 ``%mprun`` magic命令.

``%mprun``时很有用的测试,逐行检测你程序中关键函数的内存使用.与之前章节中``%lprun``非常相似. 例如 ``memory_profiler`` ``examples``
directory::

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

另一个有用的magic是 ``memory_profiler`` 定义``%memit``,这是
类似于``%timeit``. 可以如下使用::

    In [1]: import numpy as np

    In [2]: %memit np.zeros(1e7)
    maximum of 3: 76.402344 MB per loop

有关更多详细信息,请参阅magic文献, 使用 ``%memit?`` 和
``%mprun?``.


Cython开发人员的性能提示
=========================================

如果Python代码的分析显示了Python解释器开销大于实际数值计算一个数量级或更多的成本(e.g. ``for``循环遍历向量组件,嵌套条件表达式,标量算法...), 将代码的hotspot部分作为独立功能提取到``.pyx``文件中可能是明智的,添加静态类型声明和然后使用Cython生成一个c程序编译为
Python扩展模块.

http://docs.cython.org/ 上提供的官方文档包含开发此类模块的教程和参考指南.接下来我们会强调在 scikit-learn 项目中很重要的实践技巧.

TODO: html report, type declarations, bound checks, division by zero checks,
memory alignment, direct blas calls...

- https://www.youtube.com/watch?v=gMvkiQ-gOW8
- http://conference.scipy.org/proceedings/SciPy2009/paper_1/
- http://conference.scipy.org/proceedings/SciPy2009/paper_2/


.. _profiling-compiled-extension:

分析编译扩展
=============================

当使用编译扩展(用C/C++编写的包装器或直接使用Cython扩展),默认的Python分析器是无用的:我们需要一个专门的工具来预测内部发生的情况编译扩展它自己.

使用yep和google-perftools
--------------------------------

没有特殊编译选项的简单分析使用yep:

- https://pypi.python.org/pypi/yep
- http://fa.bianp.net/blog/2011/a-profiler-for-python-extensions

.. note::

  google-perftools提供了一个很好的“逐行”报告模式可以用``--lines``选项触发.不过这个在编写此文当时似乎不能正常工作.
  issue的追踪在这 `project issue tracker
  <https://github.com/gperftools/gperftools>`_.



使用 gprof
-------------

为了分析编译的Python扩展可以使用``gprof``在用``gcc -pg``重新编译项目并使用它.然后在debian / ubuntu上使用解释器的``python-dbg``变体
这种方法还需要使用``-pg``重新编译``numpy``和``scipy``这是相当复杂的工作.

幸运的是,存在两种不需要你的从头重新编译一切的性分析器.


Using valgrind / callgrind / kcachegrind
----------------------------------------

TODO


Multi-core parallelism using ``joblib.Parallel``
================================================

TODO: give a simple teaser example here.

Checkout the official joblib documentation:

- https://pythonhosted.org/joblib


.. _warm-restarts:

A sample algorithmic trick: warm restarts for cross validation
==============================================================

TODO: demonstrate the warm restart tricks for cross validation of linear
regression with Coordinate Descent.
