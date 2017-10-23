.. _developers-tips:

===========================
开发者建议和技巧
===========================

生产力和完整性保存提示
=======================================

在本节中,我们收集一些有用的建议和工具,可能会提升您的在review PR和运行单元测试时的方便性.
其中一些技巧由需要浏览器扩展的用户脚本组成,例如`TamperMonkey`_或`GreaseMonkey`_; 
设置用户脚本你必须安装,启用和运行这些扩展之一.我们提供用户手册
作为GitHub要点; 要安装它们,请点击Gist页面上的“Raw”按钮

.. _TamperMonkey: https://tampermonkey.net
.. _GreaseMonkey: http://www.greasespot.net

查看 PR 的HTML文档
----------------------------------------------------------

我们使用CircleCI为每个PR构建HTML文档.为了访问该文档,我们提供了一个重定向功能参考如下
:ref:`documentation section of the contributor guide
<contribute_documentation>`. 我们不用手动输入地址,而是提供一个`userscript <https://gist.github.com/lesteve/470170f288884ec052bcf4bc4ffe958a>`_
它为每个PR增加了一个按钮. 安装用户脚本后,导航到任何一个
GitHub PR 应该会在右上角区域出现一个标签为“查看此公关的CircleCI文档”的新按钮.
在PR上折叠和展开以前的异同对比
-----------------------------------------------------

GitHub会同时隐藏关于PR的讨论和相应的代码行. This `userscript
<https://gist.github.com/lesteve/b4ef29bccd42b354a834>`_提供一个按钮
一次展开所有这些隐藏的讨.

检查remote-tracking 分支的PR
------------------------------------------------------

在你的本地分支中, 修改 ``.git/config``, 在 ``[remote
"upstream"]`` 下一行的头部添加::

  fetch = +refs/pull/*/head:refs/remotes/upstream/pr/*

然后你可以使用 ``git checkout pr/PR_NUMBER``以给定的数字导航到PR的代码. (`Read more in this gist.
<https://gist.github.com/piscisaureus/3342247>`_)

显示PR的代码覆盖
--------------------------------------

覆盖由CodeCov连续生成的代码覆盖率报告, 参考 `this browser extension
<https://github.com/codecov/browser-extension>`_. 每行的覆盖面将显示为行号后面的颜色背景.

有用的pytest别名和标志
-------------------------------

我们建议使用pytest来运行单元测试. 当单元测试失败时,以下技巧可以使调试更容易:

  1. 命令行参数 ``pytest -l`` 构造pytest打印本地发生故障时的变量.

  2. 命令 ``pytest --pdb`` 失败时捕捉到Python调试器. 为了避免被IPython debugger ``ipdb``捕捉, 你可以设置shell别名::

         pytest --pdbcls=IPython.terminal.debugger:TerminalPdb --capture no

使用valgrind调试Cython中的内存错误
===============================================

虽然python/numpy的内置内存管理相对强大, 但仍会导致一些程序的性能下降. 因为这个原因,多数的高性能 scikit-learn 代码使用 cython.  这个性能增益来自于权衡,因为cython非常容易出现内存bugs,特别是在代码很大程度上依赖于指针算术的情况下.

内存错误可以通过多种方式表现出来. 最简单的调试通常是分段错误和相关的glibc错误.未初始化变量可能导致难以追踪的意外行为.调试这些错误时非常有用的工具是valgrind_.


Valgrind是一个可以追踪各种内存错误的命令行工具
码. 按着这些步骤:

  1. 在你的系统安装 `valgrind`_ .

  2. 下载python valgrind suppression file: `valgrind-python.supp`_.

  3. 按照“README.valgrind”文件中的说明进行自定义
      python suppression. 如果不这样做,你会有与python解释器相关虚假的输出,而不是你自己的代码.

  4. 运行::

       $> valgrind -v --suppressions=valgrind-python.supp python my_test_script.py

.. _valgrind: http://valgrind.org
.. _`README.valgrind`: http://svn.python.org/projects/python/trunk/Misc/README.valgrind
.. _`valgrind-python.supp`: http://svn.python.org/projects/python/trunk/Misc/valgrind-python.supp

输出结果将是列出所有与内存相关的错误,这是引用的由.pyx文件中的cython生成的C代码中的行. 如果你检查在.c文件中引用的行,您将看到指示的注释.pyx源文件中的相应位置. 希望输出会给你关于你的内存错误来源的线索.

有关valgrind的更多信息及其选项阵列,请参见
教程和文档 `valgrind web site <http://valgrind.org>`_.
