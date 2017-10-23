.. _developers-tips:

=====================================
开发者建议和技巧
=====================================

生产力和完整性保存提示
=======================================

在本节中,我们收集一些有用的建议和工具,可能会提升您的在 review PR 和运行单元测试时的方便性.
其中一些技巧由需要浏览器扩展的用户脚本组成,例如 `TamperMonkey`_ 或 `GreaseMonkey`_ ; 
设置用户脚本你必须安装,启用和运行这些扩展之一.我们提供用户手册
作为 GitHub 要点; 要安装它们,请点击 Gist 页面上的 "Raw" 按钮

.. _TamperMonkey: https://tampermonkey.net
.. _GreaseMonkey: http://www.greasespot.net

查看 PR 的 HTML 文档
----------------------------------------------------------

我们使用 CircleCI 为每个 PR 构建 HTML 文档.为了访问该文档,我们提供了一个重定向功能参考如下 
:ref:`documentation section of the contributor guide <contribute_documentation>` . 我们不用手动输入地址,而是提供一个 `userscript <https://gist.github.com/lesteve/470170f288884ec052bcf4bc4ffe958a>`_ 
它为每个 PR 增加了一个按钮. 安装用户脚本后,导航到任何一个 GitHub PR 应该会在右上角区域出现一个标签为 "查看此公关的 CircleCI 文档" 的新按钮.
在 PR 上折叠和展开以前的异同对比
-----------------------------------------------------

GitHub 会同时隐藏关于 PR 的讨论和相应的代码行. 这个 `userscript <https://gist.github.com/lesteve/b4ef29bccd42b354a834>`_ 提供一个按钮一次展开所有这些隐藏的讨.

检查 remote-tracking 分支的 PR
------------------------------------------------------

在你的本地分支中, 修改 ``.git/config`` , 在 ``[remote "upstream"]`` 下一行的头部添加::

  fetch = +refs/pull/*/head:refs/remotes/upstream/pr/*

然后你可以使用 ``git checkout pr/PR_NUMBER`` 以给定的数字导航到 PR 的代码. ( `Read more in this gist. <https://gist.github.com/piscisaureus/3342247>`_ )

显示 PR 的代码覆盖
--------------------------------------

覆盖由 CodeCov 连续生成的代码覆盖率报告, 参考 `this browser extension <https://github.com/codecov/browser-extension>`_ . 每行的覆盖面将显示为行号后面的颜色背景.

有用的 pytest 别名和标志
---------------------------------------

我们建议使用 pytest 来运行单元测试. 当单元测试失败时,以下技巧可以使调试更容易:

  1. 命令行参数 ``pytest -l`` 构造 pytest 打印本地发生故障时的变量.

  2. 命令 ``pytest --pdb`` 失败时捕捉到 Python 调试器. 为了避免被 IPython debugger ``ipdb`` 捕捉, 你可以设置 shell 别名::

         pytest --pdbcls=IPython.terminal.debugger:TerminalPdb --capture no

使用 valgrind 调试 Cython 中的内存错误
========================================================

虽然 python/numpy 的内置内存管理相对强大, 但仍会导致一些程序的性能下降. 因为这个原因,多数的高性能 scikit-learn 代码使用 cython.  这个性能增益来自于权衡,因为 cython 非常容易出现内存 bugs,特别是在代码很大程度上依赖于指针算术的情况下.

内存错误可以通过多种方式表现出来. 最简单的调试通常是分段错误和相关的 glibc 错误.未初始化变量可能导致难以追踪的意外行为.调试这些错误时非常有用的工具是 valgrind_ .


Valgrind 是一个可以追踪各种内存错误的命令行工具
码. 按着这些步骤:

  1. 在你的系统安装 `valgrind`_ .

  2. 下载 python valgrind suppression file: `valgrind-python.supp`_ .

  3. 按照 `README.valgrind`_ 文件中的说明进行自定义
      python suppression. 如果不这样做,你会有与 python 解释器相关虚假的输出,而不是你自己的代码.

  4. 运行如下::

       $> valgrind -v --suppressions=valgrind-python.supp python my_test_script.py

.. _valgrind: http://valgrind.org
.. _`README.valgrind`: http://svn.python.org/projects/python/trunk/Misc/README.valgrind
.. _`valgrind-python.supp`: http://svn.python.org/projects/python/trunk/Misc/valgrind-python.supp

输出结果将是列出所有与内存相关的错误,这是引用的由 .pyx 文件中的 cython 生成的 C 代码中的行. 如果你检查在 .c 文件中引用的行,您将看到指示的注释 .pyx 源文件中的相应位置. 希望输出会给你关于你的内存错误来源的线索.

有关 valgrind 的更多信息及其选项阵列,请参见
教程和文档 `valgrind web site <http://valgrind.org>`_ .
