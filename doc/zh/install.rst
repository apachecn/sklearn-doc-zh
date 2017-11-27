.. _installation-instructions:

=======================
安装 scikit-learn
=======================

.. note::

   如果你想为这个项目做出贡献，建议你 :ref:`安装最新的开发版本 <install_bleeding_edge>` .


安装最新版本
=============================

Scikit-learn 要求:

- Python (>= 2.7 or >= 3.3),
- NumPy (>= 1.8.2),
- SciPy (>= 0.13.3).

如果你已经有一个安全的 numpy 和 scipy，安装 scikit-learn 最简单的方法是使用 ``pip`` ::

    pip install -U scikit-learn

或者 ``conda``::

    conda install scikit-learn

如果您还没有安装 NumPy 或 SciPy，还可以使用 conda 或 pip 来安装它们。
当使用 pip 时，请确保使用了 *binary wheels*，并且 NumPy 和 SciPy 不会从源重新编译，这可能在使用操作系统和硬件的特定配置（如 Raspberry Pi 上的 Linux）时发生。
从源代码构建 numpy 和 scipy 可能是复杂的（特别是在 Windows 上），并且需要仔细配置，以确保它们与线性代数程序的优化实现链接。而是使用如下所述的第三方发行版。

如果您必须安装 scikit-learn 及其与 pip 的依赖关系，则可以将其安装为 ``scikit-learn[alldeps]``。
最常见的用例是 ``requirements.txt`` 用作 PaaS 应用程序或 Docker 映像的自动构建过程的一部分的文件。此选项不适用于从命令行进行手动安装。

第三方发行版
==========================
如果您尚未安装具有 numpy 和 scipy 的 python 安装，建议您通过软件包管理器或通过 python 软件包进行安装。
这些与 numpy, scipy, scikit-learn, matplotlib 和许多其他有用的科学和数据处理库。

可用选项有:

Canopy 和 Anaconda 适用于所有支持的平台
-----------------------------------------------

`Canopy
<https://www.enthought.com/products/canopy>`_ 和 `Anaconda
<https://www.continuum.io/downloads>`_ 都运送了最新版本的 scikit-learn，另外还有一大批适用于 Windows，Mac OSX 和 Linux 的科学 python 库。

Anaconda 提供 scikit-learn 作为其免费分发的一部分.


.. warning::

    升级或卸载使用 Anaconda 安装的 scikit-learn，或者 ``conda`` **不应该使用 pip 命令**。代替:

    升级 ``scikit-learn``::

        conda update scikit-learn

    卸载 ``scikit-learn``::

        conda remove scikit-learn

    使用 ``pip install -U scikit-learn`` 升级 or ``pip uninstall scikit-learn`` 卸载 可能无法正确删除 ``conda`` 命令安装的文件.

    pip 升级和卸载操作仅适用于通过 ``pip install`` 安装的软件包.


WinPython 适用于 Windows
------------------------

该 `WinPython <https://winpython.github.io/>`_ 项目分布 scikit-learn 作为额外的插件。


有关特定操作系统的安装说明或汇编出血边缘版本，请参阅 :ref:`advanced-installation`.
