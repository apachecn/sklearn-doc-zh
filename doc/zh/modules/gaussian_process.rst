

.. _gaussian_process:

============
高斯过程
============

.. currentmodule:: sklearn.gaussian_process

**高斯过程 (GP)** 是一种常用的监督学习方法，旨在解决*回归问题*和*概率分类问题*。

高斯过程模型的优点如下：

    - 预测内插了观察结果（至少对于正则核）。

    - 预测结果是概率形式的（高斯形式的）。这样的话，
	  人们可以计算得到经验置信区间并且根据那些应该
	  在某些兴趣区域改编（在线拟合，自适应）预测的那些来确定。
	
    - 通用性: 可以指定不同的:ref:`内核<gp_kernels>`。
	  虽然该函数提供了常用的内核，但是也可以指定自定义内核。

高斯过程模型的缺点包括：

    - 它们不稀疏，例如，模型通常使用整个样本/特征信息来进行预测。

    - 高维空间模型会失效，高维也就是指特征的数量超过几十个。
	
.. _gpr:

高斯过程回归（GPR）
=================================

.. currentmodule:: sklearn.gaussian_process

:class:`GaussianProcessRegressor` 类实现了回归情况下的高斯过程(GP)模型。
为此，需要实现指定GP的先验。当参数 ``normalize_y=False`` 时，先验的均值
通常假定为常数或者零; 当 ``normalize_y=True`` 时，先验均值通常为训练数
据的均值。而先验的方差通过传递 :ref:`内核 <gp_kernels>` 对象来指定。通过
最大化基于传递 ``optimizer`` 的对数边缘似然估计(LML)，内核的超参可以在
GaussianProcessRegressor 类执行拟合过程中被优化。由于 LML 可能会存在多个
局部最优解，因此优化过程可以通过指定 ``n_restarts_optimizer`` 参数进行
多次重复。通过设置内核的超参初始值来进行第一次优化的运行。后续的运行
过程中超参值都是从合理范围值中随机选取的。如果需要保持初始化超参值，
那么需要把优化器设置为 `None` 。

目标变量中的噪声级别通过参数 ``alpha`` 来传递并指定，要么全局是常数要么是一个数据点。
请注意，适度的噪声水平也可以有助于处理拟合期间的数字问题，因为它被有效地实现为吉洪诺夫正则化，
即通过将其添加到核心矩阵的对角线。明确指定噪声水平的替代方法是将 WhiteKernel 组件包含在内核中，
这可以从数据中估计全局噪声水平（见下面的示例）。

算法实现是基于 [RW2006]_ 中的算法 2.1 。除了标准 scikit learn 估计器的 API 之外，
GaussianProcessRegressor 的作用还包括：

* 允许预测，无需事先拟合（基于GP之前）

* 提供了一种额外的方法 ``sample_y(X)`` , 其评估
  在给定输入处从 GPR （先验或后验）绘制的样本

* 公开了一种方法 ``log_marginal_likelihood(theta)`` , 
  可以在外部使用其他方式选择超参数，例如通过马尔科夫链蒙特卡罗链。
 


GPR 示例
=============

具有噪声级的 GPR 估计
-------------------------

该示例说明具有包含 WhiteKernel 的和核的 GPR 可以估计数据的噪声水平。
对数边缘似然（LML）景观的图示表明存在 LML 的两个局部最大值。

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpr_noisy_000.png
   :target: ../auto_examples/gaussian_process/plot_gpr_noisy.html
   :align: center

第一个对应于具有高噪声电平和大长度尺度的模型，其解释数据中噪声的所有变化。

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpr_noisy_001.png
   :target: ../auto_examples/gaussian_process/plot_gpr_noisy.html
   :align: center

第二个具有较小的噪声水平和较短的长度尺度，这解释了无噪声功能关系的大部分变化。
第二种模式有较高的可能性; 然而，根据超参数的初始值，基于梯度的优化也可能会收敛到高噪声解。
因此，对于不同的初始化，重复优化多次。

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpr_noisy_002.png
   :target: ../auto_examples/gaussian_process/plot_gpr_noisy.html
   :align: center


GPR 和内核岭回归的比较
---------------------------------------------

内核脊回归（KRR）和 GPR 通过内部使用 "kernel trick(内核技巧)" 来学习目标函数。
KRR学习由相应内核引起的空间中的线性函数，该空间对应于原始空间中的非线性函数。
基于平均误差损失与脊正弦化，选择内核空间中的线性函数。
GPR使用内核来定义先验分布在目标函数上的协方差，并使用观察到的训练数据来定义似然函数。
基于贝叶斯定理，定义了目标函数上的（高斯）后验分布，其平均值用于预测。

一个主要区别是，GPR 可以基于边际似然函数上的梯度上升选择内核的超参数，
而KRR需要在交叉验证的损失函数（均方误差损失）上执行网格搜索。
另一个区别是，GPR 学习目标函数的生成概率模型，因此可以提供有意义的置信区间和后验样本以及预测值，
而KRR仅提供预测。

下图说明了人造数据集上的两种方法，其中包括正弦目标函数和强噪声。
该图比较了基于 ExpSineSquared 内核的 KRR 和 GPR 的学习模型，适用于学习周期函数。
内核的超参数控制内核的平滑度（length_scale）和周期性（周期性）。
此外，数据的噪声水平由 GPR 通过内核中的另外的 WhiteKernel 组件和 KRR 的正则化参数 α 明确地学习。

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_compare_gpr_krr_001.png
   :target: ../auto_examples/gaussian_process/plot_compare_gpr_krr.html
   :align: center

该图显示，两种方法都可以学习合理的目标函数模型。
GPR将功能的周期性正确地识别为 :math:`2*\pi` （6.28），而 KRR 选择倍增的周期为 :math:`4*\pi` 。
此外，GPR 为 KRR 不可用的预测提供了合理的置信区间。
两种方法之间的主要区别是拟合和预测所需的时间：
原则上KRR的拟合速度较快，超参数优化的网格搜索与超参数（ "curse of dimensionality(维度诅咒)" ）呈指数级关系。
GPR中的参数的基于梯度的优化不受此指数缩放的影响，因此在具有三维超参数空间的该示例上相当快。
预测的时间是相似的; 然而，生成 GPR 预测分布的方差需要的时间比生成平均值要长。


Mauna Loa CO2 数据中的 GRR
-------------------------------

该示例基于 [RW2006] 的第 5.4.3 节。
它演示了使用梯度上升的对数边缘似然性的复杂内核工程和超参数优化的示例。
数据包括在 1958 年至 1997 年间夏威夷 Mauna Loa 天文台收集的每月平均大气二氧
化碳浓度（以百万分之几（ppmv）计）。目的是将二氧化碳浓度建模为时间t的函数。   

内核由几个术语组成，负责说明信号的不同属性：

- 一个长期的，顺利的上升趋势是由一个 RBF 内核来解释的。
  具有较大长度尺寸的RBF内核将使该组件平滑; 
  没有强制这种趋势正在上升，这给 GP 带来了这个选择。
  具体的长度尺度和振幅是自由的超参数。

- 季节性因素，由定期的 ExpSineSquared 内核解释，固定周期为1年。
  该周期分量的长度尺度控制其平滑度是一个自由参数。
  为了使准确周期性的衰减，带有RBF内核的产品被采用。
  该RBF组件的长度尺寸控制衰减时间，并且是另一个自由参数。

- 较小的中期不规则性将由 RationalQuadratic 内核组件来解释，
  RationalQuadratic 内核组件的长度尺度和 alpha 参数决定长度尺度的扩散性。
  根据 [RW2006] ，这些不规则性可以更好地由 RationalQuadratic 来解释，
  而不是 RBF 内核组件，这可能是因为它可以容纳几个长度尺度。

- 一个 "noise(噪声)" 术语，由一个 RBF 内核贡献组成，它将解释相关的噪声分量，
  如局部天气现象以及 WhiteKernel 对白噪声的贡献。
  相对幅度和RBF的长度尺度是进一步的自由参数。

在减去目标平均值后最大化对数边际似然率产生下列内核，其中LML为-83.214:

::

   34.4**2 * RBF(length_scale=41.8)
   + 3.27**2 * RBF(length_scale=180) * ExpSineSquared(length_scale=1.44,
                                                      periodicity=1)
   + 0.446**2 * RationalQuadratic(alpha=17.7, length_scale=0.957)
   + 0.197**2 * RBF(length_scale=0.138) + WhiteKernel(noise_level=0.0336)

因此，大多数目标信号（34.4ppm）由长期上升趋势（长度为41.8年）解释。
周期分量的振幅为3.27ppm，衰减时间为180年，长度为1.44。
长时间的衰变时间表明我们在当地非常接近周期性的季节性成分。
相关噪声的幅度为0.197ppm，长度为0.138年，白噪声贡献为0.197ppm。
因此，整体噪声水平非常小，表明该模型可以很好地解释数据。
该图还显示，该模型直到2015年左右才能做出置信度比较高的预测

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpr_co2_001.png
   :target: ../auto_examples/gaussian_process/plot_gpr_co2.html
   :align: center

.. _gpc:

高斯过程分类（GPC）
======================

.. currentmodule:: sklearn.gaussian_process

所述 :class:`GaussianProcessClassifier` 器实现了用于分类目的的高斯过程（GP），当测试的预测采取类概率的形式，更能够用于概率分类。
GaussianProcessClassifier 在隐函数 :math:`f` 之前设置GP先验，然后通过链接函数进行压缩以获得概率分类。
隐函数 :math:`f` 因此就是所谓的干扰函数，其值不能被观测到，并且自身不具有相关性。
其目的是允许模型的表达形式更加简便，并且 :math:`f` 在预测过程中被去除（整合）。
GaussianProcessClassifier 实现了逻辑链路功能，
对于该逻辑，积分不能在分析上计算，但在二进制情况下很容易近似。

与回归设置相反，即使设置了高斯过程先验，隐函数 :math:`f` 的后验也不符合高斯分布，
因为高斯似然不适用于离散类标签。相反，使用的是与逻辑链路功能（logit）对应的非高斯似然。
GaussianProcessClassifier 通过拉普拉斯近似来估计非高斯后验分布。
更多详细信息，请参见 [RW2006] 的第 3 章。

GP先验平均值假定为零。先前的协方差是通过传递 :ref:`内核 <gp_kernels>` 对象来指定的。
在通过最大化基于传递的对数边缘似然（LML）的 GaussianProcessRegressor 拟合期间，
优化内核的超参数 ``optimizer`` 。由于LML可能具有多个局部最优值，
所以优化器可以通过指定重复启动 ``n_restarts_optimizer`` 。
第一次运行始终从内核的初始超参数值开始执行; 
从已经从允许值的范围中随机选择的超参数值进行后续运行。
如果初始超参数应该保持固定，`None` 可以作为优化器来传递。

:class:`GaussianProcessClassifier` 通过执行一对一或一对一的训练和预测来支持多类分类。
在一对一休息中，每个类都配有一个二进制高斯过程分类器，该类别被训练为将该类与其余类分开。
在 "one_vs_one" 中，对于每对类拟合一个二进制高斯过程分类器，这被训练为分离这两个类。
这些二进制预测因子的预测被组合成多类预测。更多详细信息，请参阅 :ref:`多类别分类 <multiclass>` 。

在高斯过程分类的情况下，"one_vs_one" 可能在计算上更便宜，
因为它必须解决涉及整个训练集的仅一个子集的许多问题，
而不是整个数据集的较少的问题。由于高斯过程分类与数据集的大小相互立方，这可能要快得多。
但是，请注意，"one_vs_one" 不支持预测概率估计，而只是简单的预测。
此外，请注意， :class:`GaussianProcessClassifier` 在内部还没有实现真正的多类 Laplace 近似，
但如上所述，基于在内部解决几个二进制分类任务，它们使用一对一或一对一组合。


GPC 示例
============

GPC 概率预测
----------------------------------


该示例说明了对于具有不同选项的超参数的RBF内核的GPC预测概率。
第一个图显示GPC具有任意选择的超参数的预测概率，并且超参数对应于最大对数边缘似然（LML）。

虽然通过优化LML选择的超参数具有相当大的LML，但是它们根据测试数据的对数损失稍微更差。
该图显示，这是因为它们在阶级边界（这是好的）表现出类概率的急剧变化，
但预测概率接近0.5远离类边界（这是坏的）这种不良影响是由由GPC内部使用的拉普拉斯逼近。

第二个图显示了内核超参数的不同选择的对数边缘似然，突出了第一个图中使用的超参数的两个选择的黑点。

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpc_000.png
   :target: ../auto_examples/gaussian_process/plot_gpc.html
   :align: center

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpc_001.png
   :target: ../auto_examples/gaussian_process/plot_gpc.html
   :align: center


GPC 在 XOR 数据集上的举例说明
--------------------------------------

.. currentmodule:: sklearn.gaussian_process.kernels


此示例说明了在XOR数据上的GPC。相对固定的，各向同性的核（:class:`RBF`）和非固定的核（:class:`DotProduct`）。
在这个特定的数据集上，`DotProduct`内核获得了更好的结果，因为类边界是线性的，与坐标轴重合。
然而，实际上，诸如:class:`RBF`经常获得更好结果的固定内核。

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpc_xor_001.png
   :target: ../auto_examples/gaussian_process/plot_gpc_xor.html
   :align: center

.. currentmodule:: sklearn.gaussian_process


iris 数据集上的高斯过程分类（GPC）
-----------------------------------------------------

该示例说明了用于虹膜数据集的二维版本上各向同性和各向异性RBF核的GPC的预测概率。
这说明了GPC对非二进制分类的适用性。
各向异性RBF内核通过为两个特征维度分配不同的长度尺度来获得稍高的对数边缘似然。

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpc_iris_001.png
   :target: ../auto_examples/gaussian_process/plot_gpc_iris.html
   :align: center


.. _gp_kernels:

高斯过程内核
================
.. currentmodule:: sklearn.gaussian_process.kernels

内核（也可以叫做GPs上下文中的"协方差函数"）
是决定高斯过程（GP）先验和后验形状的关键组成部分。
它们通过定义两个数据点的“相似性”，并结合相似的
数据点应该具有相似的目标值的假设，对所学习的函数进行编码。
内核可以分为两类：固定内核只取决于两个数据点的距离，
而不是依赖于它们的绝对值:math:`k(x_i, x_j)= k(d(x_i, x_j))`
，因此它们对于输入空间中的转换是不变的；而非固定的内核也取
决于数据点的具体值。固定内核可以进一步细分为各向同性和各向
异性内核，其中各向同性内核也不会在输入空间中旋转。想要了解
更多细节，请参看[RW2006]_的第四章。


高斯过程内核 API
---------------------------

:class:`Kernel` 主要是用来计算数据点之间的高斯过程协方差。
为此，内核中 ``__call__`` 方法会被调用。该方法可以用于计算
2d阵列X中所有数据点对的“自动协方差”，或二维阵列X的数据点
与二维阵列Y中的数据点的所有组合的“互协方差”。以下论断对于
所有内核k（除了 :class:`WhiteKernel`）都是成立的：``k(X) == K(X, Y=X)``。
如果仅仅是自协方差的对角线元素被使用，那么内核的方法``diag()``将会被调用，
该方法比等价的调用 ``__call__``: ``np.diag(k(X, X)) == k.diag(X)``
具有更高的计算效率。
 
内核通过超参数向量 :math:`\theta` 进行参数化。这些超参数可以
控制例如内核的长度尺度或周期性（见下文）。通过设置 ``__call__`` 
方法的参数 ``eval_gradient=True`` ，所有的内核支持计算解析
内核自协方差对于 :math:`\theta` 的解析梯度。该梯度被用来在
高斯过程中（不论是回归型还是分类型的）计算对数边缘似然函数
的梯度，进而被用来通过梯度下降的方法极大化对数边缘似然函数
从而确定 :math:`\theta` 的值。对于每个超参，当对内核的实例
进行赋值时，初始值和边界值需要被指定。通过内核对象属性 ``theta`` ，
:math:`\theta` 的当前值可以被获取或者设置。更重要的是，
超参的边界值可以被内核属性 ``bounds`` 获取。需要注意的是，
以上两种属性值(theta和bounds)都会返回内部使用值的日志转换值，
这是因为这两种属性值通常更适合基于梯度的优化。每个超参的
规范 :class:`Hyperparameter` 以相应内核中的实例的形式存储。
请注意使用了以"x"命名的超参的内核必然具有 self.x 和 self.x_bounds 这两种属性。

所有内核的抽象基类为 :class:`Kernel` 。Kernel 基类实现了
一个相似的接口 :class:`Estimator` ，提供了方法 ``get_params()`` ,
``set_params()`` 以及 ``clone()`` 。这也允许通过诸如
:class:`Pipeline` 或者 :class:`GridSearch` 之类的元估计来设置内核值。
需要注意的是，由于内核的嵌套结构（通过内核操作符，如下所见），
内核参数的名称可能会变得相对复杂些。通常来说，对于二元内核操作，
参数的左运算元以 ``k1__`` 为前缀，而有运算元以 ``k2__`` 为前缀。
一个额外的便利方法是 ``clone_with_theta(theta)``，
该方法返回克隆版本的内核，但是设置超参数为 ``theta``。
示例如下：

    >>> from sklearn.gaussian_process.kernels import ConstantKernel, RBF
    >>> kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.0, 10.0)) * RBF(length_scale=0.5, length_scale_bounds=(0.0, 10.0)) + RBF(length_scale=2.0, length_scale_bounds=(0.0, 10.0))
    >>> for hyperparameter in kernel.hyperparameters: print(hyperparameter)
    Hyperparameter(name='k1__k1__constant_value', value_type='numeric', bounds=array([[  0.,  10.]]), n_elements=1, fixed=False)
    Hyperparameter(name='k1__k2__length_scale', value_type='numeric', bounds=array([[  0.,  10.]]), n_elements=1, fixed=False)
    Hyperparameter(name='k2__length_scale', value_type='numeric', bounds=array([[  0.,  10.]]), n_elements=1, fixed=False)
    >>> params = kernel.get_params()
    >>> for key in sorted(params): print("%s : %s" % (key, params[key]))
    k1 : 1**2 * RBF(length_scale=0.5)
    k1__k1 : 1**2
    k1__k1__constant_value : 1.0
    k1__k1__constant_value_bounds : (0.0, 10.0)
    k1__k2 : RBF(length_scale=0.5)
    k1__k2__length_scale : 0.5
    k1__k2__length_scale_bounds : (0.0, 10.0)
    k2 : RBF(length_scale=2)
    k2__length_scale : 2.0
    k2__length_scale_bounds : (0.0, 10.0)
    >>> print(kernel.theta)  # Note: log-transformed
    [ 0.         -0.69314718  0.69314718]
    >>> print(kernel.bounds)  # Note: log-transformed
    [[       -inf  2.30258509]
     [       -inf  2.30258509]
     [       -inf  2.30258509]]


所有的高斯过程内核操作都可以通过 :mod:`sklearn.metrics.pairwise` 来进行互操作，反之亦然。
:class:`Kernel` 的子类实例可以通过 ``metric`` 参数传给 :mod:`sklearn.metrics.pairwise` 中的
 ``pairwise_kernels`` 。更重要的是，超参数的梯度不是分析的，而是数字，所有这些内核只支持
各向同性距离。该参数 ``gamma`` 被认为是一个超参数，可以进行优化。其他内核参数在初始化时直接设置，
并保持固定。


基础内核
------------
:class:`ConstantKernel` 内核类可以被用作 :class:`Product` 内核类的一部分，
在它可以对其他因子（内核）进行度量的场景下或者作为更改高斯过程均值的
 :class:`Sum` 类的一部分。这取决于参数 :math:`constant\_value` 的设置。该方法定义为：

.. math::
   k(x_i, x_j) = constant\_value \;\forall\; x_1, x_2

:class:`WhiteKernel` 内核类的主要应用实例在于当解释信号的噪声部分时
可以作为一个和内核的一部分。通过调节参数 :math:`noise\_level`，
该类可以用来估计噪声级别。具体如下所示：

.. math::
    k(x_i, x_j) = noise\_level \text{ if } x_i == x_j \text{ else } 0


内核操作
------------
内核操作是把1~2个基内核与新内核进行合并。内核类 :class:`Sum` 通过
:math:`k_{sum}(X, Y) = k1(X, Y) + k2(X, Y)` 相加来合并 :math:`k1`
和 :math:`k2` 内核。内核类 :class:`Product` 通过
:math:`k_{product}(X, Y) = k1(X, Y) * k2(X, Y)` 把 :math:`k1` 
和 :math:`k2` 内核进行合并。内核类 :class:`Exponentiation` 通过
:math:`k_{exp}(X, Y) = k(X, Y)^\text{exponent}` 把基内核与
常量参数 :math:`exponent` 进行合并。

径向基函数内核
------------------
:class:`RBF` 内核是一个固定内核，它也被称为“平方指数”内核。它通过定长的参数 :math:`l>0` 
来对内核进行参数化。该参数既可以是标量（内核的各向同性变体）或者与输入 :math:`x` （内核的各向异性变体）
具有相同数量的维度的向量。该内核可以被定义为：

.. math::
   k(x_i, x_j) = \text{exp}\left(-\frac{1}{2} d(x_i / l, x_j / l)^2\right)

这个内核是无限可微的，这意味着这个内核作为协方差函数的 GP 具有所有阶数的均方差导数，
因此非常平滑。由RBF内核产生的GP的先验和后验示意图如下所示：

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpr_prior_posterior_000.png
   :target: ../auto_examples/gaussian_process/plot_gpr_prior_posterior.html
   :align: center


Matérn 内核
----------------
:class:`Matern` 内核是一个固定内核，是 :class:`RBF` 内核的泛化。它有一个额外的参数 :math:`\nu`，
该参数控制结果函数的平滑程度。它由定长参数 :math:`l>0` 来实现参数化。该参数既可以是标量
（内核的各向同性变体）或者与输入 :math:`x` （内核的各向异性变体）具有相同数量的维度的向量。
该内核可以被定义为：

.. math::

    k(x_i, x_j) = \sigma^2\frac{1}{\Gamma(\nu)2^{\nu-1}}\Bigg(\gamma\sqrt{2\nu} d(x_i / l, x_j / l)\Bigg)^\nu K_\nu\Bigg(\gamma\sqrt{2\nu} d(x_i / l, x_j / l)\Bigg),

因为 :math:`\nu\rightarrow\infty` ，Matérn 内核收敛到 RBF 内核。
当 :math:`\nu = 1/2` 时，Matérn 内核变得与绝对指数内核相同时，即

.. math::
    k(x_i, x_j) = \sigma^2 \exp \Bigg(-\gamma d(x_i / l, x_j / l) \Bigg) \quad \quad \nu= \tfrac{1}{2}

特别的，当 :math:`\nu = 3/2` 时：

.. math::
    k(x_i, x_j) = \sigma^2 \Bigg(1 + \gamma \sqrt{3} d(x_i / l, x_j / l)\Bigg) \exp \Bigg(-\gamma \sqrt{3}d(x_i / l, x_j / l) \Bigg) \quad \quad \nu= \tfrac{3}{2}

和 :math:`\nu = 5/2` :

.. math::
    k(x_i, x_j) = \sigma^2 \Bigg(1 + \gamma \sqrt{5}d(x_i / l, x_j / l) +\frac{5}{3} \gamma^2d(x_i / l, x_j / l)^2 \Bigg) \exp \Bigg(-\gamma \sqrt{5}d(x_i / l, x_j / l) \Bigg) \quad \quad \nu= \tfrac{5}{2}

是学习函数的常用选择，并且不是无限可微的（由 RBF 内核假定）
但是至少具有一阶( :math:`\nu = 3/2` )或者二阶( :math:`\nu = 5/2` )可微性。

通过 :math:`\nu` 灵活控制学习函数的平滑性可以更加适应真正的底层函数关联属性。
通过 Matérn 内核产生的高斯过程的先验和后验如下图所示：

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpr_prior_posterior_004.png
   :target: ../auto_examples/gaussian_process/plot_gpr_prior_posterior.html
   :align: center

想要更进一步地了解不同类型的Matérn内核请参阅[RW2006]_, pp84。

有理二次内核
----------------

:class:`RationalQuadratic` 内核可以被看做不同特征尺度下的 :class:`RBF` 内核的规模混合（一个无穷和）
它通过长度尺度参数 :math:`l>0` 和比例混合参数 :math:`\alpha>0` 进行参数化。
此时仅支持 :math:`l` 标量的各向同性变量。内核公式如下：

.. math::
   k(x_i, x_j) = \left(1 + \frac{d(x_i, x_j)^2}{2\alpha l^2}\right)^{-\alpha}

从 RBF 内核中产生的高斯过程的先验和后验如下图所示：

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpr_prior_posterior_001.png
   :target: ../auto_examples/gaussian_process/plot_gpr_prior_posterior.html
   :align: center

正弦平方内核
----------------

:class:`ExpSineSquared` 内核可以对周期性函数进行建模。它由定长参数 :math:`l>0` 
以及周期参数 :math:`p>0` 来实现参数化。此时仅支持 :math:`l` 标量的各向同性变量。内核公式如下：

.. math::
   k(x_i, x_j) = \text{exp}\left(-2 \left(\text{sin}(\pi / p * d(x_i, x_j)) / l\right)^2\right)

从 ExpSineSquared 内核中产生的高斯过程的先验和后验如下图所示：

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpr_prior_posterior_002.png
   :target: ../auto_examples/gaussian_process/plot_gpr_prior_posterior.html
   :align: center

点乘内核
------------

:class:`DotProduct` 内核是非固定内核，它可以通过在线性回归的 :math:`x_d (d = 1, . . . , D)` 的相关系数上加上
服从于 :math:`N(0, 1)` 的先验以及在线性回归的偏置上加上服从于 :math:`N(0, \sigma_0^2)` 的先验来获得。
该 :class:`DotProduct` 内核是不变的关于原点的坐标，而不是翻译的旋转。它通过设置参数 :math:`\sigma_0^2` 来进行参数化。
当 :math:`\sigma_0^2 = 0` 时，该内核叫做同质线性内核；否则该内核是非同质的。内核公式如下：

.. math::
   k(x_i, x_j) = \sigma_0 ^ 2 + x_i \cdot x_j

:class:`DotProduct` 内核通常和指数分布相结合。实例如下图所示：

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpr_prior_posterior_003.png
   :target: ../auto_examples/gaussian_process/plot_gpr_prior_posterior.html
   :align: center

参考文献
------------

.. [RW2006] Carl Eduard Rasmussen and Christopher K.I. Williams, "Gaussian Processes for Machine Learning", MIT Press 2006, Link to an official complete PDF version of the book `here <http://www.gaussianprocess.org/gpml/chapters/RW.pdf>`_ .

.. currentmodule:: sklearn.gaussian_process



传统高斯过程
=================

在本节中，描述了版本 0.16.1 及之前 scikit 中高斯过程的实现，
请注意，此实现已被弃用，将在版本 0.18 中删除。

回归实例介绍
----------------

假定我们要替代这个函数：:math:`g(x) = x \sin(x)` 。
为了做到这一点，该功能被评估到一个实验设计上。然后，
我们定义一个回归和相关模型可能被其他参数指定的高斯模型，
并且要求模型能够拟合数据。拟合过程由于受到实例化过程中
参数数目的影响可能依赖于参数的最大似然估计或者直接使用给定的参数。

::

    >>> import numpy as np
    >>> from sklearn import gaussian_process
    >>> def f(x):
    ...	    return x * np.sin(x)
    >>> X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
    >>> y = f(X).ravel()
    >>> x = np.atleast_2d(np.linspace(0, 10, 1000)).T
    >>> gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
    >>> gp.fit(X, y)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    GaussianProcess(beta0=None, corr=<function squared_exponential at 0x...>,
            normalize=True, nugget=array(2.22...-15),
            optimizer='fmin_cobyla', random_start=1, random_state=...
            regr=<function constant at 0x...>, storage_mode='full',
            theta0=array([[ 0.01]]), thetaL=array([[ 0.0001]]),
            thetaU=array([[ 0.1]]), verbose=False)
    >>> y_pred, sigma2_pred = gp.predict(x, eval_MSE=True)


噪声数据拟合
----------------

当含噪声的数据被用来做拟合时，对于每个数据点，高斯过程模型可以指定噪声的方差。
:class:`GaussianProcess` 包含一个被添加到训练数据得到的自相关矩阵对角线中的
参数 ``nugget`` 。通常来说这是一种类型的吉洪诺夫正则化方法。
在平方指数相关函数的特殊情况下，该归一化等效于指定输入中的小数方差。也就是：

.. math::
   \mathrm{nugget}_i = \left[\frac{\sigma_i}{y_i}\right]^2

使用 ``nugget`` 以及 ``corr`` 正确设置，高斯过程可以更好地用于从噪声数据恢复给定的向量函数。


数学形式
------------


初始假设
^^^^^^^^^^^^

假设需要对电脑实验的结果进行建模，例如使用一个数学函数：

.. math::

        g: & \mathbb{R}^{n_{\rm features}} \rightarrow \mathbb{R} \\
           & X \mapsto y = g(X)


同时假设这个函数是 *一个* 有关于 *一个* 高斯过程 :math:`G` 的条件采用方法，
那么从这个假设出发，GPML 通常可以表示为如下形式：

.. math::

        G(X) = f(X)^T \beta + Z(X)

其中， :math:`f(X)^T \beta` 是一个线性回归模型，并且 :math:`Z(X)` 是一个
以零为均值，协方差函数完全平稳的高斯过程：

.. math::

        C(X, X') = \sigma^2 R(|X - X'|)

:math:`\sigma^2` 表示其方差， :math:`R` 表示仅仅基于样本之间的绝对相关距离的相关函数，可能是特征（这是平稳性假设）

从这些基本的公式中可以注意到GPML仅仅是基本的线性二乘回归问题的扩展。

.. math::

        g(X) \approx f(X)^T \beta

除此之外，我们另外假定由相关函数指定的样本之间的一些空间相干性（相关性）。
事实上，普通最小二乘法假设当 :math:`X = X'` 时，相关性模型 :math:`R(|X - X'|)` 
是 1；否则为 0 ，相关性模型为 *狄拉克* 相关模型--有时在克里金文献中被称为 *熔核* 相关模型


最佳线性无偏预测（BLUP）
^^^^^^^^^^^^^^^^^^^^^^^^^^^
我们现在推导出基于观测结果的 *最佳线性无偏预测* :math:`g`

.. math::

    \hat{G}(X) = G(X | y_1 = g(X_1), ...,
                                y_{n_{\rm samples}} = g(X_{n_{\rm samples}}))

它可以由 *给定的属性* 加以派生：

- 它是线性的(观测结果的线性组合)

.. math::

    \hat{G}(X) \equiv a(X)^T y

- 它是无偏的

.. math::

    \mathbb{E}[G(X) - \hat{G}(X)] = 0

- 它是最好的（在均方误差的意义上）

.. math::

    \hat{G}(X)^* = \arg \min\limits_{\hat{G}(X)} \;
                                            \mathbb{E}[(G(X) - \hat{G}(X))^2]

因此最优的带权向量 :math:`a(X)` 是以下等式约束优化问题的解

.. math::

    a(X)^* = \arg \min\limits_{a(X)} & \; \mathbb{E}[(G(X) - a(X)^T y)^2] \\
                       {\rm s. t.} & \; \mathbb{E}[G(X) - a(X)^T y] = 0

					
以拉格朗日形式重写这个受约束的优化问题，并进一步寻求要满足的一阶最优条件，
从而得到一个以闭合形式表达式为终止形式的预测器 - 参见参考文献中完整的证明。

最后，BLUP 为高斯随机变量，其中均值为：
					
.. math::

    \mu_{\hat{Y}}(X) = f(X)^T\,\hat{\beta} + r(X)^T\,\gamma

方差为:

.. math::

    \sigma_{\hat{Y}}^2(X) = \sigma_{Y}^2\,
    ( 1
    - r(X)^T\,R^{-1}\,r(X)
    + u(X)^T\,(F^T\,R^{-1}\,F)^{-1}\,u(X)
    )

其中:

* 根据自相关函数以及内置参数 :math:`\theta` 所定义的相关性矩阵为：

.. math::

    R_{i\,j} = R(|X_i - X_j|, \theta), \; i,\,j = 1, ..., m

* 在进行预测的点与 DOE 中的点之间的互相关的向量:

.. math::

    r_i = R(|X - X_i|, \theta), \; i = 1, ..., m

* 回归矩阵 (例如范德蒙矩阵 如果 :math:`f` 以多项式为基):

.. math::

    F_{i\,j} = f_i(X_j), \; i = 1, ..., p, \, j = 1, ..., m

* 广义最小二乘回归权重:

.. math::

    \hat{\beta} =(F^T\,R^{-1}\,F)^{-1}\,F^T\,R^{-1}\,Y

* 和向量:

.. math::

    \gamma & = R^{-1}(Y - F\,\hat{\beta}) \\
    u(X) & = F^T\,R^{-1}\,r(X) - f(X)

需要重点注意的是，高斯过程预测器的概率输出是完全可分析的并且依赖于基本的线性代数操作。
更准确地说，预测结果的均值是两个简单线性组合（点积）的和，方差需要两个矩阵反转操作，但关联
矩阵只能使用 Cholesky 分解算法分解一次。

经验最佳线性无偏估计（EBLUP）
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

到现在为止，自相关和回归模型都已假定给出。然而，在实践中，它们从来都是未知的
因此需要为这些模型 :ref:`correlation_models` 做出（积极的）经验选择。

根据这些选择，可以来估计 BLUP 中涉及到的遗留未知参数。为此，需要使用一系列
被提供的观测值同时结合一系列推断技术。目前使用的方法是基于 DACE's Matlab 工
具包的*最大似然估计* - 参见 DACE 手册中的完全推导公式。最大似然估计的问题在
自相关参数中是一个全局优化的问题。这种全局优化通过 scipy.optimize 中的
fmin_cobyla 优化函数加以实现。然而，在各向异性的情况下，我们提供了
Welch 的分量优化算法的实现 - 参见参考。

.. _correlation_models:

关联模型
------------

由于几乎相等的假设，常用的关联模型和一些有名的SVM内核相匹配。它们必须要满足 Mercer 条件
并且需要保持固定形式。然而，请注意，相关模型的选择应该与观察到来的原始实验的已知特性一致。
例如：

* 如果原始实验被认为是无限可微（平滑），则应使用 *平方指数关联模型* 。
* 如果不是无限可微的, 那么需要使用 *指数关联模型*.
* 还要注意，存在一个将衍生度作为输入的相关模型：这是 Matern 相关模型，但这里并没有实现（ TODO ）.

关于选择合适的关联模型更详细的讨论，请参阅 Rasmussen＆Williams 的文献。

.. _regression_models:


回归模型
------------

常用的线性回归模型包括零阶（常数）、一阶和二阶多项式。
但是可以以 Python 函数的形式指定它自己的特性，它将特征X
作为输入，并返回一个包含函数集值的向量。唯一加以限制地是，
函数的个数不能超过有效观测值的数目，因此基本的回归问题不被*确定*。

实现细节
------------

模型通过 DACE 的 Matlab 工具包来实现。

.. topic:: 参考文献:

    * `DACE, A Matlab Kriging Toolbox
      <http://imedea.uib-csic.es/master/cambioglobal/Modulo_V_cod101615/Lab/lab_maps/krigging/DACE-krigingsoft/dace/dace.pdf>`_ S Lophaven, HB Nielsen, J
      Sondergaard 2002,

    * W.J. Welch, R.J. Buck, J. Sacks, H.P. Wynn, T.J. Mitchell, and M.D.
      Morris (1992). Screening, predicting, and computer experiments.
      Technometrics, 34(1) 15--25.
