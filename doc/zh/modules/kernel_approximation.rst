.. _kernel_approximation:

内核近似
====================

这个子模块包含与某些 kernel 对应的特征映射的函数，这个会用于例如支持向量机的算法当中(see :ref:`svm`)。
下面这些特征函数对输入执行非线性转换，可以用于线性分类或者其他算法。

.. currentmodule:: sklearn.linear_model
与`kernel trick <https://en.wikipedia.org/wiki/Kernel_trick>`_相比，近似的进行特征映射更适合在线学习，并能够有效
减少学习大量数据的开销。标准化使用内核的 svm 不能有效的适用到海量数据，但是使用近似内核映射的方法，对于线性SVM来说效果可能更好。
而且，使用 :class:`SGDClassifier`进行近似的内核映射，使得对海量数据进行非线性学习也成为了可能。

由于近似嵌入的方法没有太多经验性的验证，所以建议将结果和使用精确的内核方法的结果进行比较。

.. seealso::

   :ref:`polynomial_regression` for an exact polynomial transformation.

.. currentmodule:: sklearn.kernel_approximation

.. _nystroem_kernel_approx:

内核近似的 Nystroem 方法
----------------------------------------
:class:`Nystroem` 中实现了 Nystroem 方法用于低等级的近似核。它是通过采样kernel已经评估好的数据。默认情况下，
:class:`Nystroem`使用``rbf`` kernel，但它可以使用任何内核函数和预计算内核矩阵.
使用的样本数量 - 计算的特征维数 - 由参数``n_components`给出.

.. _rbf_kernel_approx:

径向基函数内核
----------------------------
:class:`RBFSampler` 为径向基函数核构造一个近似映射，又称为 Random Kitchen Sinks [RR2007].
在应用线性算法（例如线性SVM）之前，可以使用此转换来明确建模内核映射::

    >>> from sklearn.kernel_approximation import RBFSampler
    >>> from sklearn.linear_model import SGDClassifier
    >>> X = [[0, 0], [1, 1], [1, 0], [0, 1]]
    >>> y = [0, 0, 1, 1]
    >>> rbf_feature = RBFSampler(gamma=1, random_state=1)
    >>> X_features = rbf_feature.fit_transform(X)
    >>> clf = SGDClassifier()   # doctest: +NORMALIZE_WHITESPACE
    >>> clf.fit(X_features, y)
    SGDClassifier(alpha=0.0001, average=False, class_weight=None, epsilon=0.1,
           eta0=0.0, fit_intercept=True, l1_ratio=0.15,
           learning_rate='optimal', loss='hinge', max_iter=5, n_iter=None,
           n_jobs=1, penalty='l2', power_t=0.5, random_state=None,
           shuffle=True, tol=None, verbose=0, warm_start=False)
    >>> clf.score(X_features, y)
    1.0

这个映射依赖于内核值的 Monte Carlo 近似.  ``fit`` 方法执行 Monte Carlo 采样，而该 ``transform`` 方法执行
数据的映射.由于过程的固有随机性，结果可能会在不同的 ``fit`` 函数调用之间变化。

该 ``fit`` 函数有两个参数:
``n_components`` 是特征变换的目标维数. ``gamma`` 是 RBF-kernel的参数.``n_components`越高，会导致更好的内核近似，
并且将产生与内核SVM产生的结果更相似的结果。请注意，"拟合"特征函数实际上不取决于 ``fit`` 函数传递的数据。只有数据的维数被使用。
详情可以参考 [RR2007]_.

The ``fit`` function takes two arguments:
``n_components``, which is the target dimensionality of the feature transform,
and ``gamma``, the parameter of the RBF-kernel.  A higher ``n_components`` will
result in a better approximation of the kernel and will yield results more
similar to those produced by a kernel SVM. Note that "fitting" the feature
function does not actually depend on the data given to the ``fit`` function.
Only the dimensionality of the data is used.
Details on the method can be found in [RR2007]_.

对于给定的值 ``n_components`` :class:`RBFSampler` 在:class:`Nystroem`中使用通常不太准确，
但是:class:`RBFSampler` 使用更大的特征空间，更容易计算。
For a given value of ``n_components`` :class:`RBFSampler` is often less accurate
as :class:`Nystroem`. :class:`RBFSampler` is cheaper to compute, though, making
use of larger feature spaces more efficient.

.. figure:: ../auto_examples/images/sphx_glr_plot_kernel_approximation_002.png
    :target: ../auto_examples/plot_kernel_approximation.html
    :scale: 50%
    :align: center

    Comparing an exact RBF kernel (left) with the approximation (right)

.. topic:: Examples:

    * :ref:`sphx_glr_auto_examples_plot_kernel_approximation.py`

.. _additive_chi_kernel_approx:


可添加的 Chi Squared Kernel（内核）
-----------------------------------

可添加的 Chi Squared Kernel（内核）是直方图的核心，通常用于计算机视觉。

这里使用的可添加的 Chi Squared Kernel（内核）给出

.. math::

        k(x, y) = \sum_i \frac{2x_iy_i}{x_i+y_i}

这个和 :func:`sklearn.metrics.additive_chi2_kernel` 不完全一样.[VZ2010]_ 的作者喜欢上面的版本，因为它总是积极的。
由于这个kernel是可添加的，因此可以分别处理嵌入的 :math:`x_i`. 这使得在规则的间隔类对傅里叶变换进行性才赢，代替近似的Monte Carlo采样。

 :class:`AdditiveChi2Sampler` 类实现了这个组件采样方法. 每个组件都被采样 :math:`n` 次，每一个输入维数都会产生 `2n+1` 维（来自傅立叶变换的实部和复数部分的两个数据段的倍数）.
 在文献中，:math:`n` 经常取为 1 或者 2，将数据集转换为 ``n_samples * 5 * n_features` 大小（在 :math:`n=2`的情况下 ）.

:class:`AdditiveChi2Sampler` 提供的近似特征映射可以和 :class:`RBFSampler` 提供的近似特征映射合并，得到一个取幂的
chi squared kerne。可以查看 [VZ2010]_ 和 [VVZ2010]_ :class:`RBFSampler`的合并.

.. _skewed_chi_kernel_approx:

Skewed Chi Squared Kernel
-------------------------

skewed chi squared kernel 给出下面公式

.. math::

        k(x,y) = \prod_i \frac{2\sqrt{x_i+c}\sqrt{y_i+c}}{x_i + y_i + 2c}


它有和幂次方 chi squared kernel 相似的属性，用于计算机视觉.但是允许进行简单的 Monte Carlo 近似 的特征映射。

:class:`SkewedChi2Sampler` 的使用和之前描述的 :class:`RBFSampler` 一样.唯一的区别是自由参数，称之为 :math:`c`.
这种映射和数学细节可以参考 [LS2010]_.


数学方面的细节
--------------------
内核方法像支持向量机，或者标准化PCA 依赖于重新生产 kernel Hilbert spaces.
对于任何内核函数: math:`k`（叫做 Mercer kernel），保证了 math:`\phi` 进入 Hilbert space:math:`\mathcal{H}`的映射,例如：

.. math::

        k(x,y) = \langle \phi(x), \phi(y) \rangle

math:`\langle \cdot, \cdot \rangle`  是在Hilbert space 中做内积.

如果一个算法，例如线性支持向量机或者PCA，依赖于数据集的数量级：math:`x_i`，可能会使用:math:`k(x_i, x_j)`，
符合孙发的映射:math:`\phi(x_i)`. 使用 math:`k` 的优点在于 math:`\phi` 永远不会直接计算，允许大量的特征计算（甚至是无线的）.

kernel 方法的一个缺点是，在优化过程中有可能存储大量的kernel 值：math:`k(x_i, x_j)`.
如果内核化的分类器应用于新的数据: math:`y_j`，:math:`k(x_i, y_j)`需要计算用来做预测，训练集中的：math:`x_i`有可能有很多不同的。

这个子模块的这些类中允许嵌入:math:`\phi`，从而明确的与:math:`\phi(x_i)` 一起工作，
这消除了使用kernel的需要和存储训练样本.



.. topic:: 参考:

    .. [RR2007] `"Random features for large-scale kernel machines"
      <http://www.robots.ox.ac.uk/~vgg/rg/papers/randomfeatures.pdf>`_
      Rahimi, A. and Recht, B. - Advances in neural information processing 2007,
    .. [LS2010] `"Random Fourier approximations for skewed multiplicative histogram kernels"
      <http://www.maths.lth.se/matematiklth/personal/sminchis/papers/lis_dagm10.pdf>`_
      Random Fourier approximations for skewed multiplicative histogram kernels
      - Lecture Notes for Computer Sciencd (DAGM)
    .. [VZ2010] `"Efficient additive kernels via explicit feature maps"
      <https://www.robots.ox.ac.uk/~vgg/publications/2011/Vedaldi11/vedaldi11.pdf>`_
      Vedaldi, A. and Zisserman, A. - Computer Vision and Pattern Recognition 2010
    .. [VVZ2010] `"Generalized RBF feature maps for Efficient Detection"
      <https://www.robots.ox.ac.uk/~vgg/publications/2010/Sreekanth10/sreekanth10.pdf>`_
      Vempati, S. and Vedaldi, A. and Zisserman, A. and Jawahar, CV - 2010
