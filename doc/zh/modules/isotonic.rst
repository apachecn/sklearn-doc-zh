.. _isotonic:

===================
等式回归
===================

.. currentmodule:: sklearn.isotonic

:class:`IsotonicRegression` 类对数据进行非降函数拟合.
它解决了如下的问题:

  最小化 :math:`\sum_i w_i (y_i - \hat{y}_i)^2`

  服从于 :math:`\hat{y}_{min} = \hat{y}_1 \le \hat{y}_2 ... \le \hat{y}_n = \hat{y}_{max}`

其中每一个 :math:`w_i` 都是正数，每个 :math:`y_i` 是任意实数。
它生成一个由均方差接近的不减元素组成的向量。实际上这些元素形成一个分段线性函数。

.. figure:: ../auto_examples/images/sphx_glr_plot_isotonic_regression_001.png
   :target: ../auto_examples/plot_isotonic_regression.html
   :align: center
