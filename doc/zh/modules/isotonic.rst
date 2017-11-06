.. _isotonic:

===================
等式回归
===================

.. currentmodule:: sklearn.isotonic

:class:`IsotonicRegression` 类对数据进行非降函数拟合.
它解决了如下的问题:

  最小化 :math:`\sum_i w_i (y_i - \hat{y}_i)^2`

  服从于 :math:`\hat{y}_{min} = \hat{y}_1 \le \hat{y}_2 ... \le \hat{y}_n = \hat{y}_{max}`

其中每一个 :math:`w_i` 是 strictly 正数而且每个 :math:`y_i` 是任意实
数. 它生成一个由平方误差接近的不减元素组成的向量.实际上这一些元素形成
一个分段线性的函数.

.. figure:: ../auto_examples/images/sphx_glr_plot_isotonic_regression_001.png
   :target: ../auto_examples/plot_isotonic_regression.html
   :align: center
