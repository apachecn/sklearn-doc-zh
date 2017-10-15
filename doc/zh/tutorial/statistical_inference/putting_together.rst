=========================
Putting it all together
=========================

..  Imports
    >>> import numpy as np

Pipelining
============

模型管道化
============

我们已经知道一些模型可以做数据转换，一些模型可以用来预测变量。我们可以建立一个组合模型
同时完成以上工作。

We have seen that some estimators can transform data and that some estimators
can predict variables. We can also create combined estimators:

.. image:: /auto_examples/images/sphx_glr_plot_digits_pipe_001.png
   :target: ../../auto_examples/plot_digits_pipe.html
   :scale: 65
   :align: right

.. literalinclude:: ../../auto_examples/plot_digits_pipe.py
    :lines: 23-63

Face recognition with eigenfaces
=================================

本征脸技术做人脸识别
=================================

该实例用到的数据集来自LFW_(Labeled Faces in the Wild)。数据已经进行了初步预处理。
  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

The dataset used in this example is a preprocessed excerpt of the
"Labeled Faces in the Wild", also known as LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

.. _LFW: http://vis-www.cs.umass.edu/lfw/

.. literalinclude:: ../../auto_examples/applications/plot_face_recognition.py

.. |prediction| image:: ../../images/plot_face_recognition_1.png
   :scale: 50

.. |eigenfaces| image:: ../../images/plot_face_recognition_2.png
   :scale: 50

.. list-table::
   :class: centered

   *

     - |prediction|

     - |eigenfaces|

   *

     - **Prediction**

     - **Eigenfaces**

数据集中前5名最有代表性样本的预期结果：

Expected results for the top 5 most represented people in the dataset::

                     precision    recall  f1-score   support

  Gerhard_Schroeder       0.91      0.75      0.82        28
    Donald_Rumsfeld       0.84      0.82      0.83        33
         Tony_Blair       0.65      0.82      0.73        34
       Colin_Powell       0.78      0.88      0.83        58
      George_W_Bush       0.93      0.86      0.90       129

        avg / total       0.86      0.84      0.85       282


Open problem: Stock Market Structure
=====================================

开放性问题: 股票市场结构
=====================================

我们可以预测Google在特定时间段内的股价变动吗？

Can we predict the variation in stock prices for Google over a given time frame?

:ref:`stock_market`
