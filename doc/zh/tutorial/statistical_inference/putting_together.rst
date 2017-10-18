=========================
把它们放在一起
=========================


模型管道化
============

我们已经知道一些模型可以做数据转换，一些模型可以用来预测变量。我们可以建立一个组合模型同时完成以上工作:

.. image:: /auto_examples/images/sphx_glr_plot_digits_pipe_001.png
   :target: ../../auto_examples/plot_digits_pipe.html
   :scale: 65
   :align: right

.. literalinclude:: ../../auto_examples/plot_digits_pipe.py
    :lines: 23-63

用特征面进行人脸识别
=================================

该实例用到的数据集来自 LFW_(Labeled Faces in the Wild)。数据已经进行了初步预处理
  
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

数据集中前5名最有代表性样本的预期结果::

                     precision    recall  f1-score   support

  Gerhard_Schroeder       0.91      0.75      0.82        28
    Donald_Rumsfeld       0.84      0.82      0.83        33
         Tony_Blair       0.65      0.82      0.73        34
       Colin_Powell       0.78      0.88      0.83        58
      George_W_Bush       0.93      0.86      0.90       129

        avg / total       0.86      0.84      0.85       282


开放性问题: 股票市场结构
=====================================

我们可以预测 Google 在特定时间段内的股价变动吗？

:ref:`stock_market`
