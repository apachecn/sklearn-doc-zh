光学识别手写数字数据集
===================================================

注释
-----
数据集特征：
    :实例数量: 5620
    :属性数量: 64
    :属性信息: 8x8 范围在（0-16）的整型像素值图片
    :缺失属性值: 无
    :创建者: E. Alpaydin (alpaydin@boun.edu.tr)
    :日期: 1998年7月

这是UCI ML（欧文加利福尼亚大学 机器学习库）手写数字数据集的测试集的副本。
http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits

数据集包含手写数字的图像：10个类别中每个类都是一个数字。

从预印表格中提取手写数字的标准化的位图这一过程，应用了NIST提供的预处理程序。
这些数据是从43人中得到，其中30人为训练集，13人为测试集。32x32位图被划分为4x4的非重叠块，
并且在每个块中计数像素数。这产生8×8的输入矩阵，其中每个元素是0-16范围内的整数。
这个过程降低了维度，并且在小的变形中提供了不变性。

有关NIST处理程序的信息，请参见 M. D. Garris, J. L. Blue, G.T. Candela, 
D. L. Dimmick, J. Geist, P. J. Grother, S. A. Janet, and C.
L. Wilson, NIST Form-Based Handprint Recognition System, NISTIR 5469,
1994.

参考资料
----------
  - C. Kaynak (1995) Methods of Combining Multiple Classifiers and Their
    Applications to Handwritten Digit Recognition, MSc Thesis, Institute of
    Graduate Studies in Science and Engineering, Bogazici University.
  - E. Alpaydin, C. Kaynak (1998) Cascading Classifiers, Kybernetika.
  - Ken Tang and Ponnuthurai N. Suganthan and Xi Yao and A. Kai Qin.
    Linear dimensionalityreduction using relevance weighted LDA. School of
    Electrical and Electronic Engineering Nanyang Technological University.
    2005.
  - Claudio Gentile. A New Approximate Maximal Margin Classification
    Algorithm. NIPS. 2000.
