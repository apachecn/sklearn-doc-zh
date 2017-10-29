鸢尾花数据集
====================

注释
-----
数据集特征:
    :实例数量: 150 (三个类各有50个)
    :属性数量: 4 (数值型，数值型，帮助预测的属性和类)
    :Attribute Information:
        - sepal length 萼片长度（厘米）
        - sepal width 萼片宽度（厘米）
        - petal length 花瓣长度（厘米）
        - petal width 花瓣宽度（厘米）
        - class:
                - Iris-Setosa 山鸢尾
                - Iris-Versicolour 变色鸢尾
                - Iris-Virginica 维吉尼亚鸢尾
   
    :统计摘要:

    ============== ==== ==== ======= ===== ====================
                    Min  Max   Mean    SD    Class Correlation
    ============== ==== ==== ======= ===== ====================
    sepal length:   4.3  7.9   5.84   0.83    0.7826
    sepal width:    2.0  4.4   3.05   0.43   -0.4194
    petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
    petal width:    0.1  2.5   1.20  0.76     0.9565  (high!)
    ============== ==== ==== ======= ===== ====================

    :缺失属性值: 无
    :类别分布: 3个类别各占33.3%
    :创建者: R.A. Fisher
    :捐助者: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
    :日期: 1988年7月

这是UCI ML（欧文加利福尼亚大学 机器学习库）鸢尾花数据集的副本。
http://archive.ics.uci.edu/ml/datasets/Iris

著名的鸢尾花数据库，首先由R. Fisher先生使用。

这可能是在模式识别文献中最有名的数据库。Fisher的论文是这个领域的经典之作，到今天也经常被引用。（例如：Duda＆Hart）
数据集包含3个类，每类有50个实例，每个类指向一种类型的鸢尾花。一类与另外两类线性分离，而后者不能彼此线性分离。

参考资料
----------
   - Fisher,R.A. "The use of multiple measurements in taxonomic problems"
     Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to
     Mathematical Statistics" (John Wiley, NY, 1950).
   - Duda,R.O., & Hart,P.E. (1973) Pattern Classification and Scene Analysis.
     (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.
   - Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System
     Structure and Classification Rule for Recognition in Partially Exposed
     Environments".  IEEE Transactions on Pattern Analysis and Machine
     Intelligence, Vol. PAMI-2, No. 1, 67-71.
   - Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions
     on Information Theory, May 1972, 431-433.
   - See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II
     conceptual clustering system finds 3 classes in the data.
   - Many, many more ...
