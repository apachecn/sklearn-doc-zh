威斯康辛州乳腺癌（诊断）数据库
=============================================

注释
-----
数据集特征：
    :实例数量: 569

    :属性数量: 30 (数值型，帮助预测的属性和类)

    :Attribute Information:
        - radius 半径（从中心到边缘上点的距离的平均值）
        - texture 纹理（灰度值的标准偏差）
        - perimeter 周长
        - area 区域
        - smoothness 平滑度（半径长度的局部变化）
        - compactness 紧凑度（周长 ^ 2 /面积 - 1.0）
        - concavity 凹面（轮廓的凹部的严重性）
        - concave points 凹点（轮廓的凹部的数量）
        - symmetry 对称性
        - fractal dimension 分形维数（海岸线近似 - 1）
        - 类:
                - WDBC-Malignant 恶性
                - WDBC-Benign 良性
                
        对每个图像计算这些特征的平均值，标准误差，以及“最差”（因为是肿瘤）或最大值（最大的前三个值的平均值）
        得到30个特征。例如，字段 3 是平均半径，字段 13 是半径的标准误差，字段 23 是最差半径。
                
   :统计摘要:

    ===================================== ====== ======
                                           Min    Max
    ===================================== ====== ======
    radius (mean):                        6.981  28.11
    texture (mean):                       9.71   39.28
    perimeter (mean):                     43.79  188.5
    area (mean):                          143.5  2501.0
    smoothness (mean):                    0.053  0.163
    compactness (mean):                   0.019  0.345
    concavity (mean):                     0.0    0.427
    concave points (mean):                0.0    0.201
    symmetry (mean):                      0.106  0.304
    fractal dimension (mean):             0.05   0.097
    radius (standard error):              0.112  2.873
    texture (standard error):             0.36   4.885
    perimeter (standard error):           0.757  21.98
    area (standard error):                6.802  542.2
    smoothness (standard error):          0.002  0.031
    compactness (standard error):         0.002  0.135
    concavity (standard error):           0.0    0.396
    concave points (standard error):      0.0    0.053
    symmetry (standard error):            0.008  0.079
    fractal dimension (standard error):   0.001  0.03
    radius (worst):                       7.93   36.04
    texture (worst):                      12.02  49.54
    perimeter (worst):                    50.41  251.2
    area (worst):                         185.2  4254.0
    smoothness (worst):                   0.071  0.223
    compactness (worst):                  0.027  1.058
    concavity (worst):                    0.0    1.252
    concave points (worst):               0.0    0.291
    symmetry (worst):                     0.156  0.664
    fractal dimension (worst):            0.055  0.208
    ===================================== ====== ======

    :缺失属性值: 无

    :类别分布: 212 - 恶性, 357 - 良性

    :创建者:  Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian

    :捐助者: Nick Street

    :日期: 1995年11月

这是UCI ML（欧文加利福尼亚大学 机器学习库）威斯康星州乳腺癌（诊断）数据集的副本。
https://goo.gl/U2Uwz2

这些特征是从乳房肿块的细针抽吸术（FNA）的数字图像中计算得到，描述了图像中存在的细胞核的特征。

上述的分离平面是由多表面方法树（MSM-T）[K.P.Bennett, "Decision Tree Construction Via 
Linear Programming." Proceedings of the 4th Midwest Artificial Intelligence and 
Cognitive Science Society, pp.97-101, 1992], a classification method which uses 
linear programming to construct a decision tree.  
相关特征是在1-4的特征和1-3的分离平面中使用穷举法搜索选取出的。

用于分离平面的线性规划在三维空间中描述如下：
[K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination 
of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].

该数据库也可通过UW CS ftp服务器获得：

ftp ftp.cs.wisc.edu
cd math-prog/cpo-dataset/machine-learn/WDBC/

参考资料
----------
   - W.N. Street, W.H. Wolberg and O.L. Mangasarian. Nuclear feature extraction 
     for breast tumor diagnosis. IS&T/SPIE 1993 International Symposium on 
     Electronic Imaging: Science and Technology, volume 1905, pages 861-870,
     San Jose, CA, 1993.
   - O.L. Mangasarian, W.N. Street and W.H. Wolberg. Breast cancer diagnosis and 
     prognosis via linear programming. Operations Research, 43(4), pages 570-577, 
     July-August 1995.
   - W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Machine learning techniques
     to diagnose breast cancer from fine-needle aspirates. Cancer Letters 77 (1994) 
     163-171.
