波士顿房价数据集
===========================

注释
------
数据集特征:  

    :实例数量: 506 

    :属性数量: 13 数值型或类别型，帮助预测的属性
    
    :中位数（第14个属性）经常是学习目标

    :属性信息 (按顺序):
        - CRIM     城镇人均犯罪率
        - ZN       占地面积超过2.5万平方英尺的住宅用地比例
        - INDUS    城镇非零售业务地区的比例
        - CHAS     查尔斯河虚拟变量 (= 1 如果土地在河边；否则是0)
        - NOX      一氧化氮浓度（每1000万份）
        - RM       平均每居民房数
        - AGE      在1940年之前建成的所有者占用单位的比例
        - DIS      与五个波士顿就业中心的加权距离
        - RAD      辐射状公路的可达性指数
        - TAX      每10,000美元的全额物业税率
        - PTRATIO  城镇师生比例
        - B        1000(Bk - 0.63)^2 其中 Bk 是城镇的黑人比例
        - LSTAT    人口中地位较低人群的百分数
        - MEDV     以1000美元计算的自有住房的中位数

    :缺失属性值: 无

    :创建者: Harrison, D. and Rubinfeld, D.L.

这是UCI ML（欧文加利福尼亚大学 机器学习库）房价数据集的副本。
http://archive.ics.uci.edu/ml/datasets/Housing


该数据集是从位于卡内基梅隆大学维护的StatLib图书馆取得的。

Harrison, D. 和 Rubinfeld, D.L. 的波士顿房价数据：'Hedonic 
prices and the demand for clean air', J. Environ. Economics & Management,
vol.5, 81-102, 1978，也被使用在 Belsley, Kuh & Welsch 的 'Regression diagnostics
...', Wiley, 1980。
注释：许多变化已经被应用在后者第244-261页的表中。

波士顿房价数据已被用于许多涉及回归问题的机器学习论文中。
     
**参考资料**

   - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
   - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.
   - many more! (see http://archive.ics.uci.edu/ml/datasets/Housing)
