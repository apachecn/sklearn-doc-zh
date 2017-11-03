#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=====================
分类比较
=====================

scikit-learn 中的几个分类器在合成数据集上的比较。
这个例子的目的是说明不同分类器的决策边界的性质。
这仿佛是大海中的一滴水，因为这些例子所传达的直觉不一定会转移到真正的数据集上。

特别是在高维空间中，数据可以更容易地线性分离，诸如朴素贝叶斯和线性 SVM 之类的分类器的简单性可能导致比其他分类器更好的泛化。

绘图显示了纯色和测试点半透明的训练点。右下方显示了测试仪的分类精度。
"""
print(__doc__)


# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler, Joy yx
# License: BSD 3 clause

# 导入一些必要的模块
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

h = .02  # 网格中的步长

# 各分类器对应的名称
# 各分类器为 1.k-近邻，也就是 kNN 2. 线性支持向量机 3. 带有 RBF 核的 SVM 4.高斯过程 5.决策树 6.随机森林 7.神经网络 8.集成方法 AdaBoost 9.朴素贝叶斯 10.二次判别分析
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

# 具体的分类器参数和调用
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

# make_classification() 产生一份分类数据，参数如下：
# make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)
# n_samples --- 样本的条数, n_features --- 所有特征的个数, 其中包括 n_informative 个信息特征，n_redundant 个冗余特征，n_repeated 个重复特征，以及随机绘制的 n_features-n_informative-n_redundant- n_repeated 无用特征。
# n_classes --- 分类问题的类（或标签）的数量, n_clusters_per_class --- 每个类的簇数, weights --- 分配给每个类的样本的比例。如果没有，那么类之间是平衡的。请注意，如果 len(weights) == n_classes-1, 则自动推断最后一个类的权重。如果权重之和超过1，则可以返回 n_samples 个样本。
# flip_y --- 随机交换类的样本部分。较大的值会在标签中引入噪声，使分类任务更加困难。 class_sep --- factor 乘以 hypercube size, 较大的值扩展了簇/类，并使分类任务更加容易。 hypercube --- 如果为 true, 则将聚类放在 hypercube 的顶点上。如果为 false, 则将聚类放在随即多面体上。
# shift --- 按特定的值移动特征。如果没有，则特征被移动在 [-class_sep, class_sep] 中绘制的随机值。 scale --- 将特征乘以特定值。如果没有，则按照 [1, 100] 中绘制的随机值对特征进行缩放。请注意，缩放发生在移位之后。
# shuffle --- 将 samples 以及 features 分别 shuffle。 random_state --- 如果是 int, random_state 是随机数生成器使用的种子; 如果是 RandomState 的实例，random_state 是随机数生成器; 如果没有，随机数生成器是由 np.random 使用的 RandomState 实例。
# 再具体的，请看英文文档：http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
# numpy.random.uniform(low,high,size) 从一个均匀分布 [low,high) 中随机采样，注意定义域是左闭右开，即包含 low，不包含 high。参数如下：
# low: 采样下界，float类型，默认值为0；
# high: 采样上界，float类型，默认值为1；
# size: 输出样本数目，为int或元组(tuple)类型，例如，size=(m,n,k), 则输出 m*n*k 个样本，缺省时输出1个值。
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]

# plt.figure() 新建绘画窗口，独立显示绘画的图片
figure = plt.figure(figsize=(27, 9))
i = 1
# 迭代数据集
for ds_cnt, ds in enumerate(datasets):
    # 预处理数据集，分为训练和测试部分
    X, y = ds
    # StandardScaler() 计算训练集的平均值和标准差，以便测试数据集使用相同的变换.具体的参见：http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    # fit_transform() 先拟合数据，然后转化它将其转化为标准形式.
    # train_test_split() 随机划分训练集和测试集
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    # [X,Y] = meshgrid(x,y) 将向量x和y定义的区域转换成矩阵X和Y，这两个矩阵可以用来表示mesh和surf的三维空间点以及两个变量的赋值。其中矩阵X的行向量是向量x的简单复制，而矩阵Y的列向量是向量y的简单复制。
    # 详情参见：https://docs.scipy.org/doc/numpy/reference/generated/numpy.meshgrid.html
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 那我们就首先将数据集绘制出来
    cm = plt.cm.RdBu
    # ListedColormap() 是一个以参数中的列出的颜色来映射的函数。
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    # subplot() 在绘图区域的子区域中画图
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        # 设置 title
        ax.set_title("Input data")
    # 绘制训练集中的点
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # 绘制测试集中的点
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    # set_xlim() 设置 x 轴范围
    ax.set_xlim(xx.min(), xx.max())
    # set_ylim() 设置 y 轴范围
    ax.set_ylim(yy.min(), yy.max())
    # set_xticks 设置 x 轴坐标点
    ax.set_xticks(())
    # set_yticks() 设置 y 轴坐标点
    ax.set_yticks(())
    i += 1

    # 迭代上面我们列出来的几个分类器
    # zip() 函数接受任意多个（包括0个或1个）序列作为参数，返回一个 tuple 列表。具体请参见：http://www.cnblogs.com/frydsh/archive/2012/07/10/2585370.html
    for name, clf in zip(names, classifiers):
        # 为每个分类器分配一个画图的小的子绘图区域
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        # 拟合分类器
        clf.fit(X_train, y_train)
        # 计算每个分类器的分数
        score = clf.score(X_test, y_test)

        # 绘制决策边界。为此，我们将为网格 [x_min, x_max]x[y_min, y_max] 中的每个点分配一个颜色。
        if hasattr(clf, "decision_function"):
            # decision_function() 计算点到决策边界的函数间隔
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # 将结果放入彩色图中
        Z = Z.reshape(xx.shape)
        # contourf() 绘制等高线
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # 绘制训练集中的数据点
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # 也将测试点绘制出来
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

# tight_layout() 自动调整子图参数以给定指定的填充
plt.tight_layout()
plt.show()
