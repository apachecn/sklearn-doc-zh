# -*- coding:UTF-8 -*-
"""
====================================
k-means 假设的示范
====================================

这个示例是为了说明这种情况，其中 k-means 会产生不直观和可能的意想不到的簇。在前 3 幅图中，输入数据不符合 k-means 产生的一些隐含假设，因此产生了不合需要的簇。在最后的一张图中， k-means 返回直观的簇，尽管它的大小并不均匀。
"""
print(__doc__)

# Author: Phil Roth <mr.phil.roth@gmail.com> & Joy yx <chinachenyyx@gmail.com>
# License: BSD 3 clause

# 引入 numpy 和 matplotlib
import numpy as np
import matplotlib.pyplot as plt

# 引入 sklearn 的聚类相关包 KMeans
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

plt.figure(figsize=(12, 12))

# make_blobs() 产生多类数据集，对每个类的中心和标准差有很好的控制。具体的请参见：http://blog.csdn.net/sa14023053/article/details/52086695
# n_samples -- int类型，是可选参数，默认值是100，表示 -- 总的点数，平均分到每个 cluster 中。实际上就是样本的个数。
# random_state -- RandomState对象或 None，是可选参数，默认值为 None，表示 -- 如果是 int, random_state作为随机数产生器的seed;如果是 RandomState对象，random_state 是随机数产生器; 如果是 None，RandomState对象是随机数产生器通过np.random来使用
# n_features 这个参数这里也没用到，但是还要说一下 -- int 类型 默认值为 2，表示 -- 每个样本的特征维度
# centers 这个这里虽然没用到，但是还要说一下 -- int类型或者 聚类中心坐标元组构成的数组类型，默认值为 3，表示 -- 产生的中心点的数量，或者固定中心点位置。
n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)

# 簇的个数不正确（因为我们生成的随机数的类别是 3 个类，但是咱们这里的聚类个数为 2） 进行一次 KMeans 聚类
y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)

# subplot(numRows, numCols, plotNum) -- 图表的整个绘图区域被分成 numRows 行和 numCols; plotNum 代表的是整个绘图区域的第几个，按照从左到右，从上到下的顺序。当这几个参数都是小于9的时候，是可以连起来的
# 下面这行代码的意思就是说，将整个绘图区域分为4等分，然后我们现在在第一个小区域里画图
plt.subplot(221)
# plt.scatter(x, y, s=20, c='b', marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, hold=None, **kwargs)
# x,y --- 形如 shape[x, y] 的 array, 代表输入数据
# c ----- 色彩或颜色序列，可选，默认。注意 c 不应该是一个单一的 RGB 数字或 RGBA 序列，因为不便区分。c 可以是一个 RGB 或 RGBA 二维行数组。
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
# 绘图小区域的 title
plt.title("Incorrect Number of Blobs")

# 不均匀分布的数据
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
# np.dot() 对于二维矩阵，计算真正意义上的矩阵乘积，同线性代数中矩阵乘法的定义。对于一维矩阵，计算两者的内积。这里是真正意义上的矩阵乘积。
X_aniso = np.dot(X, transformation)
# 进行不均匀分布的数据的 KMeans 的聚类处理
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)

# 在整体绘图区域的第一行的第二个小区域里作图
plt.subplot(222)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
plt.title("Anisotropicly Distributed Blobs")

# 不同的方差聚类
# cluster_std --- 聚类的每个类别的方差，例如我们希望生成 2 类数据，其中一类比另一类具有更大的方差，可以将 cluster_std 设置为 [1.0, 3.0]
X_varied, y_varied = make_blobs(n_samples=n_samples,
                                cluster_std=[1.0, 2.5, 0.5],
                                random_state=random_state)
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_varied)

# 在整个绘图区域的第三个小绘图区域绘制图形
plt.subplot(223)
plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
plt.title("Unequal Variance")

# 大小不一致的聚类
# np.vstack() 和 np.hstack() 是有合并数组的作用的，但是 vstack() 是在垂直方向上合并，而 hstack() 是在水平方向上合并
# 可以参考这里：http://blog.csdn.net/huruzun/article/details/39801217
X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
y_pred = KMeans(n_clusters=3,
                random_state=random_state).fit_predict(X_filtered)

# 在整个绘图区域的第四个小区域绘制图形
plt.subplot(224)
plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred)
plt.title("Unevenly Sized Blobs")

plt.show()
