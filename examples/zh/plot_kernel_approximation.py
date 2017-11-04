#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
==================================================
RBF kernels 的显示特征图近似
==================================================

说明 RBF kernel 的特征图近似的示例。

.. currentmodule:: sklearn.kernel_approximation

它显示如何使用 :class:`RBFSampler` 和 :class:`Nystroem` 来近似 RBF kernel 的 feature map ，以便在数字数据集上使用 SVM 进行分类。 比较在原始空间中使用线性 SVM 的结果，使用 approximate mapping 并使用核心化 SVM 的线性 SVM 进行比较。
对于不同数量的 Monte Carlo 取样（在 :class:`RBFSampler` ）和用于 approximate mapping 的训练集（用于 :class:`Nystroem` ）的不同大小的子集的时间和精度所示。

请注意，这里的数据集不足以显示内核近似的好处，因为确切的 SVM 仍然相当快。

抽样更多的维度明显地能够得到更好的分类结果，但是成本相对来说会更高。这意味着由参数 n_components 给出的运行时间和精度之间存在折中效果。请注意，通过使用随机梯度下降 class:`sklearn.linear_model.SGDClassifier` ，可以通过这种方式大大加快求解线性 SVM 及近似 kernel SVM 。
在 kernelized SVM 的情况下，这是不容易的。

第二个图展示了 RBF kernel SVM 的决策面和具有 approximate kernel maps 的线性 SVM 。
该图显示了投影到数据的前两个主要分量上的分类器的决策面。 这种可视化应该采用 a grain of salt ，因为它只是一个通过决策表面在64维度的有趣的切片。 特别要注意的是，数据点（表示为点）不一定被分类到它所在的区域中，因为它不会位于前两个主要分量跨越的平面上。

:class:`RBFSampler` 和 :class:`Nystroem` 的使用，在 :ref:`kernel_approximation` 详细介绍了。

"""
print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
#         Andreas Mueller <amueller@ais.uni-bonn.de>
#         Joy yx <chinachenyyx@gmail.com>
# License: BSD 3 clause

# 导入 numpy, matplotlib 等需要使用的模块
import matplotlib.pyplot as plt
import numpy as np
from time import time

# 导入数据集，分类器和性能指标
from sklearn import datasets, svm, pipeline
from sklearn.kernel_approximation import (RBFSampler,
                                          Nystroem)
from sklearn.decomposition import PCA

# 导入 digits 数据集
digits = datasets.load_digits(n_class=9)

# 要对此数据应用分类器，我们需要 flatten images，以 (样本, 特征) 矩阵转换数据
n_samples = len(digits.data)
data = digits.data / 16.
# mean() 求取均值
data -= data.mean(axis=0)

# 我们在数字的前半部分学习数字，即 data_train 是数据集的前半部分
# targets_train 是数据集的标签的前半部分
data_train, targets_train = (data[:n_samples // 2],
                             digits.target[:n_samples // 2])


# 现在我们预测剩下的半部分数据集的标签
data_test, targets_test = (data[n_samples // 2:],
                           digits.target[n_samples // 2:])
# data_test = scaler.transform(data_test)

# 创建一个 svc 分类器
kernel_svm = svm.SVC(gamma=.2)
linear_svm = svm.LinearSVC()

# 从 kernel approximation 和 线性 SVM 中创建 pipeline
# RBFSampler() --- 通过其傅立叶变换的 Monte Carlo 近似的 近似 RBF kernel 的特征图。它实现了 Random Kitchen Sinks 的变体。
feature_map_fourier = RBFSampler(gamma=.2, random_state=1)
# Nystroem() --- 使用训练数据的子集近似一个 kernel map, 使用数据的子集作为基础构建任意 kernel 的近似特征图。
feature_map_nystroem = Nystroem(gamma=.2, random_state=1)
# pipeline.Pipeline() --- final estimator 的变换 pipeline 。pipeline 的目的是组装几个可以交叉验证的步骤，同时设置不同的参数。
# 详情参见：http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
fourier_approx_svm = pipeline.Pipeline([("feature_map", feature_map_fourier),
                                        ("svm", svm.LinearSVC())])

nystroem_approx_svm = pipeline.Pipeline([("feature_map", feature_map_nystroem),
                                        ("svm", svm.LinearSVC())])

# 使用 线性 svm 和 kernel svm 来拟合并预测

kernel_svm_time = time()
kernel_svm.fit(data_train, targets_train)
kernel_svm_score = kernel_svm.score(data_test, targets_test)
kernel_svm_time = time() - kernel_svm_time

linear_svm_time = time()
linear_svm.fit(data_train, targets_train)
linear_svm_score = linear_svm.score(data_test, targets_test)
linear_svm_time = time() - linear_svm_time

sample_sizes = 30 * np.arange(1, 10)
fourier_scores = []
nystroem_scores = []
fourier_times = []
nystroem_times = []

for D in sample_sizes:
    fourier_approx_svm.set_params(feature_map__n_components=D)
    nystroem_approx_svm.set_params(feature_map__n_components=D)
    start = time()
    nystroem_approx_svm.fit(data_train, targets_train)
    nystroem_times.append(time() - start)

    start = time()
    fourier_approx_svm.fit(data_train, targets_train)
    fourier_times.append(time() - start)

    fourier_score = fourier_approx_svm.score(data_test, targets_test)
    nystroem_score = nystroem_approx_svm.score(data_test, targets_test)
    nystroem_scores.append(nystroem_score)
    fourier_scores.append(fourier_score)

# 将预测的结果绘制出来
plt.figure(figsize=(8, 8))
accuracy = plt.subplot(211)
# 第二个 y 轴为时间
timescale = plt.subplot(212)

accuracy.plot(sample_sizes, nystroem_scores, label="Nystroem approx. kernel")
timescale.plot(sample_sizes, nystroem_times, '--',
               label='Nystroem approx. kernel')

accuracy.plot(sample_sizes, fourier_scores, label="Fourier approx. kernel")
timescale.plot(sample_sizes, fourier_times, '--',
               label='Fourier approx. kernel')

# 精确 rbf 和 线性 kernel 的水平线
accuracy.plot([sample_sizes[0], sample_sizes[-1]],
              [linear_svm_score, linear_svm_score], label="linear svm")
timescale.plot([sample_sizes[0], sample_sizes[-1]],
               [linear_svm_time, linear_svm_time], '--', label='linear svm')

accuracy.plot([sample_sizes[0], sample_sizes[-1]],
              [kernel_svm_score, kernel_svm_score], label="rbf svm")
timescale.plot([sample_sizes[0], sample_sizes[-1]],
               [kernel_svm_time, kernel_svm_time], '--', label='rbf svm')

# 数据集维数的垂直线 = 64
accuracy.plot([64, 64], [0.7, 1], label="n_features")

# 图例 和 标签
accuracy.set_title("Classification accuracy")
timescale.set_title("Training times")
accuracy.set_xlim(sample_sizes[0], sample_sizes[-1])
accuracy.set_xticks(())
accuracy.set_ylim(np.min(fourier_scores), 1)
timescale.set_xlabel("Sampling steps = transformed feature dimension")
accuracy.set_ylabel("Classification accuracy")
timescale.set_ylabel("Training time in seconds")
accuracy.legend(loc='best')
timescale.legend(loc='best')

# 可视化决策面，预测到数据集的前两个主要分量
pca = PCA(n_components=8).fit(data_train)

X = pca.transform(data_train)

# 沿着前两个主要分量生成网格
multiples = np.arange(-2, 2, 0.1)
# 沿着第一个分量的步骤
first = multiples[:, np.newaxis] * pca.components_[0, :]
# 沿着第二个分量的步骤
second = multiples[:, np.newaxis] * pca.components_[1, :]
# 组合
grid = first[np.newaxis, :, :] + second[:, np.newaxis, :]
flat_grid = grid.reshape(-1, data.shape[1])

# 加上绘制图上的 title
titles = ['SVC with rbf kernel',
          'SVC (linear kernel)\n with Fourier rbf feature map\n'
          'n_components=100',
          'SVC (linear kernel)\n with Nystroem rbf feature map\n'
          'n_components=100']

plt.tight_layout()
plt.figure(figsize=(12, 5))

# 预测并且绘制出来
for i, clf in enumerate((kernel_svm, nystroem_approx_svm,
                         fourier_approx_svm)):
    # 绘制决策边界。为此，我们将为网格 [x_min, x_max]x[y_min, y_max] 中的每个点分配一个颜色。
    plt.subplot(1, 3, i + 1)
    Z = clf.predict(flat_grid)

    # 将结果绘制到彩色图中
    Z = Z.reshape(grid.shape[:-1])
    plt.contourf(multiples, multiples, Z, cmap=plt.cm.Paired)
    plt.axis('off')

    # 绘制训练点
    plt.scatter(X[:, 0], X[:, 1], c=targets_train, cmap=plt.cm.Paired,
                edgecolors=(0, 0, 0))

    plt.title(titles[i])
plt.tight_layout()
plt.show()
