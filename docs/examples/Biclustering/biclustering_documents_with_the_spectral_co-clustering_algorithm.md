
# 使用频谱共聚算法对文档进行聚合

> 翻译者：[@N!no](https://github.com/lovelybuggies) 
> 校验者：待校验

这个例子演示了20个新闻组数据集上的光谱协聚类算法。‘comp.os.ms-windows.misc’ 类别被排除在外，因为它包含许多只包含数据的帖子。

TF-IDF 矢量帖构成一个词频矩阵，然后使用 Dhillon 光谱协聚算法对其进行重组。由此产生的文档词双聚类表明在这些子集文档中被使用频率更高的子集词。

对于一些最好的双聚类来说，它最常见的文档类别和十个最重要的单词会被打印出来。最佳双类别由其归一化的切割决定。最好的单词是通过比较它们在两区内和两区外的总和来确定的。

为了进行比较，我们还使用 MiniBatchKMeans 对文档进行集群。从双聚类衍生出的文档聚类比使用 MiniBatchKMeans 得到的聚类具有更好的 V-measure。

```
Vectorizing...
Coclustering...
Done in 2.75s. V-measure: 0.4387
MiniBatchKMeans...
Done in 5.69s. V-measure: 0.3344

Best biclusters:
----------------
bicluster 0 : 1829 documents, 2524 words
categories   : 22% comp.sys.ibm.pc.hardware, 19% comp.sys.mac.hardware, 18% comp.graphics
words        : card, pc, ram, drive, bus, mac, motherboard, port, windows, floppy

bicluster 1 : 2391 documents, 3275 words
categories   : 18% rec.motorcycles, 17% rec.autos, 15% sci.electronics
words        : bike, engine, car, dod, bmw, honda, oil, motorcycle, behanna, ysu

bicluster 2 : 1887 documents, 4232 words
categories   : 23% talk.politics.guns, 19% talk.politics.misc, 13% sci.med
words        : gun, guns, firearms, geb, drugs, banks, dyer, amendment, clinton, cdt

bicluster 3 : 1146 documents, 3263 words
categories   : 29% talk.politics.mideast, 26% soc.religion.christian, 25% alt.atheism
words        : god, jesus, christians, atheists, kent, sin, morality, belief, resurrection, marriage

bicluster 4 : 1732 documents, 3967 words
categories   : 26% sci.crypt, 23% sci.space, 17% sci.med
words        : clipper, encryption, key, escrow, nsa, crypto, keys, intercon, secure, wiretap
```


```python
from collections import defaultdict
import operator
from time import time

import numpy as np

from sklearn.cluster import SpectralCoclustering
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.cluster import v_measure_score

print(__doc__)


def number_normalizer(tokens):
    """ 将所有数字标记映射到占位符。

    对于许多应用程序来说，以数字开头的令牌并没有直接的用处，但是这样的令牌存在的事实可能是相关的。通过应用这种降维形式，一些方法可能会表现得更好。
    """
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)


class NumberNormalizingVectorizer(TfidfVectorizer):
    def build_tokenizer(self):
        tokenize = super().build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))


# 不包含 'comp.os.ms-windows.misc' 类别
categories = ['alt.atheism', 'comp.graphics',
              'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
              'comp.windows.x', 'misc.forsale', 'rec.autos',
              'rec.motorcycles', 'rec.sport.baseball',
              'rec.sport.hockey', 'sci.crypt', 'sci.electronics',
              'sci.med', 'sci.space', 'soc.religion.christian',
              'talk.politics.guns', 'talk.politics.mideast',
              'talk.politics.misc', 'talk.religion.misc']
newsgroups = fetch_20newsgroups(categories=categories)
y_true = newsgroups.target

vectorizer = NumberNormalizingVectorizer(stop_words='english', min_df=5)
cocluster = SpectralCoclustering(n_clusters=len(categories),
                                 svd_method='arpack', random_state=0)
kmeans = MiniBatchKMeans(n_clusters=len(categories), batch_size=20000,
                         random_state=0)

print("Vectorizing...")
X = vectorizer.fit_transform(newsgroups.data)

print("Coclustering...")
start_time = time()
cocluster.fit(X)
y_cocluster = cocluster.row_labels_
print("Done in {:.2f}s. V-measure: {:.4f}".format(
    time() - start_time,
    v_measure_score(y_cocluster, y_true)))

print("MiniBatchKMeans...")
start_time = time()
y_kmeans = kmeans.fit_predict(X)
print("Done in {:.2f}s. V-measure: {:.4f}".format(
    time() - start_time,
    v_measure_score(y_kmeans, y_true)))

feature_names = vectorizer.get_feature_names()
document_names = list(newsgroups.target_names[i] for i in newsgroups.target)


def bicluster_ncut(i):
    rows, cols = cocluster.get_indices(i)
    if not (np.any(rows) and np.any(cols)):
        import sys
        return sys.float_info.max
    row_complement = np.nonzero(np.logical_not(cocluster.rows_[i]))[0]
    col_complement = np.nonzero(np.logical_not(cocluster.columns_[i]))[0]
    # 注意：接下来的操作等同于 X[rows[:, np.newaxis], cols].sum() 
    #      但是会针对于 scipy <= 0.16 的版本更快一些
    weight = X[rows][:, cols].sum()
    cut = (X[row_complement][:, cols].sum() +
           X[rows][:, col_complement].sum())
    return cut / weight


def most_common(d):
    """默认字典有最大值的项。

    在 Python >= 2.7 中类似于 Counter.most_common 。
    """
    return sorted(d.items(), key=operator.itemgetter(1), reverse=True)


bicluster_ncuts = list(bicluster_ncut(i)
                       for i in range(len(newsgroups.target_names)))
best_idx = np.argsort(bicluster_ncuts)[:5]

print()
print("Best biclusters:")
print("----------------")
for idx, cluster in enumerate(best_idx):
    n_rows, n_cols = cocluster.get_shape(cluster)
    cluster_docs, cluster_words = cocluster.get_indices(cluster)
    if not len(cluster_docs) or not len(cluster_words):
        continue

    # 种类
    counter = defaultdict(int)
    for i in cluster_docs:
        counter[document_names[i]] += 1
    cat_string = ", ".join("{:.0f}% {}".format(float(c) / n_rows * 100, name)
                           for name, c in most_common(counter)[:3])

    # 单词
    out_of_cluster_docs = cocluster.row_labels_ != cluster
    out_of_cluster_docs = np.where(out_of_cluster_docs)[0]
    word_col = X[:, cluster_words]
    word_scores = np.array(word_col[cluster_docs, :].sum(axis=0) -
                           word_col[out_of_cluster_docs, :].sum(axis=0))
    word_scores = word_scores.ravel()
    important_words = list(feature_names[cluster_words[i]]
                           for i in word_scores.argsort()[:-11:-1])

    print("bicluster {} : {} documents, {} words".format(
        idx, n_rows, n_cols))
    print("categories   : {}".format(cat_string))
    print("words        : {}\n".format(', '.join(important_words)))
```

