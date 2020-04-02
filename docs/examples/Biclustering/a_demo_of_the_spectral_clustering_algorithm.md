
# 频谱双聚类算法演示

> 翻译者：[@N!no](https://github.com/lovelybuggies) 
> 校验者：待校验

这个例子演示了如何使用光谱聚类算法生成棋盘数据集并对其进行聚类处理。

数据是用`make_checkerboard`函数生成的，然后打乱顺序并传递给光谱双聚类算法。变换后的矩阵的行和列被重新排列，以显示该算法找到的双聚类。

行和列标签向量的外积表示棋盘结构。

![png](https://scikit-learn.org/stable/_images/sphx_glr_plot_spectral_biclustering_001.png)

![png](https://scikit-learn.org/stable/_images/sphx_glr_plot_spectral_biclustering_002.png)

![png](https://scikit-learn.org/stable/_images/sphx_glr_plot_spectral_biclustering_003.png)

![png](https://scikit-learn.org/stable/_images/sphx_glr_plot_spectral_biclustering_004.png)

```
consensus score: 1.0
```


```python
print(__doc__)

# Author: Kemal Eren <kemal@kemaleren.com>
# License: BSD 3 clause

import numpy as np
from matplotlib import pyplot as plt

from sklearn.datasets import make_checkerboard
from sklearn.cluster import SpectralBiclustering
from sklearn.metrics import consensus_score


n_clusters = (4, 3)
data, rows, columns = make_checkerboard(
    shape=(300, 300), n_clusters=n_clusters, noise=10,
    shuffle=False, random_state=0)

plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Original dataset")

# 打乱聚类顺序
rng = np.random.RandomState(0)
row_idx = rng.permutation(data.shape[0])
col_idx = rng.permutation(data.shape[1])
data = data[row_idx][:, col_idx]

plt.matshow(data, cmap=plt.cm.Blues)
plt.title("Shuffled dataset")

model = SpectralBiclustering(n_clusters=n_clusters, method='log',
                             random_state=0)
model.fit(data)
score = consensus_score(model.biclusters_,
                        (rows[:, row_idx], columns[:, col_idx]))

print("consensus score: {:.1f}".format(score))

fit_data = data[np.argsort(model.row_labels_)]
fit_data = fit_data[:, np.argsort(model.column_labels_)]

plt.matshow(fit_data, cmap=plt.cm.Blues)
plt.title("After biclustering; rearranged to show biclusters")

plt.matshow(np.outer(np.sort(model.row_labels_) + 1,
                     np.sort(model.column_labels_) + 1),
            cmap=plt.cm.Blues)
plt.title("Checkerboard structure of rearranged data")

plt.show()
```

