
# 分类特征稀疏的文本

>翻译者:[@Loopy](https://github.com/loopyme)        
>校验者:[@barrycg](https://github.com/barrycg)

这个例子展示了如何使用scikit-learn中的单词包方法，根据主题对文档进行分类。本例使用scipy.sparse中的矩阵来存储特征，并演示各种能够有效处理稀疏矩阵的分类器。

本例中使用的数据集是20条新闻组数据集。通过scikit-learn可以自动下载该数据集，并进行缓存。

下述条形图展示了各个不同分类器，其信息包括精度、训练时间(已归一化)和测试时间(已归一化)。


```python
import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
```


```python
# 在stdout上显示进度日志
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
```


```python
# 解析命令行参数
op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--all_categories",
              action="store_true", dest="all_categories",
              help="Whether to use all categories or not.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
op.add_option("--filtered",
              action="store_true",
              help="Remove newsgroup information that is easily overfit: "
                   "headers, signatures, and quoting.")
```




    <Option at 0x7febca4f9320: --filtered>




```python
def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')
```


```python
# Jupyter notebook上的运行方法
argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()
print()
```

    Automatically created module for IPython interactive environment
    Usage: ipykernel_launcher.py [options]

    Options:
      -h, --help            show this help message and exit
      --report              Print a detailed classification report.
      --chi2_select=SELECT_CHI2
                            Select some number of features using a chi-squared
                            test
      --confusion_matrix    Print the confusion matrix.
      --top10               Print ten most discriminative terms per class for
                            every classifier.
      --all_categories      Whether to use all categories or not.
      --use_hashing         Use a hashing vectorizer.
      --n_features=N_FEATURES
                            n_features when using the hashing vectorizer.
      --filtered            Remove newsgroup information that is easily overfit:
                            headers, signatures, and quoting.




```python
# 从训练集中加载一些类别
if opts.all_categories:
    categories = None
else:
    categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]
```


```python
if opts.filtered:
    remove = ('headers', 'footers', 'quotes')
else:
    remove = ()

print("Loading 20 newsgroups dataset for categories:")
print(categories if categories else "all")
```

    Loading 20 newsgroups dataset for categories:
    ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']



```python
# 下载数据集
data_train = fetch_20newsgroups(subset='train', categories=categories,shuffle=True, random_state=42,remove=remove)
data_test = fetch_20newsgroups(subset='test', categories=categories,shuffle=True, random_state=42,remove=remove)
```


```python
# target_names中的标签顺序可以与categories中的不同
target_names = data_train.target_names

def size_mb(docs):
    return sum(len(s.encode('utf-8')) for s in docs) / 1e6

data_train_size_mb = size_mb(data_train.data)
data_test_size_mb = size_mb(data_test.data)

print("%d documents - %0.3fMB (training set)" % (
    len(data_train.data), data_train_size_mb))
print("%d documents - %0.3fMB (test set)" % (
    len(data_test.data), data_test_size_mb))
print("%d categories" % len(target_names))
```

    2034 documents - 3.980MB (training set)
    1353 documents - 2.867MB (test set)
    4 categories



```python
# 划分测试,训练集
y_train, y_test = data_train.target, data_test.target
```


```python
print("使用稀疏向量机从训练数据中提取特征")
t0 = time()
if opts.use_hashing:
    vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False,
                                   n_features=opts.n_features)
    X_train = vectorizer.transform(data_train.data)
else:
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    X_train = vectorizer.fit_transform(data_train.data)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_train.shape)
```

    使用稀疏向量机从训练数据中提取特征
    done in 0.476004s at 8.360MB/s
    n_samples: 2034, n_features: 33809



```python
print("使用相同的矢量化器从测试数据中提取特征")
t0 = time()
X_test = vectorizer.transform(data_test.data)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_test.shape)
```

    使用相同的矢量化器从测试数据中提取特征
    done in 0.311447s at 9.207MB/s
    n_samples: 1353, n_features: 33809



```python
# 从整数的特征名称映射到原始的token字符串
if opts.use_hashing:
    feature_names = None
else:
    feature_names = vectorizer.get_feature_names()
```


```python
if opts.select_chi2:
    print("使用卡方检验提取 %d 个特征" %
          opts.select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k=opts.select_chi2)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    if feature_names:
        # keep selected feature names
        feature_names = [feature_names[i] for i
                         in ch2.get_support(indices=True)]
    print("done in %fs" % (time() - t0))
```


```python
if feature_names:
    feature_names = np.asarray(feature_names)
```


```python
# 修剪字符串以适应终端(假设显示80列)
def trim(s):
    return s if len(s) <= 80 else s[:77] + "..."
```

## 基准分类器


```python
def benchmark(clf):
    print('_' * 80)
    print("训练: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("训练时间: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("最佳时间:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("准确率:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("维数: %d" % clf.coef_.shape[1])
        print("密度: %f" % density(clf.coef_))

        if opts.print_top10 and feature_names is not None:
            print("每个类的前十个词:")
            for i, label in enumerate(target_names):
                top10 = np.argsort(clf.coef_[i])[-10:]
                print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
        print()

    if opts.print_report:
        print("分类报告:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=target_names))

    if opts.print_cm:
        print("混淆矩阵:")
        print(metrics.confusion_matrix(y_test, pred))

    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time
```


```python
results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="sag"), "岭分类器"),
        (Perceptron(max_iter=50, tol=1e-3), "感知器"),
        (PassiveAggressiveClassifier(max_iter=50, tol=1e-3),
         "PAC分类器"),
        (KNeighborsClassifier(n_neighbors=10), "K近邻"),
        (RandomForestClassifier(n_estimators=100), "随机森林")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))
```

    ================================================================================
    岭分类器
    ________________________________________________________________________________
    训练:
    RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
                    max_iter=None, normalize=False, random_state=None, solver='sag',
                    tol=0.01)
    训练时间: 0.202s
    最佳时间:  0.002s
    准确率:   0.897
    维数: 33809
    密度: 1.000000

    ================================================================================
    感知器
    ________________________________________________________________________________
    训练:
    Perceptron(alpha=0.0001, class_weight=None, early_stopping=False, eta0=1.0,
               fit_intercept=True, max_iter=50, n_iter_no_change=5, n_jobs=None,
               penalty=None, random_state=0, shuffle=True, tol=0.001,
               validation_fraction=0.1, verbose=0, warm_start=False)
    训练时间: 0.030s
    最佳时间:  0.003s
    准确率:   0.888
    维数: 33809
    密度: 0.255302

    ================================================================================
    PAC分类器
    ________________________________________________________________________________
    训练:
    PassiveAggressiveClassifier(C=1.0, average=False, class_weight=None,
                                early_stopping=False, fit_intercept=True,
                                loss='hinge', max_iter=50, n_iter_no_change=5,
                                n_jobs=None, random_state=None, shuffle=True,
                                tol=0.001, validation_fraction=0.1, verbose=0,
                                warm_start=False)
    训练时间: 0.063s
    最佳时间:  0.003s
    准确率:   0.902
    维数: 33809
    密度: 0.700487

    ================================================================================
    K近邻
    ________________________________________________________________________________
    训练:
    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=None, n_neighbors=10, p=2,
                         weights='uniform')
    训练时间: 0.002s
    最佳时间:  0.235s
    准确率:   0.858
    ================================================================================
    随机森林
    ________________________________________________________________________________
    训练:
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)
    训练时间: 1.752s
    最佳时间:  0.084s
    准确率:   0.822



```python
for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s 罚项" % penalty.upper())
    # 训练Liblinear模型
    results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                       tol=1e-3)))

    # 训练SGD model模型
    results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=50,
                                           penalty=penalty)))
```

    ================================================================================
    L2 罚项
    ________________________________________________________________________________
    训练:
    LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, loss='squared_hinge', max_iter=1000,
              multi_class='ovr', penalty='l2', random_state=None, tol=0.001,
              verbose=0)
    训练时间: 0.274s
    最佳时间:  0.003s
    准确率:   0.900
    维数: 33809
    密度: 1.000000

    ________________________________________________________________________________
    训练:
    SGDClassifier(alpha=0.0001, average=False, class_weight=None,
                  early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
                  l1_ratio=0.15, learning_rate='optimal', loss='hinge', max_iter=50,
                  n_iter_no_change=5, n_jobs=None, penalty='l2', power_t=0.5,
                  random_state=None, shuffle=True, tol=0.001,
                  validation_fraction=0.1, verbose=0, warm_start=False)
    训练时间: 0.050s
    最佳时间:  0.003s
    准确率:   0.899
    维数: 33809
    密度: 0.573353

    ================================================================================
    L1 罚项
    ________________________________________________________________________________
    训练:
    LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, loss='squared_hinge', max_iter=1000,
              multi_class='ovr', penalty='l1', random_state=None, tol=0.001,
              verbose=0)
    训练时间: 0.257s
    最佳时间:  0.002s
    准确率:   0.873
    维数: 33809
    密度: 0.005568

    ________________________________________________________________________________
    训练:
    SGDClassifier(alpha=0.0001, average=False, class_weight=None,
                  early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
                  l1_ratio=0.15, learning_rate='optimal', loss='hinge', max_iter=50,
                  n_iter_no_change=5, n_jobs=None, penalty='l1', power_t=0.5,
                  random_state=None, shuffle=True, tol=0.001,
                  validation_fraction=0.1, verbose=0, warm_start=False)
    训练时间: 0.187s
    最佳时间:  0.003s
    准确率:   0.882
    维数: 33809
    密度: 0.023049




```python
# 训练带弹性网络(Elastic Net)罚项的SGD模型
print('=' * 80)
print("弹性网络(Elastic Net)罚项")
results.append(benchmark(SGDClassifier(alpha=.0001, max_iter=50,
                                       penalty="elasticnet")))
```

    ================================================================================
    弹性网络(Elastic Net)罚项
    ________________________________________________________________________________
    训练:
    SGDClassifier(alpha=0.0001, average=False, class_weight=None,
                  early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
                  l1_ratio=0.15, learning_rate='optimal', loss='hinge', max_iter=50,
                  n_iter_no_change=5, n_jobs=None, penalty='elasticnet',
                  power_t=0.5, random_state=None, shuffle=True, tol=0.001,
                  validation_fraction=0.1, verbose=0, warm_start=False)
    训练时间: 0.295s
    最佳时间:  0.003s
    准确率:   0.897
    维数: 33809
    密度: 0.185956




```python
# 训练不带阈值的Rocchio分类器
print('=' * 80)
print("不带阈值的Rocchio分类器")
results.append(benchmark(NearestCentroid()))
```

    ================================================================================
    不带阈值的Rocchio分类器
    ________________________________________________________________________________
    训练:
    NearestCentroid(metric='euclidean', shrink_threshold=None)
    训练时间: 0.007s
    最佳时间:  0.002s
    准确率:   0.855



```python
# 训练稀疏朴素贝叶斯分类器
print('=' * 80)
print("稀疏朴素贝叶斯分类器")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))
results.append(benchmark(ComplementNB(alpha=.1)))
```

    ================================================================================
    稀疏朴素贝叶斯分类器
    ________________________________________________________________________________
    训练:
    MultinomialNB(alpha=0.01, class_prior=None, fit_prior=True)
    训练时间: 0.007s
    最佳时间:  0.003s
    准确率:   0.899
    维数: 33809
    密度: 1.000000

    ________________________________________________________________________________
    训练:
    BernoulliNB(alpha=0.01, binarize=0.0, class_prior=None, fit_prior=True)
    训练时间: 0.010s
    最佳时间:  0.008s
    准确率:   0.884
    维数: 33809
    密度: 1.000000

    ________________________________________________________________________________
    训练:
    ComplementNB(alpha=0.1, class_prior=None, fit_prior=True, norm=False)
    训练时间: 0.007s
    最佳时间:  0.002s
    准确率:   0.911
    维数: 33809
    密度: 1.000000




```python
print('=' * 80)
print("基于l1的特征选择的LinearSVC")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
results.append(benchmark(Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                  tol=1e-3))),
  ('classification', LinearSVC(penalty="l2"))])))
```

    ================================================================================
    基于l1的特征选择的LinearSVC
    ________________________________________________________________________________
    训练:
    Pipeline(memory=None,
             steps=[('feature_selection',
                     SelectFromModel(estimator=LinearSVC(C=1.0, class_weight=None,
                                                         dual=False,
                                                         fit_intercept=True,
                                                         intercept_scaling=1,
                                                         loss='squared_hinge',
                                                         max_iter=1000,
                                                         multi_class='ovr',
                                                         penalty='l1',
                                                         random_state=None,
                                                         tol=0.001, verbose=0),
                                     max_features=None, norm_order=1, prefit=False,
                                     threshold=None)),
                    ('classification',
                     LinearSVC(C=1.0, class_weight=None, dual=True,
                               fit_intercept=True, intercept_scaling=1,
                               loss='squared_hinge', max_iter=1000,
                               multi_class='ovr', penalty='l2', random_state=None,
                               tol=0.0001, verbose=0))],
             verbose=False)
    训练时间: 0.277s
    最佳时间:  0.002s
    准确率:   0.880



```python
# 参考翻译
classifier_dic={
    'RidgeClassifier':'岭分类器(Ridge)',
    'Perceptron':'感知器(Perceptron)',
    'PassiveAggressiveClassifier':'PAC分类器',
    'KNeighborsClassifier':'K近邻(KNN)',
    'RandomForestClassifier':'随机森林',
    'LinearSVC':'线性SVC',
    'SGDClassifier':'SGD分类器',
    'NearestCentroid':'线性SVC',
    'MultinomialNB':'(多项式)稀疏朴素贝叶斯分类器',
    'BernoulliNB':'(伯努利)稀疏朴素贝叶斯分类器',
    'ComplementNB':'(补偿)稀疏朴素贝叶斯分类器',
    'Pipeline':'基于l1的特征选择的LinearSVC',
}
```


```python
# 绘图

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("模型对比")
plt.barh(indices, score, .2, label="得分(score)", color='navy')
plt.barh(indices + .3, training_time, .2, label="训练时间",
         color='c')
plt.barh(indices + .6, test_time, .2, label="最佳时间", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, classifier_dic[c])

plt.show()
```


![png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAuAAAAI3CAYAAADa5e6eAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjAsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+17YcXAAAgAElEQVR4nOzde7xWY/7/8de7g0LlUIkKGzFlHKItDaEZmvE1xshhZPoxGTlMjiNnBtPEaL5y/g4pIzFkDEbO542GpCinGoxjSidFDkn1+f1xXXu72+3ae7O7d/J+Ph73o3W41lrXtfYffdbn/qzrVkRgZmZmZmbF0aC+O2BmZmZm9n3iANzMzMzMrIgcgJuZmZmZFZEDcDMzMzOzInIAbmZmZmZWRA7AzczMzMyKyAG4mZmt8iRNkbSupAaStpA0VlIjSf0lXZiXJelDSQslzZb0kaQFkv4jaQNJLSXtL+mftbhuSGpWsH6QpOsqtRkvqUcdDtfMVnMOwM3M7LvkMuB+YDtgInAOcFRe3i8iNgTGAXsBBwCPRsQPgBOBIwpPJGlnSW8WfGZIOqCa6zcDNljezvwQ8NNvODYz+55wAG5mZqusnHF+B+gAvARMAw4HXgD2Bi4FbgR2i4i7a3PuiHgO2DMiOkREB2AW0DdnvUNS+S/VzZd0QQ362hz4JzBQUtPa9MXMvl8cgJuZ2SorIv4J9AQaAjsAM4DBwINAX2BNYH3gslxeEsCuwIvAE8DP87bOlc+dg+QnJP1MUhtgI6AX0DIiFBHKTZtHxAUFh/6yIEi/Om/rSsrCfwb0iIgFdXYTzGy14wDczMxWdf3yv/cDTYHNSCUm5Z8uef89QFtSENwW2IcUqDcnZc8h/b+3GCAHyb2BEcC5wF1AKfBvSWuvoD/3kAL/NYGT87Y/ApdExOEOvs2sOo3quwNmZmbLk8s6+pDKQ54FDgWeA24paLYNqUQlgItJdd/TJXUAPo2ITyWdBwwF9gP+Vn5gRIyTNAi4GtglIp6T9ABwOam2vCpLCoNsSQC/joi76mLMZrb6cwbczMxWZWcAdwAfARcAxwBPAe2B4cC2wDzgJqAx8DE5cI6Ip0nZ7MsiYlFEHBkRu0fEiErXaAcsAH6T188B1imc/aQG5pYvSPqLpA1rM0gz+35xBtzMzFZlTwPPAz/L6yNJNd8A6wC/JZeUAKcD1wIv5qw0pFlLGkk6uOCcvSLieQBJu5EC9q7AvZJ6R8Qo4Fd5/3I7JmkTUjlMZQcBw4APazxKM/tecQBuZmarrIh4CL4OhCOia14/CzgbGA8cEhFLCg5rX74g6VSgJCKOr3xuSVsBdwLHRsTLkg4DuktqRKo1L89if1VwWAOgg6T7gd2BP+T9JfmlznVJ0xTO+5ZDN7PVmANwMzP7zpC0LXAq0IP0wuQgoCzXca8DXFPpkDWBBpJ6V9p+PDAH+EtE3AEQEU8BT0n6OXAvKbC+MyK+LDiuFFgDeAToExFzc+B9LXBDPuaeiJhVR0M2s9WQIqL6VmZmZvVI0hRStvmvwPXA4Bz8ilSGciowPCKGrOR+tADmh//zNLNvwQG4mZl9J0hqCDSIiK+Ws79RRCwqcrfMzGrNAbiZmZmZWRF5GkIzMzMzsyJyAG5mZmZmVkSeBcVWaa1atYqSkpL67oaZmZlZrUyYMGF2RLSuap8DcFullZSUMH78+PruhpmZmVmtSHp3eftcgmJmZmZmVkQOwM3MzMzMisgBuJmZmZlZEbkG3MzMzGw199VXXzF16lQWLFhQ311Z7TRt2pT27dvTuHHjGh/jANzMzMxsNTd16lSaN29OSUkJkuq7O6uNiGDOnDlMnTqVzTbbrMbHuQTFzMzMbDW3YMECWrZs6eC7jkmiZcuWtf5mwQG4mZmZ2feAg++V45vcVwfgZmZmZmZF5ADczMzMzIritdde4+OPP6623b333su0adMYM2YMo0ePLkLPissvYZqZmZl9z0h/rNPzRZxfbZuXX36Z/v37M3LkSNq3b0+XLl0AWLJkCZ988gkTJ04EYPr06Zx11lk899xzbLrppvzud79j7733Zo011qg41+eff87//M//MHr0aNZZZ506HUtlixYtYq+99uLWW29lo402qpNzOgNuZmZmZivVc889x4knnsiVV15JixYtaN68OT169KBHjx7svvvuFe0WLlxInz59GDhwIGuttRYbb7wx+++/P6eeeupS57viiivo06fPSg++ARo1asRZZ53FeeedV2fndABuZmZmZivVNttsw7333suZZ57JW2+9xfz58ykrK6OsrIynnnoKgPnz59OrVy/23XdfevXqVXHs+eefzwcffMDRRx/NF198AcBNN91Enz59ABg0aBCbbropm266KS+99BIA119/PZ06dWLTTTdl6NChAFx11VV07NiRzTffnP79+/PVV18BUFJSwumnn86GG27Ip59+yiOPPELnzp3Zcsst+cMf/gDAT3/6U5566im+/PLLOrkfLkExMzMzs5Vq7bXX5sILL6RDhw6st9563Hrrrcu0ufXWW5HEddddx3XXXbfUvp122omWLVsyfPhwDjzwQNZbbz3WXntt5s6dy5AhQ5g+fToNGjRg4cKFPPPMMwwZMoQnn3ySDTbYgFmzZvHEE09www03MG7cONZee20OOeQQhg4dyvHHHw/Auuuuy4cffshHH33EueeeS1lZGc2aNaNHjx68+OKL7LDDDnTp0oUJEyawyy67fOv74Qy4mZmZma1Ut99+O1dddRWtW7dm5syZvPLKKxWf//u//+OWW26htLSUe++9l8mTJ7NgwQKmTJnClClTuOmmm1iwYAF//vOfOeGEE5g6dSrt27cHoEWLFmyyySacfPLJTJ8+nWbNmjF69GiOPvpoNthgAwBat27N/fffT9++fWnRogUNGzbkqKOO4vHHH6/o3wEHHADAM888w3/+8x+6devGNttswzvvvMN///tfANq1a8cHH3xQJ/fDGXAzMzMzW6nWXXddLr30Ul5//XUuvvhiZs6cWbFv2rRpNGnShPvuu48dd9yRF154oSLABpg5cyYtW7asWI+Iirm3GzZsyLhx47j++uvp3r07o0aNYsGCBTRqtHSIu2jRIho0WDrv3LBhw4rlZs2aVbTbc889ueOOO5YZQ4MGDYiIb3EXCs5VJ2cxMzMzM1uOnj170qJFCwBGjx7NdtttR/fu3Rk7dizHHHMMf/rTn/jDH/7Axx9/TP/+/RkwYEDFsW+88QZt2rSpWN9oo42YNm0aAJ999hlz5syhf//+HHzwwTz//PPsueee3HDDDRXTHU6bNo2ePXtyww038Mknn7B48WKuv/569t5772X6udNOO/Hkk0/y5ptvAlBWVlaxb9q0aXU2C4oz4GZmZmbfMzWZNnBluu6663jttdfYeeedmTp1KnfeeSdjx47lN7/5Daeccgq9evXiV7/6FZMmTWLhwoXceeedFcdusskmzJgxgwULFvD555/z4x//mCVLlrD55ptzzjnn0LJlSyZNmkTnzp1p3LgxF1xwAb/+9a+ZNGkSO+ywA40aNWL//ffniCOOWKZf7dq147LL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)
