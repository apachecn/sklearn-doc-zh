=============
API 参考手册
=============

这是scikit-learn的类和函数手册，篇幅有限不足以详述所有的类和函数指南，更多细节和用法请查阅 :ref:`full user guide <user_guide>` 


.. _base_ref:

:mod:`sklearn.base`: 基类和实用函数
=======================================================

.. automodule:: sklearn.base
    :no-members:
    :no-inherited-members:

基类
------------
.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   base.BaseEstimator
   base.ClassifierMixin
   base.ClusterMixin
   base.RegressorMixin
   base.TransformerMixin

函数
---------
.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   base.clone
   config_context
   get_config
   set_config

.. _calibration_ref:

:mod:`sklearn.calibration`: 概率检验
===================================================

.. automodule:: sklearn.calibration
   :no-members:
   :no-inherited-members:

**用户指南:** 参阅 :ref:`calibration` 章节来查阅更多相关内容。

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   calibration.CalibratedClassifierCV


.. autosummary::
   :toctree: generated/
   :template: function.rst

   calibration.calibration_curve

.. _cluster_ref:

:mod:`sklearn.cluster`: 聚类
==================================

.. automodule:: sklearn.cluster
   :no-members:
   :no-inherited-members:

**用户指南:** 参阅 :ref:`clustering` 章节来查阅更多内容。

类
-------
.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   cluster.AffinityPropagation
   cluster.AgglomerativeClustering
   cluster.Birch
   cluster.DBSCAN
   cluster.FeatureAgglomeration
   cluster.KMeans
   cluster.MiniBatchKMeans
   cluster.MeanShift
   cluster.SpectralClustering

函数
---------
.. autosummary::
   :toctree: generated/
   :template: function.rst

   cluster.affinity_propagation
   cluster.dbscan
   cluster.estimate_bandwidth
   cluster.k_means
   cluster.mean_shift
   cluster.spectral_clustering
   cluster.ward_tree

.. _bicluster_ref:

:mod:`sklearn.cluster.bicluster`: 双向聚类
==============================================

.. automodule:: sklearn.cluster.bicluster
   :no-members:
   :no-inherited-members:

**用户指南:** 参阅 :ref:`biclustering` 章节获取更多内容。

类
-------
.. currentmodule:: sklearn.cluster.bicluster

.. autosummary::
   :toctree: generated/
   :template: class.rst

   SpectralBiclustering
   SpectralCoclustering

.. _covariance_ref:

:mod:`sklearn.covariance`: 协方差估计
================================================

.. automodule:: sklearn.covariance
   :no-members:
   :no-inherited-members:

**用户指南 :** 参阅 :ref:`covariance` 章节来获取更多内容。

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   covariance.EmpiricalCovariance
   covariance.EllipticEnvelope
   covariance.GraphLasso
   covariance.GraphLassoCV
   covariance.LedoitWolf
   covariance.MinCovDet
   covariance.OAS
   covariance.ShrunkCovariance

.. autosummary::
   :toctree: generated/
   :template: function.rst

   covariance.empirical_covariance
   covariance.graph_lasso
   covariance.ledoit_wolf
   covariance.oas
   covariance.shrunk_covariance

.. _cross_decomposition_ref:

:mod:`sklearn.cross_decomposition`: 交叉分解
=======================================================

.. automodule:: sklearn.cross_decomposition
   :no-members:
   :no-inherited-members:

**用户指南:** 参阅 :ref:`cross_decomposition` 章节来获取更多内容。

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   cross_decomposition.CCA
   cross_decomposition.PLSCanonical
   cross_decomposition.PLSRegression
   cross_decomposition.PLSSVD

.. _datasets_ref:

:mod:`sklearn.datasets`: 数据集
=================================

.. automodule:: sklearn.datasets
   :no-members:
   :no-inherited-members:

**用户指南：** 参阅 :ref:`datasets` 章节来获取更多内容。

数据装载
-------

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   datasets.clear_data_home
   datasets.dump_svmlight_file
   datasets.fetch_20newsgroups
   datasets.fetch_20newsgroups_vectorized
   datasets.fetch_california_housing
   datasets.fetch_covtype
   datasets.fetch_kddcup99
   datasets.fetch_lfw_pairs
   datasets.fetch_lfw_people
   datasets.fetch_mldata
   datasets.fetch_olivetti_faces
   datasets.fetch_rcv1
   datasets.fetch_species_distributions
   datasets.get_data_home
   datasets.load_boston
   datasets.load_breast_cancer
   datasets.load_diabetes
   datasets.load_digits
   datasets.load_files
   datasets.load_iris
   datasets.load_linnerud
   datasets.load_mlcomp
   datasets.load_sample_image
   datasets.load_sample_images
   datasets.load_svmlight_file
   datasets.load_svmlight_files
   datasets.load_wine
   datasets.mldata_filename

样品生成
-----------------

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   datasets.make_biclusters
   datasets.make_blobs
   datasets.make_checkerboard
   datasets.make_circles
   datasets.make_classification
   datasets.make_friedman1
   datasets.make_friedman2
   datasets.make_friedman3
   datasets.make_gaussian_quantiles
   datasets.make_hastie_10_2
   datasets.make_low_rank_matrix
   datasets.make_moons
   datasets.make_multilabel_classification
   datasets.make_regression
   datasets.make_s_curve
   datasets.make_sparse_coded_signal
   datasets.make_sparse_spd_matrix
   datasets.make_sparse_uncorrelated
   datasets.make_spd_matrix
   datasets.make_swiss_roll


.. _decomposition_ref:

:mod:`sklearn.decomposition`: 矩阵分解
==================================================

.. automodule:: sklearn.decomposition
   :no-members:
   :no-inherited-members:

**用户指南:** 参阅 :ref:`decompositions` 章节获取更多内容。

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   decomposition.DictionaryLearning
   decomposition.FactorAnalysis
   decomposition.FastICA
   decomposition.IncrementalPCA
   decomposition.KernelPCA
   decomposition.LatentDirichletAllocation
   decomposition.MiniBatchDictionaryLearning
   decomposition.MiniBatchSparsePCA
   decomposition.NMF
   decomposition.PCA
   decomposition.SparsePCA
   decomposition.SparseCoder
   decomposition.TruncatedSVD

.. autosummary::
   :toctree: generated/
   :template: function.rst

   decomposition.dict_learning
   decomposition.dict_learning_online
   decomposition.fastica
   decomposition.sparse_encode

.. _lda_ref:

:mod:`sklearn.discriminant_analysis`: 判别分析
===========================================================

.. automodule:: sklearn.discriminant_analysis
   :no-members:
   :no-inherited-members:

**用户指南:** 参阅 :ref:`lda_qda` 章节获取更多内容。

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated
   :template: class.rst

   discriminant_analysis.LinearDiscriminantAnalysis
   discriminant_analysis.QuadraticDiscriminantAnalysis

.. _dummy_ref:

:mod:`sklearn.dummy`: 虚拟估计器
======================================

.. automodule:: sklearn.dummy
   :no-members:
   :no-inherited-members:

**用户指南:** 参阅 :ref:`model_evaluation` 章节获取更多内容。

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   dummy.DummyClassifier
   dummy.DummyRegressor

.. autosummary::
   :toctree: generated/
   :template: function.rst

.. _ensemble_ref:

:mod:`sklearn.ensemble`: 集成方法
=========================================

.. automodule:: sklearn.ensemble
   :no-members:
   :no-inherited-members:

**用户指南：** 参阅 :ref:`ensemble` 章节获取更多内容。

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ensemble.AdaBoostClassifier
   ensemble.AdaBoostRegressor
   ensemble.BaggingClassifier
   ensemble.BaggingRegressor
   ensemble.ExtraTreesClassifier
   ensemble.ExtraTreesRegressor
   ensemble.GradientBoostingClassifier
   ensemble.GradientBoostingRegressor
   ensemble.IsolationForest
   ensemble.RandomForestClassifier
   ensemble.RandomForestRegressor
   ensemble.RandomTreesEmbedding
   ensemble.VotingClassifier

.. autosummary::
   :toctree: generated/
   :template: function.rst


部分依赖
------------------

.. automodule:: sklearn.ensemble.partial_dependence
   :no-members:
   :no-inherited-members:

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   ensemble.partial_dependence.partial_dependence
   ensemble.partial_dependence.plot_partial_dependence


.. _exceptions_ref:

:mod:`sklearn.exceptions`: 异常和警告
==================================================

.. automodule:: sklearn.exceptions
   :no-members:
   :no-inherited-members:

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class_without_init.rst

   exceptions.ChangedBehaviorWarning
   exceptions.ConvergenceWarning
   exceptions.DataConversionWarning
   exceptions.DataDimensionalityWarning
   exceptions.EfficiencyWarning
   exceptions.FitFailedWarning
   exceptions.NotFittedError
   exceptions.NonBLASDotWarning
   exceptions.UndefinedMetricWarning

.. _feature_extraction_ref:

:mod:`sklearn.feature_extraction`: 特征提取
=====================================================

.. automodule:: sklearn.feature_extraction
   :no-members:
   :no-inherited-members:

**用户指南：** 参阅 :ref:`feature_extraction` 章节获取更多内容。

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   feature_extraction.DictVectorizer
   feature_extraction.FeatureHasher

图像特征提取
-----------

.. automodule:: sklearn.feature_extraction.image
   :no-members:
   :no-inherited-members:

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   feature_extraction.image.extract_patches_2d
   feature_extraction.image.grid_to_graph
   feature_extraction.image.img_to_graph
   feature_extraction.image.reconstruct_from_patches_2d

   :template: class.rst

   feature_extraction.image.PatchExtractor

.. _text_feature_extraction_ref:

文本特征提取
---------

.. automodule:: sklearn.feature_extraction.text
   :no-members:
   :no-inherited-members:

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   feature_extraction.text.CountVectorizer
   feature_extraction.text.HashingVectorizer
   feature_extraction.text.TfidfTransformer
   feature_extraction.text.TfidfVectorizer


.. _feature_selection_ref:

:mod:`sklearn.feature_selection`: 特征选择
===================================================

.. automodule:: sklearn.feature_selection
   :no-members:
   :no-inherited-members:

**用户指南:** 参阅 :ref:`feature_selection` 章节获取更多内容。

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   feature_selection.GenericUnivariateSelect
   feature_selection.SelectPercentile
   feature_selection.SelectKBest
   feature_selection.SelectFpr
   feature_selection.SelectFdr
   feature_selection.SelectFromModel
   feature_selection.SelectFwe
   feature_selection.RFE
   feature_selection.RFECV
   feature_selection.VarianceThreshold

.. autosummary::
   :toctree: generated/
   :template: function.rst

   feature_selection.chi2
   feature_selection.f_classif
   feature_selection.f_regression
   feature_selection.mutual_info_classif
   feature_selection.mutual_info_regression


.. _gaussian_process_ref:

:mod:`sklearn.gaussian_process`: 高斯过程
===================================================

.. automodule:: sklearn.gaussian_process
   :no-members:
   :no-inherited-members:

**用户指南:** 参阅 :ref:`gaussian_process` 章节获取更多内容。

.. currentmodule:: sklearn

.. autosummary::
  :toctree: generated/
  :template: class.rst

  gaussian_process.GaussianProcessClassifier
  gaussian_process.GaussianProcessRegressor

核函数

.. autosummary::
  :toctree: generated/
  :template: class_with_call.rst

  gaussian_process.kernels.CompoundKernel
  gaussian_process.kernels.ConstantKernel
  gaussian_process.kernels.DotProduct
  gaussian_process.kernels.ExpSineSquared
  gaussian_process.kernels.Exponentiation
  gaussian_process.kernels.Hyperparameter
  gaussian_process.kernels.Kernel
  gaussian_process.kernels.Matern
  gaussian_process.kernels.PairwiseKernel
  gaussian_process.kernels.Product
  gaussian_process.kernels.RBF
  gaussian_process.kernels.RationalQuadratic
  gaussian_process.kernels.Sum
  gaussian_process.kernels.WhiteKernel

.. _isotonic_ref:

:mod:`sklearn.isotonic`: 保序回归
============================================

.. automodule:: sklearn.isotonic
   :no-members:
   :no-inherited-members:

**用户指南;** 参阅 :ref:`isotonic` 章节获取更多内容。

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   isotonic.IsotonicRegression

.. autosummary::
   :toctree: generated
   :template: function.rst

   isotonic.check_increasing
   isotonic.isotonic_regression

.. _kernel_approximation_ref:

:mod:`sklearn.kernel_approximation` 核近似
========================================================

.. automodule:: sklearn.kernel_approximation
   :no-members:
   :no-inherited-members:

**用户指南:** 参阅 :ref:`kernel_approximation` 获取更多内容。

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   kernel_approximation.AdditiveChi2Sampler
   kernel_approximation.Nystroem
   kernel_approximation.RBFSampler
   kernel_approximation.SkewedChi2Sampler

.. _kernel_ridge_ref:

:mod:`sklearn.kernel_ridge` 内核岭回归
========================================================

.. automodule:: sklearn.kernel_ridge
   :no-members:
   :no-inherited-members:

**用户指南:** 参阅 :ref:`kernel_ridge` 章节获取更多内容。

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   kernel_ridge.KernelRidge

.. _linear_model_ref:

:mod:`sklearn.linear_model`: 广义线性模型
======================================================

.. automodule:: sklearn.linear_model
   :no-members:
   :no-inherited-members:

**用户指南:** 参阅 :ref:`linear_model` 章节获取更多内容。

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   linear_model.ARDRegression
   linear_model.BayesianRidge
   linear_model.ElasticNet
   linear_model.ElasticNetCV
   linear_model.HuberRegressor
   linear_model.Lars
   linear_model.LarsCV
   linear_model.Lasso
   linear_model.LassoCV
   linear_model.LassoLars
   linear_model.LassoLarsCV
   linear_model.LassoLarsIC
   linear_model.LinearRegression
   linear_model.LogisticRegression
   linear_model.LogisticRegressionCV
   linear_model.MultiTaskLasso
   linear_model.MultiTaskElasticNet
   linear_model.MultiTaskLassoCV
   linear_model.MultiTaskElasticNetCV
   linear_model.OrthogonalMatchingPursuit
   linear_model.OrthogonalMatchingPursuitCV
   linear_model.PassiveAggressiveClassifier
   linear_model.PassiveAggressiveRegressor
   linear_model.Perceptron
   linear_model.RANSACRegressor
   linear_model.Ridge
   linear_model.RidgeClassifier
   linear_model.RidgeClassifierCV
   linear_model.RidgeCV
   linear_model.SGDClassifier
   linear_model.SGDRegressor
   linear_model.TheilSenRegressor

.. autosummary::
   :toctree: generated/
   :template: function.rst

   linear_model.enet_path
   linear_model.lars_path
   linear_model.lasso_path
   linear_model.lasso_stability_path
   linear_model.logistic_regression_path
   linear_model.orthogonal_mp
   linear_model.orthogonal_mp_gram


.. _manifold_ref:

:mod:`sklearn.manifold`: 集成学习
==========================================

.. automodule:: sklearn.manifold
   :no-members:
   :no-inherited-members:

**用户指南:** 参阅 :ref:`manifold` 章节获取更多内容。

.. currentmodule:: sklearn

.. autosummary::
    :toctree: generated
    :template: class.rst

    manifold.Isomap
    manifold.LocallyLinearEmbedding
    manifold.MDS
    manifold.SpectralEmbedding
    manifold.TSNE

.. autosummary::
    :toctree: generated
    :template: function.rst

    manifold.locally_linear_embedding
    manifold.smacof
    manifold.spectral_embedding


.. _metrics_ref:

:mod:`sklearn.metrics`: 指标
===============================

参阅 :ref:`model_evaluation` 和 :ref:`metrics` 章节的用户指南获取更多信息。

.. automodule:: sklearn.metrics
   :no-members:
   :no-inherited-members:

.. currentmodule:: sklearn

模型选择接口
-------------------------
参阅 :ref:`scoring_parameter` 章节获取更多内容。

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.get_scorer
   metrics.make_scorer

分类指标
----------------------

参阅 :ref:`classification_metrics` 用户指南章节获取更多内容。

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.accuracy_score
   metrics.auc
   metrics.average_precision_score
   metrics.brier_score_loss
   metrics.classification_report
   metrics.cohen_kappa_score
   metrics.confusion_matrix
   metrics.dcg_score
   metrics.f1_score
   metrics.fbeta_score
   metrics.hamming_loss
   metrics.hinge_loss
   metrics.jaccard_similarity_score
   metrics.log_loss
   metrics.matthews_corrcoef
   metrics.ndcg_score
   metrics.precision_recall_curve
   metrics.precision_recall_fscore_support
   metrics.precision_score
   metrics.recall_score
   metrics.roc_auc_score
   metrics.roc_curve
   metrics.zero_one_loss

回归度量
------------------

参阅 :ref:`regression_metrics` 用户指南章节获取更多内容。

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.explained_variance_score
   metrics.mean_absolute_error
   metrics.mean_squared_error
   metrics.mean_squared_log_error
   metrics.median_absolute_error
   metrics.r2_score

多标签排序指标
--------------------------
参阅 :ref:`multilabel_ranking_metrics` 用户指南章节获取更多内容。

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.coverage_error
   metrics.label_ranking_average_precision_score
   metrics.label_ranking_loss


聚类指标
------------------

参阅 :ref:`clustering_evaluation` 用户指南章节获取更多内容

.. automodule:: sklearn.metrics.cluster
   :no-members:
   :no-inherited-members:

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.adjusted_mutual_info_score
   metrics.adjusted_rand_score
   metrics.calinski_harabaz_score
   metrics.completeness_score
   metrics.fowlkes_mallows_score
   metrics.homogeneity_completeness_v_measure
   metrics.homogeneity_score
   metrics.mutual_info_score
   metrics.normalized_mutual_info_score
   metrics.silhouette_score
   metrics.silhouette_samples
   metrics.v_measure_score

双向聚类指标
--------------------

参阅 :ref:`biclustering_evaluation` 用户指南章节获取更多内容。

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.consensus_score


成对度量
----------------

参阅 :ref:`metrics` 用户指南章节获取更多内容。

.. automodule:: sklearn.metrics.pairwise
   :no-members:
   :no-inherited-members:

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.pairwise.additive_chi2_kernel
   metrics.pairwise.chi2_kernel
   metrics.pairwise.cosine_similarity
   metrics.pairwise.cosine_distances
   metrics.pairwise.distance_metrics
   metrics.pairwise.euclidean_distances
   metrics.pairwise.kernel_metrics
   metrics.pairwise.laplacian_kernel
   metrics.pairwise.linear_kernel
   metrics.pairwise.manhattan_distances
   metrics.pairwise.pairwise_distances
   metrics.pairwise.pairwise_kernels
   metrics.pairwise.polynomial_kernel
   metrics.pairwise.rbf_kernel
   metrics.pairwise.sigmoid_kernel
   metrics.pairwise.paired_euclidean_distances
   metrics.pairwise.paired_manhattan_distances
   metrics.pairwise.paired_cosine_distances
   metrics.pairwise.paired_distances
   metrics.pairwise_distances
   metrics.pairwise_distances_argmin
   metrics.pairwise_distances_argmin_min


.. _mixture_ref:

:mod:`sklearn.mixture`: 高斯混合模型
===============================================

.. automodule:: sklearn.mixture
   :no-members:
   :no-inherited-members:

**用户指南:** 参阅 :ref:`mixture` 章节获取更多内容。

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mixture.BayesianGaussianMixture
   mixture.GaussianMixture

.. _modelselection_ref:

:mod:`sklearn.model_selection`: 模型选择
===============================================

.. automodule:: sklearn.model_selection
   :no-members:
   :no-inherited-members:

**用户指南:** 参阅 :ref:`cross_validation`, :ref:`grid_search` 和
:ref:`learning_curve` 章节获取更多内容。

分离器类
----------------

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   model_selection.GroupKFold
   model_selection.GroupShuffleSplit
   model_selection.KFold
   model_selection.LeaveOneGroupOut
   model_selection.LeavePGroupsOut
   model_selection.LeaveOneOut
   model_selection.LeavePOut
   model_selection.PredefinedSplit
   model_selection.RepeatedKFold
   model_selection.RepeatedStratifiedKFold
   model_selection.ShuffleSplit
   model_selection.StratifiedKFold
   model_selection.StratifiedShuffleSplit
   model_selection.TimeSeriesSplit

分离器函数
------------------

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   model_selection.check_cv
   model_selection.train_test_split

超参数优化器
--------------------------

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   model_selection.GridSearchCV
   model_selection.ParameterGrid
   model_selection.ParameterSampler
   model_selection.RandomizedSearchCV


.. autosummary::
   :toctree: generated/
   :template: function.rst

   model_selection.fit_grid_point

模型验证
----------------

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   model_selection.cross_validate
   model_selection.cross_val_predict
   model_selection.cross_val_score
   model_selection.learning_curve
   model_selection.permutation_test_score
   model_selection.validation_curve

.. _multiclass_ref:

:mod:`sklearn.multiclass`: 多类和多标签分类
===================================================================

.. automodule:: sklearn.multiclass
   :no-members:
   :no-inherited-members:

**用户指南:** 参阅 :ref:`multiclass` 章节获取更多内容。

.. currentmodule:: sklearn

.. autosummary::
    :toctree: generated
    :template: class.rst

    multiclass.OneVsRestClassifier
    multiclass.OneVsOneClassifier
    multiclass.OutputCodeClassifier

.. _multioutput_ref:

:mod:`sklearn.multioutput`: 多输出回归和分类
=====================================================================

.. automodule:: sklearn.multioutput
   :no-members:
   :no-inherited-members:

**用户指南:** 参阅 :ref:`multiclass` 章节获取更多内容。

.. currentmodule:: sklearn

.. autosummary::
    :toctree: generated
    :template: class.rst

    multioutput.ClassifierChain
    multioutput.MultiOutputRegressor
    multioutput.MultiOutputClassifier

.. _naive_bayes_ref:

:mod:`sklearn.naive_bayes`: 朴素贝叶斯
=======================================

.. automodule:: sklearn.naive_bayes
   :no-members:
   :no-inherited-members:

**用户指南:** 参阅 :ref:`naive_bayes` 章节获取更多内容。

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   naive_bayes.BernoulliNB
   naive_bayes.GaussianNB
   naive_bayes.MultinomialNB


.. _neighbors_ref:

:mod:`sklearn.neighbors`: 最近邻
===========================================

.. automodule:: sklearn.neighbors
   :no-members:
   :no-inherited-members:

**用户指南:** 参阅 :ref:`neighbors` 章节获取更多内容。

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   neighbors.BallTree
   neighbors.DistanceMetric
   neighbors.KDTree
   neighbors.KernelDensity
   neighbors.KNeighborsClassifier
   neighbors.KNeighborsRegressor
   neighbors.LocalOutlierFactor
   neighbors.RadiusNeighborsClassifier
   neighbors.RadiusNeighborsRegressor
   neighbors.NearestCentroid
   neighbors.NearestNeighbors

.. autosummary::
   :toctree: generated/
   :template: function.rst

   neighbors.kneighbors_graph
   neighbors.radius_neighbors_graph

.. _neural_network_ref:

:mod:`sklearn.neural_network`: 神经网络模型
=====================================================

.. automodule:: sklearn.neural_network
   :no-members:
   :no-inherited-members:

**用户指南:** 参阅 :ref:`neural_networks_supervised` 和 :ref:`neural_networks_unsupervised` 章节获取更多内容。

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   neural_network.BernoulliRBM
   neural_network.MLPClassifier
   neural_network.MLPRegressor

.. _pipeline_ref:

:mod:`sklearn.pipeline`: 管道线
=================================

.. automodule:: sklearn.pipeline
   :no-members:
   :no-inherited-members:

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   pipeline.FeatureUnion
   pipeline.Pipeline

.. autosummary::
   :toctree: generated/
   :template: function.rst

   pipeline.make_pipeline
   pipeline.make_union


.. _preprocessing_ref:

:mod:`sklearn.preprocessing`: 预处理和正则化
=============================================================

.. automodule:: sklearn.preprocessing
   :no-members:
   :no-inherited-members:

**用户指南:** 参阅 :ref:`preprocessing` 章节获取更多内容。

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   preprocessing.Binarizer
   preprocessing.FunctionTransformer
   preprocessing.Imputer
   preprocessing.KernelCenterer
   preprocessing.LabelBinarizer
   preprocessing.LabelEncoder
   preprocessing.MultiLabelBinarizer
   preprocessing.MaxAbsScaler
   preprocessing.MinMaxScaler
   preprocessing.Normalizer
   preprocessing.OneHotEncoder
   preprocessing.PolynomialFeatures
   preprocessing.QuantileTransformer
   preprocessing.RobustScaler
   preprocessing.StandardScaler

.. autosummary::
   :toctree: generated/
   :template: function.rst

   preprocessing.add_dummy_feature
   preprocessing.binarize
   preprocessing.label_binarize
   preprocessing.maxabs_scale
   preprocessing.minmax_scale
   preprocessing.normalize
   preprocessing.quantile_transform
   preprocessing.robust_scale
   preprocessing.scale


.. _random_projection_ref:

:mod:`sklearn.random_projection`: 随机投影
===================================================

.. automodule:: sklearn.random_projection
   :no-members:
   :no-inherited-members:

**用户指南:** 参阅 :ref:`random_projection` 章节获取更多内容。

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   random_projection.GaussianRandomProjection
   random_projection.SparseRandomProjection

.. autosummary::
   :toctree: generated/
   :template: function.rst

   random_projection.johnson_lindenstrauss_min_dim


.. _semi_supervised_ref:

:mod:`sklearn.semi_supervised` 办监督学习
========================================================

.. automodule:: sklearn.semi_supervised
   :no-members:
   :no-inherited-members:

**用户指南:** 参阅 :ref:`semi_supervised` 章节获取更多内容。

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   semi_supervised.LabelPropagation
   semi_supervised.LabelSpreading


.. _svm_ref:

:mod:`sklearn.svm`: 支持向量机
===========================================

.. automodule:: sklearn.svm
   :no-members:
   :no-inherited-members:

**用户指南:** 参阅 :ref:`svm` 章节获取更多内容。

估计器
----------

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   svm.LinearSVC
   svm.LinearSVR
   svm.NuSVC
   svm.NuSVR
   svm.OneClassSVM
   svm.SVC
   svm.SVR

.. autosummary::
   :toctree: generated/
   :template: function.rst

   svm.l1_min_c

初级方法
-----------------

.. autosummary::
   :toctree: generated
   :template: function.rst

   svm.libsvm.cross_validation
   svm.libsvm.decision_function
   svm.libsvm.fit
   svm.libsvm.predict
   svm.libsvm.predict_proba


.. _tree_ref:

:mod:`sklearn.tree`: 决策树
===================================

.. automodule:: sklearn.tree
   :no-members:
   :no-inherited-members:

**用户指南:** 参阅 :ref:`tree` 章节获取更多内容。

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   tree.DecisionTreeClassifier
   tree.DecisionTreeRegressor
   tree.ExtraTreeClassifier
   tree.ExtraTreeRegressor

.. autosummary::
   :toctree: generated/
   :template: function.rst

   tree.export_graphviz


.. _utils_ref:

:mod:`sklearn.utils`: 效率
===============================

.. automodule:: sklearn.utils
   :no-members:
   :no-inherited-members:

**开发者指南:** 参阅 :ref:`developers-utils` 页获取更多内容

.. currentmodule:: sklearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   utils.as_float_array
   utils.assert_all_finite
   utils.check_X_y
   utils.check_array
   utils.check_consistent_length
   utils.check_random_state
   utils.class_weight.compute_class_weight
   utils.class_weight.compute_sample_weight
   utils.estimator_checks.check_estimator
   utils.extmath.safe_sparse_dot
   utils.indexable
   utils.resample
   utils.safe_indexing
   utils.shuffle
   utils.sparsefuncs.incr_mean_variance_axis
   utils.sparsefuncs.inplace_column_scale
   utils.sparsefuncs.inplace_row_scale
   utils.sparsefuncs.inplace_swap_row
   utils.sparsefuncs.inplace_swap_column
   utils.sparsefuncs.mean_variance_axis
   utils.validation.check_is_fitted
   utils.validation.check_symmetric
   utils.validation.column_or_1d
   utils.validation.has_fit_parameter

最近弃用
===================


0.21将会移除
---------------------

.. autosummary::
   :toctree: generated/
   :template: deprecated_class.rst

   linear_model.RandomizedLasso
   linear_model.RandomizedLogisticRegression
   neighbors.LSHForest


0.20版本将会移除
---------------------

.. autosummary::
   :toctree: generated/
   :template: deprecated_class.rst

   cross_validation.KFold
   cross_validation.LabelKFold
   cross_validation.LeaveOneLabelOut
   cross_validation.LeaveOneOut
   cross_validation.LeavePOut
   cross_validation.LeavePLabelOut
   cross_validation.LabelShuffleSplit
   cross_validation.ShuffleSplit
   cross_validation.StratifiedKFold
   cross_validation.StratifiedShuffleSplit
   cross_validation.PredefinedSplit
   decomposition.RandomizedPCA
   gaussian_process.GaussianProcess
   grid_search.ParameterGrid
   grid_search.ParameterSampler
   grid_search.GridSearchCV
   grid_search.RandomizedSearchCV
   mixture.DPGMM
   mixture.GMM
   mixture.VBGMM


.. autosummary::
   :toctree: generated/
   :template: deprecated_function.rst

   cross_validation.check_cv
   cross_validation.cross_val_predict
   cross_validation.cross_val_score
   cross_validation.permutation_test_score
   cross_validation.train_test_split
   grid_search.fit_grid_point
   learning_curve.learning_curve
   learning_curve.validation_curve
