# <center>scikit-learn (sklearn) 官方文档中文版</center>

<center><img src="img/logo/scikit-learn-logo.png" alt="logo" /></center>

<br/>
<table>
    <tr align="center">
        <td><a title="sklearn 0.21.3[master] 中文文档" href="https://sklearn.apachecn.org/" target="_blank"><font size="5">sklearn 0.21.3 中文文档</font></a></td>
        <td><a title="sklearn 0.21.3[master] 中文示例" href="https://sklearn.apachecn.org/docs/examples" target="_blank"><font size="5">sklearn 0.21.3 中文示例</font></a></td>
        <td><a title="sklearn 英文官网" href="https://scikit-learn.org" target="_blank"><font size="5">sklearn 英文官网</font></a></td>
    </tr>
</table>
<br/>

---

## 介绍

sklearn (scikit-learn) 是基于 Python 语言的机器学习工具

1. 简单高效的数据挖掘和数据分析工具
2. 可供大家在各种环境中重复使用
3. 建立在 NumPy ，SciPy 和 matplotlib 上
4. 开源，可商业使用 - BSD许可证

> 组织构建[网站]

+ GitHub Pages(国外): https://sklearn.apachecn.org
+ Gitee Pages(国内): https://apachecn.gitee.io/sklearn-doc-zh

> 第三方站长[网站]

+ sklearn 中文文档: http://www.scikitlearn.com.cn
+ 地址A: xxx (欢迎留言，我们完善补充)

> 其他补充

+ [官方Github](https://github.com/apachecn/scikit-learn-doc-zh)
+ [EPUB 下载地址](https://github.com/apachecn/sklearn-doc-zh/raw/epub/sklearn_0.21.3_2019_12_13.epub)

## 下载

### Docker

```
docker pull apachecn0/sklearn-doc-zh
docker run -tid -p <port>:80 apachecn0/sklearn-doc-zh
# 访问 http://localhost:{port} 查看文档
```

### PYPI

```
pip install sklearn-doc-zh
sklearn-doc-zh <port>
# 访问 http://localhost:{port} 查看文档
```

### NPM

```
npm install -g sklearn-doc-zh
sklearn-doc-zh <port>
# 访问 http://localhost:{port} 查看文档
```

## 目录

*   [安装 scikit-learn](docs/master/62.md)
*   用户指南
    *   [1. 监督学习](docs/master/1.md)
        * [1.1. 广义线性模型](docs/master/2.md)
        * [1.2. 线性和二次判别分析](docs/master/3.md)
        * [1.3. 内核岭回归](docs/master/4.md)
        * [1.4. 支持向量机](docs/master/5.md)
        * [1.5. 随机梯度下降](docs/master/6.md)
        * [1.6. 最近邻](docs/master/7.md)
        * [1.7. 高斯过程](docs/master/8.md)
        * [1.8. 交叉分解](docs/master/9.md)
        * [1.9. 朴素贝叶斯](docs/master/10.md)
        * [1.10. 决策树](docs/master/11.md)
        * [1.11. 集成方法](docs/master/12.md)
        * [1.12. 多类和多标签算法](docs/master/13.md)
        * [1.13. 特征选择](docs/master/14.md)
        * [1.14. 半监督学习](docs/master/15.md)
        * [1.15. 等式回归](docs/master/16.md)
        * [1.16. 概率校准](docs/master/17.md)
        * [1.17. 神经网络模型（有监督）](docs/master/18.md)
    *   [2. 无监督学习](docs/master/19.md)
        * [2.1. 高斯混合模型](docs/master/20.md)
        * [2.2. 流形学习](docs/master/21.md)
        * [2.3. 聚类](docs/master/22.md)
        * [2.4. 双聚类](docs/master/23.md)
        * [2.5. 分解成分中的信号（矩阵分解问题）](docs/master/24.md)
        * [2.6. 协方差估计](docs/master/25.md)
        * [2.7. 新奇和异常值检测](docs/master/26.md)
        * [2.8. 密度估计](docs/master/27.md)
        * [2.9. 神经网络模型（无监督）](docs/master/28.md)
    * [3. 模型选择和评估](docs/master/29.md)
        * [3.1. 交叉验证：评估估算器的表现](docs/master/30.md)
        * [3.2. 调整估计器的超参数](docs/master/31.md)
        * [3.3. 模型评估: 量化预测的质量](docs/master/32.md)
        * [3.4. 模型持久化](docs/master/33.md)
        * [3.5. 验证曲线: 绘制分数以评估模型](docs/master/34.md)
    * [4.  检验](docs/master/35.md)
        * [4.1. 部分依赖图](docs/master/36.md)
    * [5. 数据集转换](docs/master/37.md)
        * [5.1. Pipeline（管道）和 FeatureUnion（特征联合）: 合并的评估器](docs/master/38.md)
        * [5.2. 特征提取](docs/master/39.md)
        * [5.3 预处理数据](docs/master/40.md)
        * [5.4 缺失值插补](docs/master/41.md)
        * [5.5. 无监督降维](docs/master/42.md)
        * [5.6. 随机投影](docs/master/43.md)
        * [5.7. 内核近似](docs/master/44.md)
        * [5.8. 成对的矩阵, 类别和核函数](docs/master/45.md)
        * [5.9. 预测目标 (`y`) 的转换](docs/master/46.md)
    * [6. 数据集加载工具](docs/master/47.md)
        * [6.1. 通用数据集 API](docs/master/47.md)
        * [6.2. 玩具数据集](docs/master/47.md)
        * [6.3 真实世界中的数据集](docs/master/47.md)
        * [6.4. 样本生成器](docs/master/47.md)
        * [6.5. 加载其他数据集](docs/master/47.md)
    * [7. 使用scikit-learn计算](docs/master/48.md)
        * [7.1. 大规模计算的策略: 更大量的数据](docs/master/48.md)
        * [7.2. 计算性能](docs/master/48.md)
        * [7.3. 并行性、资源管理和配置](docs/master/48.md)
*   [教程](docs/master/50.md)
    *   [使用 scikit-learn 介绍机器学习](docs/master/51.md)
    *   [关于科学数据处理的统计学习教程](docs/master/52.md)
        *   [机器学习: scikit-learn 中的设置以及预估对象](docs/master/53.md)
        *   [监督学习：从高维观察预测输出变量](docs/master/54.md)
        *   [模型选择：选择估计量及其参数](docs/master/55.md)
        *   [无监督学习: 寻求数据表示](docs/master/56.md)
        *   [把它们放在一起](docs/master/57.md)
        *   [寻求帮助](docs/master/58.md)
    *   [处理文本数据](docs/master/59.md)
    *   [选择正确的评估器(estimator.md)](docs/master/60.md)
    *   [外部资源，视频和谈话](docs/master/61.md)
*   [API 参考](https://scikit-learn.org/stable/modules/classes.html)
*   [常见问题](docs/master/63.md)
*   [时光轴](docs/master/64.md)

## 历史版本

* [scikit-learn (sklearn) 0.19 官方文档中文版](https://github.com/apachecn/sklearn-doc-zh/tree/master/docs/0.19.x.zip)
* [scikit-learn (sklearn) 0.18 官方文档中文版](http://cwiki.apachecn.org/pages/viewpage.action?pageId=10030181)

如何编译使用历史版本: 

* 解压 `0.19.x.zip` 文件夹
* 将 `master/img` 的图片资源, 复制到 `0.19.x` 里面去
* gitbook 正常编译过程，可以使用 `sh run_website.sh`

## 贡献指南

项目当前处于校对阶段，请查看[贡献指南](CONTRIBUTING.md)，并在[整体进度](https://github.com/apachecn/sklearn-doc-zh/issues/352)中领取任务。

> 请您勇敢地去翻译和改进翻译。虽然我们追求卓越，但我们并不要求您做到十全十美，因此请不要担心因为翻译上犯错——在大部分情况下，我们的服务器已经记录所有的翻译，因此您不必担心会因为您的失误遭到无法挽回的破坏。（改编自维基百科）

## 项目负责人

格式: GitHub + QQ

> 第一期 (2017-09-29)

* [@那伊抹微笑](https://github.com/wangyangting)
* [@片刻](https://github.com/jiangzhonglian)
* [@小瑶](https://github.com/chenyyx)

> 第二期 (2019-06-29)

* [@N!no](https://github.com/lovelybuggies)：1352899627
* [@mahaoyang](https://github.com/mahaoyang)：992635910
* [@loopyme](https://github.com/loopyme)：3322728009
* [飞龙](https://github.com/wizardforcel)：562826179
* [片刻](https://github.com/jiangzhonglian)：529815144

-- 负责人要求: (欢迎一起为 `sklearn 中文版本` 做贡献)

* 热爱开源，喜欢装逼
* 长期使用 sklearn(至少0.5年) + 提交Pull Requests>=3
* 能够有时间及时优化页面 bug 和用户 issues
* 试用期: 2个月
* 欢迎联系: [片刻](https://github.com/jiangzhonglian) 529815144

## 贡献者

[【0.19.X】贡献者名单](https://github.com/apachecn/sklearn-doc-zh/issues/354)

## 建议反馈

* 在我们的 [apachecn/pytorch-doc-zh](https://github.com/apachecn/sklearn-doc-zh) github 上提 issue.
* 发邮件到 Email: `apachecn@163.com`.
* 在我们的 [QQ群-搜索: 交流方式](https://github.com/apachecn/home) 中联系群主/管理员即可.

## **项目协议**

* **最近有很多人联系我们，关于内容授权问题！**
* 开源是指知识应该重在传播和迭代（而不是禁止别人转载）
* 不然你TM在GitHub开源，然后又说不让转载，你TM有病吧！
* 禁止商业化，符合协议规范，备注地址来源，**重点: 不需要**发邮件给我们申请
* ApacheCN 账号下没有协议的项目，一律视为 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh)。

温馨提示:

* 对于个人想自己copy一份再更新的人
* 我也是有这样的经历，但是这种激情维持不了几个月，就泄气了！
* 不仅浪费了你的心血，还浪费了更多人看到你的翻译成果！很可惜！你觉得呢？
* 个人的建议是: fork -> pull requests 到 `https://github.com/apachecn/sklearn-doc-zh`
* 那为什么要选择 `ApacheCN` 呢？
* 因为我们做翻译这事情是觉得开心和装逼，比较纯粹！
* 你如果喜欢，你可以来参与/甚至负责这个项目，没有任何学历和背景的限制

## 赞助我们

<img src="http://data.apachecn.org/img/about/donate.jpg" alt="微信&支付宝" />
