# 贡献指南

为了使项目更加便于维护，我们将文档格式全部转换成了 Markdown，同时更换了页面生成器。后续维护工作将完全在 Markdown 上进行。

小部分格式仍然存在问题，主要是链接和表格。需要大家帮忙找到，并提 PullRequest 来修复。

## 参与翻译 & 修正错误

翻译待解决的问题 & 对应的解决思路：
1. 缺少示例的翻译：  用Jupyter notebook跑一遍示例程序，再把介绍和注释翻译成中文，最后生成为md并修复其他文章中对应的链接
2. 小部分格式和翻译仍然存在问题，需要大家帮忙找到，并提 PullRequest 来修复。
3. 部分翻译的语言可能不够流畅，需要大家帮忙润色，并提 PullRequest 来优化。

贡献方式：
1. 在 github 上 fork 该 repository.
2. 按照上面提到的解决思路修复对应的问题
3. 然后, 在你的 github 发起 New pull request 请求.
4. 工具使用, 可参考下面的内容.

## 工具使用（针对新手）

工欲善其事, 必先利其器 ...  
工具随意, 能达到效果就好.  
我这里使用的是 `VSCode` 编辑器.  
简易的使用指南请参阅: [VSCode Windows 平台入门使用指南](help/vscode-windows-usage.md), 介绍了 `VSCode` 与 `github` 一起搭配的简易使用的方法.  
如果要将 VSCode 的 Markdown 预览风格切换为 github 的风格，请参阅: [VSCode 修改 markdown 的预览风格为 github 的风格](help/vscode-markdown-preview-github-style.md).

**注意注意注意:**  

为了尽量正规化各顶级项目的翻译，更便于以后的迭代更新，我们在 `scikit-learn` 文档翻译中使用了 `Git` 的分支，具体应用方法请参阅: [使用 Git 分支进行迭代翻译](help/git-branch-usage.md).

## 角色分配

目前有如下可分配的角色: 

* 翻译: 负责文章内容的翻译.
* 校验: 负责文章内容的校验, 比如格式, 正确度之类的.
* 负责人: 负责整个 Projcet, 不至于让该 Project 成为垃圾项目, 需要在 sklearn 方面经验稍微丰富点.

有兴趣参与的朋友, 可以看看最后的联系方式.
