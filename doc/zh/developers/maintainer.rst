维护者 / 核心开发者 信息
========================================

更多详细信息请看 https://github.com/scikit-learn/scikit-learn/wiki/How-to-make-a-release

发布流程
------------------

1. 更新文档:

    - 编辑 doc/whats_new.rst 文件，添加发布版本的标题和 commit 的统计。 使用如下命令获取 commit 的统计::

        $ git shortlog -ns 0.998..

    - 编辑 doc/conf.py 以更新版本号

    - 编辑 doc/themes/scikit-learn/layout.html 以改变首页 'News' 入口。

2. 在 sklearn/__init__.py 中通过修改 __version__变量来改变版本号 

3. 创建 tag 并 push::

    $ git tag 0.999

    $ git push origin --tags

4. 创建 tar 包:

   - 清理你的仓库::

       $ git clean -xfd

   - 在 PyPI 上注册和上传::

       $ python setup.py sdist register upload


5. 将文档 push 到网站上 (详见 doc 文件夹中的 README 文件)


6. 使用专用CI服务器，通过将git子模块的引用更新到新发布的scikit-learn 标签上来构建二进制文件:

   https://github.com/MacPython/scikit-learn-wheels

   当 CI 处理完毕后，收集所有生成的二进制 wheel 包并使用如下命令（在scikit-learn 源码文件夹中）将其上传到 PyPI上（查看发布标签）::

       $ pip install -U wheelhouse_uploader
       $ python setup.py sdist fetch_artifacts upload_all


7. 最终发布: 在 What's New 中更新发布日期
