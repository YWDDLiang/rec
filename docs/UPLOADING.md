# 本地上传说明：两个仓库分开处理

## rec实施仓库

解压`rec_repository.zip`，将`rec/`内的文件作为YWDDLiang/rec的仓库根目录。不要把外层文件夹再嵌成`rec/rec/`。本轮读取时远端为空，本次没有push。

仅对仍为空的新仓库，可在解压目录执行：

```bash
git init -b main
git add .
git commit -m "Add auditable LLM4Rec research implementations and validation"
git remote add origin https://github.com/YWDDLiang/rec.git
git push -u origin main
```

若本地已有.git或远端已经有提交，先检查现有remote和分支；不要使用force push覆盖。Windows可以用Git Bash执行相同命令。

## Research研究资料

解压`Research_llm4rec_overlay.zip`，把里面的**llm4rec文件夹**复制到Research根目录，使结构为：

```text
Research/
  crystal/           # 原样保留
  llm4rec/           # 本次新增
  README.md          # 原样保留，可手动追加导航
```

这不是整个Research仓库替换包，不包含旧crystal的副本。可在原README追加一条导航：

```markdown
- [LLM4Rec：数据选择、互补学习与推荐研究](./llm4rec/README.md)
```

然后只stage新增目录与主动改动的README，不覆盖其他文件。若llm4rec已存在，先比对；不要盲目覆盖。overlay根README明确研究文档中的执行命令应在rec实施仓库运行。

## 上传前

检查`docs/VALIDATION.md`与负结果仍保留；确认没有把真实数据、模型权重、API key或私人研究日志加进Git；确认PPT提取文字是否适合公开。所有原始输入来源指纹和阅读日期见provenance。没有第三方baseline代码被假冒成已复现。
