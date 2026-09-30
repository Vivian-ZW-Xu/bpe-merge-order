# Europe PMC 正文句子离线审计

本目录只包含审计代码与新生成的记录。没有调用模型、tokenizer、生成、teacher forcing、MAUVE、网络接口或下载。旧 XML、清单、脚本和结果均只读。

## 复核范围与判定

- `audit.py` 用标准库读取 120+20 份本地 XML，调用旧 `experiments/europe_pmc_pilot/collect.py` 的既定正文遍历、清理、切句函数，独立记录原始 `<body>` 段落路径、前六句、第五句前至第六句后的上下文、原始 XML 段落与机器可疑标记。逐篇核对文件大小、SHA-256、XML PMCID、选择清单、评分/生成记录的 PMCID、顺序、输入和完整目标，以及提示是否只含前五句。`scan_records_v5.json` 是扩展模式的最终扫描版本；早期中间扫描已清理。
- `finalize.py` 依据原始 XML 回看的显式决策表写 `audit_records.json`，并只用旧 NLL 结果计算 `sensitivity_after_confirmed_errors.json`。两者均不替换主实验结果。脚本拒绝覆盖已有同名审计文件。
- 2021 批次：120 篇中 21 篇确定有句子/正文内容错误；53 篇在人工查看边界上下文后通过；46 篇待人工判断，其中 3 篇涉及开头 `<boxed-text>` 是否计入正文的定义，43 篇仅通过全面的自动结构与边界筛查，尚未逐篇人工判断语义完整性。旧 pilot：20 篇中 1 篇图号删除导致残句，19 篇人工边界审阅通过。
- 实际人工查看了 97 篇的前六句边界上下文：2021 批次全部 50 篇自动标记文章、每月首尾各一篇未标记文章共 24 篇、另 3 篇 `<boxed-text>` 文章，以及旧 pilot 全部 20 篇。其余 43 篇仅做全量自动核查，保守标为待人工判断。人工查看的是相关段落和边界，不是整篇文章的同行评审。
- `audit_records.json` 为 140 篇逐条记录；`boundary_context` 保留清理后的第六句前文、完整第六句与后文，以及来源 XML 段落。确定错误还含 `error_source_xpath`、原始 `error_source_paragraph_xml`、错误位置与可能正确的边界。判定字段为 `通过`、`确定错误`、`待人工判断`。

## 事后敏感性与复跑

原 120 篇预先锁定的统计是主结果。事后只排除 21 篇已确认错误，保留全部 46 篇待判断者，得到 99 篇的 D−E、D−B、D−A 均值及 D 较低的篇数。99 篇数字只是诊断筛选错误影响，不能当作预先固定的正式结果。

在项目根目录执行（审计文件若已存在，脚本会拒绝覆盖）：

```bash
.venv/bin/python experiments/sentence_data_audit/audit.py
.venv/bin/python experiments/sentence_data_audit/finalize.py
```

需要重新执行时，在另一个空审计目录中复制脚本并修改输出路径；原始审计记录保留不动。
