# Europe PMC 2021 holdout：预先固定的第六句概率实验

本方案在读取任何新样本模型分数之前写定。数据仅来自 Europe PMC 官方 REST `search` 和 `{PMCID}/fullTextXML`，不使用 S2ORC、摘要、搜索片段或其他镜像。旧 20 篇只用于排除和对照，旧实验文件不改动。

## 样本规则（v1，不因评分结果更改）

- 按 2021 年 1–12 月顺序处理，每月最多入选 10 篇，总目标 120 篇；每月达到 10 篇即停止检查该月。完整采集 12 个月后，少于 100 篇则停止，不加载模型。
- 沿用旧 `experiments/europe_pmc_pilot/collect.py` 的检索约束，逐月使用 `OPEN_ACCESS:Y AND LANG:eng AND PUB_TYPE:"research article" AND HAS_ABSTRACT:Y AND FIRST_PDATE:[YYYY-MM-01 TO YYYY-MM-最后一天]`。`HAS_ABSTRACT:Y` 只用于候选检索，模型输入和目标均仅来自 XML 正文。官方搜索端点 `https://www.ebi.ac.uk/europepmc/webservices/rest/search`；`sort=P_PDATE_D asc`，`resultType=lite`，`synonym=false`，`cursorMark=*` 开始，`pageSize=100` 逐页，最多保留搜索返回顺序的前 1000 个带 PMCID 的候选。缺 PMCID 的行另计，不参与这 1000 个。记录每月查询时间、逐页响应元信息、原始候选顺序及去重后的候选池。
- 每月候选 PMCID 去重后，以 `SHA-256(UTF-8("europe-pmc-holdout-2021-v1|" + PMCID))` 的十六进制升序排序；哈希相同则按 PMCID 升序。只在此顺序中检查全文，不按模型分数挑选。跨月重复 PMCID、旧 pilot 的 20 个 PMCID、元数据日期不在对应月份、非 OA/非 research article、XML 不可取得或不可解析、XML PMCID 不符、XML 非英文或非 `research-article`、无正文或不足六个合格完整句子，都有固定排除原因。
- 还排除缺少可靠 journal 名称的文章。journal 由 XML `front/journal-meta/journal-title-group/journal-title` 取得，退回到搜索结果的 `journalTitle` 仅在 XML 中缺失时使用；名称按 Unicode NFKC、去两端空格、内部空格归一化、casefold 作为配额键。全部月份中每个 journal 最多入选 3 篇。XML 和搜索 journal 名不一致时，以 XML 为准并记录两者。旧 20 篇不占新配额。
- 正文提取完全调用旧 `collect.py` 的 `body_paragraphs`、`clean_paragraph`、`split_complete_sentences` 和 `extract_first_five`：只读 `<article>/<body>` 内非排除容器中的 `<p>`，跳过图表、标题、文献、附录、公式等指定节点；删除交叉引用及括号引文、折叠空格；保护固定缩写，仅保留满足旧规则的前五个完整英文句子，要求第六个完整句子。第六句必须保存**全文**，不使用旧 manifest 的 250 字符摘录。五句和第六句记录各自正文段落序号及清理计数；第六句绝不进入提示。
- HTTP 请求串行，正常请求至少相隔 0.5 秒。对 429/500/502/503/504 最多 4 次尝试，指数退避 1、2、4 秒，并尊重更长的有效 `Retry-After`；仍失败则保存采集状态并停止。XML 单篇最多 20 MB，不批量下载语料。只有入选 XML 保存为原始文件，并记录实际字节数及 SHA-256。
- 每个候选的检查结果及入选名单逐条原子保存；每个月的候选池在检查 XML 前整体保存。全部月份结束才写不可更改的 `selection_manifest.json`，其 SHA-256 用于后续预检和评分结果绑定。中断恢复时读取已有池和检查记录，不重新排序或重复已完成的 XML 检查。

## 模型评分（仅名单锁定并预检通过后）

模型固定为本地原版 `Qwen/Qwen2-7B-Instruct` revision `f2826a00ceef68f0f2b946d945ecc0477ce4450c`。完全沿用旧生成实验的 system/user 文本和 Qwen chat template；A 官方、B 原 merge seed42 shuffle、D 补全后 seed42 shuffle、E D 中原规则相对顺序，各组只有提示 token ID 不同。对第六句**仅由 A 官方 tokenizer 编码一次**，四组共用同一目标 ID；拼接提示 ID 和目标 ID，目标第 `j` 个 token 用绝对位置 `prompt_length+j-1` 的 logits 预测，提示 token 不计损失。MPS/BF16、`eval()`、`inference_mode()`、`use_cache=False`、float32 logits、单篇单组 teacher forcing；每次评分后原子保存以支持中断恢复。

预先确定每篇等权的平均 NLL 汇总。主要对比 D−E，另给 D−B 与 D−A，各报告均值、差值中位数和 D 较低的篇数；可用固定 seed=2021 的 10000 次论文级有放回 bootstrap 报描述性 95% 百分位区间，不进行结果驱动的样本剔除。这是真实第六句概率指标，不是 MAUVE，也不能证明开放式生成质量。

官方接口说明：[Europe PMC RESTful Web Service](https://europepmc.org/RestfulWebService)。
