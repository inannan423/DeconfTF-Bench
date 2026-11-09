# E5 预注册 v3.1：agent 外部一次性评估

**日期**：2026-11-04
**状态**：v3.1 已按审稿意见改完。条件 B 的开发检查已完成。基线复现通过之前不调用外部队列。

## 进入的变体

按 E4 运行前写死的规则：条件 C 相对 M10 的 Δlog2 均值从高到低，最多两个，再加上固定的 v3 propose–verify。不因为 Dev 上 V2 更差而把 V2 拿掉，也不改终点。

| 变体 | Dev 条件 C | 说明 |
|---|---|---|
| v3 propose–verify | 增量拉回 0/27，端到端 −0.21 bit | 固定进入。flash，prompt 在 `run_a2_pilot_design.py` |
| E4 V1 池内重排 | Δlog2 +0.004（p=0.955） | 几乎没有增益 |
| E4 V2 工具 agent | Δlog2 −0.171（p=0.018） | 比 M10 差。规则没有因此把它拿掉 |

`deepseek-v4-pro` 不进入。模型是 `deepseek-v4.1-flash`，temperature 0，`reasoning_effort=low`，`max_completion_tokens=16384`。

条件 A 和 C 的提示文本与 E4 / v3 相同。这一点在 Dev 样本上逐字核对过。

## 条件 B

B 只给 V1，只在 GEO 51 上作为描述，不进 Holm 族。v3 和 V2 要输出真实 TF 名，B 对它们不是良定义的。

做法：把该实验 M10 前 25 个候选的符号换成 `ID01`–`ID25`。映射是每个实验一次置换，种子串是 `e5-condition-b-v1:42` 加上实验编号的 SHA-256，证据数字保持不变。模型输出 ID，再反映射回符号。上下文与条件 A 相同，也就是元数据可见、真实 TF 名从元数据里抹掉。

V1 的证据行里没有差异表达基因的符号，只有 25 个候选名和 `chip_z`、`dorothea_z`、`cc`、`hub`、`exclusive_t`、`n_exclusive`。不把基因名单加进 B，否则 B 看见的信息就和 A、C 不同。ID 映射按「候选和基因共用一个符号一个 ID」来定义；这一版的基因列表是空的。

开发检查：锁定的 v3 Dev 60，每个实验一次，purpose 前缀 `e5bdev:`。事先定下：60 次里解析失败超过 12 次就删掉 B。实际存储 60/60，解析失败 8/60，花费 4.28923 元。名次没有计算。B 保留。明细在 `e5b_dev_raw.jsonl`、`e5b_dev_summary.txt`。

## 元数据

每个提示生成后检查真实 TF 名。元数据里命中的片段换成 `[PERT]` 并计数。替换之后名字还在，就停。V1 和 V2 的候选名单是假设空间，真实 TF 如果本来就在前 25 里，留在名单里，不把这个符号换成 `[PERT]`。v3 的名单和差异表达基因本来就不含真实 TF。条件 B 的整条提示里都不能出现真实 TF 或原来的候选符号。

GEO 的 `series_title`、`method_detail`、`evidence`、`mapping_evidence` 不进入任何提示。这些字段的原文如果出现在提示里，调用之前就停。

| 队列 | 条件 A | P1 |
|---|---|---|
| GEO | Biosample = cell_line，Tissue = not provided，Study = GSE 号 | 与 A 相同的三个字段，没有表达 |
| Replogle K562 | Biosample = K562，Tissue = not provided，Study = `Replogle 2022 K562 genome-wide Perturb-seq` | 不跑 |
| Unseen | Biosample、Tissue、Study，沿用 Dev 条件 A | 沿用 E1 的 P1：Biosample、Tissue、PubMed、Knock.Method、Biosample.Type、Platform、Data.Source、Profile.ID。不含 Study |

条件 C 在三个队列都是 `Biosample = REDACTED. Tissue = REDACTED. Study = REDACTED. Title = REDACTED.`

## 基线

任何外部调用之前，用冻结的 `score_knocktf_experiments` 和 `m10_soft_order(λ_bind=1.0, λ_hub=0.3)` 重建 bundle。Unseen 先对 holdout 135 打分，再切出 64。M10 名次必须和下面三份已发表表 **0 差异**，否则停，不调用：

- GEO：`r2_geo_phase2_ranks.csv` 的 `r_m10_soft`
- Unseen：`r7_m10_ranks_knocktf.csv` 里 `KnockTF_Unseen_64`
- Replogle K562 Primary 267：`r7_m10_ranks_replogle.csv` 里 `K562`

K562 的 267 沿用发表时的纳入：`target_log2fc <= -0.5` 且 `max_abs_blind >= 0.5`。`max_abs_blind` 用的是当时那份 864 行表，这样纳入集合和发表表是同一个。打分本身仍走冻结代码里的 1,888 并集。

## 名次和 Holm 族

Δlog2 = log2(M10 名次) − log2(方法名次)。解析失败或弃权时，名次留在 M10，Δlog2 = 0。

- v3 的名次是假设列表按 k=10 验证后并入 M10 的名次，和 v3 试点的端到端名次相同。
- V1、V2 的名次是返回的最多 5 个符号放在 M10 前面，和 E4 相同。条件 B 先把 ID 映回符号。

Holm 族 m = 9，只在 GEO 51 上，双侧，研究聚类符号翻转 20,000 次，seed 42。顺序固定：

1. v3 条件 C 相对 M10 的 Δlog2
2. v3 的 A−C，也就是条件 A 的 Δlog2 减去条件 C 的 Δlog2
3. v3 条件 A 的 Top-5 命中相对 P1 的 Top-5 命中，配对 0/1
4. V1 的同样三项
5. V2 的同样三项

Top-5 命中指真实 TF 出现在返回的前 5 个名字里，不是合并名次 ≤ 5。解析失败记为 0。

P1 和条件 A 是否一致，改成描述：Cohen's κ 用上述 0/1 命中；重叠率是每个实验 `|A 的前 5 个名字 ∩ P1 的前 5 个名字| / 5` 的平均。这两项不进族。

## 时间分层

GEO 51 按 `papers/deconf-regulon/agent_cutoff_geo51_dates.csv` 的 `pub_date` 分层，界是 2025-05-31。这一天及之前 29 个，之后 22 个。运行前核对这两个人数，对不上就停。分别报告每个方法的 A−C 均值，以及 P1 的 Top-5 命中。不进 Holm 族。如果增益主要来自记忆，2025-05-31 之后那一层的 A−C 和 P1 应该下降。

## 花费和执行

E5 阶段硬上限 **175 元**，按 purpose 前缀 `e5` 累计，含这次开发检查已经记入的 4.28923 元。总账本软上限仍是 450 元。

顺序：GEO → Unseen → Replogle。三个队列之间不改代码。一个队列的解析失败率在至少 20 次之后超过 20%，就停在该队列，不修复，不重跑，也不开始下一个队列。解析按一次实验的一次条件计，V2 的中间工具调用不计。

每个队列跑完立即提交原始 jsonl。三个队列都完成后才计算终点。中途不看名次、Δlog2、p 值、κ 或时间分层。

Unseen 64 在旧 M8 里已经用过，报告里写明。Replogle 的研究编号是上面那一句固定文字，整个队列是一个聚类。

## 偏离 D7

Replogle 在第一次调用前停下。`build_prompt` 把「矩阵里没有真实 TF 这一列」当成没有屏蔽。267 个里 169 列存在且为 0，98 列不存在，没有非零列。缺列时表达值进不了提示。发现这个问题时还没有看过任何 E5 终点。

只改这一条断言：列不存在，或者列存在且等于 0，都通过；列存在且非零仍然停止。差异表达名单检查和提示文本里的真实 TF 断言不动。V1 和 V2 的提示构建没有这条断言，所以没有改。

等价性：用改后的代码重放 GEO 和 Unseen 已经发出的 856 个 episode 的首轮提示（V2 只比第一轮），和 raw 里的 prompt 逐字节比较，差异 0。新哈希写在冻结清单第 7 节。通过之后才跑 Replogle 全部 267 个。

Replogle 的描述性结果再按真实 TF 是否在 `mat_blind` 的列里分成 169 和 98。只描述，不进 Holm 族。人数如果不是 169/98，终点脚本停下来。

## 偏离 D8

D7 之后，`K562_normalized_z_CTBP1` 的下调基因 `CTBP1-DT` 被原有文本断言拦住。`CTBP1-DT` 是 CTBP1 的 divergent transcript，CRISPRi 打在 CTBP1 启动子上时它会被一起敲低，所以这是真实的标签泄漏，断言拦下它是对的。v3.1 写的规则是命中就换成 `[PERT]` 并计数，代码却直接停止。发现时还没有看过任何 E5 终点。

只改 v3 的差异表达名单：基因名按连字符拆开后，如果其中一段就是真实 TF，把整个基因名换成 `[PERT]`，log2FC 保留，并计入替换次数。精确同名检查、屏蔽断言、替换之后的文本断言都不变。替换之后文本断言必须通过。受影响的是这一个实验的条件 A 和条件 C，共 2 条提示。V2 的提示不列差异表达基因，所以不动。

等价性：856 条已发首轮提示再次逐字节比较，差异 0。通过后才跑 Replogle。再被别的断言拦住就停，不自行修改。

终点算完之后补一项零成本描述，不进 Holm 族：三个队列里，下调最强 30 个或上调最强 30 个基因中，出现「连字符拆开后含真实 TF」的邻近转录本的实验各有多少。这是确定性屏蔽没有盖住的泄漏通道，写入局限性。人数在终点之后再数，规则现在写死。
