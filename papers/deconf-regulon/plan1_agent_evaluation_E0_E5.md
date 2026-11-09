# 方案 1 执行计划：把 LLM agent 当作被严格检验的对象（E0–E5）

**日期**：2026-11-04
**用户决定**：主线走方案 1，可以用 DeepSeek API 自由试不同方向，总预算上限 500 元（账本累计 98.28 元，剩余约 401.7 元）。
**执行方**：RegAgent（chat 7b7f125e）；审稿方：chat 803f48fe。
**论文要回答的问题**：在"从扰动转录组识别原发 TF"这个任务上，LLM agent 的增益来自哪里？什么时候有增益，什么时候没有？在预注册、控制了污染的外部评估下还剩下多少？

## 总规则
- 所有调用都要经过 budget_gate 并记账；单次 temperature = 0，自洽性实验除外。
- **会话软上限**：账本累计达到 **450 元**就立即停止并汇报。每个阶段的花费如果超过预估的 1.5 倍，也停下来汇报。
- E0–E4 **只用 Dev 266**，都属于探索性分析，标为 exploratory 或 development，可以迭代，但每个变体的结果都要保留，不能覆盖。
- **Unseen 64、Replogle、GEO 只能在 E5 预注册冻结之后各跑一次。** 在那之前不能调用 LLM 处理这些队列，也不能查看它们的标签。
- 冻结文件不动（deconftf_tools 45939716…、M10 1e513f52…、GEO QC/ranks/contrasts）。
- 每个阶段单独 commit，只提交明确列出的路径，不用 `git add -A`。另一个 agent 在同一个仓库里改稿件。在 research_state.md 里加节记录。

## E0 基础设施（0 元）
- budget_gate 改成按模型计价：flash 维持 2/8 元每百万 token；`deepseek-v4-pro` 用官方 cache-miss 高峰价 $1.32 / $3.96 × 7.3 × 1.1 的安全系数，约 10.6 / 31.8 元每百万 token；其他模型一律禁止调用。
- 确认账本写入有文件锁。软上限 450 元写进代码。
- 统一的调用封装：max_completion_tokens = 16384，reasoning_effort = low（默认值），保存 raw JSON。

## E1 记忆探针：只给元数据，不给表达数据（Dev 266，flash，约 25 元）
- P1：给 Biosample、Tissue、PubMed ID 和 KnockTF 里有的实验描述字段（不能包含 TF 名，要先 assert），问"被敲低的是哪个 TF"，要求给出 10 个排序。
- P2：只给 Biosample 和 Tissue。
- 和 LOSO 细胞系频率先验（r_prior_loso）、M10、v3 条件 A 对比 Top-1 / Top-5 / Top-10。
- 意义：不看任何表达数据就能答对多少，等于"记忆加先验"的上限。

## E2 反事实元数据互换（Dev，用 v3 的 60 个，flash，约 4 元）
- 条件 A′：表达数据用实验 i 的，元数据（Biosample、Tissue、Study）换成另一个研究、另一个 TF 的实验 j 的。
- 统计 LLM 的假设或排序跟着谁走：i 的真实 TF 还是 j 的真实 TF，给出两者的命中率。
- 意义：直接检验 A 的增益是"元数据驱动"还是"表达驱动"。

## E3 更强的 DeepSeek 模型（deepseek-v4-pro，Dev，约 25–30 元）
- 从 Dev 中剩下的 146 个实验里，用 Hamilton 法、seed 44 抽 60 个（不能和 v2、v3 重叠），prompt 和条件 A/C 完全照搬 v3，通过线也和 v3 一样（只看条件 C）。
- 先小测 5 次，看 finish_reason。
- 意义：回答"是不是模型太弱"。

## E4 agent 设计变体（Dev，用 v3 的 60 个，以便和 v3 直接比较，flash，约 50–70 元）
- **V1 池内重排**：给出 M10 Top-25 和每个候选的证据（ChIP_Z、DoRothEA_Z、CC、hub 标记、专属靶基因检验），让模型重排并输出 Top-5。对应约 40% 的池内失败（旁系同源、级联）。条件 A 和 C。
- **V2 ReAct 工具调用 agent**：模型可以多轮调用 `verify_hypothesis`、`exclusive_target_test`、`cascade_check`，最多 6 次，看到工具结果后再给出 Top-5 或弃权。工具结果是确定性的，模型不能编数值。条件 A 和 C。这是真正意义上的"agent"。
- 每个变体都报告：Top-1、Top-5、相对 M10 的 Δlog2 rank（按研究聚类）、A−C 差值、解析失败率、平均调用次数和成本。
- 只能根据 Dev 结果选出 **最多 2 个**变体进入 E5，选择标准事先写好：C 条件下相对 M10 的 Δlog2 均值最高，并列时取成本较低的。v3 的 propose–verify 作为一个固定变体，一并进入 E5。

## E5 预注册的外部一次性评估（要先冻结，并经审稿方审完才能跑，约 150–180 元）
- 冻结：被选中的变体代码和 prompt（commit 加 hash），写进 FROZEN_MANIFEST v3。
- 队列：GEO 51（主队列，干净外部）；Replogle K562 Primary 267（外部，元数据单一）；Unseen 64（开发后已看过，注明 M8 曾在这上面跑过）。
- 条件：A、C；在 GEO 51 上加跑 B（符号盲化）；在 GEO 51 和 Unseen 上加跑 E1 的元数据探针 P1。
- 主终点（Holm 族事先定死）：在 GEO 51 上，每个变体 C 条件相对 M10 的 Δlog2 rank（按研究聚类置换）；A−C 差值；P1 的 Top-5 和 A 条件 Top-5 的关系。
- 所有结果都可以发表：有增益就写"增益来自哪里"，没有增益就写"预注册的负结果"。

## 汇报节点
- E0–E4 可以连续跑，不用停，但每个阶段都要 commit，并在 research_state 里记录。
- **E4 结束、E5 预注册写好并 commit 之后，必须停下来，等审稿方审完再跑 E5。**
- 如果出现以下任一情况，立即停下汇报：累计花费超过 450 元；某阶段超过预算的 1.5 倍；解析失败率 > 20%；发现泄漏。
