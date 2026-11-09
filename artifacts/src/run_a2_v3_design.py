#!/usr/bin/env python3
"""Lock the A2 v3 sample and budget. Does not call a model.

The prompt function is the v2 function. Only the completion cap, the sample,
and the seed change. v2's NO-GO is not rewritten.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/Users/bytedance/codes/new-bishe/artifacts/src")
from budget_gate import compute_cost_cny, estimate_tokens_conservative
from develop_m10_soft_gate import compute_bundle_coverage
from regagent_tools import score_candidates
from run_a2_pilot_design import N_PILOT, build_prompt, hamilton_sample

ROOT = Path("/Users/bytedance/codes/new-bishe")
OUT = ROOT / "artifacts" / "results" / "cand6_deconftf" / "agent"
PAPER = ROOT / "papers" / "deconf-regulon"
BUNDLE_PATH = ROOT / "artifacts" / "results" / "cand6_deconftf" / "knocktf_precomputed_bundle.pkl"
MAX_COMPLETION = 16384
SEED = 43
SMOKE_N = 5


def descriptive_v2() -> str:
    """First-hypothesis hits on the calls whose JSON parsed. Not a gate."""
    df = pd.read_csv(OUT / "a2_pilot_per_experiment.csv")
    lines = [
        "v2 可解析回答的描述统计。不参与 v2 或 v3 的通过线。",
        "可解析指 parse_ok=1。v2 共 18 次条件 A 和 16 次条件 C，合计 34 次回答。",
    ]
    for cond, label in (("A", "条件 A"), ("C", "条件 C")):
        sub = df[df[f"{cond}_parse_ok"] == 1]
        fail = sub[sub["in_recalled"] == 0]
        lines.append(
            f"{label}：可解析 {len(sub)} 个实验，第 1 个假设命中 {int(sub[f'{cond}_first_hit'].sum())}/{len(sub)}。"
            f"其中召回失败 {len(fail)} 个，命中 {int(fail[f'{cond}_first_hit'].sum())}/{len(fail)}。"
            f"同一批实验上，随机有 ChIP 轨道的第 1 个抽取命中 "
            f"{int(sub['random_first_hit'].sum())}/{len(sub)}，"
            f"召回失败子集 {int(fail['random_first_hit'].sum())}/{len(fail)}。"
        )
    either = df[(df["A_parse_ok"] == 1) | (df["C_parse_ok"] == 1)]
    lines.append(f"至少有一个条件可解析的实验是 {len(either)} 个，不是 34。34 是两次条件的回答数。")
    both = df[(df["A_parse_ok"] == 1) & (df["C_parse_ok"] == 1)]
    lines.append(f"两个条件都可解析的实验是 {len(both)} 个。")
    text = "\n".join(lines) + "\n"
    (OUT / "a2_v2_parseable_descriptive.txt").write_text(text, encoding="utf-8")
    return text


def main() -> None:
    desc = descriptive_v2()
    bundle = pickle.loads(BUNDLE_PATH.read_bytes())
    if len(bundle["obs_sub"]) != 266:
        raise SystemExit("bundle is not Dev 266")
    labels = pd.read_csv(
        ROOT / "artifacts" / "results" / "cand6_deconftf" / "knocktf401_ranks_with_study_ids.csv",
        usecols=["exp_id", "study_id", "split_new"],
    )
    dev = labels[labels["split_new"] == "Dev"]
    if set(dev["exp_id"]) != set(bundle["obs_sub"].index):
        raise SystemExit("bundle index is not exactly Dev")
    v2 = pd.read_csv(OUT / "a2_pilot_sample60.csv")
    if len(v2) != 60 or v2["exp_id"].nunique() != 60:
        raise SystemExit("v2 sample is not 60 unique experiments")
    study = dev.set_index("exp_id")["study_id"].astype(str)
    remain = study.drop(index=list(v2["exp_id"]))
    if len(remain) != 206:
        raise SystemExit(f"remaining Dev experiments are {len(remain)}, expected 206")
    sample = hamilton_sample(remain, N_PILOT, SEED)
    overlap = set(sample["exp_id"]) & set(v2["exp_id"])
    if overlap:
        raise SystemExit(f"v3 sample overlaps v2: {sorted(overlap)[:5]}")
    sample.to_csv(OUT / "a2v3_pilot_sample60.csv", index=False)

    ctrl = pd.read_csv(OUT / "a15_hypothesis_controls_per_experiment.csv")
    joined = sample.merge(ctrl, on="exp_id", suffixes=("", "_ctrl"))
    if len(joined) != 60:
        raise SystemExit("control table is missing v3 experiments")
    outside = (joined["in_recalled"] == 0) & (joined["r_m10"] > 25)
    control_rows = []
    control_bits = []
    for name, k in (
        ("random_chip", 5),
        ("random_chip", 10),
        ("m10_rank_26_60_chip", 10),
    ):
        col = f"{name}_k{k}_rank"
        flag = (outside & (joined[col] <= 25)).astype(int)
        control_rows.append(flag.rename(f"{name}_k{k}_incremental"))
        control_bits.append(f"{name} k={k} {int(flag.sum())}/{int(outside.sum())}")
    indicators = pd.DataFrame({
        "exp_id": joined["exp_id"],
        "study_id": joined["study_id"],
        "in_recalled": joined["in_recalled"],
        "r_m10": joined["r_m10"],
        "outside_m10_top25_recall_fail": outside.astype(int),
    })
    for series in control_rows:
        indicators[series.name] = series.to_numpy()
    indicators.to_csv(OUT / "a2v3_control_indicators.csv", index=False)

    cov = compute_bundle_coverage(bundle)
    records = []
    for row in sample.itertuples(index=False):
        base = score_candidates(row.exp_id, bundle, cov_frac=cov)
        for condition in ("A", "C"):
            text, _n_rep = build_prompt(
                row.exp_id, condition, bundle, str(row.study_id), base["m10_order"],
            )
            n_in = estimate_tokens_conservative(text)
            records.append({
                "exp_id": row.exp_id,
                "study_id": row.study_id,
                "condition": condition,
                "est_input_tokens": n_in,
                "max_completion_tokens": MAX_COMPLETION,
                "preflight_cost_cny": compute_cost_cny(n_in, MAX_COMPLETION),
                "prompt_chars": len(text),
            })
    costs = pd.DataFrame(records)
    costs.to_csv(OUT / "a2v3_pilot_cost_estimate.csv", index=False)
    total = float(costs["preflight_cost_cny"].sum())
    per = costs.groupby("condition")["preflight_cost_cny"].sum()
    smoke_ids = sample["exp_id"].head(SMOKE_N).tolist()
    smoke_cost = float(costs[(costs["condition"] == "C") & (costs["exp_id"].isin(smoke_ids))]["preflight_cost_cny"].sum())
    n_fail = int((joined["in_recalled"] == 0).sum())
    lines = [
        "# A2 试跑设计 v3（运行前锁定）",
        "",
        "v2 的 NO-GO 保留，不覆盖。这一版不改 prompt 文本、条件、k、合并规则和通过线。只换样本，并把 completion 上限从 4096 提到 16384。",
        "",
        f"样本：从 v2 没用到的 {len(remain)} 个 Dev 实验里，按同样的 Hamilton 分配、研究内抽取，seed {SEED}。共 {sample['exp_id'].nunique()} 个实验、{sample['study_id'].nunique()} 个研究，与 v2 的交集是 0。名单在 `artifacts/results/cand6_deconftf/agent/a2v3_pilot_sample60.csv`。",
        "",
        "prompt 仍调用 `run_a2_pilot_design.build_prompt`。温度仍是 0。推理不关闭。",
        "",
        "推理长度：DeepSeek 官方 Thinking Mode 文档（https://api-docs.deepseek.com/guides/thinking_mode ，2026-11-03 读取）里，OpenAI 格式的档位是 `reasoning_effort` = low / high / max。`medium` 映射到 high，不是更短的一档。`none` 会关掉推理，本轮不用。文档没有单独的推理 token 整数上限；`max_completion_tokens` 同时覆盖推理和答案。v2 已经传了 `reasoning_effort=low`，答案仍被 4096 截断。v3 继续传 low，这是仍保留推理时的最低档，并把共用上限改为 16384。",
        "",
        f"费用按账本单价：输入 2 元、输出 8 元，每百万 token。预检用 `estimate_tokens_conservative` 加上 `max_completion_tokens={MAX_COMPLETION}`。",
        "",
        f"全部 120 次的预检上限合计 **{total:.4f} 元**。其中 A {per['A']:.4f} 元，C {per['C']:.4f} 元。单次输入 token 中位数 {int(costs['est_input_tokens'].median())}，最长 {int(costs['est_input_tokens'].max())}。这次试跑的花费硬上限是 25 元。下一调用的预检会越过 25 元时停止。",
        "",
        f"运行前小测：名单前 {SMOKE_N} 个实验，只跑条件 C。实验 id：{', '.join(smoke_ids)}。这 {SMOKE_N} 次的预检上限合计 {smoke_cost:.4f} 元。若 finish_reason=length 达到 2 次，或这 {SMOKE_N} 次里有调用没有返回 finish_reason，就停止，不跑其余调用。小测通过后才跑完全部条件 A 和其余条件 C。已完成的小测调用不重复。",
        "",
        f"新样本上，召回失败 {n_fail}/60，增量拉回分母 {int(outside.sum())}。全量 Dev 锁定的对照在这批实验上的增量拉回（运行前从已有对照列读出，不重新抽）：{'; '.join(control_bits)}。判定时 k=5 仍对 random_chip，k=10 仍对 random_chip 和 m10_rank_26_60_chip，不在这 60 个上重选对照。",
        "",
        "通过线与 v2 相同，只看条件 C。k=5 和 k=10 的增量拉回都要在配对 0/1 上均值更高，研究聚类两侧符号翻转（20,000 次，seed 42）p < 0.05。k=10 两个对照都要超过。端到端 Δlog2 = log2(r_M10) − log2(r_C,k=10)，研究聚类均值要 > 0，p 值只报告。条件 A 只描述。解析失败仍算没有提出假设，不重试，不改 prompt。v3 之后不再安排 v4。",
        "",
        "## v2 可解析回答（只描述）",
        "",
        desc.rstrip(),
        "",
    ]
    (PAPER / "a2_v3_pilot_design.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
