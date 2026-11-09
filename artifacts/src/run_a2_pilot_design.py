#!/usr/bin/env python3
"""Lock the A2 pilot sample and its v2 prompt. Does not call a model.

v2, declared before any completion call: 30 down and 30 up genes, DE counts
at |log2FC| >= 1, and a leakage assert. The 60 experiment ids stay the draw
from seed 42.
"""
from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/bytedance/codes/new-bishe/artifacts/src")
from budget_gate import compute_cost_cny, estimate_tokens_conservative
from develop_m10_soft_gate import compute_bundle_coverage
from regagent_tools import score_candidates

ROOT = Path("/Users/bytedance/codes/new-bishe")
OUT = ROOT / "artifacts" / "results" / "cand6_deconftf" / "agent"
PAPER = ROOT / "papers" / "deconf-regulon"
BUNDLE_PATH = ROOT / "artifacts" / "results" / "cand6_deconftf" / "knocktf_precomputed_bundle.pkl"
MAX_COMPLETION = 4096
N_PILOT = 60
SEED = 42
N_GENES = 30
_AFFIXES = ("SHRNA", "SIRNA", "KNOCKDOWN", "KNOCKOUT", "SH", "SI", "KD", "KO", "OE")


def hamilton_sample(study: pd.Series, n: int, seed: int) -> pd.DataFrame:
    """Proportional allocation across studies, then a seeded draw inside each study."""
    counts = study.value_counts().sort_index()
    raw = counts / counts.sum() * n
    base = np.minimum(np.floor(raw).astype(int), counts.to_numpy())
    base = pd.Series(base, index=counts.index)
    leftover = n - int(base.sum())
    frac = (raw - np.floor(raw)).sort_values(ascending=False)
    for sid in frac.index:
        if leftover <= 0:
            break
        if int(base.loc[sid]) < int(counts.loc[sid]):
            base.loc[sid] = int(base.loc[sid]) + 1
            leftover -= 1
    if leftover != 0:
        raise SystemExit(f"could not allocate {n} experiments, leftover {leftover}")
    rng = np.random.default_rng(seed)
    rows = []
    grouped = study.groupby(study).groups
    for sid in base.index:
        k = int(base.loc[sid])
        if k <= 0:
            continue
        members = sorted(grouped[sid])
        pick = sorted(rng.choice(members, size=k, replace=False).tolist())
        for exp_id in pick:
            rows.append({"exp_id": exp_id, "study_id": sid, "n_allocated_in_study": k})
    out = pd.DataFrame(rows)
    if len(out) != n:
        raise SystemExit(f"sample size {len(out)} != {n}")
    return out


def _core_symbol(token: str) -> str:
    core = token.upper()
    changed = True
    while changed and core:
        changed = False
        for affix in _AFFIXES:
            if core.startswith(affix) and len(core) > len(affix):
                core = core[len(affix):]
                changed = True
                break
        for affix in _AFFIXES:
            if core.endswith(affix) and len(core) > len(affix):
                core = core[: -len(affix)]
                changed = True
                break
    return core


def symbol_in_text(text: str, tf: str) -> bool:
    """True when ``tf`` stands as its own symbol, including sh/si/KO/KD attachments.

    A longer gene symbol that only contains the same letters does not match.
    ``AR`` does not match ``ARID1A``. ``ESR1`` does match ``shESR1`` and ``ESR1-KO``.
    """
    wanted = str(tf).upper()
    for match in re.finditer(r"[A-Za-z0-9]+", str(text)):
        token = match.group(0).upper()
        if token == wanted or _core_symbol(token) == wanted:
            return True
    return False


def redact_tf_tokens(text: str, tf: str) -> tuple[str, int]:
    """Replace whitespace-separated pieces that carry the TF symbol with [PERT]."""
    n = 0
    out = []
    for tok in str(text).split(" "):
        if tok and symbol_in_text(tok, tf):
            out.append("[PERT]")
            n += 1
        else:
            out.append(tok)
    return " ".join(out), n


def _extreme_genes(y: pd.Series, tf: str, sign: int) -> pd.Series:
    keep = y[[c for c in y.index if str(c).upper() != tf.upper()]]
    if sign < 0:
        picked = keep[keep < 0].sort_values().head(N_GENES)
    else:
        picked = keep[keep > 0].sort_values(ascending=False).head(N_GENES)
    return picked


def hyphen_tf_token(gene: str, tf: str) -> bool:
    """True when a hyphen-separated piece of the gene name is the TF symbol."""
    parts = str(gene).upper().split("-")
    wanted = str(tf).upper()
    return wanted in parts and str(gene).upper() != wanted


def _format_de(series: pd.Series, tf: str) -> tuple[str, int]:
    n = 0
    bits = []
    for gene, value in series.items():
        name = str(gene)
        if hyphen_tf_token(name, tf):
            name = "[PERT]"
            n += 1
        bits.append(f"{name} ({float(value):.2f})")
    return ", ".join(bits), n


def build_prompt(exp_id: str, condition: str, bundle, study_id: str, m10_order: list) -> tuple[str, int]:
    """Prompt used for both the cost estimate and the pilot. Asserts no label leak."""
    obs = bundle["obs_sub"].loc[exp_id]
    tf = str(obs["TF"]).upper()
    y = bundle["mat_blind"].loc[exp_id]
    matched = [c for c in y.index if str(c).upper() == tf]
    if any(abs(float(y[col])) > 1e-8 for col in matched):
        raise SystemExit(f"{exp_id}: true TF {tf} is not masked to 0 in mat_blind")
    down = _extreme_genes(y, tf, -1)
    up = _extreme_genes(y, tf, +1)
    for gene in list(down.index) + list(up.index):
        if str(gene).upper() == tf:
            raise SystemExit(f"{exp_id}: true TF remained in the DE list")
    n_down = int((y <= -1).sum())
    n_up = int((y >= 1).sum())
    shown = [g for g in m10_order if str(g).upper() != tf][:25]
    n_redact = 0
    if condition == "A":
        bio, n_bio = redact_tf_tokens(obs["Biosample.Name"], tf)
        tissue, n_tissue = redact_tf_tokens(obs["Tissue.Type"], tf)
        n_redact = n_bio + n_tissue
        ctx = f"Biosample = {bio}. Tissue = {tissue}. Study = {study_id}."
    elif condition == "C":
        ctx = "Biosample = REDACTED. Tissue = REDACTED. Study = REDACTED. Title = REDACTED."
    else:
        raise SystemExit(condition)
    down_txt, n_down_name = _format_de(down, tf)
    up_txt, n_up_name = _format_de(up, tf)
    n_redact += n_down_name + n_up_name
    prompt = (
        "Propose 10 transcription-factor hypotheses for the primary knockdown, "
        "most plausible first. Prefer factors outside the fixed-pipeline list when "
        "the gene changes support them. Return JSON only, no markdown. "
        'Schema: {"hypotheses": ["SYMBOL", "..."]} with exactly 10 HGNC symbols, '
        "most plausible first.\n"
        f"{ctx}\n"
        f"Strongest downregulated genes: {down_txt}\n"
        f"Strongest upregulated genes: {up_txt}\n"
        f"Downregulated genes with |log2FC| >= 1: {n_down}\n"
        f"Upregulated genes with |log2FC| >= 1: {n_up}\n"
        "Fixed-pipeline top 25, do not spend the whole list repeating these: "
        + ", ".join(shown)
        + "\n"
    )
    if symbol_in_text(prompt, tf):
        raise SystemExit(f"{exp_id} condition {condition}: true TF {tf} is in the prompt")
    return prompt, n_redact


def _power_note(sample: pd.DataFrame) -> str:
    ctrl = pd.read_csv(OUT / "a15_hypothesis_controls_per_experiment.csv")
    sub = sample.merge(ctrl, on="exp_id", suffixes=("", "_ctrl"))
    fail = sub[sub["in_recalled"] == 0]
    outside = fail[fail["r_m10"] > 25]
    bits = []
    for name in ("random_chip", "m10_rank_26_60_chip", "chea3_toprank"):
        for k in (5, 10):
            col = f"{name}_k{k}_rank"
            n = int(((outside[col] <= 25)).sum())
            bits.append(f"{name} k={k} {n}")
    return (
        f"功效在调用之前用这 60 个实验和已锁定的对照列核算。召回失败 {len(fail)}/60，"
        f"其中 M10 名次已经 ≤ 25 的有 {int((fail['r_m10'] <= 25).sum())} 个，"
        f"增量拉回的分母是 {len(outside)}。"
        f"三种对照在这 {len(outside)} 个上的增量拉回都是 0（{'; '.join(bits)}）。"
        "对照指示全是 0 时，拉回若落在不同的研究里，研究聚类双侧符号翻转"
        "（20,000 次，seed 42）在 5 次拉回时 p≈0.057，在 6 次时 p≈0.028。"
        f"要过 p < 0.05，大约需要在不同研究里拉回 6 个，相当于这 {len(outside)} 个里的 17%。"
        "同一次研究里的重复拉回更不容易显著。门槛偏严，通过线不因此放宽。"
    )


def main() -> None:
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
    study = dev.set_index("exp_id")["study_id"].astype(str)
    sample = hamilton_sample(study.loc[list(bundle["obs_sub"].index)], N_PILOT, SEED)
    locked = pd.read_csv(OUT / "a2_pilot_sample60.csv")
    if sample["exp_id"].tolist() != locked["exp_id"].tolist():
        raise SystemExit("v2 redrew the locked 60 experiment ids")
    cov = compute_bundle_coverage(bundle)

    records = []
    n_redact = 0
    n_omitted = 0
    for row in sample.itertuples(index=False):
        base = score_candidates(row.exp_id, bundle, cov_frac=cov)
        true = str(bundle["obs_sub"].loc[row.exp_id, "TF"]).upper()
        if true in [str(g).upper() for g in base["m10_order"][:25]]:
            n_omitted += 1
        for condition in ("A", "C"):
            text, n_rep = build_prompt(row.exp_id, condition, bundle, row.study_id, base["m10_order"])
            n_redact += n_rep
            n_in = estimate_tokens_conservative(text)
            cost = compute_cost_cny(n_in, MAX_COMPLETION)
            records.append({
                "exp_id": row.exp_id,
                "study_id": row.study_id,
                "condition": condition,
                "est_input_tokens": n_in,
                "max_completion_tokens": MAX_COMPLETION,
                "preflight_cost_cny": cost,
                "prompt_chars": len(text),
                "biosample_redactions": n_rep,
            })
    costs = pd.DataFrame(records)
    costs.to_csv(OUT / "a2_pilot_cost_estimate.csv", index=False)
    n_calls = len(costs)
    total = float(costs["preflight_cost_cny"].sum())
    per = costs.groupby("condition")["preflight_cost_cny"].sum()
    power = _power_note(sample)
    lines = [
        "# A2 试跑设计 v2（运行前锁定）",
        "",
        "v1 的 60 个实验不变。这一版只改 prompt 的信息量和防泄漏断言，通过线不动。k=5 仍是同一次回答里 10 个假设的前 5 个。",
        "",
        f"样本：Dev 266 按研究大小做 Hamilton 分配，研究内用 seed {SEED} 抽取，共 {sample['exp_id'].nunique()} 个实验、{sample['study_id'].nunique()} 个研究。名单在 `artifacts/results/cand6_deconftf/agent/a2_pilot_sample60.csv`。",
        "",
        "条件：A（保留生物样本、组织、研究编号）和 C（这三项以及标题都写成 REDACTED，保留基因符号和 log2FC）。B 不在这次试跑里。",
        "",
        f"每个条件给出下调最强的 {N_GENES} 个基因和上调最强的 {N_GENES} 个基因，并给出 |log2FC| ≥ 1 的下调基因数和上调基因数。只列非零的 log2FC，避免被置 0 的转录因子符号进入名单。",
        "",
        f"防泄漏：生成 prompt 后，每个实验、两个条件都断言真实 TF 的符号不在文本里。匹配把 sh/si/KO/KD 这类前后缀算进去，所以 shESR1 和 ESR1-KO 会命中；更长的基因符号只是含有同样字母时不算，AR 不命中 ARID1A。条件 A 的生物样本和组织里，含有该符号的片段换成 [PERT]。这次替换了 {n_redact} 次。mat_blind 里真实 TF 的 log2FC 已是 0，60 个实验的差异基因名单里都没有它。固定流程名单如果含有真实 TF，就从展示名单里去掉，不另补；60 个里有 {n_omitted} 个因此少显示一个符号。",
        "",
        f"费用按账本正在使用的单价：输入 2 元、输出 8 元，每百万 token。预检口径用 `budget_gate.estimate_tokens_conservative`（每 2 个字符 1 个 token，再加 64）加上 `max_completion_tokens={MAX_COMPLETION}`。",
        "",
        f"调用次数 {n_calls}（60 实验 × 2 条件）。预检上限合计 **{total:.4f} 元**。其中 A {per['A']:.4f} 元，C {per['C']:.4f} 元。单次输入 token 中位数 {int(costs['est_input_tokens'].median())}，最长 {int(costs['est_input_tokens'].max())}。按这个预检口径，基因名单从 10 个加到 30 个之后仍是 {total:.2f} 元，低于按输入长度粗估的 6–8 元：单次费用里，输出按 4096 token 封顶占了绝大部分。硬上限仍是放行时定的 12 元。",
        "",
        "运行时 temperature = 0，每次调用都经过 budget_gate 写入账本。这次试跑的花费硬上限是 12 元；下一调用的预检上限会越过 12 元时立即停止。解析失败的样本记为没有提出假设，不重试，也不改 prompt。",
        "",
        "合并规则沿用 `merge_verified_into_m10`：只把 t < 0 的假设插到 M10 前面。不在试跑后再改。",
        "",
        "## 功效",
        "",
        power,
        "",
        "## 通过线（写死）",
        "",
        "只在条件 C 上判定。k=5 和 k=10 都要过拉回这一条；端到端排名用 k=10 的合并结果。",
        "",
        "1. 增量拉回：召回失败、M10 名次 > 25、合并后名次 ≤ 25。对照是谁，由全量 Dev 的增量拉回锁定，不在这 60 个实验上重选：k=5 是 `random_chip`（4/142），k=10 是 `random_chip` 与 `m10_rank_26_60_chip` 并列（都是 5/142）。在这 60 个实验上，把条件 C 和该对照的增量拉回做成配对 0/1，条件 C 的均值要更高，并且按研究聚类的两侧符号翻转检验（20,000 次，seed 42）p < 0.05。k=10 并列时，两个对照都要这样超过。",
        "",
        "2. 端到端 Δlog2 = log2(r_M10) − log2(r_C,k=10)。按研究聚类的均值要 > 0。p 值要报告，但不另设阈值。",
        "",
        "条件 A 只描述细胞系上下文多带来的拉回，不参与通过线。",
        "",
    ]
    (PAPER / "a2_pilot_design.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
