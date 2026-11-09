#!/usr/bin/env python3
"""E1 memory probe on Dev 266. Metadata only, no expression. Uses flash.

P1 gets the experiment description fields that do not name the TF.
P2 gets biosample and tissue only. Both ask for 10 ranked TF symbols.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/Users/bytedance/codes/new-bishe/artifacts/src")
from budget_gate import (
    SESSION_SOFT_CAP_CNY,
    BudgetExceededError,
    BudgetLedger,
    gated_chat,
)
from run_a2_pilot import parse_hypotheses
from run_a2_pilot_design import redact_tf_tokens, symbol_in_text
from run_e_common import StageGuard, load_jsonl, stage_spent

ROOT = Path("/Users/bytedance/codes/new-bishe")
OUT = ROOT / "artifacts" / "results" / "cand6_deconftf" / "agent"
BUNDLE_PATH = ROOT / "artifacts" / "results" / "cand6_deconftf" / "knocktf_precomputed_bundle.pkl"
RAW_PATH = OUT / "e1_memory_raw.jsonl"
FROZEN = {
    ROOT / "artifacts" / "src" / "deconftf_tools.py":
        "45939716963851d980d5767d9285832efd5b4b3537ce014b0e44f7affa225490",
    ROOT / "artifacts" / "src" / "develop_m10_soft_gate.py":
        "1e513f5236e56842e5ff96de3263175498ba9c1e6662ab8f228de3a1d7500677",
}
ESTIMATE_CNY = 25.0
WORKERS = 4
P1_FIELDS = (
    "Knock.Method",
    "Biosample.Type",
    "Platform",
    "Data.Source",
    "Profile.ID",
)
EXCLUDED = ("TF", "TF.Class", "TF.Superclass", "logFC")


def assert_frozen() -> None:
    for path, expected in FROZEN.items():
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != expected:
            raise SystemExit(f"frozen hash changed: {path.name} {got}")


def dev_table(bundle) -> pd.DataFrame:
    labels = pd.read_csv(
        ROOT / "artifacts" / "results" / "cand6_deconftf" / "knocktf401_ranks_with_study_ids.csv",
        usecols=["exp_id", "study_id", "split_new"],
    )
    dev = labels[labels["split_new"] == "Dev"]
    if set(dev["exp_id"]) != set(bundle["obs_sub"].index):
        raise SystemExit("bundle is not exactly Dev 266")
    obs = bundle["obs_sub"].copy()
    obs.index = obs.index.astype(str)
    obs["study_id"] = dev.set_index("exp_id")["study_id"].astype(str).reindex(obs.index).to_numpy()
    obs["exp_id"] = obs.index
    return obs


def _field(obs_row, name: str, tf: str) -> tuple[str, int]:
    text, n = redact_tf_tokens(obs_row[name], tf)
    if symbol_in_text(text, tf):
        raise SystemExit(f"leakage: {obs_row.name} field {name} still contains {tf}")
    return text, n


def build_probe(obs_row, probe: str) -> tuple[str, int]:
    tf = str(obs_row["TF"]).upper()
    bio, n_bio = _field(obs_row, "Biosample.Name", tf)
    tissue, n_tissue = _field(obs_row, "Tissue.Type", tf)
    n_redact = n_bio + n_tissue
    lines = [f"Biosample = {bio}.", f"Tissue = {tissue}."]
    if probe == "P1":
        pubmed, n_pub = _field(obs_row, "Pubmed.ID", tf)
        n_redact += n_pub
        lines.append(f"PubMed = {pubmed}.")
        for name in P1_FIELDS:
            value, n_one = _field(obs_row, name, tf)
            n_redact += n_one
            lines.append(f"{name} = {value}.")
    elif probe != "P2":
        raise SystemExit(probe)
    prompt = (
        "Which transcription factor was knocked down in this experiment? "
        "You are given metadata only. No expression values are provided. "
        "Return JSON only, no markdown. "
        'Schema: {"hypotheses": ["SYMBOL", "..."]} with exactly 10 HGNC symbols, '
        "most plausible first.\n"
        + "\n".join(lines)
        + "\n"
    )
    if symbol_in_text(prompt, tf):
        raise SystemExit(f"leakage: {obs_row.name} {probe} still contains {tf}")
    for name in EXCLUDED:
        raw = str(obs_row[name])
        if raw and raw in prompt:
            raise SystemExit(f"leakage: {obs_row.name} {probe} contains excluded field {name}")
    return prompt, n_redact


def _one(obs_row, probe: str, prompt: str, guard: StageGuard) -> str:
    est = 0.20
    reason = guard.reserve(est)
    if reason:
        return reason
    record = {
        "exp_id": obs_row.name,
        "study_id": str(obs_row["study_id"]),
        "probe": probe,
        "prompt": prompt,
        "error": None,
    }
    try:
        res = gated_chat(
            [{"role": "user", "content": prompt}],
            purpose=f"e1:{obs_row.name}:{probe}",
            model="deepseek-v4.1-flash",
            raw_path=RAW_PATH,
            raw_record=record,
            timeout_sec=300,
        )
    except BudgetExceededError as err:
        guard.settle(est, 0.0, False)
        return f"budget_gate stopped the run: {err}"
    except Exception as err:
        record["error"] = f"{type(err).__name__}: {err}"[:500]
        blob = json.dumps(record, ensure_ascii=False)
        if "Bearer " in blob or "LLMBOX_API_KEY" in blob:
            raise SystemExit("refusing to store a response that contains a credential")
        import fcntl
        with RAW_PATH.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(blob + "\n")
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return guard.settle(est, 0.0, False)
    parsed = parse_hypotheses(res.get("content") or "")
    cost = float((res.get("ledger_entry") or {}).get("cost_cny") or 0.0)
    print(
        f"{obs_row.name} {probe} parse={parsed is not None} spent={guard.spent + cost:.2f}",
        flush=True,
    )
    return guard.settle(est, cost, parsed is not None)


def run_calls(obs: pd.DataFrame) -> str:
    done = {(row["exp_id"], row["probe"]) for row in load_jsonl(RAW_PATH)}
    prompts = {}
    n_redact = 0
    for exp_id, row in obs.iterrows():
        for probe in ("P1", "P2"):
            prompt, n_one = build_probe(row, probe)
            prompts[(exp_id, probe)] = prompt
            n_redact += n_one
    print(f"leakage asserts passed for {len(prompts)} prompts, redacted tokens {n_redact}", flush=True)
    jobs = [(exp_id, probe) for exp_id in obs.index for probe in ("P1", "P2") if (exp_id, probe) not in done]
    guard = StageGuard(ESTIMATE_CNY, stage_spent(load_jsonl(RAW_PATH)))
    if not jobs:
        return guard.reason
    stop = ""
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = []
        for exp_id, probe in jobs:
            if guard.reason:
                break
            futures.append(pool.submit(_one, obs.loc[exp_id], probe, prompts[(exp_id, probe)], guard))
        for fut in as_completed(futures):
            reason = fut.result()
            if reason:
                stop = reason
    return stop or guard.reason


def _hits(hyps: list, true: str) -> dict:
    return {
        "top1": int(bool(hyps) and hyps[0] == true),
        "top5": int(true in hyps[:5]),
        "top10": int(true in hyps[:10]),
    }


def _rate(frame: pd.DataFrame, col: str) -> str:
    if frame.empty or col not in frame.columns:
        return "n=0"
    return f"{int(frame[col].fillna(0).sum())}/{len(frame)}"


def prior_ranks(obs: pd.DataFrame, cand: list) -> pd.Series:
    """Dev-only leave-one-study-out frequency. Does not read holdout labels."""
    ranks = {}
    tf = obs["TF"].astype(str).str.upper()
    for exp_id, row in obs.iterrows():
        other = obs[obs["study_id"] != row["study_id"]]
        bio = other.loc[other["Biosample.Name"] == row["Biosample.Name"], "TF"].astype(str).str.upper().value_counts()
        tis = other.loc[other["Tissue.Type"] == row["Tissue.Type"], "TF"].astype(str).str.upper().value_counts()
        glob = other["TF"].astype(str).str.upper().value_counts()
        score = {c: 100.0 * bio.get(c, 0) + 10.0 * tis.get(c, 0) + 1.0 * glob.get(c, 0) for c in cand}
        order = sorted(cand, key=lambda c: (-score[c], c))
        ranks[exp_id] = order.index(tf.loc[exp_id]) + 1
    return pd.Series(ranks, name="r_prior_dev")


def evaluate(obs: pd.DataFrame, stop: str) -> None:
    raw = load_jsonl(RAW_PATH)
    by_key = {(row["exp_id"], row["probe"]): row for row in raw}
    ctrl = pd.read_csv(OUT / "a15_hypothesis_controls_per_experiment.csv", usecols=["exp_id", "r_m10"])
    ctrl = ctrl.set_index("exp_id")
    bundle = pickle.loads(BUNDLE_PATH.read_bytes())
    cand = [str(tf).upper() for tf in bundle["cand_tfs"]]
    prior = prior_ranks(obs, cand)
    v3 = pd.read_csv(OUT / "a2v3_pilot_per_experiment.csv", usecols=["exp_id", "A_hypotheses", "A_first_hit"])
    rows = []
    n_fail = {"P1": 0, "P2": 0}
    n_call = {"P1": 0, "P2": 0}
    for exp_id, row in obs.iterrows():
        true = str(row["TF"]).upper()
        out = {
            "exp_id": exp_id,
            "study_id": row["study_id"],
            "true_tf": true,
            "biosample": row["Biosample.Name"],
            "r_m10": int(ctrl.loc[exp_id, "r_m10"]),
            "r_prior_dev": int(prior.loc[exp_id]),
        }
        for probe in ("P1", "P2"):
            got = by_key.get((exp_id, probe))
            if got is None:
                out[f"{probe}_missing"] = 1
                continue
            out[f"{probe}_missing"] = 0
            n_call[probe] += 1
            hyps = parse_hypotheses(got.get("content") or "")
            if hyps is None:
                n_fail[probe] += 1
                hyps = []
            hit = _hits([h.upper() for h in hyps], true)
            out[f"{probe}_parse_ok"] = int(bool(hyps) or parse_hypotheses(got.get("content") or "") is not None)
            out[f"{probe}_top1"] = hit["top1"]
            out[f"{probe}_top5"] = hit["top5"]
            out[f"{probe}_top10"] = hit["top10"]
            out[f"{probe}_hypotheses"] = "|".join(h.upper() for h in hyps)
            entry = got.get("ledger_entry") or {}
            out[f"{probe}_cost_cny"] = entry.get("cost_cny")
        rows.append(out)
    df = pd.DataFrame(rows)
    df["m10_top1"] = (df["r_m10"] <= 1).astype(int)
    df["m10_top5"] = (df["r_m10"] <= 5).astype(int)
    df["m10_top10"] = (df["r_m10"] <= 10).astype(int)
    df["prior_top1"] = (df["r_prior_dev"] <= 1).astype(int)
    df["prior_top5"] = (df["r_prior_dev"] <= 5).astype(int)
    df["prior_top10"] = (df["r_prior_dev"] <= 10).astype(int)
    df.to_csv(OUT / "e1_memory_per_experiment.csv", index=False)
    lines = [
        "E1 memory probe. Exploratory. Dev 266 only. No expression values were sent.",
        stop or "all scheduled calls finished",
        f"P1 calls {n_call['P1']}, parse failures {n_fail['P1']}. "
        f"P2 calls {n_call['P2']}, parse failures {n_fail['P2']}.",
        f"Stage spend {stage_spent(raw):.4f} CNY. Session soft cap {SESSION_SOFT_CAP_CNY:.0f}.",
        "Excluded from every prompt: TF symbol, TF.Class, TF.Superclass, logFC.",
        "Prior is leave-one-study-out frequency on Dev 266 only (biosample 100, tissue 10, global 1).",
    ]
    p1 = df[df["P1_missing"].fillna(1) == 0] if "P1_missing" in df.columns else df.iloc[0:0]
    p2 = df[df["P2_missing"].fillna(1) == 0] if "P2_missing" in df.columns else df.iloc[0:0]
    for k, label in ((1, "top1"), (5, "top5"), (10, "top10")):
        lines.append(
            f"Top-{k}: P1 {_rate(p1, f'P1_{label}')}, P2 {_rate(p2, f'P2_{label}')}, "
            f"Dev prior {_rate(p1, f'prior_{label}')}, M10 {_rate(p1, f'm10_{label}')}"
        )
    if len(p1):
        lines.append(
            f"Of P1 Top-10 hits, Dev prior also Top-10 in "
            f"{int(((p1['P1_top10'] == 1) & (p1['prior_top10'] == 1)).sum())}"
            f"/{int((p1['P1_top10'] == 1).sum())}."
        )
    overlap = p1[p1["exp_id"].isin(set(v3["exp_id"]))]
    if len(overlap):
        merged = overlap.merge(v3, on="exp_id", how="left")

        def listed(text: str, k: int, true: str) -> int:
            hyps = [part for part in str(text).split("|") if part and part != "nan"]
            return int(true in hyps[:k])

        merged["v3A_top1"] = [
            listed(text, 1, true) for text, true in zip(merged["A_hypotheses"], merged["true_tf"])
        ]
        merged["v3A_top5"] = [
            listed(text, 5, true) for text, true in zip(merged["A_hypotheses"], merged["true_tf"])
        ]
        merged["v3A_top10"] = [
            listed(text, 10, true) for text, true in zip(merged["A_hypotheses"], merged["true_tf"])
        ]
        lines.append(f"On the {len(merged)} v3 experiments that also have a P1 answer:")
        for k, label in ((1, "top1"), (5, "top5"), (10, "top10")):
            lines.append(
                f"  Top-{k}: P1 {_rate(merged, f'P1_{label}')}, P2 {_rate(merged, f'P2_{label}')}, "
                f"v3 condition A {_rate(merged, f'v3A_{label}')}, "
                f"Dev prior {_rate(merged, f'prior_{label}')}, M10 {_rate(merged, f'm10_{label}')}"
            )
    text = "\n".join(lines) + "\n"
    (OUT / "e1_memory_summary.txt").write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    assert_frozen()
    spent = BudgetLedger().total_spent_cny()
    if spent >= SESSION_SOFT_CAP_CNY:
        raise SystemExit(f"session soft cap already reached: {spent:.4f}")
    bundle = pickle.loads(BUNDLE_PATH.read_bytes())
    if len(bundle["obs_sub"]) != 266:
        raise SystemExit("bundle is not Dev 266")
    obs = dev_table(bundle)
    stop = run_calls(obs)
    if stop:
        print(stop, flush=True)
    evaluate(obs, stop)
    assert_frozen()


if __name__ == "__main__":
    main()
