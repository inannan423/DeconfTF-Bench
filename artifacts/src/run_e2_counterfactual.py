#!/usr/bin/env python3
"""E2: expression from experiment i, metadata from a different study's experiment j.

Uses the locked v3 Dev sample and flash. Exploratory. Does not touch holdout data.
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
from budget_gate import SESSION_SOFT_CAP_CNY, BudgetExceededError, BudgetLedger, gated_chat
from develop_m10_soft_gate import compute_bundle_coverage
from regagent_tools import score_candidates
from run_a2_pilot import parse_hypotheses
from run_a2_pilot_design import _extreme_genes, redact_tf_tokens, symbol_in_text
from run_e_common import StageGuard, load_jsonl, stage_spent

ROOT = Path("/Users/bytedance/codes/new-bishe")
OUT = ROOT / "artifacts" / "results" / "cand6_deconftf" / "agent"
BUNDLE_PATH = ROOT / "artifacts" / "results" / "cand6_deconftf" / "knocktf_precomputed_bundle.pkl"
RAW_PATH = OUT / "e2_counterfactual_raw.jsonl"
PAIR_PATH = OUT / "e2_pairs.csv"
FROZEN = {
    ROOT / "artifacts" / "src" / "deconftf_tools.py":
        "45939716963851d980d5767d9285832efd5b4b3537ce014b0e44f7affa225490",
    ROOT / "artifacts" / "src" / "develop_m10_soft_gate.py":
        "1e513f5236e56842e5ff96de3263175498ba9c1e6662ab8f228de3a1d7500677",
}
ESTIMATE_CNY = 4.0
WORKERS = 4


def assert_frozen() -> None:
    for path, expected in FROZEN.items():
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != expected:
            raise SystemExit(f"frozen hash changed: {path.name} {got}")


def make_pairs(bundle) -> pd.DataFrame:
    sample = pd.read_csv(OUT / "a2v3_pilot_sample60.csv")
    if len(sample) != 60:
        raise SystemExit("v3 sample is not 60")
    obs = bundle["obs_sub"]
    rows = []
    for rec in sample.itertuples(index=False):
        rows.append({
            "exp_id": rec.exp_id,
            "study_id": str(rec.study_id),
            "true_tf": str(obs.loc[rec.exp_id, "TF"]).upper(),
        })
    ordered = sorted(rows, key=lambda rec: rec["exp_id"])
    pairs = []
    n = len(ordered)
    for i, rec in enumerate(ordered):
        donor = None
        for step in range(1, n):
            other = ordered[(i + step) % n]
            if other["study_id"] != rec["study_id"] and other["true_tf"] != rec["true_tf"]:
                donor = other
                break
        if donor is None:
            raise SystemExit(f"no donor for {rec['exp_id']}")
        pairs.append({
            "exp_i": rec["exp_id"],
            "study_i": rec["study_id"],
            "tf_i": rec["true_tf"],
            "exp_j": donor["exp_id"],
            "study_j": donor["study_id"],
            "tf_j": donor["true_tf"],
        })
    out = pd.DataFrame(pairs)
    if PAIR_PATH.exists():
        locked = pd.read_csv(PAIR_PATH)
        if locked[["exp_i", "exp_j"]].astype(str).values.tolist() != out[["exp_i", "exp_j"]].astype(str).values.tolist():
            raise SystemExit("pair file does not match the deterministic swap")
    else:
        out.to_csv(PAIR_PATH, index=False)
    return out


def build_prompt(bundle, pair, m10_order) -> str:
    obs_i = bundle["obs_sub"].loc[pair.exp_i]
    obs_j = bundle["obs_sub"].loc[pair.exp_j]
    tf_i = str(pair.tf_i).upper()
    y = bundle["mat_blind"].loc[pair.exp_i]
    matched = [c for c in y.index if str(c).upper() == tf_i]
    if not matched or abs(float(y[matched[0]])) > 1e-8:
        raise SystemExit(f"{pair.exp_i}: true TF is not masked in mat_blind")
    down = _extreme_genes(y, tf_i, -1)
    up = _extreme_genes(y, tf_i, +1)
    bio, _ = redact_tf_tokens(obs_j["Biosample.Name"], tf_i)
    tissue, _ = redact_tf_tokens(obs_j["Tissue.Type"], tf_i)
    study, _ = redact_tf_tokens(str(pair.study_j), tf_i)
    shown = [g for g in m10_order if str(g).upper() != tf_i][:25]
    prompt = (
        "Propose 10 transcription-factor hypotheses for the primary knockdown, "
        "most plausible first. Prefer factors outside the fixed-pipeline list when "
        "the gene changes support them. Return JSON only, no markdown. "
        'Schema: {"hypotheses": ["SYMBOL", "..."]} with exactly 10 HGNC symbols, '
        "most plausible first.\n"
        f"Biosample = {bio}. Tissue = {tissue}. Study = {study}.\n"
        "Strongest downregulated genes: "
        + ", ".join(f"{g} ({v:.2f})" for g, v in down.items())
        + "\nStrongest upregulated genes: "
        + ", ".join(f"{g} ({v:.2f})" for g, v in up.items())
        + f"\nDownregulated genes with |log2FC| >= 1: {int((y <= -1).sum())}\n"
        f"Upregulated genes with |log2FC| >= 1: {int((y >= 1).sum())}\n"
        "Fixed-pipeline top 25, do not spend the whole list repeating these: "
        + ", ".join(shown)
        + "\n"
    )
    if symbol_in_text(prompt, tf_i):
        raise SystemExit(f"leakage: expression experiment {pair.exp_i} TF {tf_i} is in the swapped prompt")
    return prompt


def _one(bundle, pair, prompt: str, guard: StageGuard) -> str:
    reason = guard.reserve(0.20)
    if reason:
        return reason
    record = {"exp_i": pair.exp_i, "exp_j": pair.exp_j, "tf_i": pair.tf_i, "tf_j": pair.tf_j, "prompt": prompt}
    try:
        res = gated_chat(
            [{"role": "user", "content": prompt}],
            purpose=f"e2:{pair.exp_i}",
            model="deepseek-v4.1-flash",
            raw_path=RAW_PATH,
            raw_record=record,
            timeout_sec=300,
        )
    except BudgetExceededError as err:
        guard.settle(0.20, 0.0, False)
        return f"budget_gate stopped the run: {err}"
    except Exception as err:
        record["error"] = f"{type(err).__name__}: {err}"[:500]
        with RAW_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return guard.settle(0.20, 0.0, False)
    parsed = parse_hypotheses(res.get("content") or "")
    cost = float((res.get("ledger_entry") or {}).get("cost_cny") or 0.0)
    print(f"{pair.exp_i} parse={parsed is not None} spent={guard.spent + cost:.2f}", flush=True)
    return guard.settle(0.20, cost, parsed is not None)


def run_calls(bundle, pairs: pd.DataFrame, cov: float) -> str:
    done = {row["exp_i"] for row in load_jsonl(RAW_PATH)}
    prompts = {}
    for pair in pairs.itertuples(index=False):
        base = score_candidates(pair.exp_i, bundle, cov_frac=cov)
        prompts[pair.exp_i] = build_prompt(bundle, pair, base["m10_order"])
    print(f"leakage asserts passed for {len(prompts)} swapped prompts", flush=True)
    guard = StageGuard(ESTIMATE_CNY, stage_spent(load_jsonl(RAW_PATH)))
    jobs = [pair for pair in pairs.itertuples(index=False) if pair.exp_i not in done]
    stop = ""
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(_one, bundle, pair, prompts[pair.exp_i], guard) for pair in jobs]
        for fut in as_completed(futures):
            reason = fut.result()
            if reason:
                stop = reason
    return stop or guard.reason


def evaluate(pairs: pd.DataFrame, stop: str) -> None:
    raw = load_jsonl(RAW_PATH)
    by_id = {row["exp_i"]: row for row in raw}
    rows = []
    n_fail = 0
    for pair in pairs.itertuples(index=False):
        got = by_id.get(pair.exp_i)
        hyps = []
        parsed = False
        if got is not None:
            parsed_list = parse_hypotheses(got.get("content") or "")
            parsed = parsed_list is not None
            hyps = [h.upper() for h in (parsed_list or [])]
            if not parsed:
                n_fail += 1
        rows.append({
            "exp_i": pair.exp_i,
            "tf_i": pair.tf_i,
            "exp_j": pair.exp_j,
            "tf_j": pair.tf_j,
            "study_i": pair.study_i,
            "study_j": pair.study_j,
            "parse_ok": int(parsed),
            "first": hyps[0] if hyps else "",
            "hit_i_first": int(bool(hyps) and hyps[0] == pair.tf_i),
            "hit_j_first": int(bool(hyps) and hyps[0] == pair.tf_j),
            "hit_i_top10": int(pair.tf_i in hyps[:10]),
            "hit_j_top10": int(pair.tf_j in hyps[:10]),
            "hypotheses": "|".join(hyps),
            "cost_cny": (got.get("ledger_entry") or {}).get("cost_cny") if got else None,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e2_counterfactual_per_experiment.csv", index=False)
    answered = df[df["parse_ok"] == 1] if "parse_ok" in df.columns else df.iloc[0:0]
    def n(col: str) -> str:
        if answered.empty:
            return "0/0"
        return f"{int(answered[col].sum())}/{len(answered)}"
    lines = [
        "E2 counterfactual metadata swap. Exploratory. Expression is experiment i; metadata is experiment j.",
        stop or "all scheduled calls finished",
        f"Stored {len(raw)} calls, parse failures {n_fail}, stage spend {stage_spent(raw):.4f} CNY.",
        "A donor is the next experiment in exp_id order with a different study and a different true TF.",
        f"Among parsed answers, first hypothesis is i in {n('hit_i_first')}, j in {n('hit_j_first')}.",
        f"Top-10 contains i in {n('hit_i_top10')}, j in {n('hit_j_top10')}.",
    ]
    text = "\n".join(lines) + "\n"
    (OUT / "e2_counterfactual_summary.txt").write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    assert_frozen()
    if BudgetLedger().total_spent_cny() >= SESSION_SOFT_CAP_CNY:
        raise SystemExit("session soft cap already reached")
    bundle = pickle.loads(BUNDLE_PATH.read_bytes())
    if len(bundle["obs_sub"]) != 266:
        raise SystemExit("bundle is not Dev 266")
    pairs = make_pairs(bundle)
    cov = compute_bundle_coverage(bundle)
    stop = run_calls(bundle, pairs, cov)
    if stop:
        print(stop, flush=True)
    evaluate(pairs, stop)
    assert_frozen()


if __name__ == "__main__":
    main()
