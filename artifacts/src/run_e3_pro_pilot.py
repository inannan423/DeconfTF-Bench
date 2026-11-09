#!/usr/bin/env python3
"""E3: deepseek-v4-pro on a new Dev 60, same prompt and condition-C line as v3.

The 60 are a Hamilton draw, seed 44, from the Dev experiments left after v2 and v3.
Five condition-C calls run first. Two or more finish_reason=length stops the rest.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/Users/bytedance/codes/new-bishe/artifacts/src")
from budget_gate import SESSION_SOFT_CAP_CNY, BudgetExceededError, BudgetLedger, gated_chat
from develop_m10_soft_gate import compute_bundle_coverage
from regagent_tools import (
    load_track_catalog,
    prepare_hypothesis_resources,
    score_candidates,
)
from run_a2_pilot import _apply, _decision, _known_symbols, parse_hypotheses
from run_a2_pilot_design import build_prompt, hamilton_sample
from run_e_common import StageGuard, load_jsonl, stage_spent

ROOT = Path("/Users/bytedance/codes/new-bishe")
OUT = ROOT / "artifacts" / "results" / "cand6_deconftf" / "agent"
BUNDLE_PATH = ROOT / "artifacts" / "results" / "cand6_deconftf" / "knocktf_precomputed_bundle.pkl"
RAW_PATH = OUT / "e3_pro_raw.jsonl"
SAMPLE_PATH = OUT / "e3_pro_sample60.csv"
MODEL = "deepseek-v4-pro"
ESTIMATE_CNY = 30.0
SMOKE_N = 5
FROZEN = {
    ROOT / "artifacts" / "src" / "deconftf_tools.py":
        "45939716963851d980d5767d9285832efd5b4b3537ce014b0e44f7affa225490",
    ROOT / "artifacts" / "src" / "develop_m10_soft_gate.py":
        "1e513f5236e56842e5ff96de3263175498ba9c1e6662ab8f228de3a1d7500677",
}


def assert_frozen() -> None:
    for path, expected in FROZEN.items():
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != expected:
            raise SystemExit(f"frozen hash changed: {path.name} {got}")


def load_sample(bundle) -> pd.DataFrame:
    labels = pd.read_csv(
        ROOT / "artifacts" / "results" / "cand6_deconftf" / "knocktf401_ranks_with_study_ids.csv",
        usecols=["exp_id", "study_id", "split_new"],
    )
    dev = labels[labels["split_new"] == "Dev"]
    if set(dev["exp_id"]) != set(bundle["obs_sub"].index):
        raise SystemExit("bundle is not exactly Dev 266")
    study = dev.set_index("exp_id")["study_id"].astype(str)
    used = set(pd.read_csv(OUT / "a2_pilot_sample60.csv")["exp_id"])
    used |= set(pd.read_csv(OUT / "a2v3_pilot_sample60.csv")["exp_id"])
    remain = study.drop(index=[exp_id for exp_id in used if exp_id in study.index])
    if len(remain) != 146:
        raise SystemExit(f"expected 146 Dev experiments left, found {len(remain)}")
    drawn = hamilton_sample(remain, 60, 44)
    if set(drawn["exp_id"]) & used:
        raise SystemExit("E3 sample overlaps v2 or v3")
    if SAMPLE_PATH.exists():
        locked = pd.read_csv(SAMPLE_PATH)
        if locked["exp_id"].tolist() != drawn["exp_id"].tolist():
            raise SystemExit("E3 sample file does not match seed 44")
    else:
        drawn.to_csv(SAMPLE_PATH, index=False)
    return drawn


def _finish_reason(row: dict):
    choices = (row.get("raw_response") or {}).get("choices") or [{}]
    return choices[0].get("finish_reason")


def _one(client_unused, bundle, cov, exp_id, study_id, condition, guard: StageGuard, label: str):
    del client_unused
    base = score_candidates(exp_id, bundle, cov_frac=cov)
    prompt, _n = build_prompt(exp_id, condition, bundle, study_id, base["m10_order"])
    reason = guard.reserve(0.80)
    if reason:
        return reason
    record = {
        "exp_id": exp_id,
        "study_id": str(study_id),
        "condition": condition,
        "model": MODEL,
        "prompt": prompt,
    }
    try:
        print(f"{label} sending", flush=True)
        res = gated_chat(
            [{"role": "user", "content": prompt}],
            purpose=f"e3:{exp_id}:{condition}",
            model=MODEL,
            raw_path=RAW_PATH,
            raw_record=record,
            timeout_sec=300,
            max_attempts=1,
            stream=True,
        )
    except BudgetExceededError as err:
        guard.settle(0.80, 0.0, False)
        return f"budget_gate stopped the run: {err}"
    except Exception as err:
        record["error"] = f"{type(err).__name__}: {err}"[:500]
        with RAW_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{label} client error: {record['error']}", flush=True)
        guard.settle(0.80, 0.0, False)
        return f"client error on {label}: {record['error']}"
    parsed = parse_hypotheses(res.get("content") or "")
    cost = float((res.get("ledger_entry") or {}).get("cost_cny") or 0.0)
    finish = _finish_reason({"raw_response": res.get("raw_response")})
    print(f"{label} parse={parsed is not None} finish={finish} spent={guard.spent + cost:.2f}", flush=True)
    return guard.settle(0.80, cost, parsed is not None)


def _answered(rows: list) -> set:
    """An HTTP error is not a finished call. A later streaming response can replace it."""
    return {
        (row["exp_id"], row["condition"])
        for row in rows
        if row.get("raw_response") and not row.get("error")
    }


def run_calls(bundle, sample: pd.DataFrame, cov: float) -> str:
    smoke_ids = tuple(sample["exp_id"].head(SMOKE_N).tolist())
    done = _answered(load_jsonl(RAW_PATH))
    study = dict(zip(sample["exp_id"], sample["study_id"].astype(str)))
    guard = StageGuard(ESTIMATE_CNY, stage_spent(load_jsonl(RAW_PATH)))
    for i, exp_id in enumerate(smoke_ids, start=1):
        if (exp_id, "C") in done:
            continue
        if guard.reason:
            return guard.reason
        reason = _one(None, bundle, cov, exp_id, study[exp_id], "C", guard, f"smoke {i}/{SMOKE_N}")
        if reason:
            return reason
    raw = load_jsonl(RAW_PATH)
    by_id = {row["exp_id"]: row for row in raw if row.get("condition") == "C" and row["exp_id"] in smoke_ids}
    if len(by_id) >= SMOKE_N:
        reasons = [_finish_reason(by_id[exp_id]) for exp_id in smoke_ids]
        n_length = sum(reason == "length" for reason in reasons)
        n_missing = sum(reason is None for reason in reasons)
        if n_length >= 2 or n_missing:
            return (
                f"smoke stop: finish_reason length {n_length}/{SMOKE_N}, "
                f"missing {n_missing}. Remaining calls were not sent."
            )
        print("smoke passed; continuing", flush=True)
    done = _answered(load_jsonl(RAW_PATH))
    jobs = [(r.exp_id, "A") for r in sample.itertuples(index=False)]
    jobs += [(r.exp_id, "C") for r in sample.itertuples(index=False) if r.exp_id not in smoke_ids]
    for i_job, (exp_id, condition) in enumerate(jobs, start=1):
        if (exp_id, condition) in done:
            continue
        if guard.reason:
            return guard.reason
        reason = _one(
            None, bundle, cov, exp_id, study[exp_id], condition, guard,
            f"{i_job}/{len(jobs)} {exp_id} {condition}",
        )
        if reason:
            return reason
    return guard.reason


def evaluate(bundle, sample: pd.DataFrame, cov: float, stop: str) -> None:
    raw = load_jsonl(RAW_PATH)
    by_key = {(row["exp_id"], row["condition"]): row for row in raw}
    human = [t for t in load_track_catalog() if t.cell and t.species == "HUMAN"]
    resources = prepare_hypothesis_resources(bundle, human)
    known = _known_symbols(bundle, resources)
    ctrl = pd.read_csv(OUT / "a15_hypothesis_controls_per_experiment.csv").set_index("exp_id")
    import numpy as np
    cand = list(bundle["cand_tfs"])
    chip_pool = sorted(tf for tf in cand if len(resources["chip"].get(tf) or []) >= 5)
    rng = np.random.default_rng(42)
    random_draw = {}
    for exp_id in bundle["obs_sub"].index:
        random_draw[exp_id] = [str(tf).upper() for tf in rng.choice(chip_pool, size=10, replace=False)]
    rows = []
    n_parse_fail = n_calls = 0
    for rec in sample.itertuples(index=False):
        true = str(bundle["obs_sub"].loc[rec.exp_id, "TF"]).upper()
        base = score_candidates(rec.exp_id, bundle, cov_frac=cov)
        info = ctrl.loc[rec.exp_id]
        row = {
            "exp_id": rec.exp_id,
            "study_id": rec.study_id,
            "true_tf": true,
            "in_recalled": int(info["in_recalled"]),
            "r_m10": int(info["r_m10"]),
            "random_first": random_draw[rec.exp_id][0],
            "random_k10_hit": int(true in random_draw[rec.exp_id]),
            "random_first_hit": int(random_draw[rec.exp_id][0] == true),
        }
        for condition in ("A", "C"):
            got = by_key.get((rec.exp_id, condition))
            if got is None:
                row[f"{condition}_missing"] = 1
                continue
            row[f"{condition}_missing"] = 0
            n_calls += 1
            hyps = [str(tf).upper() for tf in (parse_hypotheses(got.get("content") or "") or [])]
            if parse_hypotheses(got.get("content") or "") is None:
                n_parse_fail += 1
            applied = _apply(hyps, rec.exp_id, base["m10_order"], true, bundle, resources, known)
            row[f"{condition}_parse_ok"] = int(parse_hypotheses(got.get("content") or "") is not None)
            row[f"{condition}_hypotheses"] = "|".join(hyps)
            row[f"{condition}_first_hit"] = applied["first_hit"]
            row[f"{condition}_k10_hit"] = applied["k10_hit"]
            row[f"{condition}_k5_rank"] = applied["k5_rank"]
            row[f"{condition}_k10_rank"] = applied["k10_rank"]
            row[f"{condition}_k5_proposed"] = applied["k5_proposed"]
            row[f"{condition}_k10_proposed"] = applied["k10_proposed"]
            row[f"{condition}_cost_cny"] = (got.get("ledger_entry") or {}).get("cost_cny")
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e3_pro_per_experiment.csv", index=False)
    missing = int(df[[c for c in df.columns if c.endswith("_missing")]].sum().sum()) if len(df) else 1
    complete = len(raw) == 120 and missing == 0 and not str(stop).startswith("smoke")
    lines = [
        "E3 deepseek-v4-pro pilot. Same prompt and condition-C line as A2 v3. Exploratory Dev sample.",
        stop or "all scheduled calls finished",
        f"stored calls {n_calls}, parse failures {n_parse_fail} "
        f"({(100 * n_parse_fail / n_calls) if n_calls else 0:.2f}%), stage spend {stage_spent(raw):.4f} CNY.",
        "Sample: Hamilton, seed 44, 60 from the 146 Dev experiments not used in v2 or v3.",
    ]
    if complete:
        lines.extend(_decision(df))
    else:
        lines.append("The locked line is not decided because the run is incomplete.")
    text = "\n".join(lines) + "\n"
    (OUT / "e3_pro_summary.txt").write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    assert_frozen()
    if BudgetLedger().total_spent_cny() >= SESSION_SOFT_CAP_CNY:
        raise SystemExit("session soft cap already reached")
    bundle = pickle.loads(BUNDLE_PATH.read_bytes())
    sample = load_sample(bundle)
    cov = compute_bundle_coverage(bundle)
    stop = run_calls(bundle, sample, cov)
    if stop:
        print(stop, flush=True)
    evaluate(bundle, sample, cov, stop)
    assert_frozen()


if __name__ == "__main__":
    main()
