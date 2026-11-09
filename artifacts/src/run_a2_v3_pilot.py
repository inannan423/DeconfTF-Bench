#!/usr/bin/env python3
"""A2 v3 pilot. Same prompt as v2, completion cap 16384.

Runs five condition-C calls first. Two or more finish_reason=length stops
the rest. Temperature is 0 and reasoning_effort stays low. Does not call v4.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/Users/bytedance/codes/new-bishe/artifacts/src")
from budget_gate import (
    BudgetExceededError,
    BudgetedDeepSeekClient,
    compute_cost_cny,
    estimate_tokens_conservative,
)
from develop_m10_soft_gate import compute_bundle_coverage
from regagent_tools import (
    load_track_catalog,
    prepare_hypothesis_resources,
    score_candidates,
)
from run_a2_pilot import _apply, _decision, _known_symbols, assert_frozen, parse_hypotheses
from run_a2_pilot_design import build_prompt
from run_a2_v3_design import MAX_COMPLETION, SMOKE_N

ROOT = Path("/Users/bytedance/codes/new-bishe")
OUT = ROOT / "artifacts" / "results" / "cand6_deconftf" / "agent"
BUNDLE_PATH = ROOT / "artifacts" / "results" / "cand6_deconftf" / "knocktf_precomputed_bundle.pkl"
RAW_PATH = OUT / "a2v3_pilot_raw.jsonl"
SUMMARY_PATH = OUT / "a2v3_pilot_summary.txt"
SPEND_CAP_CNY = 25.0
SMOKE_IDS = (
    "DataSet_01_198",
    "DataSet_01_015",
    "DataSet_01_095",
    "DataSet_01_074",
    "DataSet_01_079",
)


def _load_raw() -> list:
    if not RAW_PATH.exists():
        return []
    return [json.loads(line) for line in RAW_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_raw(row: dict) -> None:
    blob = json.dumps(row, ensure_ascii=False)
    if "Bearer " in blob or "LLMBOX_API_KEY" in blob:
        raise SystemExit("refusing to store a response that contains a credential")
    with RAW_PATH.open("a", encoding="utf-8") as handle:
        handle.write(blob + "\n")


def _finish_reason(row: dict):
    if row.get("error") or not row.get("raw_response"):
        return None
    choices = row["raw_response"].get("choices") or [{}]
    return choices[0].get("finish_reason")


def _spent(rows: list) -> float:
    return sum(float((row.get("ledger_entry") or {}).get("cost_cny") or 0.0) for row in rows)


def _one_call(client, bundle, cov, exp_id, study_id, condition, spent: float, label: str):
    base = score_candidates(exp_id, bundle, cov_frac=cov)
    prompt, _n_redact = build_prompt(exp_id, condition, bundle, study_id, base["m10_order"])
    est = compute_cost_cny(estimate_tokens_conservative(prompt), MAX_COMPLETION)
    if spent + est > SPEND_CAP_CNY:
        return None, f"stopped before {label}: spent {spent:.4f} + preflight {est:.4f} > {SPEND_CAP_CNY:.0f}"
    record = {
        "exp_id": exp_id,
        "study_id": str(study_id),
        "condition": condition,
        "prompt": prompt,
        "content": "",
        "reasoning_content": "",
        "raw_response": None,
        "usage": None,
        "latency_sec": None,
        "ledger_entry": None,
        "parse_ok": False,
        "hypotheses": [],
        "error": None,
    }
    try:
        res = client.chat(
            [{"role": "user", "content": prompt}],
            purpose=f"a2v3:{exp_id}:{condition}",
            max_completion_tokens=MAX_COMPLETION,
            temperature=0.0,
            reasoning_effort="low",
            timeout_sec=300,
        )
    except BudgetExceededError as err:
        return None, f"budget_gate stopped the run: {err}"
    except Exception as err:
        record["error"] = f"{type(err).__name__}: {err}"[:500]
        _append_raw(record)
        print(f"{label} client error, not retried", flush=True)
        return record, ""
    record["content"] = res.get("content") or ""
    record["reasoning_content"] = res.get("reasoning_content") or ""
    record["raw_response"] = res.get("raw_response")
    record["usage"] = res.get("usage")
    record["latency_sec"] = res.get("latency_sec")
    record["ledger_entry"] = res.get("ledger_entry")
    parsed = parse_hypotheses(record["content"])
    record["parse_ok"] = parsed is not None
    record["hypotheses"] = parsed or []
    _append_raw(record)
    cost = float(record["ledger_entry"]["cost_cny"])
    print(
        f"{label} parse={record['parse_ok']} finish={_finish_reason(record)} "
        f"n={len(record['hypotheses'])} spent={spent + cost:.4f}",
        flush=True,
    )
    return record, ""


def _smoke_block(raw: list) -> str:
    by_id = {}
    for row in raw:
        if row["condition"] == "C" and row["exp_id"] in SMOKE_IDS:
            by_id[row["exp_id"]] = row
    if len(by_id) < SMOKE_N:
        return ""
    reasons = [_finish_reason(by_id[exp_id]) for exp_id in SMOKE_IDS]
    n_length = sum(reason == "length" for reason in reasons)
    n_missing = sum(reason is None for reason in reasons)
    detail = ", ".join(f"{exp_id}={reason}" for exp_id, reason in zip(SMOKE_IDS, reasons))
    if n_length >= 2 or n_missing:
        return (
            f"smoke stop: finish_reason length {n_length}/{SMOKE_N}, "
            f"missing finish_reason {n_missing}. {detail}. The remaining calls were not sent."
        )
    return ""


def run_calls(bundle, sample: pd.DataFrame, cov: float) -> str:
    if sample["exp_id"].head(SMOKE_N).tolist() != list(SMOKE_IDS):
        raise SystemExit("sample order does not match the locked smoke ids")
    client = BudgetedDeepSeekClient()
    done = {(row["exp_id"], row["condition"]) for row in _load_raw()}
    study = dict(zip(sample["exp_id"], sample["study_id"].astype(str)))
    spent = _spent(_load_raw())
    for i, exp_id in enumerate(SMOKE_IDS, start=1):
        if (exp_id, "C") in done:
            continue
        record, stop = _one_call(client, bundle, cov, exp_id, study[exp_id], "C", spent, f"smoke {i}/{SMOKE_N}")
        if stop:
            return stop
        if record and record.get("ledger_entry"):
            spent += float(record["ledger_entry"]["cost_cny"])
    blocked = _smoke_block(_load_raw())
    if blocked:
        print(blocked, flush=True)
        return blocked
    print("smoke passed; continuing the remaining calls", flush=True)
    done = {(row["exp_id"], row["condition"]) for row in _load_raw()}
    jobs = [(r.exp_id, "A") for r in sample.itertuples(index=False)]
    jobs += [(r.exp_id, "C") for r in sample.itertuples(index=False) if r.exp_id not in SMOKE_IDS]
    for i_job, (exp_id, condition) in enumerate(jobs, start=1):
        if (exp_id, condition) in done:
            continue
        record, stop = _one_call(
            client, bundle, cov, exp_id, study[exp_id], condition, spent, f"{i_job}/{len(jobs)} {exp_id} {condition}",
        )
        if stop:
            print(stop, flush=True)
            return stop
        if record and record.get("ledger_entry"):
            spent += float(record["ledger_entry"]["cost_cny"])
            if spent >= SPEND_CAP_CNY:
                stop = f"stopped at spend {spent:.4f}"
                print(stop, flush=True)
                return stop
    return ""


def evaluate(bundle, sample: pd.DataFrame, cov: float, stop: str) -> None:
    raw = _load_raw()
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
    n_parse_fail = n_calls = n_invalid = n_outside = n_http_fail = 0
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
            if got.get("error"):
                n_http_fail += 1
            elif not got.get("parse_ok"):
                n_parse_fail += 1
            hyps = [str(tf).upper() for tf in (got.get("hypotheses") or [])]
            applied = _apply(hyps, rec.exp_id, base["m10_order"], true, bundle, resources, known)
            n_invalid += applied["n_invalid"]
            n_outside += applied["n_outside_cand"]
            row[f"{condition}_parse_ok"] = int(bool(got.get("parse_ok")))
            row[f"{condition}_n_hyp"] = len(hyps)
            row[f"{condition}_n_invalid"] = applied["n_invalid"]
            row[f"{condition}_hypotheses"] = "|".join(hyps)
            row[f"{condition}_invalid"] = "|".join(applied["invalid"])
            row[f"{condition}_first_hit"] = applied["first_hit"]
            row[f"{condition}_k10_hit"] = applied["k10_hit"]
            row[f"{condition}_k5_rank"] = applied["k5_rank"]
            row[f"{condition}_k10_rank"] = applied["k10_rank"]
            row[f"{condition}_k5_proposed"] = applied["k5_proposed"]
            row[f"{condition}_k10_proposed"] = applied["k10_proposed"]
            entry = got.get("ledger_entry") or {}
            row[f"{condition}_cost_cny"] = entry.get("cost_cny")
            row[f"{condition}_latency_sec"] = got.get("latency_sec")
            usage = got.get("usage") or {}
            row[f"{condition}_prompt_tokens"] = usage.get("prompt_tokens")
            row[f"{condition}_completion_tokens"] = usage.get("completion_tokens")
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "a2v3_pilot_per_experiment.csv", index=False)
    n_length_empty = n_length_partial = 0
    for saved in raw:
        if _finish_reason(saved) != "length":
            continue
        if (saved.get("content") or "").strip():
            n_length_partial += 1
        else:
            n_length_empty += 1
    missing = int(df[[c for c in df.columns if c.endswith("_missing")]].sum().sum()) if len(df) else 1
    complete = len(raw) == 120 and missing == 0 and not stop.startswith("smoke")
    lines = [
        "A2 pilot v3.",
        stop or "all scheduled calls finished",
        f"stored calls {n_calls}, parse failures {n_parse_fail} "
        f"({(100 * n_parse_fail / n_calls) if n_calls else 0:.2f}%), "
        f"client errors {n_http_fail}, invalid hypotheses {n_invalid}, "
        f"known symbols outside the 647 candidates {n_outside}.",
        "Parse failures count as no hypotheses. They were not retried. The prompt text was not changed.",
        f"Length cap: {n_length_empty} empty content and {n_length_partial} truncated JSON "
        f"with finish_reason=length, at max_completion_tokens={MAX_COMPLETION}.",
    ]
    if not complete:
        lines.append("The locked go/no-go is not decided. v3 does not schedule a v4.")
    else:
        lines.extend(_decision(df))
        lines.append("v3 does not schedule a v4. A NO-GO ends propose-verify.")
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    assert_frozen()
    bundle = pickle.loads(BUNDLE_PATH.read_bytes())
    if len(bundle["obs_sub"]) != 266:
        raise SystemExit("bundle is not Dev 266")
    sample = pd.read_csv(OUT / "a2v3_pilot_sample60.csv")
    v2 = pd.read_csv(OUT / "a2_pilot_sample60.csv")
    if set(sample["exp_id"]) & set(v2["exp_id"]):
        raise SystemExit("v3 sample overlaps v2")
    labels = pd.read_csv(
        ROOT / "artifacts" / "results" / "cand6_deconftf" / "knocktf401_ranks_with_study_ids.csv",
        usecols=["exp_id", "study_id", "split_new"],
    )
    dev_ids = set(labels.loc[labels["split_new"] == "Dev", "exp_id"])
    if set(sample["exp_id"]) - dev_ids:
        raise SystemExit("v3 sample is not inside Dev")
    cov = compute_bundle_coverage(bundle)
    stop = run_calls(bundle, sample, cov)
    evaluate(bundle, sample, cov, stop)
    assert_frozen()


if __name__ == "__main__":
    main()
