#!/usr/bin/env python3
"""A2 pilot, v2 prompt. Calls the model only through budget_gate.

Temperature is 0. This run stops before a call whose preflight upper bound
would push its own spend past 12 CNY. A response that is not the declared
JSON object is stored and treated as no hypotheses. It is not retried.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
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
    merge_verified_into_m10,
    prepare_hypothesis_resources,
    score_candidates,
    verify_hypothesis,
)
from run_a2_pilot_design import MAX_COMPLETION, build_prompt
from verify_round4_accept_checks import study_clustered_sign_flip_perm

ROOT = Path("/Users/bytedance/codes/new-bishe")
OUT = ROOT / "artifacts" / "results" / "cand6_deconftf" / "agent"
BUNDLE_PATH = ROOT / "artifacts" / "results" / "cand6_deconftf" / "knocktf_precomputed_bundle.pkl"
RAW_PATH = OUT / "a2_pilot_raw.jsonl"
SPEND_CAP_CNY = 12.0
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


def parse_hypotheses(content: str):
    """The v2 schema only. Anything else is a parse failure, with no hypotheses."""
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("hypotheses"), list):
        return None
    out = []
    for item in obj["hypotheses"]:
        if not isinstance(item, str):
            return None
        out.append(item.strip().upper())
    return out[:10]


def _load_raw() -> list:
    if not RAW_PATH.exists():
        return []
    rows = []
    for line in RAW_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _append_raw(row: dict) -> None:
    blob = json.dumps(row, ensure_ascii=False)
    if "Bearer " in blob or "LLMBOX_API_KEY" in blob:
        raise SystemExit("refusing to store a response that contains a credential")
    with RAW_PATH.open("a", encoding="utf-8") as handle:
        handle.write(blob + "\n")


def _known_symbols(bundle, resources) -> set:
    known = {str(tf).upper() for tf in bundle["cand_tfs"]}
    known.update(str(tf).upper() for tf in resources["collectri"])
    known.update(str(tf).upper() for tf in resources["dorothea"])
    known.update(t.tf.upper() for t in load_track_catalog())
    return known


def run_calls(bundle, sample: pd.DataFrame, cov: float) -> str:
    done = {(r["exp_id"], r["condition"]) for r in _load_raw()}
    spent = 0.0
    for row in _load_raw():
        entry = row.get("ledger_entry") or {}
        spent += float(entry.get("cost_cny") or 0.0)
    client = BudgetedDeepSeekClient()
    stop = ""
    jobs = [(r.exp_id, r.study_id, cond) for r in sample.itertuples(index=False) for cond in ("A", "C")]
    for i_job, (exp_id, study_id, condition) in enumerate(jobs, start=1):
        if (exp_id, condition) in done:
            continue
        base = score_candidates(exp_id, bundle, cov_frac=cov)
        prompt, _n_redact = build_prompt(exp_id, condition, bundle, study_id, base["m10_order"])
        est = compute_cost_cny(estimate_tokens_conservative(prompt), MAX_COMPLETION)
        if spent + est > SPEND_CAP_CNY:
            stop = f"stopped before call {i_job}: spent {spent:.4f} + preflight {est:.4f} > {SPEND_CAP_CNY:.0f}"
            print(stop, flush=True)
            break
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
                purpose=f"a2v2:{exp_id}:{condition}",
                max_completion_tokens=MAX_COMPLETION,
                temperature=0.0,
                reasoning_effort="low",
                timeout_sec=180,
            )
        except BudgetExceededError as err:
            stop = f"budget_gate stopped the run: {err}"
            print(stop, flush=True)
            break
        except Exception as err:
            record["error"] = f"{type(err).__name__}: {err}"[:500]
            _append_raw(record)
            print(f"{i_job}/120 {exp_id} {condition} HTTP/client error, not retried", flush=True)
            continue
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
        spent += float(record["ledger_entry"]["cost_cny"])
        print(
            f"{i_job}/120 {exp_id} {condition} parse={record['parse_ok']} "
            f"n={len(record['hypotheses'])} spent={spent:.4f}",
            flush=True,
        )
        if spent >= SPEND_CAP_CNY:
            stop = f"stopped at spend {spent:.4f}"
            print(stop, flush=True)
            break
    return stop


def _apply(hyps, exp_id, m10_order, true, bundle, resources, known) -> dict:
    """Verify the prefix. Invalid symbols occupy a slot and are not scored."""
    invalid = []
    outside = []
    aligned = []
    m10_set = set(m10_order)
    for tf in hyps:
        if tf not in known:
            invalid.append(tf)
            aligned.append(None)
            continue
        if tf not in m10_set:
            outside.append(tf)
            aligned.append(None)
            continue
        aligned.append(verify_hypothesis(exp_id, tf, bundle, resources))
    out = {"n_invalid": len(invalid), "invalid": invalid, "n_outside_cand": len(outside)}
    for k in (5, 10):
        rows = [row for row in aligned[:k] if row is not None]
        merged = merge_verified_into_m10(m10_order, rows)
        out[f"k{k}_rank"] = merged.index(true) + 1
        out[f"k{k}_proposed"] = int(true in hyps[:k])
    out["first_hit"] = int(bool(hyps) and hyps[0] == true)
    out["k10_hit"] = int(true in hyps[:10])
    return out


def evaluate(bundle, sample: pd.DataFrame, cov: float, stop: str) -> None:
    raw = _load_raw()
    by_key = {(r["exp_id"], r["condition"]): r for r in raw}
    human = [t for t in load_track_catalog() if t.cell and t.species == "HUMAN"]
    resources = prepare_hypothesis_resources(bundle, human)
    known = _known_symbols(bundle, resources)
    ctrl = pd.read_csv(OUT / "a15_hypothesis_controls_per_experiment.csv")
    ctrl = ctrl.set_index("exp_id")
    cand = list(bundle["cand_tfs"])
    chip_pool = sorted(tf for tf in cand if len(resources["chip"].get(tf) or []) >= 5)
    rng = np.random.default_rng(42)
    random_draw = {}
    for exp_id in bundle["obs_sub"].index:
        random_draw[exp_id] = [str(tf).upper() for tf in rng.choice(chip_pool, size=10, replace=False)]

    rows = []
    n_parse_fail = 0
    n_calls = 0
    n_invalid = 0
    n_outside = 0
    n_http_fail = 0
    for rec in sample.itertuples(index=False):
        true = str(bundle["obs_sub"].loc[rec.exp_id, "TF"]).upper()
        base = score_candidates(rec.exp_id, bundle, cov_frac=cov)
        m10_order = base["m10_order"]
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
            prefix = condition
            if got is None:
                row[f"{prefix}_missing"] = 1
                continue
            row[f"{prefix}_missing"] = 0
            n_calls += 1
            if got.get("error"):
                n_http_fail += 1
            elif not got.get("parse_ok"):
                n_parse_fail += 1
            hyps = [str(tf).upper() for tf in (got.get("hypotheses") or [])]
            applied = _apply(hyps, rec.exp_id, m10_order, true, bundle, resources, known)
            n_invalid += applied["n_invalid"]
            n_outside += applied["n_outside_cand"]
            row[f"{prefix}_parse_ok"] = int(bool(got.get("parse_ok")))
            row[f"{prefix}_n_hyp"] = len(hyps)
            row[f"{prefix}_n_invalid"] = applied["n_invalid"]
            row[f"{prefix}_hypotheses"] = "|".join(hyps)
            row[f"{prefix}_invalid"] = "|".join(applied["invalid"])
            row[f"{prefix}_first_hit"] = applied["first_hit"]
            row[f"{prefix}_k10_hit"] = applied["k10_hit"]
            row[f"{prefix}_k5_rank"] = applied["k5_rank"]
            row[f"{prefix}_k10_rank"] = applied["k10_rank"]
            row[f"{prefix}_k5_proposed"] = applied["k5_proposed"]
            row[f"{prefix}_k10_proposed"] = applied["k10_proposed"]
            entry = got.get("ledger_entry") or {}
            row[f"{prefix}_cost_cny"] = entry.get("cost_cny")
            row[f"{prefix}_latency_sec"] = got.get("latency_sec")
            usage = got.get("usage") or {}
            row[f"{prefix}_prompt_tokens"] = usage.get("prompt_tokens")
            row[f"{prefix}_completion_tokens"] = usage.get("completion_tokens")
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "a2_pilot_per_experiment.csv", index=False)

    complete = bool(len(raw) == 120 and df[[c for c in df.columns if c.endswith("_missing")]].sum().sum() == 0)
    lines = [
        "A2 pilot v2.",
        stop or "all scheduled calls finished",
        f"stored calls {n_calls}, parse failures {n_parse_fail} ({(100 * n_parse_fail / n_calls) if n_calls else 0:.2f}%), "
        f"client errors {n_http_fail}, invalid hypotheses {n_invalid}, "
        f"known symbols outside the 647 candidates {n_outside}.",
        "Invalid means the symbol is outside the 647 candidates and also outside ChIP, CollecTRI, and DoRothEA.",
        "Parse failures count as no hypotheses. They were not retried.",
    ]
    n_length_empty = 0
    n_length_partial = 0
    for saved in raw:
        choice = ((saved.get("raw_response") or {}).get("choices") or [{}])[0]
        if choice.get("finish_reason") != "length":
            continue
        if (saved.get("content") or "").strip():
            n_length_partial += 1
        else:
            n_length_empty += 1
    lines.append(
        f"Length cap: {n_length_empty} calls returned empty content with finish_reason=length, "
        f"and {n_length_partial} returned truncated JSON. "
        "The reasoning text used the 4096-token completion cap before the answer. "
        "The prompt was not edited and those calls were not repeated."
    )
    if not complete:
        lines.append("The run is incomplete, so the locked go/no-go is not decided.")
    else:
        lines.extend(_decision(df))
    text = "\n".join(lines) + "\n"
    (OUT / "a2_pilot_summary.txt").write_text(text, encoding="utf-8")
    print(text)


def _paired(df: pd.DataFrame, llm_col: str, ctrl_col: str):
    a = df[llm_col].to_numpy(dtype=float)
    b = df[ctrl_col].to_numpy(dtype=float)
    mean, p = study_clustered_sign_flip_perm(
        a, b, df["study_id"].astype(str).to_numpy(), n_perm=20000, seed=42,
    )
    return mean, p


def _decision(df: pd.DataFrame) -> list:
    ctrl = pd.read_csv(OUT / "a15_hypothesis_controls_per_experiment.csv").set_index("exp_id")
    ordered = ctrl.loc[df["exp_id"]].reset_index()
    lines = []
    gates = []
    mask = ((df["in_recalled"] == 0) & (df["r_m10"] > 25)).to_numpy()
    n_mask = int(mask.sum())
    for k, controls in (
        (5, ("random_chip_k5_rank",)),
        (10, ("random_chip_k10_rank", "m10_rank_26_60_chip_k10_rank")),
    ):
        llm = (mask & (df[f"C_k{k}_rank"].to_numpy() <= 25)).astype(float)
        for col in controls:
            base = (mask & (ordered[col].to_numpy() <= 25)).astype(float)
            mean, p = _paired(df.assign(_a=llm, _b=base), "_a", "_b")
            ok = bool(mean > 0 and p < 0.05)
            gates.append(ok)
            lines.append(
                f"condition C k={k} vs {col}: incremental {int(llm.sum())}/{n_mask}, "
                f"control {int(base.sum())}/{n_mask}, paired mean {mean:.4f}, p={p:.4f}, pass={ok}"
            )
    dlog_a = -np.log2(df["C_k10_rank"].to_numpy(dtype=float))
    dlog_b = -np.log2(df["r_m10"].to_numpy(dtype=float))
    mean, p = study_clustered_sign_flip_perm(
        dlog_a, dlog_b, df["study_id"].astype(str).to_numpy(), n_perm=20000, seed=42,
    )
    d_ok = bool(mean > 0)
    gates.append(d_ok)
    lines.append(
        f"condition C k=10 end-to-end mean Δlog2 {mean:.4f}, cluster p={p:.4f}, mean>0 is {d_ok}"
    )
    # Condition A is descriptive.
    for k in (5, 10):
        inc = int((mask & (df[f"A_k{k}_rank"].to_numpy() <= 25)).sum())
        lines.append(f"condition A k={k} incremental {inc}/{n_mask} (descriptive)")
    a_mean, a_p = study_clustered_sign_flip_perm(
        -np.log2(df["A_k10_rank"].to_numpy(dtype=float)),
        dlog_b,
        df["study_id"].astype(str).to_numpy(),
        n_perm=20000,
        seed=42,
    )
    lines.append(f"condition A k=10 end-to-end mean Δlog2 {a_mean:.4f}, cluster p={a_p:.4f} (descriptive)")
    fail = df["in_recalled"] == 0
    for cond, label in (("C", "condition C"), ("A", "condition A")):
        first = int(df[f"{cond}_first_hit"].sum())
        first_fail = int(df.loc[fail, f"{cond}_first_hit"].sum())
        hit10 = int(df.loc[fail, f"{cond}_k10_hit"].sum())
        lines.append(
            f"{label}: first hypothesis hits the true TF in {first}/60 "
            f"({first_fail}/{int(fail.sum())} of recall-fail). "
            f"k=10 contains the true TF in {hit10}/{int(fail.sum())} recall-fail experiments."
        )
    rnd_first = int(df["random_first_hit"].sum())
    rnd_first_fail = int(df.loc[fail, "random_first_hit"].sum())
    rnd_k10 = int(df.loc[fail, "random_k10_hit"].sum())
    lines.append(
        f"random_chip on the same 60: first draw hits {rnd_first}/60 "
        f"({rnd_first_fail}/{int(fail.sum())} of recall-fail); "
        f"k=10 contains the true TF in {rnd_k10}/{int(fail.sum())} recall-fail experiments."
    )
    go = all(gates)
    lines.append("GO" if go else "NO-GO")
    lines.append("Judged only on the locked condition-C rules. The prompt was not changed after the calls.")
    return lines


def main() -> None:
    assert_frozen()
    bundle = pickle.loads(BUNDLE_PATH.read_bytes())
    if len(bundle["obs_sub"]) != 266:
        raise SystemExit("bundle is not Dev 266")
    sample = pd.read_csv(OUT / "a2_pilot_sample60.csv")
    if len(sample) != 60:
        raise SystemExit("pilot sample is not the locked 60")
    labels = pd.read_csv(
        ROOT / "artifacts" / "results" / "cand6_deconftf" / "knocktf401_ranks_with_study_ids.csv",
        usecols=["exp_id", "study_id", "split_new"],
    )
    dev_ids = set(labels.loc[labels["split_new"] == "Dev", "exp_id"])
    if set(sample["exp_id"]) - dev_ids:
        raise SystemExit("pilot sample is not inside Dev")
    cov = compute_bundle_coverage(bundle)
    stop = run_calls(bundle, sample, cov)
    evaluate(bundle, sample, cov, stop)
    assert_frozen()


if __name__ == "__main__":
    main()
