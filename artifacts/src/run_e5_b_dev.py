#!/usr/bin/env python3
"""Development check for condition B on the locked v3 Dev 60.

One V1 call per experiment. A call parses when the reply contains at least one
ID from that experiment's map. The check keeps B when failures are at most 12
of 60. It does not compute ranks.
"""
from __future__ import annotations

import json
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/Users/bytedance/codes/new-bishe/artifacts/src")
from budget_gate import SESSION_SOFT_CAP_CNY, BudgetExceededError, BudgetLedger, gated_chat
from develop_m10_soft_gate import compute_bundle_coverage
from regagent_tools import score_candidates
from run_e4_agent_variants import parse_top5
from run_e5_common import OUT, assert_frozen, build_v1_prompt
from run_e_common import StageGuard, load_jsonl

RAW = OUT / "e5b_dev_raw.jsonl"
SUMMARY = OUT / "e5b_dev_summary.txt"
ESTIMATE = 6.0
KEEP_IF_FAILS_AT_MOST = 12


def _check(bundle, sample, cov) -> None:
    for rec in sample.itertuples(index=False):
        base = score_candidates(rec.exp_id, bundle, cov_frac=cov)
        pool = [str(item) for item in base["m10_order"][:25]]
        prompt, _n, mapping = build_v1_prompt(
            rec.exp_id, "B", str(rec.study_id), bundle, base, "dev",
        )
        if set(mapping) != set(pool) or len(mapping) != 25:
            raise SystemExit(f"{rec.exp_id}: ID map is not the 25 candidates")
        if any(symbol in prompt for symbol in pool):
            raise SystemExit(f"{rec.exp_id}: a candidate symbol is still in the B prompt")
        again, _n2, mapping2 = build_v1_prompt(
            rec.exp_id, "B", str(rec.study_id), bundle, base, "dev",
        )
        if prompt != again or mapping != mapping2:
            raise SystemExit(f"{rec.exp_id}: B prompt is not stable")
    print(f"offline B check passed for {len(sample)} prompts", flush=True)


def main() -> None:
    assert_frozen()
    if BudgetLedger().total_spent_cny() >= SESSION_SOFT_CAP_CNY:
        raise SystemExit("session soft cap already reached")
    bundle = pickle.loads(
        (OUT.parent / "knocktf_precomputed_bundle.pkl").read_bytes()
    )
    sample = pd_read()
    cov = compute_bundle_coverage(bundle)
    _check(bundle, sample, cov)
    if "--check" in sys.argv:
        return
    done = {row.get("exp_id") for row in load_jsonl(RAW) if row.get("content") or row.get("error")}
    guard = StageGuard(ESTIMATE, 0.0)
    jobs = []
    for rec in sample.itertuples(index=False):
        if rec.exp_id in done:
            continue
        base = score_candidates(rec.exp_id, bundle, cov_frac=cov)
        prompt, n_redact, mapping = build_v1_prompt(
            rec.exp_id, "B", str(rec.study_id), bundle, base, "dev",
        )
        jobs.append((rec.exp_id, prompt, n_redact, mapping))

    def _one(job):
        exp_id, prompt, n_redact, mapping = job
        reason = guard.reserve(0.20)
        if reason:
            return reason
        record = {
            "exp_id": exp_id,
            "condition": "B",
            "variant": "V1",
            "cohort": "dev",
            "prompt": prompt,
            "id_map": mapping,
            "n_redact": n_redact,
        }
        try:
            res = gated_chat(
                [{"role": "user", "content": prompt}],
                purpose=f"e5bdev:{exp_id}:B",
                model="deepseek-v4.1-flash",
                raw_path=RAW,
                raw_record=record,
                timeout_sec=300,
            )
        except BudgetExceededError as err:
            guard.settle(0.20, 0.0, False)
            return f"budget_gate stopped the run: {err}"
        except Exception as err:
            record["error"] = f"{type(err).__name__}: {err}"[:500]
            with RAW.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            guard.settle(0.20, 0.0, False)
            print(f"B {exp_id} parse=False spent={guard.spent:.2f}", flush=True)
            return guard.reason
        allowed = set(mapping.values())
        parsed = parse_top5(res.get("content") or "", allowed)[0] is not None
        cost = float((res.get("ledger_entry") or {}).get("cost_cny") or 0.0)
        guard.settle(0.20, cost, parsed)
        print(f"B {exp_id} parse={parsed} spent={guard.spent:.2f}", flush=True)
        return guard.reason

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_one, job) for job in jobs]
        for fut in as_completed(futures):
            reason = fut.result()
            if reason and not guard.reason:
                guard.reason = reason
    rows = [row for row in load_jsonl(RAW) if row.get("prompt")]
    failed = 0
    for row in rows:
        allowed = set((row.get("id_map") or {}).values())
        content = row.get("content") or ""
        if row.get("error") or parse_top5(content, allowed)[0] is None:
            failed += 1
    keep = len(rows) == 60 and failed <= KEEP_IF_FAILS_AT_MOST
    text = (
        "Condition B development check on the v3 Dev 60. Ranks were not computed.\n"
        f"Stored calls {len(rows)}/60. Parse failures {failed}/{len(rows) or 1}.\n"
        f"Keep B for E5: {keep}. Threshold is at most {KEEP_IF_FAILS_AT_MOST} failures out of 60.\n"
        f"Stop note: {guard.reason or 'none'}\n"
    )
    SUMMARY.write_text(text, encoding="utf-8")
    print(text, flush=True)


def pd_read():
    import pandas as pd
    sample = pd.read_csv(OUT / "a2v3_pilot_sample60.csv")
    if len(sample) != 60:
        raise SystemExit("v3 sample is not 60")
    return sample


if __name__ == "__main__":
    main()
