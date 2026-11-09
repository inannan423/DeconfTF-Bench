#!/usr/bin/env python3
"""E5 calls for one cohort. Usage: run_e5_external.py geo|unseen|replogle

Does not compute ranks or endpoints. Stops the cohort when its parse-failure
rate exceeds 20% after 20 episodes, or when E5 spend would pass 175 CNY.
"""
from __future__ import annotations

import fcntl
import json
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, "/Users/bytedance/codes/new-bishe/artifacts/src")
from budget_gate import SESSION_SOFT_CAP_CNY, BudgetExceededError, BudgetLedger, gated_chat
from develop_m10_soft_gate import compute_bundle_coverage
from regagent_tools import (
    load_track_catalog,
    match_context_tracks,
    prepare_hypothesis_resources,
    score_candidates,
)
from run_a2_pilot import parse_hypotheses
from run_e4_agent_variants import _tool_payload, parse_obj, parse_top5
from run_e5_common import (
    BUNDLE_DIR,
    E5_CAP_CNY,
    OUT,
    assert_frozen,
    build_v1_prompt,
    e5_purpose_spent,
    forbid_long_strings,
    p1_prompt,
    v2_prompt,
    v3_prompt,
)
from run_e_common import PARSE_FAIL_MIN_N, PARSE_FAIL_RATE, StageGuard, load_jsonl

COHORTS = ("geo", "unseen", "replogle")


def _banned(cohort: str) -> dict[str, list[str]]:
    if cohort != "geo":
        return {}
    contrasts = pd.read_csv(
        OUT.parent / "external" / "r2_geo_contrasts.csv",
        usecols=["contrast_id", "series_title", "method_detail", "evidence"],
    )
    qc = pd.read_csv(
        OUT.parent / "external" / "r2_geo_qc_pass.csv",
        usecols=["contrast_id", "mapping_evidence"],
    )
    merged = contrasts.merge(qc, on="contrast_id", how="left")
    out = {}
    for row in merged.itertuples(index=False):
        banned = []
        for value in (row.series_title, row.method_detail, row.evidence, row.mapping_evidence):
            text = "" if pd.isna(value) else str(value)
            if len(text) >= 12:
                banned.append(text)
        out[str(row.contrast_id)] = banned
    return out


def _load(cohort: str):
    stamp = (OUT / "e5_baseline_report.txt").read_text(encoding="utf-8")
    if "total differences: 0" not in stamp:
        raise SystemExit("baseline is not a zero-difference reproduce")
    bundle = pickle.loads((BUNDLE_DIR / f"{cohort}.pkl").read_bytes())
    sample = pd.read_csv(BUNDLE_DIR / f"{cohort}_sample.csv")
    meta = json.loads((BUNDLE_DIR / f"{cohort}_cov.json").read_text(encoding="utf-8"))
    cov = compute_bundle_coverage(bundle)
    if abs(cov - float(meta["cov"])) > 0:
        raise SystemExit(f"coverage changed for {cohort}")
    if len(sample) != int(meta["n"]):
        raise SystemExit(f"sample size changed for {cohort}")
    return bundle, sample, cov


def _note(guard: StageGuard, parsed: bool) -> str:
    with guard._lock:
        guard.done += 1
        if not parsed:
            guard.failed += 1
        if guard.done >= PARSE_FAIL_MIN_N and guard.failed / guard.done > PARSE_FAIL_RATE:
            guard.reason = f"stage stop: parse failures {guard.failed}/{guard.done}"
        return guard.reason


def _append(path, record) -> None:
    blob = json.dumps(record, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(blob + "\n")
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _call(messages, purpose, raw_path, record, guard: StageGuard):
    reason = guard.reserve(0.20)
    if reason:
        return None, reason
    try:
        res = gated_chat(
            messages,
            purpose=purpose,
            model="deepseek-v4.1-flash",
            raw_path=raw_path,
            raw_record=record,
            timeout_sec=300,
        )
    except BudgetExceededError as err:
        guard.settle(0.20, 0.0, False, track_parse=False)
        return None, f"budget_gate stopped the run: {err}"
    except Exception as err:
        record = dict(record)
        record["error"] = f"{type(err).__name__}: {err}"[:500]
        _append(raw_path, record)
        guard.settle(0.20, 0.0, False, track_parse=False)
        return None, ""
    cost = float((res.get("ledger_entry") or {}).get("cost_cny") or 0.0)
    guard.settle(0.20, cost, True, track_parse=False)
    return res, guard.reason


def run_cohort(cohort: str) -> str:
    assert cohort in COHORTS
    assert_frozen()
    ledger = BudgetLedger()
    if ledger.total_spent_cny() >= SESSION_SOFT_CAP_CNY:
        raise SystemExit("session soft cap already reached")
    bundle, sample, cov = _load(cohort)
    banned = _banned(cohort)
    raw_path = OUT / f"e5_{cohort}_raw.jsonl"
    human = [track for track in load_track_catalog() if track.cell and track.species == "HUMAN"]
    resources = prepare_hypothesis_resources(bundle, human)
    catalog = load_track_catalog()
    bases = {
        str(exp_id): score_candidates(str(exp_id), bundle, cov_frac=cov)
        for exp_id in sample["exp_id"]
    }
    prompts = []
    plan = [("v3", ("A", "C")), ("V1", ("A", "C", "B") if cohort == "geo" else ("A", "C")), ("V2", ("A", "C"))]
    if cohort != "replogle":
        plan.append(("P1", ("P1",)))
    for variant, conditions in plan:
        for rec in sample.itertuples(index=False):
            exp_id = str(rec.exp_id)
            study_id = str(rec.study_id)
            base = bases[exp_id]
            for condition in conditions:
                if variant == "v3":
                    prompt, n_redact = v3_prompt(exp_id, condition, study_id, bundle, base, cohort)
                    mapping = {}
                elif variant == "V1":
                    prompt, n_redact, mapping = build_v1_prompt(
                        exp_id, condition, study_id, bundle, base, cohort,
                    )
                elif variant == "V2":
                    prompt, n_redact = v2_prompt(exp_id, condition, study_id, bundle, base, cohort)
                    mapping = {}
                else:
                    prompt, n_redact = p1_prompt(bundle["obs_sub"].loc[exp_id], cohort, study_id)
                    mapping = {}
                forbid_long_strings(prompt, banned.get(exp_id, []), exp_id)
                prompts.append({
                    "variant": variant,
                    "exp_id": exp_id,
                    "study_id": study_id,
                    "condition": condition,
                    "prompt": prompt,
                    "n_redact": n_redact,
                    "id_map": mapping,
                    "allowed": (
                        set(mapping.values()) if condition == "B"
                        else set(str(item) for item in base["m10_order"][:25]) if variant == "V1"
                        else None
                    ),
                })
    print(f"{cohort}: {len(prompts)} prompts built, leakage checks passed", flush=True)
    done = set()
    for row in load_jsonl(raw_path):
        if row.get("variant") == "V2":
            if row.get("episode_done"):
                done.add((row.get("variant"), row.get("exp_id"), row.get("condition")))
        elif row.get("content") or row.get("error"):
            done.add((row.get("variant"), row.get("exp_id"), row.get("condition")))
    guard = StageGuard(E5_CAP_CNY, e5_purpose_spent(ledger.load_entries()), cap_cny=E5_CAP_CNY)
    by_key = {(item["variant"], item["exp_id"], item["condition"]): item for item in prompts}
    for row in load_jsonl(raw_path):
        key = (row.get("variant"), row.get("exp_id"), row.get("condition"))
        if row.get("variant") == "V2":
            if row.get("episode_done"):
                _note(guard, bool(row.get("final_ok")))
            continue
        if not (row.get("content") or row.get("error")):
            continue
        item = by_key.get(key)
        if row.get("error") or item is None:
            parsed = False
        elif row.get("variant") == "V1":
            parsed = parse_top5(row.get("content") or "", item["allowed"])[0] is not None
        else:
            parsed = parse_hypotheses(row.get("content") or "") is not None
        _note(guard, parsed)
    ones = [item for item in prompts if item["variant"] != "V2" and (item["variant"], item["exp_id"], item["condition"]) not in done]

    def _one(item):
        if guard.reason:
            return guard.reason
        res, reason = _call(
            [{"role": "user", "content": item["prompt"]}],
            f"e5:{cohort}:{item['variant']}:{item['exp_id']}:{item['condition']}",
            raw_path,
            {
                "cohort": cohort,
                "variant": item["variant"],
                "exp_id": item["exp_id"],
                "study_id": item["study_id"],
                "condition": item["condition"],
                "prompt": item["prompt"],
                "n_redact": item["n_redact"],
                "id_map": item["id_map"],
            },
            guard,
        )
        if res is None:
            parsed = False
            if reason:
                return reason
        else:
            content = res.get("content") or ""
            if item["variant"] == "V1":
                parsed = parse_top5(content, item["allowed"])[0] is not None
            else:
                parsed = parse_hypotheses(content) is not None
        stop = _note(guard, parsed)
        print(
            f"{cohort} {item['variant']} {item['exp_id']} {item['condition']} "
            f"parse={parsed} spent={guard.spent:.2f}",
            flush=True,
        )
        return stop

    if not guard.reason:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(_one, item) for item in ones]
            for fut in as_completed(futures):
                fut.result()
    v2_items = []
    if not guard.reason:
        v2_items = [
            item for item in prompts
            if item["variant"] == "V2" and ("V2", item["exp_id"], item["condition"]) not in done
        ]

    def _episode(item):
        if guard.reason:
            return guard.reason
        obs = bundle["obs_sub"].loc[item["exp_id"]]
        matched = match_context_tracks(str(obs["Biosample.Name"]), str(obs["Tissue.Type"]), catalog)
        messages = [{"role": "user", "content": item["prompt"]}]
        final_ok = False
        n_tools = 0
        reason = ""
        for step in range(7):
            res, reason = _call(
                messages,
                f"e5:{cohort}:V2:{item['exp_id']}:{item['condition']}:{step}",
                raw_path,
                {
                    "cohort": cohort,
                    "variant": "V2",
                    "exp_id": item["exp_id"],
                    "study_id": item["study_id"],
                    "condition": item["condition"],
                    "step": step,
                    "episode_done": False,
                    "n_redact": item["n_redact"],
                    "prompt": item["prompt"] if step == 0 else "",
                },
                guard,
            )
            if res is None:
                break
            content = res.get("content") or ""
            messages.append({"role": "assistant", "content": content})
            obj = parse_obj(content)
            if obj is None:
                break
            if "top5" in obj or obj.get("abstain") is True:
                final_ok = True
                break
            tool = obj.get("tool")
            tfs = obj.get("tfs")
            if tool not in {"verify_hypothesis", "exclusive_target_test", "cascade_check"} or not isinstance(tfs, list):
                break
            if n_tools >= 6:
                break
            n_tools += 1
            payload = _tool_payload(
                tool, tfs, item["exp_id"], item["condition"], bundle, resources, matched,
            )
            messages.append({"role": "user", "content": json.dumps(payload, ensure_ascii=False)})
        _append(raw_path, {
            "cohort": cohort,
            "variant": "V2",
            "exp_id": item["exp_id"],
            "condition": item["condition"],
            "episode_done": True,
            "final_ok": final_ok,
            "n_tools": n_tools,
        })
        stop = _note(guard, final_ok)
        print(
            f"{cohort} V2 {item['exp_id']} {item['condition']} "
            f"parse={final_ok} tools={n_tools} spent={guard.spent:.2f}",
            flush=True,
        )
        return reason or stop

    if v2_items:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(_episode, item) for item in v2_items]
            for fut in as_completed(futures):
                fut.result()
    status = (
        f"cohort {cohort}\n"
        f"episodes {guard.done}, parse failures {guard.failed}\n"
        f"e5 spent {guard.spent:.4f} of {E5_CAP_CNY:.0f}\n"
        f"stop: {guard.reason or 'none'}\n"
    )
    (OUT / f"e5_{cohort}_status.txt").write_text(status, encoding="utf-8")
    print(status, flush=True)
    return guard.reason


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in COHORTS:
        raise SystemExit("usage: run_e5_external.py geo|unseen|replogle")
    reason = run_cohort(sys.argv[1])
    if reason:
        raise SystemExit(reason)


if __name__ == "__main__":
    main()
