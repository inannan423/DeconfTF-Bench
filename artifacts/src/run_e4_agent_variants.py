#!/usr/bin/env python3
"""E4 on the v3 Dev 60. V1 reranks the M10 pool. V2 is a 6-step tool agent.

Both use flash and conditions A and C. Exploratory. Selection for E5 is the
condition-C mean Δlog2 versus M10, then lower cost. v3 stays a fixed entrant.
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
from budget_gate import SESSION_SOFT_CAP_CNY, BudgetExceededError, BudgetLedger, gated_chat
from deconftf_tools import UNIVERSAL_SECONDARY_HUBS
from develop_m10_soft_gate import compute_bundle_coverage
from regagent_tools import (
    cascade_check,
    exclusive_target_test,
    load_track_catalog,
    match_context_tracks,
    prepare_hypothesis_resources,
    score_candidates,
    verify_hypothesis,
)
from run_a2_pilot_design import redact_tf_tokens, symbol_in_text
from run_e_common import StageGuard, load_jsonl, stage_spent
from verify_round4_accept_checks import study_clustered_sign_flip_perm

ROOT = Path("/Users/bytedance/codes/new-bishe")
OUT = ROOT / "artifacts" / "results" / "cand6_deconftf" / "agent"
BUNDLE_PATH = ROOT / "artifacts" / "results" / "cand6_deconftf" / "knocktf_precomputed_bundle.pkl"
RAW_V1 = OUT / "e4_v1_raw.jsonl"
RAW_V2 = OUT / "e4_v2_raw.jsonl"
FROZEN = {
    ROOT / "artifacts" / "src" / "deconftf_tools.py":
        "45939716963851d980d5767d9285832efd5b4b3537ce014b0e44f7affa225490",
    ROOT / "artifacts" / "src" / "develop_m10_soft_gate.py":
        "1e513f5236e56842e5ff96de3263175498ba9c1e6662ab8f228de3a1d7500677",
}
ESTIMATE_CNY = 70.0
MAX_TOOLS = 6
MAX_TFS = 8


def assert_frozen() -> None:
    for path, expected in FROZEN.items():
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != expected:
            raise SystemExit(f"frozen hash changed: {path.name} {got}")


def _num(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return f"{number:.3f}"


def _context(obs, study_id: str, condition: str, tf: str) -> str:
    if condition == "C":
        return "Biosample = REDACTED. Tissue = REDACTED. Study = REDACTED. Title = REDACTED."
    bio, _ = redact_tf_tokens(obs["Biosample.Name"], tf)
    tissue, _ = redact_tf_tokens(obs["Tissue.Type"], tf)
    study, _ = redact_tf_tokens(str(study_id), tf)
    return f"Biosample = {bio}. Tissue = {tissue}. Study = {study}."


def _evidence(exp_id: str, order: list, bundle, pool: list) -> str:
    z = bundle["z_libs"]
    chip = pd.concat(
        [
            z["ENCODE_ChIP-seq"].loc[exp_id],
            z["ReMap_ChIP-seq"].loc[exp_id],
            z["Literature_ChIP-seq"].loc[exp_id],
        ],
        axis=1,
    ).max(axis=1)
    doro = z["DoRothEA_Signed"].loc[exp_id]
    cc = -bundle["est_c_res"].loc[exp_id]
    excl = {row["tf"]: row for row in exclusive_target_test(exp_id, pool, bundle)}
    lines = []
    for rank, tf in enumerate(pool, start=1):
        row = excl.get(tf, {})
        hub = int(tf in UNIVERSAL_SECONDARY_HUBS)
        lines.append(
            f"{rank}. {tf} chip_z={_num(chip.get(tf))} dorothea_z={_num(doro.get(tf))} "
            f"cc={_num(cc.get(tf))} hub={hub} "
            f"exclusive_t={_num(row.get('excl_t'))} n_exclusive={row.get('n_exclusive', 0)}"
        )
    del order
    return "\n".join(lines)


def build_v1(exp_id, condition, study_id, bundle, base) -> str:
    obs = bundle["obs_sub"].loc[exp_id]
    tf = str(obs["TF"]).upper()
    pool = [str(item) for item in base["m10_order"][:25]]
    prompt = (
        "Rerank the fixed-pipeline top 25. Return JSON only, no markdown. "
        'Schema: {"top5": ["SYMBOL", "..."]} with exactly 5 symbols from this list, '
        "most plausible primary knockdown first.\n"
        f"{_context(obs, study_id, condition, tf)}\n"
        "Candidates, with deterministic evidence:\n"
        f"{_evidence(exp_id, base['m10_order'], bundle, pool)}\n"
    )
    if symbol_in_text(_context(obs, study_id, condition, tf), tf):
        raise SystemExit(f"leakage: {exp_id} {condition} context contains {tf}")
    return prompt


def parse_obj(content: str):
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if "```" in text:
            text = text[: text.rfind("```")]
        text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_top5(content: str, allowed: set | None):
    obj = parse_obj(content)
    if not obj or obj.get("abstain") is True:
        return obj, []
    top = obj.get("top5")
    if not isinstance(top, list):
        return None, []
    out = []
    for item in top:
        if not isinstance(item, str):
            return None, []
        symbol = item.strip().upper()
        if allowed is not None and symbol not in allowed:
            continue
        if symbol not in out:
            out.append(symbol)
    if len(out) < 1:
        return None, []
    return obj, out[:5]


def merged_rank(top5: list, m10_order: list, true: str) -> int:
    front = []
    allowed = set(m10_order)
    for tf in top5:
        if tf in allowed and tf not in front:
            front.append(tf)
    order = front + [tf for tf in m10_order if tf not in front]
    return order.index(true) + 1


def _call(messages, purpose, raw_path, record, guard, reserve: float):
    reason = guard.reserve(reserve)
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
        guard.settle(reserve, 0.0, False, track_parse=False)
        return None, f"budget_gate stopped the run: {err}"
    except Exception as err:
        record = dict(record)
        record["error"] = f"{type(err).__name__}: {err}"[:500]
        with raw_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        guard.settle(reserve, 0.0, False, track_parse=False)
        return None, ""
    cost = float((res.get("ledger_entry") or {}).get("cost_cny") or 0.0)
    guard.settle(reserve, cost, True, track_parse=False)
    return res, guard.reason


def _note_episode(guard: StageGuard, parsed: bool, label: str) -> str:
    guard.done += 1
    if not parsed:
        guard.failed += 1
    if guard.done >= 20 and guard.failed / guard.done > 0.20:
        guard.reason = f"stage stop: {label} parse failures {guard.failed}/{guard.done}"
    return guard.reason


def run_v1(bundle, sample, cov, guard: StageGuard) -> str:
    done = {(row.get("exp_id"), row.get("condition")) for row in load_jsonl(RAW_V1)}
    for rec in sample.itertuples(index=False):
        for condition in ("A", "C"):
            if guard.reason:
                return guard.reason
            if (rec.exp_id, condition) in done:
                continue
            base = score_candidates(rec.exp_id, bundle, cov_frac=cov)
            prompt = build_v1(rec.exp_id, condition, rec.study_id, bundle, base)
            res, reason = _call(
                [{"role": "user", "content": prompt}],
                f"e4v1:{rec.exp_id}:{condition}",
                RAW_V1,
                {"exp_id": rec.exp_id, "condition": condition, "variant": "V1", "prompt": prompt},
                guard,
                0.20,
            )
            if res is None:
                return reason or "V1 call failed"
            parsed = parse_top5(res.get("content") or "", set(base["m10_order"][:25]))[0] is not None
            print(f"V1 {rec.exp_id} {condition} parse={parsed} spent={guard.spent:.2f}", flush=True)
            episode_reason = _note_episode(guard, parsed, "V1")
            if reason or episode_reason:
                return reason or episode_reason
    return guard.reason


def _tool_payload(name: str, tfs: list, exp_id: str, condition: str, bundle, resources, matched) -> dict:
    symbols = []
    for tf in tfs[:MAX_TFS]:
        if isinstance(tf, str):
            symbols.append(tf.strip().upper())
    if name == "verify_hypothesis":
        rows = []
        for tf in symbols:
            row = verify_hypothesis(exp_id, tf, bundle, resources)
            rows.append({
                "tf": row["tf"],
                "t_stat": None if not np.isfinite(row["t_stat"]) else round(float(row["t_stat"]), 4),
                "source": row["source"],
                "n_targets": row["n_targets"],
            })
        return {"tool": name, "rows": rows}
    if name == "exclusive_target_test":
        rows = []
        for row in exclusive_target_test(exp_id, symbols, bundle):
            rows.append({
                "tf": row["tf"],
                "n_targets": row["n_targets"],
                "n_exclusive": row["n_exclusive"],
                "excl_t": None if not np.isfinite(row["excl_t"]) else round(float(row["excl_t"]), 4),
            })
        return {"tool": name, "rows": rows}
    if name == "cascade_check":
        tracks = matched if condition == "A" else None
        rows = []
        for row in cascade_check(exp_id, symbols, bundle, matched_tracks=tracks):
            upstream = row["upstream"][:6]
            rows.append({"tf": row["tf"], "is_downstream": row["is_downstream"], "upstream": upstream})
        return {"tool": name, "rows": rows}
    return {"tool": name, "error": "unknown tool"}


def build_v2(exp_id, condition, study_id, bundle, base) -> str:
    obs = bundle["obs_sub"].loc[exp_id]
    tf = str(obs["TF"]).upper()
    pool = ", ".join(str(item) for item in base["m10_order"][:25])
    prompt = (
        "Identify the primary knocked-down transcription factor. "
        "Reply with JSON only. A tool request has the form "
        '{"tool": "verify_hypothesis" or "exclusive_target_test" or "cascade_check", "tfs": ["SYMBOL"]}. '
        "At most 8 symbols per request and at most 6 tool requests. "
        "When you are done, reply {\"top5\": [five symbols]} or {\"abstain\": true}. "
        "Do not invent numbers. Only a tool response contains numeric evidence. "
        "You may name a symbol outside the starting list after a tool has scored it.\n"
        f"{_context(obs, study_id, condition, tf)}\n"
        f"Fixed-pipeline top 25, names only: {pool}\n"
    )
    if symbol_in_text(_context(obs, study_id, condition, tf), tf):
        raise SystemExit(f"leakage: {exp_id} {condition} context contains {tf}")
    return prompt


def run_v2(bundle, sample, cov, resources, catalog, guard: StageGuard) -> str:
    done_rows = load_jsonl(RAW_V2)
    done = set()
    for row in done_rows:
        if row.get("episode_done"):
            done.add((row.get("exp_id"), row.get("condition")))
    for rec in sample.itertuples(index=False):
        obs = bundle["obs_sub"].loc[rec.exp_id]
        matched = match_context_tracks(str(obs["Biosample.Name"]), str(obs["Tissue.Type"]), catalog)
        for condition in ("A", "C"):
            if guard.reason:
                return guard.reason
            if (rec.exp_id, condition) in done:
                continue
            base = score_candidates(rec.exp_id, bundle, cov_frac=cov)
            prompt = build_v2(rec.exp_id, condition, rec.study_id, bundle, base)
            messages = [{"role": "user", "content": prompt}]
            final_ok = False
            n_tools = 0
            for step in range(MAX_TOOLS + 1):
                res, reason = _call(
                    messages,
                    f"e4v2:{rec.exp_id}:{condition}:{step}",
                    RAW_V2,
                    {
                        "exp_id": rec.exp_id,
                        "condition": condition,
                        "variant": "V2",
                        "step": step,
                        "episode_done": False,
                    },
                    guard,
                    0.25,
                )
                if res is None:
                    return reason or guard.reason
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
                if n_tools >= MAX_TOOLS:
                    break
                n_tools += 1
                payload = _tool_payload(tool, tfs, rec.exp_id, condition, bundle, resources, matched)
                messages.append({"role": "user", "content": json.dumps(payload, ensure_ascii=False)})
            with RAW_V2.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "exp_id": rec.exp_id,
                    "condition": condition,
                    "variant": "V2",
                    "episode_done": True,
                    "final_ok": final_ok,
                    "n_tools": n_tools,
                }, ensure_ascii=False) + "\n")
            print(f"V2 {rec.exp_id} {condition} final={final_ok} tools={n_tools} spent={guard.spent:.2f}", flush=True)
            episode_reason = _note_episode(guard, final_ok, "V2")
            if reason or episode_reason:
                return reason or episode_reason
    return guard.reason


def _episode_top5(rows: list, allowed: set | None):
    finals = [row for row in rows if row.get("raw_response") or row.get("content")]
    for row in reversed(finals):
        obj, top5 = parse_top5(row.get("content") or "", allowed)
        if obj is None:
            continue
        if obj.get("abstain") is True:
            return True, []
        if top5:
            return True, top5
    return False, []


def evaluate(bundle, sample, cov, stop: str) -> None:
    v1 = load_jsonl(RAW_V1)
    v2 = load_jsonl(RAW_V2)
    v1_by = {}
    for row in v1:
        v1_by[(row.get("exp_id"), row.get("condition"))] = row
    v2_by = {}
    for row in v2:
        v2_by.setdefault((row.get("exp_id"), row.get("condition")), []).append(row)
    rows = []
    for rec in sample.itertuples(index=False):
        true = str(bundle["obs_sub"].loc[rec.exp_id, "TF"]).upper()
        base = score_candidates(rec.exp_id, bundle, cov_frac=cov)
        m10 = list(base["m10_order"])
        r_m10 = m10.index(true) + 1
        row = {
            "exp_id": rec.exp_id,
            "study_id": str(rec.study_id),
            "true_tf": true,
            "r_m10": r_m10,
            "in_pool25": int(r_m10 <= 25),
        }
        for condition in ("A", "C"):
            got = v1_by.get((rec.exp_id, condition))
            ok, top5 = (False, [])
            if got is not None:
                ok, top5 = parse_top5(got.get("content") or "", set(m10[:25]))
                ok = ok is not None and (bool(top5) or (ok or {}).get("abstain") is True)
            rank = merged_rank(top5, m10, true) if ok and top5 else r_m10
            row[f"V1_{condition}_parse_ok"] = int(bool(ok) and got is not None)
            row[f"V1_{condition}_rank"] = rank
            row[f"V1_{condition}_top5"] = "|".join(top5)
            eps = v2_by.get((rec.exp_id, condition), [])
            calls = [item for item in eps if item.get("raw_response") or item.get("content") or item.get("error")]
            ok2, top2 = _episode_top5(calls, None)
            rank2 = merged_rank(top2, m10, true) if ok2 and top2 else r_m10
            row[f"V2_{condition}_parse_ok"] = int(ok2)
            row[f"V2_{condition}_rank"] = rank2
            row[f"V2_{condition}_top5"] = "|".join(top2)
            row[f"V2_{condition}_n_calls"] = len(calls)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e4_variants_per_experiment.csv", index=False)
    lines = [
        "E4 agent variants on the v3 Dev 60. Exploratory.",
        stop or "all scheduled calls finished",
        f"V1 calls {len(v1)}, V2 records {len(v2)}, combined spend {stage_spent(v1) + stage_spent(v2):.4f} CNY.",
        "V1 may only emit symbols from the M10 top 25. Its rank puts those five first and keeps the other M10 order.",
        "V2 may emit any candidate after a tool call. Abstain or a parse failure leaves the M10 rank unchanged.",
        "Condition C cascade_check does not receive cell-line ChIP tracks.",
    ]
    summary = []
    for variant in ("V1", "V2"):
        for condition in ("A", "C"):
            rank_col = f"{variant}_{condition}_rank"
            ok_col = f"{variant}_{condition}_parse_ok"
            delta = np.log2(df["r_m10"].to_numpy(dtype=float)) - np.log2(df[rank_col].to_numpy(dtype=float))
            mean, p = study_clustered_sign_flip_perm(
                -np.log2(df[rank_col].to_numpy(dtype=float)),
                -np.log2(df["r_m10"].to_numpy(dtype=float)),
                df["study_id"].astype(str).to_numpy(),
                n_perm=20000,
                seed=42,
            )
            top1 = int((df[rank_col] <= 1).sum())
            top5 = int((df[rank_col] <= 5).sum())
            cost = stage_spent(v1 if variant == "V1" else v2) / 2.0
            lines.append(
                f"{variant} {condition}: parse {int(df[ok_col].sum())}/{len(df)}, "
                f"Top-1 {top1}/{len(df)}, Top-5 {top5}/{len(df)}, "
                f"mean Δlog2 {mean:.4f} (p={p:.4f})."
            )
            if condition == "C":
                call_cost = stage_spent(v1 if variant == "V1" else v2)
                summary.append((variant, mean, call_cost, int(df[ok_col].sum()) == len(df) or not str(stop)))
            del delta, cost
    summary.sort(key=lambda item: (-item[1], item[2]))
    chosen = [item[0] for item in summary[:2]]
    lines.append(
        "Selection rule, fixed before the run: at most two variants, highest condition-C mean Δlog2, "
        "then lower cost. v3 propose-verify is a fixed additional entrant and is not dropped."
    )
    lines.append("Ranked condition C: " + ", ".join(f"{name} Δ={mean:.4f} cost={cost:.3f}" for name, mean, cost, _ok in summary))
    lines.append("Enter E5: " + ", ".join(chosen + ["v3_propose_verify"]))
    lines.append(f"Pool base rate: true TF already inside M10 top 25 in {int(df['in_pool25'].sum())}/{len(df)}.")
    text = "\n".join(lines) + "\n"
    (OUT / "e4_variants_summary.txt").write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    assert_frozen()
    if BudgetLedger().total_spent_cny() >= SESSION_SOFT_CAP_CNY:
        raise SystemExit("session soft cap already reached")
    bundle = pickle.loads(BUNDLE_PATH.read_bytes())
    sample = pd.read_csv(OUT / "a2v3_pilot_sample60.csv")
    if len(sample) != 60:
        raise SystemExit("v3 sample is not 60")
    cov = compute_bundle_coverage(bundle)
    human = [t for t in load_track_catalog() if t.cell and t.species == "HUMAN"]
    resources = prepare_hypothesis_resources(bundle, human)
    catalog = load_track_catalog()
    spent = stage_spent(load_jsonl(RAW_V1)) + stage_spent(load_jsonl(RAW_V2))
    guard = StageGuard(ESTIMATE_CNY, spent)
    # V1 and V2 episode failures are judged in the summary. The guard still
    # stops on spend. V2 also sets the parse-rate reason itself.
    stop = run_v1(bundle, sample, cov, guard)
    guard.done = 0
    guard.failed = 0
    if not stop:
        stop = run_v2(bundle, sample, cov, resources, catalog, guard)
    if stop:
        print(stop, flush=True)
    evaluate(bundle, sample, cov, stop or "")
    assert_frozen()


if __name__ == "__main__":
    main()
