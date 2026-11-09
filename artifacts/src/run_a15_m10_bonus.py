#!/usr/bin/env python3
"""A1.5 Dev-only re-rank: context evidence is a bonus on the M10 score.

Does not call an LLM. Does not read Unseen ranks, Replogle, or GEO
expression or ranks. Does not overwrite the A1 result files.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/bytedance/codes/new-bishe/artifacts/src")
from develop_m10_soft_gate import compute_bundle_coverage
from regagent_tools import (
    _target_indices,
    context_rank,
    context_regulon_test,
    load_track_catalog,
    m11_m10_bonus_order,
    m10_plus_evidence_order,
    match_context_tracks,
    score_candidates,
    score_target_groups,
)
from verify_round4_accept_checks import exact_mcnemar_2s, study_clustered_sign_flip_perm, wilcoxon_both_modes

ROOT = Path("/Users/bytedance/codes/new-bishe")
OUT = ROOT / "artifacts" / "results" / "cand6_deconftf" / "agent"
BUNDLE_PATH = ROOT / "artifacts" / "results" / "cand6_deconftf" / "knocktf_precomputed_bundle.pkl"
FROZEN = {
    ROOT / "artifacts" / "src" / "deconftf_tools.py":
        "45939716963851d980d5767d9285832efd5b4b3537ce014b0e44f7affa225490",
    ROOT / "artifacts" / "src" / "develop_m10_soft_gate.py":
        "1e513f5236e56842e5ff96de3263175498ba9c1e6662ab8f228de3a1d7500677",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_frozen() -> None:
    for path, expected in FROZEN.items():
        got = _sha256(path)
        if got != expected:
            raise SystemExit(f"frozen hash changed: {path.name} {got}")


def rank_of(order, tf: str) -> int:
    return order.index(tf) + 1


def compare(df: pd.DataFrame, name: str, r_new: np.ndarray, r_base: np.ndarray) -> dict:
    dlog = np.log2(r_base.astype(float)) - np.log2(r_new.astype(float))
    studies = df["study_id"].to_numpy()
    mean_c, p_c = study_clustered_sign_flip_perm(
        -np.log2(r_new.astype(float)), -np.log2(r_base.astype(float)), studies
    )
    w = wilcoxon_both_modes(r_new.astype(float), r_base.astype(float))
    top1_w, top1_l, top1_p = exact_mcnemar_2s((r_new == 1).astype(int), (r_base == 1).astype(int))
    top5_w, top5_l, top5_p = exact_mcnemar_2s((r_new <= 5).astype(int), (r_base <= 5).astype(int))
    return {
        "contrast": name,
        "top1_new": int((r_new == 1).sum()),
        "top1_base": int((r_base == 1).sum()),
        "top5_new": int((r_new <= 5).sum()),
        "top5_base": int((r_base <= 5).sum()),
        "mean_dlog2": float(np.mean(dlog)),
        "median_dlog2": float(np.median(dlog)),
        "cluster_mean_dlog2": mean_c,
        "cluster_p_dlog2": p_c,
        "wilcox_p": w["p_wilcox"],
        "pratt_p": w["p_pratt"],
        "mcnemar_top1_wins": top1_w,
        "mcnemar_top1_losses": top1_l,
        "mcnemar_top1_p": top1_p,
        "mcnemar_top5_wins": top5_w,
        "mcnemar_top5_losses": top5_l,
        "mcnemar_top5_p": top5_p,
    }


def main() -> None:
    assert_frozen()
    OUT.mkdir(parents=True, exist_ok=True)
    import pickle

    bundle = pickle.loads(BUNDLE_PATH.read_bytes())
    if len(bundle["obs_sub"]) != 266:
        raise SystemExit(f"bundle is not Dev 266: {len(bundle['obs_sub'])}")
    labels = pd.read_csv(
        ROOT / "artifacts" / "results" / "cand6_deconftf" / "knocktf401_ranks_with_study_ids.csv",
        usecols=["exp_id", "study_id", "split_new"],
    )
    dev = labels[labels["split_new"] == "Dev"]
    if set(dev["exp_id"]) != set(bundle["obs_sub"].index):
        raise SystemExit("bundle index is not exactly Dev")
    study = dev.set_index("exp_id")["study_id"].astype(str)

    catalog = load_track_catalog()
    human = [t for t in catalog if t.cell and t.species == "HUMAN"]
    g2i = bundle["gene_to_idx"]
    all_groups = _target_indices(human, g2i)
    n_tracks_all = {}
    for track in human:
        n_tracks_all[track.tf] = n_tracks_all.get(track.tf, 0) + 1
    cov = compute_bundle_coverage(bundle)

    rows = []
    n_exp = len(bundle["obs_sub"].index)
    for i_exp, exp_id in enumerate(bundle["obs_sub"].index, start=1):
        if i_exp == 1 or i_exp % 50 == 0 or i_exp == n_exp:
            print(f"scoring {i_exp}/{n_exp}", flush=True)
        obs = bundle["obs_sub"].loc[exp_id]
        true = str(obs["TF"])
        base = score_candidates(exp_id, bundle, cov_frac=cov)
        matched = match_context_tracks(str(obs["Biosample.Name"]), str(obs["Tissue.Type"]), human)
        ctx = context_regulon_test(exp_id, matched, bundle)
        y = bundle["mat_blind"].loc[exp_id].to_numpy(dtype=np.float64)
        oracle_true = score_target_groups(
            y, {true: all_groups[true]} if true in all_groups else {}, bundle["cand_tfs"], n_tracks_all,
        )
        oracle_all = score_target_groups(y, all_groups, bundle["cand_tfs"], n_tracks_all)

        r_ulm = rank_of(list(base["ulm"].sort_values(ascending=True).index), true)
        r_doro = rank_of(list(bundle["est_d"].loc[exp_id].sort_values(ascending=True).index), true)
        r_chea = rank_of(list(bundle["chea3_toprank"].loc[exp_id].sort_values(ascending=True).index), true)
        r_m5 = rank_of(list(base["m5"].sort_values(ascending=False).index), true)
        r_m10 = rank_of(base["m10_order"], true)
        r_m11 = rank_of(m11_m10_bonus_order(exp_id, bundle, human, cov_frac=cov), true)
        recalled = int(min(r_ulm, r_doro, r_chea, r_m5) <= 25)

        t_rule = ctx["t_ulm"] if len(ctx) else pd.Series(dtype=float)
        t_oracle = t_rule.copy()
        if len(oracle_true) and true in oracle_true.index:
            t_oracle.loc[true] = float(oracle_true.loc[true, "t_ulm"])
        t_all = oracle_all["t_ulm"] if len(oracle_all) else pd.Series(dtype=float)
        rank_rule = rank_of(m10_plus_evidence_order(base["m10_order"], base["m10_score"], t_rule), true)
        rank_oracle_true = rank_of(m10_plus_evidence_order(base["m10_order"], base["m10_score"], t_oracle), true)
        rank_oracle_all = rank_of(m10_plus_evidence_order(base["m10_order"], base["m10_score"], t_all), true)
        rank_old = rank_of(context_rank(t_rule, base["m5"]), true)

        rows.append({
            "exp_id": exp_id,
            "study_id": study.loc[exp_id],
            "true_tf": true,
            "in_recalled": recalled,
            "r_m5": r_m5,
            "r_m10": r_m10,
            "r_m11_m10_bonus": r_m11,
            "n_scored_tfs": int(len(ctx)),
            "true_has_matched_track": int(true in set(ctx.index)),
            "rank_rule_m10_bonus": rank_rule,
            "rank_oracle_true_m10_bonus": rank_oracle_true,
            "rank_oracle_all_m10_bonus": rank_oracle_all,
            "rank_rule_a1_frontload": rank_old,
        })

    df = pd.DataFrame(rows)
    if int(df["in_recalled"].sum()) != 124:
        raise SystemExit(f"recall count is {int(df['in_recalled'].sum())}, expected 124")
    df.to_csv(OUT / "a15_dev266_per_experiment.csv", index=False)

    fail = df[df["in_recalled"] == 0]
    ok = df[df["in_recalled"] == 1]

    def rescue(frame, col):
        hit = frame[col] <= 25
        return int(hit.sum()), float(hit.mean()) if len(frame) else float("nan")

    n_fail = len(fail)
    rule_n, rule_p = rescue(fail, "rank_rule_m10_bonus")
    old_n, old_p = rescue(fail, "rank_rule_a1_frontload")
    ot_n, ot_p = rescue(fail, "rank_oracle_true_m10_bonus")
    oa_n, oa_p = rescue(fail, "rank_oracle_all_m10_bonus")
    harm_n = int((ok["rank_rule_m10_bonus"] > 25).sum())
    harm_old = int((ok["rank_rule_a1_frontload"] > 25).sum())
    harm_unmatched = int(((ok["rank_rule_m10_bonus"] > 25) & (ok["true_has_matched_track"] == 0)).sum())
    harm_vs_m10 = int(((df["r_m10"] <= 25) & (df["rank_rule_m10_bonus"] > 25)).sum())

    ceiling = pd.DataFrame([
        {"item": "n_recall_fail", "n": n_fail, "pct": round(100 * n_fail / 266, 2)},
        {"item": "n_recall_success", "n": len(ok), "pct": round(100 * len(ok) / 266, 2)},
        {"item": "rule_rescue_m10_bonus_top25", "n": rule_n, "pct": round(100 * rule_p, 2)},
        {"item": "rule_rescue_a1_frontload_top25", "n": old_n, "pct": round(100 * old_p, 2)},
        {"item": "oracle_true_m10_bonus_top25", "n": ot_n, "pct": round(100 * ot_p, 2)},
        {"item": "oracle_all_m10_bonus_top25", "n": oa_n, "pct": round(100 * oa_p, 2)},
        {"item": "harm_m10_bonus_among_recalled", "n": harm_n, "pct": round(100 * harm_n / len(ok), 2)},
        {"item": "harm_a1_frontload_among_recalled", "n": harm_old, "pct": round(100 * harm_old / len(ok), 2)},
        {"item": "harm_m10_bonus_and_unmatched", "n": harm_unmatched, "pct": round(100 * harm_unmatched / len(ok), 2)},
        {"item": "harm_left_m10_top25", "n": harm_vs_m10, "pct": round(100 * harm_vs_m10 / 266, 2)},
    ])
    ceiling.to_csv(OUT / "a15_dev266_ceiling_summary.csv", index=False)

    comp = pd.DataFrame([
        compare(df, "M11b_vs_M5", df["r_m11_m10_bonus"].to_numpy(), df["r_m5"].to_numpy()),
        compare(df, "M11b_vs_M10", df["r_m11_m10_bonus"].to_numpy(), df["r_m10"].to_numpy()),
        compare(df, "M10_vs_M5", df["r_m10"].to_numpy(), df["r_m5"].to_numpy()),
    ])
    comp.to_csv(OUT / "a15_dev266_m11_comparison.csv", index=False)

    lines = [
        "A1.5 Dev 266. Context t is a z-score bonus on the M10 score. Weight 1.0, not refit.",
        "M11b uses that bonus plus the original exclusive (0.25) and cascade (0.5) terms.",
        f"Recall fail {n_fail}. M10-bonus rescue {rule_n}/{n_fail} ({100 * rule_p:.2f}%). A1 front-load rescue was {old_n}/{n_fail} ({100 * old_p:.2f}%).",
        f"Oracle true-TF any cell, same bonus: {ot_n}/{n_fail} ({100 * ot_p:.2f}%). Oracle all TFs: {oa_n}/{n_fail} ({100 * oa_p:.2f}%).",
        f"Harm among {len(ok)} recalled: M10-bonus {harm_n} ({100 * harm_n / len(ok):.2f}%), of which {harm_unmatched} had no matched track. A1 front-load harm was {harm_old}.",
        f"Experiments whose true TF left an M10 Top-25: {harm_vs_m10}.",
        f"M11b top1 {int((df.r_m11_m10_bonus == 1).sum())}/266 top5 {int((df.r_m11_m10_bonus <= 5).sum())}/266; "
        f"M5 {int((df.r_m5 == 1).sum())}/{int((df.r_m5 <= 5).sum())}; M10 {int((df.r_m10 == 1).sum())}/{int((df.r_m10 <= 5).sum())}.",
        "No go/no-go call.",
    ]
    (OUT / "a15_dev266_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(comp.to_string(index=False))
    assert_frozen()


if __name__ == "__main__":
    main()
