#!/usr/bin/env python3
"""A1 ceiling test of context regulons on KnockTF Dev 266 only. No LLM.

Does not read Unseen ranks, Replogle, or GEO expression/ranks.
Oracle columns are ceilings, not a method.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/bytedance/codes/new-bishe/artifacts/src")
import decoupler as dc
from develop_m10_soft_gate import compute_bundle_coverage
from regagent_tools import (
    ChipTrack,
    _target_indices,
    cascade_check,
    context_rank,
    context_regulon_test,
    exclusive_target_test,
    load_track_catalog,
    m11_order,
    match_context_tracks,
    score_candidates,
    score_target_groups,
    track_parse_table,
    ulm_t_binary,
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


def verify_ulm_matches_decoupler(exp_id: str, tracks: list, bundle) -> float:
    """Max |t_local - t_decoupler| on up to 8 matched TFs. Raises if > 1e-3.

    ``empty=False`` keeps log2FC == 0 genes. The default drops them on a
    one-row matrix, which is not the frozen multi-sample ULM universe.
    Targets are the column names at the uppercase ``gene_to_idx`` positions.
    """
    ctx = context_regulon_test(exp_id, tracks, bundle)
    if ctx.empty:
        return float("nan")
    use = ctx.head(8)
    y = bundle["mat_blind"].loc[exp_id]
    genes = list(y.index)
    g2i = bundle["gene_to_idx"]
    rows = []
    for tf, rec in use.iterrows():
        tf_tracks = [t for t in tracks if t.tf == tf]
        idxs = sorted({
            g2i[g] for t in tf_tracks for g in t.genes if g != tf and g in g2i
        })
        local = ulm_t_binary(y.to_numpy(dtype=np.float64), idxs)
        targets = [genes[i] for i in idxs]
        net = pd.DataFrame({"source": tf, "target": targets, "weight": 1.0})
        data = pd.DataFrame(y.to_numpy(dtype=np.float64).reshape(1, -1), index=[exp_id], columns=genes)
        est, _ = dc.mt.ulm(data=data, net=net, empty=False)
        dec = float(est.loc[exp_id, tf])
        rows.append(abs(local - dec))
        if abs(local - rec["t_ulm"]) > 1e-8:
            raise SystemExit(f"context_regulon_test disagrees with ulm_t_binary for {tf}")
    gap = float(max(rows))
    if gap > 1e-3:
        raise SystemExit(f"ULM does not match decoupler (max abs diff {gap})")
    return gap


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
    parse_tbl = track_parse_table(catalog)
    parse_tbl.to_csv(OUT / "a1_track_parse_summary.csv", index=False)
    human = [t for t in catalog if t.cell and t.species == "HUMAN"]
    by_tf: dict[str, list[ChipTrack]] = {}
    for track in human:
        by_tf.setdefault(track.tf, []).append(track)
    # Target unions do not depend on the experiment. Score them per sample below.
    g2i = bundle["gene_to_idx"]
    all_groups = _target_indices(human, g2i)
    n_tracks_all = {tf: len(items) for tf, items in by_tf.items()}
    true_groups = {tf: idxs for tf, idxs in all_groups.items()}

    cov = compute_bundle_coverage(bundle)
    # Equivalence on the first Dev experiment that matches at least one human track.
    ulm_gap = float("nan")
    for exp_id in bundle["obs_sub"].index:
        obs = bundle["obs_sub"].loc[exp_id]
        matched = match_context_tracks(str(obs["Biosample.Name"]), str(obs["Tissue.Type"]), human)
        if matched:
            ulm_gap = verify_ulm_matches_decoupler(exp_id, matched, bundle)
            break
    if not np.isfinite(ulm_gap):
        raise SystemExit("no Dev experiment matched a human ChIP track; matcher is empty")

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
            y, {true: true_groups[true]} if true in true_groups else {}, bundle["cand_tfs"], n_tracks_all,
        )
        oracle_all = score_target_groups(y, all_groups, bundle["cand_tfs"], n_tracks_all)

        r_ulm = rank_of(list(base["ulm"].sort_values(ascending=True).index), true)
        r_doro = rank_of(list(bundle["est_d"].loc[exp_id].sort_values(ascending=True).index), true)
        r_chea = rank_of(list(bundle["chea3_toprank"].loc[exp_id].sort_values(ascending=True).index), true)
        r_m5 = rank_of(list(base["m5"].sort_values(ascending=False).index), true)
        r_m9 = rank_of(base["m9_order"], true)
        r_m10 = rank_of(base["m10_order"], true)
        r_m11 = rank_of(m11_order(exp_id, bundle, human, cov_frac=cov), true)
        recalled = int(min(r_ulm, r_doro, r_chea, r_m5) <= 25)

        rule_order = context_rank(ctx["t_ulm"] if len(ctx) else pd.Series(dtype=float), base["m5"])
        t_oracle = ctx["t_ulm"].copy() if len(ctx) else pd.Series(dtype=float)
        if len(oracle_true) and true in oracle_true.index:
            t_oracle.loc[true] = float(oracle_true.loc[true, "t_ulm"])
        oracle_true_order = context_rank(t_oracle, base["m5"])
        oracle_all_order = context_rank(
            oracle_all["t_ulm"] if len(oracle_all) else pd.Series(dtype=float),
            base["m5"],
        )
        rank_rule = rank_of(rule_order, true)
        rank_oracle_true = rank_of(oracle_true_order, true)
        rank_oracle_all = rank_of(oracle_all_order, true)

        # Touch exclusive/cascade once so the fixed pipeline's tools are not import-only.
        if exp_id == bundle["obs_sub"].index[0]:
            contenders = list(base["m5"].sort_values(ascending=False).head(4).index)
            exclusive_target_test(exp_id, contenders, bundle)
            cascade_check(exp_id, contenders, bundle, matched_tracks=matched)

        rows.append({
            "exp_id": exp_id,
            "study_id": study.loc[exp_id],
            "true_tf": true,
            "biosample": obs["Biosample.Name"],
            "tissue": obs["Tissue.Type"],
            "in_recalled": recalled,
            "r_ulm": r_ulm,
            "r_doro": r_doro,
            "r_chea": r_chea,
            "r_m5": r_m5,
            "r_m9": r_m9,
            "r_m10": r_m10,
            "r_m11": r_m11,
            "n_matched_tracks": len(matched),
            "n_scored_tfs": int(len(ctx)),
            "true_has_matched_track": int(true in set(ctx.index)),
            "true_n_any_human_tracks": int(n_tracks_all.get(true, 0)),
            "true_scored_any_cell": int(len(oracle_true) > 0 and true in oracle_true.index),
            "rank_rule": rank_rule,
            "rank_oracle_true": rank_oracle_true,
            "rank_oracle_all": rank_oracle_all,
        })

    df = pd.DataFrame(rows)
    if int(df["in_recalled"].sum()) != 124:
        raise SystemExit(f"recall count is {int(df['in_recalled'].sum())}, expected 124")
    df.to_csv(OUT / "a1_dev266_per_experiment.csv", index=False)

    fail = df[df["in_recalled"] == 0]
    ok = df[df["in_recalled"] == 1]

    def rescue(frame, col):
        hit = frame[col] <= 25
        return int(hit.sum()), float(hit.mean()) if len(frame) else float("nan")

    n_fail = len(fail)
    rule_n, rule_p = rescue(fail, "rank_rule")
    ot_n, ot_p = rescue(fail, "rank_oracle_true")
    oa_n, oa_p = rescue(fail, "rank_oracle_all")
    rule_small = int(((fail["rank_rule"] <= 25) & (fail["n_scored_tfs"] < 25)).sum())
    harm_n = int((ok["rank_rule"] > 25).sum())
    harm_unmatched = int(((ok["rank_rule"] > 25) & (ok["true_has_matched_track"] == 0)).sum())

    ceiling = pd.DataFrame([
        {"item": "n_dev", "n": 266, "pct": 100.0},
        {"item": "n_recall_fail", "n": n_fail, "pct": round(100 * n_fail / 266, 2)},
        {"item": "n_recall_success", "n": len(ok), "pct": round(100 * len(ok) / 266, 2)},
        {"item": "rule_rescue_top25", "n": rule_n, "pct": round(100 * rule_p, 2)},
        {"item": "rule_rescue_but_fewer_than_25_scored", "n": rule_small, "pct": round(100 * rule_small / n_fail, 2)},
        {"item": "oracle_true_any_cell_rescue_top25", "n": ot_n, "pct": round(100 * ot_p, 2)},
        {"item": "oracle_all_tfs_any_cell_rescue_top25", "n": oa_n, "pct": round(100 * oa_p, 2)},
        {"item": "recall_fail_true_tf_matched", "n": int(fail["true_has_matched_track"].sum()), "pct": round(100 * fail["true_has_matched_track"].mean(), 2)},
        {"item": "recall_fail_true_tf_any_human_track", "n": int(fail["true_scored_any_cell"].sum()), "pct": round(100 * fail["true_scored_any_cell"].mean(), 2)},
        {"item": "harm_pushed_out_of_top25", "n": harm_n, "pct": round(100 * harm_n / len(ok), 2)},
        {"item": "harm_and_true_tf_unmatched", "n": harm_unmatched, "pct": round(100 * harm_unmatched / len(ok), 2)},
        {"item": "dev_with_any_matched_track", "n": int((df["n_matched_tracks"] > 0).sum()), "pct": round(100 * (df["n_matched_tracks"] > 0).mean(), 2)},
        {"item": "ulm_vs_decoupler_max_abs_diff", "n": ulm_gap, "pct": ulm_gap},
    ])
    ceiling.to_csv(OUT / "a1_dev266_ceiling_summary.csv", index=False)

    def compare(name, r_new, r_base):
        dlog = np.log2(r_base.astype(float)) - np.log2(r_new.astype(float))
        studies = df["study_id"].to_numpy()
        mean_c, p_c = study_clustered_sign_flip_perm(
            -np.log2(r_new.astype(float)), -np.log2(r_base.astype(float)), studies
        )
        w = wilcoxon_both_modes(r_new.astype(float), r_base.astype(float))
        top1_w, top1_l, top1_p = exact_mcnemar_2s((r_new == 1).astype(int), (r_base == 1).astype(int))
        top5_w, top5_l, top5_p = exact_mcnemar_2s((r_new <= 5).astype(int), (r_base <= 5).astype(int))
        _, p_top1 = study_clustered_sign_flip_perm((r_new == 1).astype(float), (r_base == 1).astype(float), studies)
        _, p_top5 = study_clustered_sign_flip_perm((r_new <= 5).astype(float), (r_base <= 5).astype(float), studies)
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
            "cluster_p_top1": p_top1,
            "cluster_p_top5": p_top5,
        }

    comp = pd.DataFrame([
        compare("M11_vs_M5", df["r_m11"].to_numpy(), df["r_m5"].to_numpy()),
        compare("M11_vs_M10", df["r_m11"].to_numpy(), df["r_m10"].to_numpy()),
        compare("M10_vs_M5", df["r_m10"].to_numpy(), df["r_m5"].to_numpy()),
    ])
    comp.to_csv(OUT / "a1_dev266_m11_comparison.csv", index=False)

    lines = [
        "A1 Dev 266 context-regulon ceiling. Oracle rows are not a method.",
        f"ULM vs decoupler max abs diff: {ulm_gap:.3e}",
        f"Recall fail {n_fail}/266. Rule rescue {rule_n}/{n_fail} ({100 * rule_p:.2f}%), of which {rule_small} had <25 scored TFs.",
        f"Oracle true-TF any cell line: {ot_n}/{n_fail} ({100 * ot_p:.2f}%).",
        f"Oracle all TFs any cell line: {oa_n}/{n_fail} ({100 * oa_p:.2f}%).",
        f"Harm among {len(ok)} recalled: {harm_n} ({100 * harm_n / len(ok):.2f}%) pushed out of Top-25; {harm_unmatched} of those had no matched track.",
        f"M11 top1 {int((df.r_m11 == 1).sum())}/266 top5 {int((df.r_m11 <= 5).sum())}/266; M5 {int((df.r_m5 == 1).sum())}/{int((df.r_m5 <= 5).sum())}; M10 {int((df.r_m10 == 1).sum())}/{int((df.r_m10 <= 5).sum())}.",
        "Go/no-go threshold for rule rescue is 15%. This file does not make the go/no-go call.",
    ]
    (OUT / "a1_dev266_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(parse_tbl.to_string(index=False))
    print(comp.to_string(index=False))
    assert_frozen()


if __name__ == "__main__":
    main()
