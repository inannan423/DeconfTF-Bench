"""M10_Soft_Deconf: coverage-scaled soft gate (R5 development + R6 re-freeze).

R6 re-freeze (2026-11-03), BEFORE any GEO expression download/scoring:
- Vacates the R5 freeze of this file (`2d031d746d751ed84a902144bdb1b5fe11274161646fe92ccda945fc9edfd9bc`).
- Reason: the lambda_bind grid was expanded on Dev 266 only (`{0.25,0.5,0.75,1.0,1.5,2.0,3.0,5.0}`
  x `{0.15,0.30,0.50}`, plus lambda=100 hard-limit control) prior to GEO scoring.
- 170-fold LOSO-CV on Dev 266 selected `(lambda_bind, lambda_hub) = (1.0, 0.3)` in 168/170 folds
  (1 fold `(1.5, 0.3)`, 1 fold `(1.0, 0.15)`). Full-data optimum is the same interior point.
- Fold votes: `external/r6_m10_dev266_loso_fold_votes.csv`.
- Grid + LOSO summary: `external/r6_m10_dev266_grid_with_loso.csv`.
- Unseen 64 and Replogle were not used for selection (post-development descriptive only).

Do not overwrite the R5 CSVs. Formula `m10_soft_order` is unchanged; only the locked defaults
and freeze note are updated.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/bytedance/codes/new-bishe/artifacts/src")
from deconftf_tools import (
    ENHANCEOSOME_COMPLEX,
    UNIVERSAL_SECONDARY_HUBS,
    load_gmt_with_tracks,
    m9_order,
    score_knocktf_experiments,
)
from eval_p0_2_external import load_replogle_cohort
from verify_round4_accept_checks import exact_mcnemar_2s, study_clustered_sign_flip_perm, wilcoxon_both_modes

ROOT = Path("/Users/bytedance/codes/new-bishe")
DATA_DIR = ROOT / "artifacts" / "data" / "knocktf"
EXT_DATA_DIR = ROOT / "artifacts" / "data" / "external"
OUT_DIR = ROOT / "artifacts" / "results" / "cand6_deconftf"
EXT_OUT_DIR = OUT_DIR / "external"


def compute_bundle_coverage(bundle: Dict[str, Any]) -> float:
    net_c = bundle["net_c"]
    cand_set = set(bundle["cand_tfs"])
    gset = set(str(g).upper() for g in bundle["mat_blind"].columns)
    sub = net_c[net_c["source"].isin(cand_set)]
    fracs = []
    for _, grp in sub.groupby("source"):
        tgts = set(grp["target"].astype(str).str.upper())
        fracs.append(len(tgts & gset) / max(1, len(tgts)))
    return float(np.mean(fracs))


def m10_soft_order(
    exp_id: str,
    bundle: Dict[str, Any],
    cov_frac: float | None = None,
    ch_thr: float = 0.5,
    cc_thr: float = 3.0,
    lambda_bind: float = 1.0,
    lambda_hub: float = 0.3,
    gamma_cov: float = 1.0,
) -> Tuple[List[str], pd.Series]:
    """Coverage-modulated continuous soft conditional re-ranking (`M10_Soft_Deconf`)."""
    if cov_frac is None:
        cov_frac = bundle.get("_cov_frac")
        if cov_frac is None:
            cov_frac = compute_bundle_coverage(bundle)
            bundle["_cov_frac"] = cov_frac

    z = bundle["z_libs"]
    ulm = bundle["est_ulm"].loc[exp_id]
    mm = bundle["multi_modal_score"].loc[exp_id]
    cc = -bundle["est_c_res"].loc[exp_id]
    chip = pd.concat(
        [z["ENCODE_ChIP-seq"].loc[exp_id], z["ReMap_ChIP-seq"].loc[exp_id], z["Literature_ChIP-seq"].loc[exp_id]],
        axis=1,
    ).max(axis=1)
    coexp = pd.concat(
        [z["ARCHS4_Coexpression"].loc[exp_id], z["GTEx_Coexpression"].loc[exp_id]], axis=1
    ).max(axis=1)
    doro = z["DoRothEA_Signed"].loc[exp_id]

    u_order = list(ulm.sort_values(ascending=True).index)
    t1 = u_order[0]
    trig = (
        (t1 in UNIVERSAL_SECONDARY_HUBS and cc[t1] < cc_thr)
        or (chip[t1] < ch_thr)
        or (doro[t1] < 0.8 and chip[t1] < 1.5 and coexp[t1] > 1.2)
    )

    def_bind = np.maximum(0.0, ch_thr - chip) * np.maximum(0.0, 1.0 - doro)
    is_hub = pd.Series([1.0 if c in UNIVERSAL_SECONDARY_HUBS else 0.0 for c in mm.index], index=mm.index)
    def_hub = is_hub * np.maximum(0.0, cc_thr - np.maximum(0.0, cc))

    scale = float(cov_frac ** gamma_cov)
    s_soft = mm - scale * (lambda_bind * def_bind + lambda_hub * def_hub)
    soft_order = list(s_soft.sort_values(ascending=False).index)

    if not trig:
        order = [t1] + [c for c in soft_order if c != t1]
        s_out = s_soft.copy()
        s_out[t1] = float(s_soft.max() + 1.0)
        return order, s_out
    return soft_order, s_soft


def main():
    dev_b = pickle.loads((OUT_DIR / "knocktf_precomputed_bundle.pkl").read_bytes())
    cand_tfs = dev_b["cand_tfs"]
    ranks401 = pd.read_csv(OUT_DIR / "knocktf401_ranks_with_study_ids.csv").set_index("exp_id")
    h135 = pd.read_csv(OUT_DIR / "holdout135_harmonised_ranks.csv").set_index("exp_id")
    for col in ["r_m1_ulm", "r_m2b_viper_shadow", "r_m5_mm_static", "r_m9_determ"]:
        ranks401.loc[h135.index, col] = h135[col].values

    dev_ids = list(ranks401.index[ranks401["split_new"] == "Dev"])
    dev_studies = ranks401.loc[dev_ids, "study_id"].astype(str).values
    dev_tfs = ranks401.loc[dev_ids, "true_tf"].values
    cov_dev = compute_bundle_coverage(dev_b)
    print(f"Dev 266 mean CollecTRI target coverage fraction: {cov_dev:.4f}")

    # Expanded candidate grid on Dev 266 ONLY: lambda_bind in {0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0} x lambda_hub in {0.15, 0.30, 0.50}
    lb_vals = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 5.00]
    lh_vals = [0.15, 0.30, 0.50]
    grid = [(lb, lh) for lb in lb_vals for lh in lh_vals]
    # Also evaluate asymptotic hard-penalty limit control (lb=100.0, lh=100.0)
    all_eval_points = grid + [(100.0, 100.0)]

    grid_ranks: Dict[Tuple[float, float], np.ndarray] = {}
    grid_table_rows = []
    r9_dev = ranks401.loc[dev_ids, "r_m9_determ"].values
    r5_dev = ranks401.loc[dev_ids, "r_m5_mm_static"].values

    for lb, lh in all_eval_points:
        r_vec = []
        for eid, tf in zip(dev_ids, dev_tfs):
            ord_10, _ = m10_soft_order(eid, dev_b, cov_frac=cov_dev, lambda_bind=lb, lambda_hub=lh)
            r_vec.append(ord_10.index(tf) + 1)
        r_arr = np.array(r_vec, dtype=int)
        grid_ranks[(lb, lh)] = r_arr
        obj = (
            float(np.mean(-np.log2(r_arr)))
            + 0.5 * float(np.mean(r_arr <= 5))
            + 0.5 * float(np.mean(r_arr == 1))
        )
        grid_table_rows.append({
            "lambda_bind": lb,
            "lambda_hub": lh,
            "is_limit_control": int(lb == 100.0),
            "dev_top1_hits": int((r_arr == 1).sum()),
            "dev_top1_pct": round(float(np.mean(r_arr == 1) * 100), 2),
            "dev_top5_hits": int((r_arr <= 5).sum()),
            "dev_top5_pct": round(float(np.mean(r_arr <= 5) * 100), 2),
            "dev_mrr": round(float(np.mean(1.0 / r_arr)), 4),
            "dev_median_rank": float(np.median(r_arr)),
            "dev_mean_log2_rank": round(float(np.mean(np.log2(r_arr))), 4),
            "dev_cv_objective": round(obj, 5),
            "mean_dlog2_vs_m5": round(float(np.mean(np.log2(r5_dev) - np.log2(r_arr))), 4),
            "mean_dlog2_vs_m9": round(float(np.mean(np.log2(r9_dev) - np.log2(r_arr))), 4),
        })

    df_grid = pd.DataFrame(grid_table_rows)
    df_grid.to_csv(EXT_OUT_DIR / "r5_m10_dev266_grid_search.csv", index=False)
    print("\n=== Full Expanded Grid Search on Dev 266 (plus Hard-Limit Control lb=100, lh=100) ===")
    print(df_grid.to_string(index=False))

    # 170-fold Leave-One-Study-Out CV over the expanded grid
    loso_ranks = np.zeros(len(dev_ids), dtype=int)
    chosen_params = []
    for sid in np.unique(dev_studies):
        val_mask = dev_studies == sid
        tr_mask = ~val_mask
        best_p = max(
            grid,
            key=lambda p: (
                float(np.mean(-np.log2(grid_ranks[p][tr_mask])))
                + 0.5 * float(np.mean(grid_ranks[p][tr_mask] <= 5))
                + 0.5 * float(np.mean(grid_ranks[p][tr_mask] == 1))
            ),
        )
        chosen_params.append(best_p)
        loso_ranks[val_mask] = grid_ranks[best_p][val_mask]

    vc_p = pd.Series(chosen_params).value_counts()
    print("\n=== 170-Fold LOSO Parameter Selection Distribution on Expanded Grid ===")
    for p_pair, cnt in vc_p.items():
        print(f"  (lambda_bind={p_pair[0]:.2f}, lambda_hub={p_pair[1]:.2f}): {cnt}/170 folds")
    locked_lb, locked_lh = vc_p.index[0]
    print(f"Locked M10 parameters from 170-fold LOSO: lambda_bind={locked_lb}, lambda_hub={locked_lh}")
    print(
        f"M10 LOSO-CV on Dev 266: Top1={int((loso_ranks==1).sum())}/266 ({np.mean(loso_ranks==1)*100:.2f}%), "
        f"Top5={int((loso_ranks<=5).sum())}/266 ({np.mean(loso_ranks<=5)*100:.2f}%), "
        f"MedRank={np.median(loso_ranks):.1f}, MeanLog2Rank={np.mean(np.log2(loso_ranks)):.3f}"
    )

    # Evaluate locked M10 across Dev 266 and post-dev descriptive cohorts
    mat_ktf = pd.read_csv(DATA_DIR / "knockTF_expr.csv", index_col=0)
    obs_ktf = pd.read_csv(DATA_DIR / "knockTF_meta.csv", index_col=0)
    hold_ids = list(ranks401.index[ranks401["split_new"] != "Dev"])
    unseen_ids = list(ranks401.index[ranks401["split_new"] == "Unseen"])
    hold_b = score_knocktf_experiments(mat_ktf.loc[hold_ids], obs_ktf.loc[hold_ids], cand_tfs=cand_tfs)

    a_k562 = ad.read_h5ad(EXT_DATA_DIR / "K562_gwps_normalized_bulk_01.h5ad")
    k562_genes = set(a_k562.var["gene_name"].astype(str).str.upper())
    shared_cols = [c for c in mat_ktf.columns if str(c).upper() in k562_genes]
    b_8k = score_knocktf_experiments(mat_ktf.loc[ranks401.index, shared_cols], obs_ktf.loc[ranks401.index], cand_tfs=cand_tfs)

    collectri_df = pd.read_csv(DATA_DIR / "collectri_human.csv")
    dorothea_df = pd.read_csv(DATA_DIR / "dorothea_human.csv")
    gmt_names = [
        "ARCHS4_Coexpression", "GTEx_Coexpression", "ENCODE_ChIP-seq",
        "ReMap_ChIP-seq", "Literature_ChIP-seq", "Enrichr_Queries",
    ]
    gmt_srcs = set().union(*(set(load_gmt_with_tracks(DATA_DIR / f"{gn}.gmt")[0]["source"]) for gn in gmt_names))
    masked_tfs = set(collectri_df["source"].astype(str).str.upper()) | set(dorothea_df["source"].astype(str).str.upper()) | gmt_srcs
    tf_to_class = obs_ktf.groupby("TF")["TF.Class"].first().to_dict()

    mat_k562, obs_k562 = load_replogle_cohort(
        EXT_DATA_DIR / "K562_gwps_normalized_bulk_01.h5ad", cand_tfs, masked_tfs, tf_to_class, "K562", "Haematopoietic", mode="normalized_z"
    )
    b_k562 = score_knocktf_experiments(mat_k562, obs_k562, cand_tfs=cand_tfs)
    k562_prim_ids = list(b_k562["obs_sub"].index[b_k562["obs_sub"]["max_abs_blind"] >= 0.5])

    mat_rpe1, obs_rpe1 = load_replogle_cohort(
        EXT_DATA_DIR / "rpe1_normalized_bulk_01.h5ad", cand_tfs, masked_tfs, tf_to_class, "RPE1", "RPE", mode="normalized_z"
    )
    b_rpe1 = score_knocktf_experiments(mat_rpe1, obs_rpe1, cand_tfs=cand_tfs)
    rpe1_prim_ids = list(b_rpe1["obs_sub"].index[b_rpe1["obs_sub"]["max_abs_blind"] >= 0.5])

    eval_cohorts = [
        ("KnockTF_Dev_266_Full22k", "dev_training_cohort", dev_b, dev_ids, dev_studies),
        ("KnockTF_Unseen_64_Full22k", "postdev_descriptive_seen_split", hold_b, unseen_ids, ranks401.loc[unseen_ids, "study_id"].astype(str).values),
        ("KnockTF_Holdout_135_Full22k", "postdev_descriptive_seen_split", hold_b, hold_ids, ranks401.loc[hold_ids, "study_id"].astype(str).values),
        ("KnockTF_Unseen_64_Restricted8k", "postdev_descriptive_seen_split", b_8k, unseen_ids, ranks401.loc[unseen_ids, "study_id"].astype(str).values),
        ("Replogle_K562_Primary_267", "postdev_descriptive_already_inspected", b_k562, k562_prim_ids, b_k562["obs_sub"].loc[k562_prim_ids, "TF.Class"].astype(str).values),
        ("Replogle_K562_All_KD0.5_514", "postdev_descriptive_already_inspected", b_k562, list(b_k562["obs_sub"].index), b_k562["obs_sub"]["TF.Class"].astype(str).values),
        ("Replogle_RPE1_Primary_76", "postdev_descriptive_already_inspected", b_rpe1, rpe1_prim_ids, b_rpe1["obs_sub"].loc[rpe1_prim_ids, "TF.Class"].astype(str).values),
    ]

    summary_rows = []
    for cname, role, b, eids, cids in eval_cohorts:
        cov_f = compute_bundle_coverage(b)
        r_u, r_5, r_9, r_10 = [], [], [], []
        for eid in eids:
            tf = b["obs_sub"].loc[eid, "TF"]
            ord_u = list(b["est_ulm"].loc[eid].sort_values(ascending=True).index)
            ord_5 = list(b["multi_modal_score"].loc[eid].sort_values(ascending=False).index)
            ord_9 = m9_order(eid, b)
            ord_10, _ = m10_soft_order(eid, b, cov_frac=cov_f, lambda_bind=locked_lb, lambda_hub=locked_lh)
            r_u.append(ord_u.index(tf) + 1)
            r_5.append(ord_5.index(tf) + 1)
            r_9.append(ord_9.index(tf) + 1)
            r_10.append(ord_10.index(tf) + 1)
        ru = np.array(r_u)
        r5 = np.array(r_5)
        r9 = np.array(r_9)
        r10 = np.array(r_10)

        for comp_name, ra, rb in [
            ("M10_Soft_vs_M9_Hard", r10, r9),
            ("M10_Soft_vs_M5_Static", r10, r5),
            ("M10_Soft_vs_M1_ULM", r10, ru),
            ("M9_Hard_vs_M5_Static", r9, r5),
        ]:
            w1, l1, p1 = exact_mcnemar_2s((ra == 1).astype(int), (rb == 1).astype(int))
            w5, l5, p5 = exact_mcnemar_2s((ra <= 5).astype(int), (rb <= 5).astype(int))
            wx = wilcoxon_both_modes(ra, rb)
            _, pc = study_clustered_sign_flip_perm(-np.log2(ra.astype(float)), -np.log2(rb.astype(float)), cids)
            summary_rows.append({
                "cohort": cname,
                "cohort_role": role,
                "n_samples": len(eids),
                "cov_frac": round(cov_f, 3),
                "locked_lambda_bind": locked_lb,
                "locked_lambda_hub": locked_lh,
                "comparison": comp_name,
                "top1_a_vs_b": f"{int((ra==1).sum())}/{int((rb==1).sum())} ({np.mean(ra==1)*100:.1f}% vs {np.mean(rb==1)*100:.1f}%)",
                "top1_wl": f"{w1}/{l1}",
                "top5_a_vs_b": f"{int((ra<=5).sum())}/{int((rb<=5).sum())} ({np.mean(ra<=5)*100:.1f}% vs {np.mean(rb<=5)*100:.1f}%)",
                "top5_wl": f"{w5}/{l5}",
                "med_rank_a_vs_b": f"{np.median(ra):.1f} vs {np.median(rb):.1f}",
                "mean_dlog2rank_bits": round(wx["mean_bits"], 4),
                "median_dlog2rank_bits": round(wx["median_bits"], 4),
                "p_wilcox": wx["p_wilcox"],
                "p_pratt": wx["p_pratt"],
                "p_clustered": pc,
            })

    df_m10 = pd.DataFrame(summary_rows)
    df_m10.to_csv(EXT_OUT_DIR / "r5_m10_soft_gate_evaluation.csv", index=False)
    print("\n=== R5: M10_Soft_Deconf Evaluation Summary (with Expanded Grid Selection) ===")
    print(df_m10.to_string(index=False))


if __name__ == "__main__":
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 25)
    main()
