"""R3 (= P2-a): Systematic 3-Dimensional Benchmark Audit across KnockTF (Dev 266, Unseen 64)
and Replogle 2022 Perturb-seq (K562 Primary 267, RPE1 Primary 76).

Key Audit Guarantees:
1. Exact reproduction of `holdout135_harmonised_ranks.csv` and `knocktf401_ranks_with_study_ids.csv`
   under `3_StrictBlinded_1888TFs` (0 rank differences on Dev 266 and Unseen 64):
   - `est_d` is NOT pre-filled with 0.0 before `z_libs` row-standardisation (matching `deconftf_tools.py` line 224).
   - For KnockTF Unseen 64 under `3_StrictBlinded_1888TFs`, scores are computed on the 135-holdout matrix
     and sliced to Unseen 64 so `decoupler(..., empty=True)` drops the exact same all-zero gene columns
     as `rescore_holdout_harmonised.py`; we also explicitly quantify the `empty=True` vs `empty=False`
     batch-size effect in `r3_decoupler_empty_col_drop_audit.csv`.

Outputs under `artifacts/results/cand6_deconftf/audit/`:
- `r3_method_performance_by_blinding_and_metric.csv`
- `r3_ranking_inversion_and_significance_shift.csv`
- `r3_viper_native_vs_shadow_comparison.csv`
- `r3_decoupler_empty_col_drop_audit.csv`
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import decoupler as dc
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, "/Users/bytedance/codes/new-bishe/artifacts/src")
from audit_reviewer_controls import compute_viper_shadow_scores
from deconftf_tools import (
    ENHANCEOSOME_COMPLEX,
    UNIVERSAL_SECONDARY_HUBS,
    fast_unsigned_enrichment,
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
AUDIT_DIR = OUT_DIR / "audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def compute_global_auroc_auprc(
    score_df: pd.DataFrame,
    true_tfs: List[str],
    higher_is_better: bool = True,
    exact_ranks: np.ndarray | None = None,
) -> Tuple[float, float, float, np.ndarray]:
    """Compute:
    1) Global pooled AUROC across all (experiment, candidate_TF) pairs using row-standardized scores;
    2) Global pooled AUPRC;
    3) Rank-based AUROC;
    4) 1-indexed per-experiment ranks of `true_tfs`.
    """
    S = score_df.values.astype(np.float64)
    if not higher_is_better:
        S = -S
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    mu = S.mean(axis=1, keepdims=True)
    sd = S.std(axis=1, keepdims=True) + 1e-8
    Z = (S - mu) / sd

    cols = list(score_df.columns)
    col_to_idx = {c: j for j, c in enumerate(cols)}
    Y_true = np.zeros_like(Z, dtype=int)
    n_cands = len(cols)

    if exact_ranks is None:
        ranks = []
        for i, tf in enumerate(true_tfs):
            j = col_to_idx[tf]
            Y_true[i, j] = 1
            # Use exact pandas sort_values order to match deconftf_tools / rescore_holdout_harmonised
            ord_cols = list(score_df.iloc[i].sort_values(ascending=not higher_is_better).index)
            ranks.append(ord_cols.index(tf) + 1)
        ranks_arr = np.array(ranks, dtype=int)
    else:
        for i, tf in enumerate(true_tfs):
            Y_true[i, col_to_idx[tf]] = 1
        ranks_arr = exact_ranks.astype(int)

    auroc_pooled = float(roc_auc_score(Y_true.ravel(), Z.ravel()))
    auprc_pooled = float(average_precision_score(Y_true.ravel(), Z.ravel()))
    auroc_rank = float(np.mean(1.0 - (ranks_arr - 1.0) / (n_cands - 1.0)))
    return auroc_pooled, auprc_pooled, auroc_rank, ranks_arr


def score_all_methods_under_regime(
    mat_raw: pd.DataFrame,
    obs_df: pd.DataFrame,
    cand_tfs: List[str],
    blinding_regime: str,
    collectri: pd.DataFrame,
    dorothea: pd.DataFrame,
    gmt_nets_no_self: Dict[str, pd.DataFrame],
    gmt_nets_with_self: Dict[str, pd.DataFrame],
    all_masked_tfs: set[str],
) -> Dict[str, pd.DataFrame]:
    if blinding_regime == "1_Unblinded_WithSelfLoops":
        mat_in = mat_raw.copy()
        net_c = collectri.copy()
        net_d = dorothea.copy()
        gmt_nets = gmt_nets_with_self
    elif blinding_regime == "2_Unblinded_NoSelfLoops":
        mat_in = mat_raw.copy()
        net_c = collectri[collectri["source"] != collectri["target"]].copy()
        net_d = dorothea[dorothea["source"] != dorothea["target"]].copy()
        gmt_nets = gmt_nets_no_self
    elif blinding_regime == "3_StrictBlinded_1888TFs":
        mat_in = mat_raw.copy()
        tf_cols = [g for g in all_masked_tfs if g in mat_in.columns]
        mat_in[tf_cols] = 0.0
        net_c = collectri[collectri["source"] != collectri["target"]].copy()
        net_d = dorothea[dorothea["source"] != dorothea["target"]].copy()
        gmt_nets = gmt_nets_no_self
    else:
        raise ValueError(blinding_regime)

    obs_sub = obs_df[obs_df["TF"].isin(cand_tfs)].copy()
    idx = obs_sub.index
    est_ulm, _ = dc.mt.ulm(data=mat_in, net=net_c)
    mat_in = mat_in.loc[idx].copy()
    est_ulm = est_ulm.reindex(index=idx, columns=cand_tfs, fill_value=0.0)

    est_mlm, _ = dc.mt.mlm(data=mat_in, net=net_c)
    est_mlm = est_mlm.reindex(index=idx, columns=cand_tfs, fill_value=0.0)

    est_wsum, _ = dc.mt.waggr(data=mat_in, net=net_c, fun="wsum", times=0)
    est_wsum = est_wsum.reindex(index=idx, columns=cand_tfs, fill_value=0.0)

    est_wmean, _ = dc.mt.waggr(data=mat_in, net=net_c, fun="wmean", times=0)
    est_wmean = est_wmean.reindex(index=idx, columns=cand_tfs, fill_value=0.0)

    est_zscore, _ = dc.mt.zscore(data=mat_in, net=net_c)
    est_zscore = est_zscore.reindex(index=idx, columns=cand_tfs, fill_value=0.0)

    est_viper_native, _ = dc.mt.viper(data=mat_in, net=net_c)
    est_viper_native = est_viper_native.reindex(index=idx, columns=cand_tfs, fill_value=0.0)

    deg_c = net_c.groupby("target")["source"].transform("count")
    net_c_idf = net_c.copy()
    net_c_idf["weight"] = net_c_idf["weight"] / np.sqrt(deg_c)
    est_c_idf, _ = dc.mt.ulm(data=mat_in, net=net_c_idf)
    est_c_idf = est_c_idf.reindex(index=idx, columns=cand_tfs, fill_value=0.0)

    # IMPORTANT: Do NOT fillna(0.0) on est_d before computing z_libs (matches deconftf_tools.py line 224)
    est_d, _ = dc.mt.ulm(data=mat_in, net=net_d)
    est_d = est_d.reindex(index=idx, columns=cand_tfs)

    gene_to_idx = {str(g).upper(): i for i, g in enumerate(mat_in.columns)}
    tf_to_idx = {t: j for j, t in enumerate(cand_tfs)}
    W_c = np.zeros((len(mat_in.columns), len(cand_tfs)), dtype=np.float32)
    for src, tgt, w in zip(net_c_idf["source"], net_c_idf["target"], net_c_idf["weight"]):
        t_u = str(tgt).upper()
        if t_u in gene_to_idx and src in tf_to_idx:
            W_c[gene_to_idx[t_u], tf_to_idx[src]] = float(w)

    viper_shadow = compute_viper_shadow_scores(mat_in, est_ulm, W_c, gene_to_idx, tf_to_idx, cand_tfs)

    Y = mat_in.values.astype(np.float32)
    Y_abs = np.abs(Y)
    Y_abs_z = (Y_abs - Y_abs.mean(axis=1, keepdims=True)) / (Y_abs.std(axis=1, keepdims=True) + 1e-8)

    hub_list = ["E2F4", "E2F1", "FOXM1", "MYC", "ATF4"]
    W_hubs = np.zeros((len(mat_in.columns), len(hub_list)), dtype=np.float32)
    for j, h_tf in enumerate(hub_list):
        sub_h = net_c[net_c["source"] == h_tf]
        for _, r_h in sub_h.iterrows():
            t_u = str(r_h["target"]).upper()
            if t_u in gene_to_idx:
                W_hubs[gene_to_idx[t_u], j] = float(r_h["weight"])
    Q_hubs, _ = np.linalg.qr(W_hubs)
    Y_res = Y - (Y @ Q_hubs) @ Q_hubs.T
    df_mat_res = pd.DataFrame(Y_res, index=idx, columns=mat_in.columns)
    est_c_res, _ = dc.mt.ulm(data=df_mat_res, net=net_c_idf)
    est_c_res = est_c_res.reindex(index=idx, columns=cand_tfs, fill_value=0.0)

    gmt_names = list(gmt_nets.keys())
    lib_scores: Dict[str, pd.DataFrame] = {
        "CollecTRI_Signed": -est_c_idf,
        "CollecTRI_Abs": est_c_idf.abs(),
        "CollecTRI_Unsigned": pd.DataFrame(
            fast_unsigned_enrichment(Y_abs_z, gene_to_idx, net_c, cand_tfs), index=idx, columns=cand_tfs
        ),
        "DoRothEA_Signed": -est_d,
        "DoRothEA_Abs": est_d.abs(),
    }
    for gn in gmt_names:
        s_mat = fast_unsigned_enrichment(Y_abs_z, gene_to_idx, gmt_nets[gn], cand_tfs)
        lib_scores[gn] = pd.DataFrame(s_mat, index=idx, columns=cand_tfs)

    stack_chea6 = np.stack(
        [
            lib_scores[k]
            .rank(axis=1, ascending=False, method="min", na_option="keep")
            .div(lib_scores[k].notna().sum(axis=1), axis=0)
            .values
            for k in gmt_names
        ],
        axis=0,
    )
    chea3_orig_mean = pd.DataFrame(-np.nanmean(stack_chea6, axis=0), index=idx, columns=cand_tfs)
    chea3_orig_top = pd.DataFrame(-np.nanmin(stack_chea6, axis=0), index=idx, columns=cand_tfs)

    z_libs = {}
    for lname, sdf in lib_scores.items():
        mu = sdf.mean(axis=1).values[:, None]
        sd = sdf.std(axis=1).values[:, None] + 1e-8
        z_libs[lname] = ((sdf - mu) / sd).fillna(0.0)

    weights = {
        "CollecTRI_Signed": 1.5,
        "CollecTRI_Abs": 0.8,
        "CollecTRI_Unsigned": 1.0,
        "DoRothEA_Signed": 1.2,
        "DoRothEA_Abs": 0.6,
        "ARCHS4_Coexpression": 1.2,
        "GTEx_Coexpression": 0.8,
        "ENCODE_ChIP-seq": 1.0,
        "ReMap_ChIP-seq": 1.2,
        "Literature_ChIP-seq": 1.2,
        "Enrichr_Queries": 0.8,
    }
    mm_score = sum(w * z_libs[k] for k, w in weights.items()) / sum(weights.values())

    temp_bundle = {
        "z_libs": z_libs,
        "est_ulm": est_ulm,
        "multi_modal_score": mm_score,
        "est_c_res": est_c_res,
    }
    m9_scores = np.zeros((len(idx), len(cand_tfs)), dtype=float)
    for i_r, eid in enumerate(idx):
        ord_m9 = m9_order(eid, temp_bundle)
        for rk_0, c in enumerate(ord_m9):
            m9_scores[i_r, tf_to_idx[c]] = float(len(cand_tfs) - rk_0)
    df_m9_score = pd.DataFrame(m9_scores, index=idx, columns=cand_tfs)

    return {
        "M1_ULM": -est_ulm,
        "M2_MLM": -est_mlm,
        "M_WSUM": -est_wsum,
        "M_WMEAN": -est_wmean,
        "M_ZSCORE": -est_zscore,
        "M_VIPER_Native": -est_viper_native,
        "M2b_VIPER_Shadow": viper_shadow,
        "M3_ChEA3_Orig6_MeanRank": chea3_orig_mean,
        "M4_ChEA3_Orig6_TopRank": chea3_orig_top,
        "M5_MultiModal_Static": mm_score,
        "M9_Selective_Deconf": df_m9_score,
    }


def load_gmt_keep_self(gmt_path: Path) -> pd.DataFrame:
    edges = []
    for line in gmt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = [p.strip() for p in line.split("\t") if p.strip()]
        if len(parts) < 3:
            continue
        tf = parts[0].split("_")[0].upper()
        for g in parts[2:]:
            edges.append((tf, g.upper()))
    df = pd.DataFrame(edges, columns=["source", "target"]).drop_duplicates()
    df["weight"] = 1.0
    return df


def audit_decoupler_empty_col_drop(mat_ktf, obs_ktf, ranks401, h135, cand_tfs):
    """Audit the effect of `decoupler.mt.ulm(..., empty=True)` when scoring Holdout 135 vs Unseen 64 vs single-sample."""
    hold_ids = list(ranks401.index[ranks401["split_new"] != "Dev"])
    unseen_ids = list(ranks401.index[ranks401["split_new"] == "Unseen"])
    b_135 = score_knocktf_experiments(mat_ktf.loc[hold_ids], obs_ktf.loc[hold_ids], cand_tfs=cand_tfs)
    b_64 = score_knocktf_experiments(mat_ktf.loc[unseen_ids], obs_ktf.loc[unseen_ids], cand_tfs=cand_tfs)

    rows = []
    for eid in unseen_ids:
        tf = obs_ktf.loc[eid, "TF"]
        r_u_135 = list(b_135["est_ulm"].loc[eid].sort_values(ascending=True).index).index(tf) + 1
        r_u_64 = list(b_64["est_ulm"].loc[eid].sort_values(ascending=True).index).index(tf) + 1
        r_m5_135 = list(b_135["multi_modal_score"].loc[eid].sort_values(ascending=False).index).index(tf) + 1
        r_m5_64 = list(b_64["multi_modal_score"].loc[eid].sort_values(ascending=False).index).index(tf) + 1
        r_m9_135 = m9_order(eid, b_135).index(tf) + 1
        r_m9_64 = m9_order(eid, b_64).index(tf) + 1
        if (r_u_135 != r_u_64) or (r_m5_135 != r_m5_64) or (r_m9_135 != r_m9_64):
            rows.append({
                "exp_id": eid,
                "true_tf": tf,
                "r_ulm_batch135": r_u_135,
                "r_ulm_batch64": r_u_64,
                "r_m5_batch135": r_m5_135,
                "r_m5_batch64": r_m5_64,
                "r_m9_batch135": r_m9_135,
                "r_m9_batch64": r_m9_64,
                "n_nonzero_genes_batch135": int((b_135["mat_blind"] != 0).any(axis=0).sum()),
                "n_nonzero_genes_batch64": int((b_64["mat_blind"] != 0).any(axis=0).sum()),
            })
    df_empty = pd.DataFrame(rows)
    df_empty.to_csv(AUDIT_DIR / "r3_decoupler_empty_col_drop_audit.csv", index=False)
    print("\n=== Audit of decoupler(empty=True) Batch-Size Effect (Holdout 135 vs Unseen 64) ===")
    print(df_empty.to_string(index=False))


def main():
    t_start = time.time()
    dev_b = pickle.loads((OUT_DIR / "knocktf_precomputed_bundle.pkl").read_bytes())
    cand_tfs = dev_b["cand_tfs"]

    collectri = pd.read_csv(DATA_DIR / "collectri_human.csv")
    dorothea = pd.read_csv(DATA_DIR / "dorothea_human.csv")
    gmt_names = [
        "ARCHS4_Coexpression", "GTEx_Coexpression", "ENCODE_ChIP-seq",
        "ReMap_ChIP-seq", "Literature_ChIP-seq", "Enrichr_Queries",
    ]
    gmt_no_self = {gn: load_gmt_with_tracks(DATA_DIR / f"{gn}.gmt")[0] for gn in gmt_names}
    gmt_with_self = {gn: load_gmt_keep_self(DATA_DIR / f"{gn}.gmt") for gn in gmt_names}
    all_masked_tfs = sorted(
        list(
            set(collectri[collectri["source"] != collectri["target"]]["source"])
            | set(dorothea[dorothea["source"] != dorothea["target"]]["source"])
            | set().union(*(set(g["source"]) for g in gmt_no_self.values()))
        )
    )

    mat_ktf = pd.read_csv(DATA_DIR / "knockTF_expr.csv", index_col=0)
    obs_ktf = pd.read_csv(DATA_DIR / "knockTF_meta.csv", index_col=0)
    ranks401 = pd.read_csv(OUT_DIR / "knocktf401_ranks_with_study_ids.csv").set_index("exp_id")
    h135 = pd.read_csv(OUT_DIR / "holdout135_harmonised_ranks.csv").set_index("exp_id")
    for col in ["r_m1_ulm", "r_m2b_viper_shadow", "r_m5_mm_static", "r_m9_determ"]:
        ranks401.loc[h135.index, col] = h135[col].values

    audit_decoupler_empty_col_drop(mat_ktf, obs_ktf, ranks401, h135, cand_tfs)

    tf_to_class = obs_ktf.groupby("TF")["TF.Class"].first().to_dict()
    mat_k562, obs_k562 = load_replogle_cohort(
        EXT_DATA_DIR / "K562_gwps_normalized_bulk_01.h5ad", cand_tfs, all_masked_tfs, tf_to_class, "K562", "Haematopoietic", mode="normalized_z"
    )
    k562_prim_ids = list(obs_k562.index[obs_k562["max_abs_blind"] >= 0.5])

    mat_rpe1, obs_rpe1 = load_replogle_cohort(
        EXT_DATA_DIR / "rpe1_normalized_bulk_01.h5ad", cand_tfs, all_masked_tfs, tf_to_class, "RPE1", "RPE", mode="normalized_z"
    )
    rpe1_prim_ids = list(obs_rpe1.index[obs_rpe1["max_abs_blind"] >= 0.5])

    dev_ids = list(ranks401.index[ranks401["split_new"] == "Dev"])
    hold_ids = list(ranks401.index[ranks401["split_new"] != "Dev"])
    unseen_ids = list(ranks401.index[ranks401["split_new"] == "Unseen"])
    msk_dev298 = (obs_ktf["logFC"] < -1.0) & (obs_ktf["TF"].isin(set(collectri["source"])))

    datasets = [
        # Score on the exact 298-row msk_dev298 matrix and slice to dev_ids (266) so `empty=True` matches `prepare_knocktf_bundle`
        ("KnockTF_Dev_266", mat_ktf.loc[msk_dev298], obs_ktf.loc[msk_dev298], dev_ids, ranks401.loc[dev_ids, "study_id"].values),
        # Score on Holdout 135 and slice to Unseen 64 so `empty=True` matches `rescore_holdout_harmonised.py` identically
        ("KnockTF_Unseen_64", mat_ktf.loc[hold_ids], obs_ktf.loc[hold_ids], unseen_ids, ranks401.loc[unseen_ids, "study_id"].values),
        ("Replogle_K562_Primary_267", mat_k562.loc[k562_prim_ids], obs_k562.loc[k562_prim_ids], k562_prim_ids, obs_k562.loc[k562_prim_ids, "TF.Class"].values),
        ("Replogle_RPE1_Primary_76", mat_rpe1.loc[rpe1_prim_ids], obs_rpe1.loc[rpe1_prim_ids], rpe1_prim_ids, obs_rpe1.loc[rpe1_prim_ids, "TF.Class"].values),
    ]

    perf_rows = []
    shift_rows = []
    viper_comp_rows = []

    for dname, mat_batch, obs_batch, eval_ids, cluster_ids in datasets:
        true_tfs = list(obs_batch.loc[eval_ids, "TF"])
        for regime in ["1_Unblinded_WithSelfLoops", "2_Unblinded_NoSelfLoops", "3_StrictBlinded_1888TFs"]:
            print(f"[{time.time()-t_start:5.1f}s] Auditing {dname} under {regime}...", flush=True)
            scores_full = score_all_methods_under_regime(
                mat_batch, obs_batch, cand_tfs, regime, collectri, dorothea, gmt_no_self, gmt_with_self, all_masked_tfs
            )
            scores_dict = {k: v.loc[eval_ids] for k, v in scores_full.items()}
            ranks_dict: Dict[str, np.ndarray] = {}
            for mname, sdf in scores_dict.items():
                auroc_p, auprc_p, auroc_rk, r_arr = compute_global_auroc_auprc(sdf, true_tfs, higher_is_better=True)
                ranks_dict[mname] = r_arr

                # Verify 0 differences against harmonised tables for KnockTF under 3_StrictBlinded_1888TFs
                if regime == "3_StrictBlinded_1888TFs" and dname in ("KnockTF_Dev_266", "KnockTF_Unseen_64"):
                    col_map = {
                        "M1_ULM": "r_m1_ulm",
                        "M2b_VIPER_Shadow": "r_m2b_viper_shadow",
                        "M5_MultiModal_Static": "r_m5_mm_static",
                        "M9_Selective_Deconf": "r_m9_determ",
                    }
                    if mname in col_map:
                        ref_r = ranks401.loc[eval_ids, col_map[mname]].values
                        n_diff = int((r_arr != ref_r).sum())
                        assert n_diff == 0, f"Mismatch on {dname} {mname}: {n_diff} diffs!"

                perf_rows.append({
                    "dataset": dname,
                    "n_samples": len(true_tfs),
                    "blinding_regime": regime,
                    "method": mname,
                    "top1_hits": int(np.sum(r_arr == 1)),
                    "top1_pct": round(float(np.mean(r_arr == 1) * 100), 2),
                    "top5_hits": int(np.sum(r_arr <= 5)),
                    "top5_pct": round(float(np.mean(r_arr <= 5) * 100), 2),
                    "top10_pct": round(float(np.mean(r_arr <= 10) * 100), 2),
                    "mrr": round(float(np.mean(1.0 / r_arr)), 4),
                    "median_rank": float(np.median(r_arr)),
                    "mean_log2_rank": round(float(np.mean(np.log2(r_arr))), 3),
                    "global_pooled_auroc": round(auroc_p, 4),
                    "global_pooled_auprc": round(auprc_p, 4),
                    "per_sample_rank_auroc": round(auroc_rk, 4),
                })

            if regime == "3_StrictBlinded_1888TFs":
                rn = ranks_dict["M_VIPER_Native"]
                rs = ranks_dict["M2b_VIPER_Shadow"]
                r9 = ranks_dict["M9_Selective_Deconf"]
                wx_ns = wilcoxon_both_modes(rn, rs)
                wx_9n = wilcoxon_both_modes(r9, rn)
                _, pc_9n = study_clustered_sign_flip_perm(-np.log2(r9.astype(float)), -np.log2(rn.astype(float)), cluster_ids)
                viper_comp_rows.append({
                    "dataset": dname,
                    "n_samples": len(true_tfs),
                    "viper_native_top1_pct": round(float(np.mean(rn == 1) * 100), 2),
                    "viper_shadow_top1_pct": round(float(np.mean(rs == 1) * 100), 2),
                    "viper_native_top5_pct": round(float(np.mean(rn <= 5) * 100), 2),
                    "viper_shadow_top5_pct": round(float(np.mean(rs <= 5) * 100), 2),
                    "viper_native_median_rank": float(np.median(rn)),
                    "viper_shadow_median_rank": float(np.median(rs)),
                    "native_vs_shadow_mean_dlog2rank": round(wx_ns["mean_bits"], 3),
                    "native_vs_shadow_p_wilcox": wx_ns["p_wilcox"],
                    "m9_vs_viper_native_mean_dlog2rank": round(wx_9n["mean_bits"], 3),
                    "m9_vs_viper_native_p_wilcox": wx_9n["p_wilcox"],
                    "m9_vs_viper_native_p_clust": pc_9n,
                })

                for m_a, m_b in [
                    ("M9_Selective_Deconf", "M1_ULM"),
                    ("M9_Selective_Deconf", "M_VIPER_Native"),
                    ("M9_Selective_Deconf", "M2b_VIPER_Shadow"),
                    ("M9_Selective_Deconf", "M5_MultiModal_Static"),
                    ("M5_MultiModal_Static", "M1_ULM"),
                    ("M_ZSCORE", "M1_ULM"),
                    ("M_WMEAN", "M1_ULM"),
                ]:
                    ra, rb = ranks_dict[m_a], ranks_dict[m_b]
                    w5, l5, p5_naive = exact_mcnemar_2s((ra <= 5).astype(int), (rb <= 5).astype(int))
                    _, p5_clust = study_clustered_sign_flip_perm((ra <= 5).astype(float), (rb <= 5).astype(float), cluster_ids)
                    wx_ab = wilcoxon_both_modes(ra, rb)
                    _, plr_clust = study_clustered_sign_flip_perm(-np.log2(ra.astype(float)), -np.log2(rb.astype(float)), cluster_ids)
                    shift_rows.append({
                        "dataset": dname,
                        "comparison": f"{m_a}_vs_{m_b}",
                        "top5_wl": f"{w5}/{l5}",
                        "p_top5_naive_mcnemar": p5_naive,
                        "p_top5_clustered": p5_clust,
                        "mean_dlog2rank": round(wx_ab["mean_bits"], 3),
                        "p_log2rank_naive_wilcox": wx_ab["p_wilcox"],
                        "p_log2rank_clustered": plr_clust,
                    })

    df_perf = pd.DataFrame(perf_rows)
    df_perf.to_csv(AUDIT_DIR / "r3_method_performance_by_blinding_and_metric.csv", index=False)

    df_vip = pd.DataFrame(viper_comp_rows)
    df_vip.to_csv(AUDIT_DIR / "r3_viper_native_vs_shadow_comparison.csv", index=False)

    df_shift = pd.DataFrame(shift_rows)
    df_shift.to_csv(AUDIT_DIR / "r3_ranking_inversion_and_significance_shift.csv", index=False)

    print("\n=== R3 Benchmark Audit: VIPER Native vs VIPER Shadow ===")
    print(df_vip.to_string(index=False))
    print("\n=== R3 Benchmark Audit: Naive vs Clustered Significance Shift ===")
    print(df_shift.to_string(index=False))


if __name__ == "__main__":
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 25)
    main()
