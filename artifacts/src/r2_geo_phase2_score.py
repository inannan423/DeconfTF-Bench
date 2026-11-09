"""R2 GEO Phase 2: one-shot scoring of QC-pass contrasts.

Frozen calls only: score_knocktf_experiments, m9_order, m10_soft_order.
Do not re-run after seeing ranks. Do not change deconftf_tools.py or M9/M10.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, "/Users/bytedance/codes/new-bishe/artifacts/src")
from audit_reviewer_controls import compute_viper_shadow_scores
from deconftf_tools import m9_order, score_knocktf_experiments
from develop_m10_soft_gate import compute_bundle_coverage, m10_soft_order
from verify_round4_accept_checks import exact_mcnemar_2s, study_clustered_sign_flip_perm, wilcoxon_both_modes

ROOT = Path("/Users/bytedance/codes/new-bishe")
OUT_DIR = ROOT / "artifacts" / "results" / "cand6_deconftf"
EXT_OUT = OUT_DIR / "external"
QC_CSV = EXT_OUT / "r2_geo_qc_pass.csv"
LFC_PARQUET = EXT_OUT / "r2_geo_lfc_matrix.parquet"
RANKS_CSV = EXT_OUT / "r2_geo_phase2_ranks.csv"
ENDPOINTS_CSV = EXT_OUT / "r2_geo_phase2_endpoints.csv"
HOLM_CSV = EXT_OUT / "r2_geo_phase2_holm.csv"
DECISION_TXT = EXT_OUT / "r2_geo_phase2_decision.txt"


def rank_of(order: List[str], tf: str) -> int:
    return order.index(tf) + 1


def dlog2(r_base: np.ndarray, r_method: np.ndarray) -> np.ndarray:
    return np.log2(r_base.astype(float)) - np.log2(r_method.astype(float))


def clustered_p(r_method: np.ndarray, r_base: np.ndarray, study_ids: np.ndarray) -> Tuple[float, float]:
    mean_bits, p = study_clustered_sign_flip_perm(
        -np.log2(r_method.astype(float)),
        -np.log2(r_base.astype(float)),
        study_ids,
        n_perm=20000,
        seed=42,
    )
    return float(mean_bits), float(p)


def score_pass_contrasts(qc: pd.DataFrame, lfc: pd.DataFrame, cand_tfs: List[str]) -> pd.DataFrame:
    pass_qc = qc[qc["qc_reason"] == "PASS"].copy()
    ids = [c for c in pass_qc["contrast_id"].tolist() if c in lfc.columns]
    mat = lfc[ids].T.astype(float)
    mat.index = ids
    mat.columns = [str(c).upper() for c in mat.columns]
    if mat.columns.duplicated().any():
        mat = mat.T.groupby(level=0).sum().T
    mat = mat.fillna(0.0)

    obs_rows = []
    for _, r in pass_qc.iterrows():
        cid = str(r["contrast_id"])
        if cid not in mat.index:
            continue
        obs_rows.append(
            {
                "exp_id": cid,
                "TF": str(r["target_tf"]).upper(),
                "logFC": float(r["target_log2fc"]),
                "Biosample.Name": str(r["cell_line"]),
                "Tissue.Type": str(r.get("cell_line", "")),
                "study_id": str(r["study_id"]),
                "stratum": str(r["stratum"]),
                "is_lambert2018_tf": int(r["is_lambert2018_tf"]),
                "n_treated": int(r["n_treated"]),
                "n_control": int(r["n_control"]),
                "single_replicate_arm": int(r["single_replicate_arm"]),
                "tmd14_typo_sensitivity": int(r["tmd14_typo_sensitivity"]),
                "expression_source": str(r["expression_source"]),
                "n_genes": int(r["n_genes"]),
                "max_abs_log2fc_blind": float(r["max_abs_log2fc_blind"]),
            }
        )
    obs = pd.DataFrame(obs_rows).set_index("exp_id")
    print(f"[score] mat {mat.shape} obs {obs.shape}", flush=True)
    bundle = score_knocktf_experiments(mat, obs, cand_tfs=cand_tfs)
    viper = compute_viper_shadow_scores(
        bundle["mat_blind"],
        bundle["est_ulm"],
        bundle["W_c"],
        bundle["gene_to_idx"],
        bundle["tf_to_idx"],
        bundle["cand_tfs"],
    )
    cov = compute_bundle_coverage(bundle)
    print(f"[score] cov_exp={cov:.4f}", flush=True)
    rows = []
    for eid in bundle["obs_sub"].index:
        tf = str(bundle["obs_sub"].loc[eid, "TF"])
        ord_u = list(bundle["est_ulm"].loc[eid].sort_values(ascending=True).index)
        ord_vip = list(viper.loc[eid].sort_values(ascending=False).index)
        ord_m5 = list(bundle["multi_modal_score"].loc[eid].sort_values(ascending=False).index)
        ord_m9 = m9_order(eid, bundle)
        ord_m10, _ = m10_soft_order(eid, bundle, cov_frac=cov, lambda_bind=1.0, lambda_hub=0.3)
        rec = obs.loc[eid].to_dict()
        rec.update(
            {
                "exp_id": eid,
                "true_tf": tf,
                "cov_exp": cov,
                "r_m1_ulm": rank_of(ord_u, tf),
                "r_m2b_viper_shadow": rank_of(ord_vip, tf),
                "r_m5_mm_static": rank_of(ord_m5, tf),
                "r_m9_determ": rank_of(ord_m9, tf),
                "r_m10_soft": rank_of(ord_m10, tf),
                "top1_ulm": ord_u[0],
                "top1_m5": ord_m5[0],
                "top1_m9": ord_m9[0],
                "top1_m10": ord_m10[0],
            }
        )
        rows.append(rec)
    return pd.DataFrame(rows).set_index("exp_id")


def stratum_metrics(df: pd.DataFrame, name: str) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    study = df["study_id"].astype(str).values
    pairs = [
        ("M9", "M5", df["r_m9_determ"].values, df["r_m5_mm_static"].values),
        ("M9", "ULM", df["r_m9_determ"].values, df["r_m1_ulm"].values),
        ("M9", "VIPER", df["r_m9_determ"].values, df["r_m2b_viper_shadow"].values),
        ("M5", "ULM", df["r_m5_mm_static"].values, df["r_m1_ulm"].values),
        ("M10", "M5", df["r_m10_soft"].values, df["r_m5_mm_static"].values),
        ("M10", "ULM", df["r_m10_soft"].values, df["r_m1_ulm"].values),
        ("M10", "M9", df["r_m10_soft"].values, df["r_m9_determ"].values),
    ]
    out = []
    for a, b, ra, rb in pairs:
        bits = dlog2(rb, ra)
        wx = wilcoxon_both_modes(ra, rb)
        mean_c, p_c = clustered_p(ra, rb, study)
        w1, l1, p1 = exact_mcnemar_2s((ra == 1).astype(int), (rb == 1).astype(int))
        w5, l5, p5 = exact_mcnemar_2s((ra <= 5).astype(int), (rb <= 5).astype(int))
        out.append(
            {
                "stratum": name,
                "n": len(df),
                "n_studies": int(df["study_id"].nunique()),
                "comparison": f"{a}_vs_{b}",
                "mean_dlog2_rank": float(np.mean(bits)),
                "median_dlog2_rank": float(np.median(bits)),
                "p_clustered_dlog2": p_c,
                "clustered_mean_bits": mean_c,
                "p_wilcox": wx["p_wilcox"],
                "p_pratt": wx["p_pratt"],
                f"top1_{a}": int(np.sum(ra == 1)),
                f"top1_{b}": int(np.sum(rb == 1)),
                "top1_wl": f"{w1}/{l1}",
                "p_top1_mcnemar": p1,
                f"top5_{a}": int(np.sum(ra <= 5)),
                f"top5_{b}": int(np.sum(rb <= 5)),
                "top5_wl": f"{w5}/{l5}",
                "p_top5_mcnemar": p5,
                f"median_rank_{a}": float(np.median(ra)),
                f"median_rank_{b}": float(np.median(rb)),
            }
        )
    return out


def holm_family(endpoints: pd.DataFrame, stratum: str, tests: List[Tuple[str, str]], m_label: str) -> pd.DataFrame:
    rows = []
    pvals = []
    for comp, pcol in tests:
        hit = endpoints[(endpoints.stratum == stratum) & (endpoints.comparison == comp)]
        if hit.empty:
            p = 1.0
            extra = {}
        else:
            p = float(hit.iloc[0][pcol])
            extra = {
                "mean_dlog2_rank": float(hit.iloc[0]["mean_dlog2_rank"]),
                "median_dlog2_rank": float(hit.iloc[0]["median_dlog2_rank"]),
            }
        rows.append({"stratum": stratum, "family": m_label, "comparison": comp, "p_raw": p, **extra})
        pvals.append(p)
    _, p_holm, _, _ = multipletests(pvals, alpha=0.05, method="holm")
    for rec, ph in zip(rows, p_holm):
        rec["p_holm"] = float(ph)
        rec["sig_holm_05"] = int(ph < 0.05)
    return pd.DataFrame(rows)


def decide_abc(holm5: pd.DataFrame) -> str:
    def p(comp: str) -> Tuple[float, float]:
        row = holm5[holm5.comparison == comp].iloc[0]
        return float(row["mean_dlog2_rank"]), float(row["p_holm"])

    m9_m5_bits, m9_m5_p = p("M9_vs_M5")
    m9_ulm_bits, m9_ulm_p = p("M9_vs_ULM")
    m5_ulm_bits, m5_ulm_p = p("M5_vs_ULM")
    if m9_m5_bits > 0 and m9_m5_p < 0.05 and m9_ulm_bits > 0 and m9_ulm_p < 0.05:
        return "A"
    if (m5_ulm_bits > 0 and m5_ulm_p < 0.05) or (m9_ulm_bits > 0 and m9_ulm_p < 0.05):
        return "B"
    return "C"


def main() -> None:
    qc = pd.read_csv(QC_CSV)
    lfc = pd.read_parquet(LFC_PARQUET)
    bundle_dev = pickle.loads((OUT_DIR / "knocktf_precomputed_bundle.pkl").read_bytes())
    cand_tfs = list(bundle_dev["cand_tfs"])
    ranks = score_pass_contrasts(qc, lfc, cand_tfs)
    ranks.to_csv(RANKS_CSV)
    print(f"wrote {RANKS_CSV} n={len(ranks)}", flush=True)

    kd = ranks[ranks.stratum == "KD"]
    slices = [
        ("Combined_Primary", ranks),
        ("KD", kd),
        ("KO", ranks[ranks.stratum == "KO"]),
        ("Degron", ranks[ranks.stratum == "Degron"]),
        ("Lambert2018", ranks[ranks.is_lambert2018_tf == 1]),
        ("KD_strong_le_minus1", kd[kd.logFC <= -1.0]),
        ("KD_moderate", kd[(kd.logFC > -1.0) & (kd.logFC <= -0.5)]),
        ("n_ge2_both_arms", ranks[(ranks.n_treated >= 2) & (ranks.n_control >= 2)]),
        ("exclude_TMD14_typo", ranks[ranks.tmd14_typo_sensitivity == 0]),
    ]
    ep_rows: List[Dict[str, Any]] = []
    for name, sub in slices:
        print(f"[endpoints] {name} n={len(sub)}", flush=True)
        ep_rows.extend(stratum_metrics(sub, name))
    endpoints = pd.DataFrame(ep_rows)
    endpoints.to_csv(ENDPOINTS_CSV, index=False)

    tests5 = [
        ("M9_vs_M5", "p_clustered_dlog2"),
        ("M9_vs_ULM", "p_clustered_dlog2"),
        ("M9_vs_ULM", "p_top5_mcnemar"),
        ("M5_vs_ULM", "p_clustered_dlog2"),
        ("M5_vs_ULM", "p_top5_mcnemar"),
    ]
    # Holm m=5 uses five tests; duplicate comparison names distinguished by p column.
    tests5_named = [
        ("M9_vs_M5", "p_clustered_dlog2", "M9_vs_M5_dlog2"),
        ("M9_vs_ULM", "p_clustered_dlog2", "M9_vs_ULM_dlog2"),
        ("M9_vs_ULM", "p_top5_mcnemar", "M9_vs_ULM_top5"),
        ("M5_vs_ULM", "p_clustered_dlog2", "M5_vs_ULM_dlog2"),
        ("M5_vs_ULM", "p_top5_mcnemar", "M5_vs_ULM_top5"),
    ]
    tests7_named = tests5_named + [
        ("M10_vs_M5", "p_clustered_dlog2", "M10_vs_M5_dlog2"),
        ("M10_vs_ULM", "p_clustered_dlog2", "M10_vs_ULM_dlog2"),
    ]

    holm_rows = []
    for family, named in [("m5", tests5_named), ("m7", tests7_named)]:
        pvals = []
        recs = []
        for comp, pcol, label in named:
            hit = endpoints[(endpoints.stratum == "Combined_Primary") & (endpoints.comparison == comp)]
            p = float(hit.iloc[0][pcol]) if len(hit) else 1.0
            recs.append(
                {
                    "family": family,
                    "test": label,
                    "comparison": comp,
                    "p_col": pcol,
                    "p_raw": p,
                    "mean_dlog2_rank": float(hit.iloc[0]["mean_dlog2_rank"]) if len(hit) else np.nan,
                }
            )
            pvals.append(p)
        _, p_holm, _, _ = multipletests(pvals, alpha=0.05, method="holm")
        for rec, ph in zip(recs, p_holm):
            rec["p_holm"] = float(ph)
            rec["sig_holm_05"] = int(ph < 0.05)
            holm_rows.append(rec)
    holm = pd.DataFrame(holm_rows)
    holm.to_csv(HOLM_CSV, index=False)

    holm5 = holm[holm.family == "m5"].copy()
    # Map named tests back for A/B/C
    def holm5_bits_p(label: str) -> Tuple[float, float]:
        row = holm5[holm5.test == label].iloc[0]
        return float(row["mean_dlog2_rank"]), float(row["p_holm"])

    m9_m5_bits, m9_m5_p = holm5_bits_p("M9_vs_M5_dlog2")
    m9_ulm_bits, m9_ulm_p = holm5_bits_p("M9_vs_ULM_dlog2")
    m5_ulm_bits, m5_ulm_p = holm5_bits_p("M5_vs_ULM_dlog2")
    if m9_m5_bits > 0 and m9_m5_p < 0.05 and m9_ulm_bits > 0 and m9_ulm_p < 0.05:
        outcome = "A"
    elif m5_ulm_bits > 0 and m5_ulm_p < 0.05:
        outcome = "B"
    else:
        outcome = "C"
    holm7 = holm[holm.family == "m7"]
    m10_m5 = holm7[holm7.test == "M10_vs_M5_dlog2"].iloc[0]
    m10_ulm = holm7[holm7.test == "M10_vs_ULM_dlog2"].iloc[0]
    m10_generalises = bool(
        float(m10_m5["mean_dlog2_rank"]) > 0
        and int(m10_m5["sig_holm_05"]) == 1
        and float(m10_ulm["mean_dlog2_rank"]) > 0
        and int(m10_ulm["sig_holm_05"]) == 1
    )

    def top_line(col: str) -> str:
        r = ranks[col]
        return f"Top-1 {int((r==1).sum())}/{len(r)}  Top-5 {int((r<=5).sum())}/{len(r)}  median {float(np.median(r)):.1f}"

    text = [
        f"R2 GEO Phase 2 one-shot scoring  n={len(ranks)} QC-pass contrasts",
        f"Outcome (m=5 family): {outcome}",
        f"M10 generalisation (m=7 Holm): {m10_generalises}",
        f"M1_ULM     {top_line('r_m1_ulm')}",
        f"M2b_VIPER  {top_line('r_m2b_viper_shadow')}",
        f"M5         {top_line('r_m5_mm_static')}",
        f"M9         {top_line('r_m9_determ')}",
        f"M10        {top_line('r_m10_soft')}",
        f"M9 vs M5 dlog2 mean={m9_m5_bits:.4f} p_Holm5={m9_m5_p:.4g}",
        f"M9 vs ULM dlog2 mean={m9_ulm_bits:.4f} p_Holm5={m9_ulm_p:.4g}",
        f"M5 vs ULM dlog2 mean={m5_ulm_bits:.4f} p_Holm5={m5_ulm_p:.4g}",
        f"M10 vs M5 dlog2 mean={float(m10_m5['mean_dlog2_rank']):.4f} p_Holm7={float(m10_m5['p_holm']):.4g}",
        f"M10 vs ULM dlog2 mean={float(m10_ulm['mean_dlog2_rank']):.4f} p_Holm7={float(m10_ulm['p_holm']):.4g}",
        f"cov_exp={float(ranks['cov_exp'].iloc[0]):.4f}",
        "No parameter re-run. deconftf_tools.py untouched.",
    ]
    DECISION_TXT.write_text("\n".join(text) + "\n")
    print("\n".join(text), flush=True)
    print(f"wrote {ENDPOINTS_CSV}, {HOLM_CSV}, {DECISION_TXT}")


if __name__ == "__main__":
    main()
