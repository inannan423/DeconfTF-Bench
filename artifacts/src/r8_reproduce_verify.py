"""R8a: recompute manuscript tables/figures into a scratch directory and diff vs committed CSVs.

Never writes to frozen primary paths. GEO scoring uses the local LFC parquet (no download).
Exit 0 only if every compared cell matches within 1e-9 (ranks compared exactly).
"""
from __future__ import annotations

import pickle
import shutil
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/bytedance/codes/new-bishe/artifacts/src")
from deconftf_tools import m9_order, score_knocktf_experiments
from develop_m10_soft_gate import compute_bundle_coverage, m10_soft_order
from audit_reviewer_controls import compute_viper_shadow_scores
from eval_p0_2_external import load_replogle_cohort
from r2_geo_phase2_score import score_pass_contrasts, stratum_metrics
from r7_forest_all_cohorts import cluster_boot, dersimonian_laird, COMPARISONS
from statsmodels.stats.multitest import multipletests
from recompute_study_level_stats import endpoint_tests

ROOT = Path("/Users/bytedance/codes/new-bishe")
DATA_DIR = ROOT / "artifacts" / "data" / "knocktf"
EXT_DATA = ROOT / "artifacts" / "data" / "external"
OUT_DIR = ROOT / "artifacts" / "results" / "cand6_deconftf"
EXT_OUT = OUT_DIR / "external"
AUDIT = OUT_DIR / "audit"
SCRATCH = AUDIT / "r8_repro_scratch"
ATOL = 1e-9

DIFFS: List[str] = []


def _fail(msg: str) -> None:
    DIFFS.append(msg)
    print(f"DIFF  {msg}", flush=True)


def compare_numeric(got: pd.Series, ref: pd.Series, label: str) -> None:
    g = pd.to_numeric(got, errors="coerce").to_numpy(dtype=float)
    r = pd.to_numeric(ref, errors="coerce").to_numpy(dtype=float)
    if g.shape != r.shape:
        _fail(f"{label}: shape {g.shape} vs {r.shape}")
        return
    if not np.allclose(g, r, atol=ATOL, rtol=0.0, equal_nan=True):
        bad = np.where(~np.isclose(g, r, atol=ATOL, rtol=0.0, equal_nan=True))[0]
        _fail(f"{label}: {len(bad)} mismatches, e.g. idx={bad[0]} {g[bad[0]]} vs {r[bad[0]]}")


def compare_frames(got: pd.DataFrame, ref: pd.DataFrame, name: str, cols: Sequence[str] | None = None) -> None:
    cols = list(cols) if cols is not None else [c for c in ref.columns if c in got.columns]
    missing = [c for c in cols if c not in got.columns]
    if missing:
        _fail(f"{name}: missing columns {missing}")
        return
    if not got.index.equals(ref.index):
        if set(got.index) != set(ref.index):
            _fail(f"{name}: index set differs ({len(got)} vs {len(ref)})")
            return
        got = got.loc[ref.index]
    for c in cols:
        if pd.api.types.is_numeric_dtype(ref[c]) or pd.api.types.is_numeric_dtype(got[c]):
            compare_numeric(got[c], ref[c], f"{name}.{c}")
        else:
            if not (got[c].astype(str).fillna("").to_numpy() == ref[c].astype(str).fillna("").to_numpy()).all():
                _fail(f"{name}.{c}: string mismatch")


def ranks_from_bundle(bundle: dict, ids: List[str], tfs: List[str], cov: float) -> pd.DataFrame:
    rows = []
    viper = compute_viper_shadow_scores(
        bundle["mat_blind"], bundle["est_ulm"], bundle["W_c"],
        bundle["gene_to_idx"], bundle["tf_to_idx"], bundle["cand_tfs"],
    )
    for eid, tf in zip(ids, tfs):
        ord_u = list(bundle["est_ulm"].loc[eid].sort_values(ascending=True).index)
        ord_vip = list(viper.loc[eid].sort_values(ascending=False).index)
        ord_m5 = list(bundle["multi_modal_score"].loc[eid].sort_values(ascending=False).index)
        ord_m9 = m9_order(eid, bundle)
        ord_m10, _ = m10_soft_order(eid, bundle, cov_frac=cov, lambda_bind=1.0, lambda_hub=0.3)
        rows.append(
            {
                "exp_id": eid,
                "true_tf": tf,
                "r_m1_ulm": ord_u.index(tf) + 1,
                "r_m2b_viper_shadow": ord_vip.index(tf) + 1,
                "r_m5_mm_static": ord_m5.index(tf) + 1,
                "r_m9_determ": ord_m9.index(tf) + 1,
                "r_m10_soft": ord_m10.index(tf) + 1,
            }
        )
    return pd.DataFrame(rows).set_index("exp_id")


def stage_dev_unseen() -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("== Dev bundle + Unseen rescore ==", flush=True)
    ranks401 = pd.read_csv(OUT_DIR / "knocktf401_ranks_with_study_ids.csv").set_index("exp_id")
    h135 = pd.read_csv(OUT_DIR / "holdout135_harmonised_ranks.csv").set_index("exp_id")
    for col in ["r_m1_ulm", "r_m2b_viper_shadow", "r_m5_mm_static", "r_m9_determ"]:
        ranks401.loc[h135.index, col] = h135[col].values
    dev = ranks401[ranks401["split_new"] == "Dev"].copy()
    unseen = ranks401[ranks401["split_new"] == "Unseen"].copy()
    bundle = pickle.loads((OUT_DIR / "knocktf_precomputed_bundle.pkl").read_bytes())
    cand_tfs = list(bundle["cand_tfs"])
    cov = compute_bundle_coverage(bundle)
    got_dev = ranks_from_bundle(bundle, list(dev.index), list(dev["true_tf"].astype(str)), cov)
    compare_frames(got_dev, dev, "Dev266", ["r_m1_ulm", "r_m2b_viper_shadow", "r_m5_mm_static", "r_m9_determ"])
    m10_ref = pd.read_csv(AUDIT / "r7_m10_ranks_knocktf.csv").set_index("exp_id")
    compare_numeric(got_dev["r_m10_soft"], m10_ref.loc[got_dev.index, "r_m10_soft"], "Dev266.r_m10_soft")

    mat = pd.read_csv(DATA_DIR / "knockTF_expr.csv", index_col=0)
    obs = pd.read_csv(DATA_DIR / "knockTF_meta.csv", index_col=0)
    # Harmonised holdout ranks were scored on the 135-row matrix then sliced to
    # Unseen 64 (decoupler empty=True batch effect; R3). M10 in r7 used Unseen 64 alone.
    hold_ids = list(ranks401.index[ranks401["split_new"] != "Dev"])
    print(f"  scoring Holdout {len(hold_ids)} then slice Unseen {len(unseen)}", flush=True)
    hold_b = score_knocktf_experiments(mat.loc[hold_ids], obs.loc[hold_ids], cand_tfs=cand_tfs)
    cov_u = compute_bundle_coverage(hold_b)
    u_ids = list(unseen.index)
    tfs = [str(obs.loc[eid, "TF"]) for eid in u_ids]
    got_u = ranks_from_bundle(hold_b, u_ids, tfs, cov_u)
    compare_frames(got_u, unseen, "Unseen64", ["r_m1_ulm", "r_m2b_viper_shadow", "r_m5_mm_static", "r_m9_determ"])
    print(f"  scoring Unseen-only {len(u_ids)} for M10 vs r7 cache", flush=True)
    u_b = score_knocktf_experiments(mat.loc[u_ids], obs.loc[u_ids], cand_tfs=cand_tfs)
    cov64 = compute_bundle_coverage(u_b)
    got_u_m10 = ranks_from_bundle(u_b, u_ids, tfs, cov64)
    compare_numeric(got_u_m10["r_m10_soft"], m10_ref.loc[got_u_m10.index, "r_m10_soft"], "Unseen64.r_m10_soft")
    got_u["r_m10_soft"] = got_u_m10["r_m10_soft"]
    got_dev.to_csv(SCRATCH / "dev266_ranks.csv")
    got_u.to_csv(SCRATCH / "unseen64_ranks.csv")
    return got_dev, got_u


def stage_geo() -> None:
    print("== GEO phase 2 from local LFC (no download) ==", flush=True)
    qc = pd.read_csv(EXT_OUT / "r2_geo_qc_pass.csv")
    lfc = pd.read_parquet(EXT_OUT / "r2_geo_lfc_matrix.parquet")
    cand_tfs = list(pickle.loads((OUT_DIR / "knocktf_precomputed_bundle.pkl").read_bytes())["cand_tfs"])
    ranks = score_pass_contrasts(qc, lfc, cand_tfs)
    ranks.to_csv(SCRATCH / "r2_geo_phase2_ranks.csv")
    ref = pd.read_csv(EXT_OUT / "r2_geo_phase2_ranks.csv").set_index("exp_id")
    compare_frames(
        ranks,
        ref,
        "GEO_ranks",
        ["r_m1_ulm", "r_m2b_viper_shadow", "r_m5_mm_static", "r_m9_determ", "r_m10_soft"],
    )

    print("== GEO endpoints/holm from committed ranks ==", flush=True)
    kd = ref[ref.stratum == "KD"]
    slices = [
        ("Combined_Primary", ref),
        ("KD", kd),
        ("KO", ref[ref.stratum == "KO"]),
        ("Degron", ref[ref.stratum == "Degron"]),
        ("Lambert2018", ref[ref.is_lambert2018_tf == 1]),
        ("KD_strong_le_minus1", kd[kd.logFC <= -1.0]),
        ("KD_moderate", kd[(kd.logFC > -1.0) & (kd.logFC <= -0.5)]),
        ("n_ge2_both_arms", ref[(ref.n_treated >= 2) & (ref.n_control >= 2)]),
        ("exclude_TMD14_typo", ref[ref.tmd14_typo_sensitivity == 0]),
    ]
    ep = pd.DataFrame([row for name, sub in slices for row in stratum_metrics(sub, name)])
    ep.to_csv(SCRATCH / "r2_geo_phase2_endpoints.csv", index=False)
    ref_ep = pd.read_csv(EXT_OUT / "r2_geo_phase2_endpoints.csv")
    compare_numeric(ep["mean_dlog2_rank"], ref_ep["mean_dlog2_rank"], "GEO_endpoints.mean_dlog2_rank")
    compare_numeric(ep["p_clustered_dlog2"], ref_ep["p_clustered_dlog2"], "GEO_endpoints.p_clustered_dlog2")

    tests5 = [
        ("M9_vs_M5", "p_clustered_dlog2", "M9_vs_M5_dlog2"),
        ("M9_vs_ULM", "p_clustered_dlog2", "M9_vs_ULM_dlog2"),
        ("M9_vs_ULM", "p_top5_mcnemar", "M9_vs_ULM_top5"),
        ("M5_vs_ULM", "p_clustered_dlog2", "M5_vs_ULM_dlog2"),
        ("M5_vs_ULM", "p_top5_mcnemar", "M5_vs_ULM_top5"),
    ]
    tests7 = tests5 + [
        ("M10_vs_M5", "p_clustered_dlog2", "M10_vs_M5_dlog2"),
        ("M10_vs_ULM", "p_clustered_dlog2", "M10_vs_ULM_dlog2"),
    ]
    rows = []
    for family, named in [("m5", tests5), ("m7", tests7)]:
        pvals = []
        recs = []
        for comp, pcol, label in named:
            hit = ep[(ep.stratum == "Combined_Primary") & (ep.comparison == comp)]
            p = float(hit.iloc[0][pcol])
            recs.append({"family": family, "test": label, "p_raw": p, "mean_dlog2_rank": float(hit.iloc[0]["mean_dlog2_rank"])})
            pvals.append(p)
        _, p_holm, _, _ = multipletests(pvals, alpha=0.05, method="holm")
        for rec, ph in zip(recs, p_holm):
            rec["p_holm"] = float(ph)
            rows.append(rec)
    holm = pd.DataFrame(rows)
    holm.to_csv(SCRATCH / "r2_geo_phase2_holm.csv", index=False)
    ref_h = pd.read_csv(EXT_OUT / "r2_geo_phase2_holm.csv")
    compare_numeric(holm["p_holm"], ref_h["p_holm"], "GEO_holm.p_holm")
    compare_numeric(holm["mean_dlog2_rank"], ref_h["mean_dlog2_rank"], "GEO_holm.mean_dlog2_rank")


def stage_replogle() -> None:
    print("== Replogle rescore from local h5ad ==", flush=True)
    cand_tfs = list(pickle.loads((OUT_DIR / "knocktf_precomputed_bundle.pkl").read_bytes())["cand_tfs"])
    masked = set(pd.read_csv(OUT_DIR / "supplement_masked_tfs_list.csv").iloc[:, 0].astype(str).str.upper())
    obs_ktf = pd.read_csv(DATA_DIR / "knockTF_meta.csv", index_col=0)
    tf_to_class = obs_ktf.groupby("TF")["TF.Class"].first().to_dict()
    m10_ref = pd.read_csv(AUDIT / "r7_m10_ranks_replogle.csv").set_index("exp_id")
    for label, h5, bio, tissue, ref_path, n_primary in [
        ("K562", EXT_DATA / "K562_gwps_normalized_bulk_01.h5ad", "K562", "Haematopoietic",
         EXT_OUT / "replogle2022_k562_gwps_ranks.csv", 267),
        ("RPE1", EXT_DATA / "rpe1_normalized_bulk_01.h5ad", "RPE1", "RPE",
         EXT_OUT / "replogle2022_rpe1_ranks.csv", 76),
    ]:
        mat, obs = load_replogle_cohort(h5, cand_tfs, masked, tf_to_class, bio, tissue, mode="normalized_z")
        print(f"  scoring {label} n={len(obs)}", flush=True)
        bundle = score_knocktf_experiments(mat, obs, cand_tfs=cand_tfs)
        cov = compute_bundle_coverage(bundle)
        tfs = [str(obs.loc[eid, "TF"]) for eid in bundle["obs_sub"].index]
        got = ranks_from_bundle(bundle, list(bundle["obs_sub"].index), tfs, cov)
        got["max_abs_blind"] = bundle["obs_sub"]["max_abs_blind"].astype(float)
        ref = pd.read_csv(ref_path)
        if "exp_id" not in ref.columns:
            ref = ref.rename(columns={ref.columns[0]: "exp_id"})
        ref = ref.set_index("exp_id")
        compare_frames(got, ref, f"Replogle_{label}", ["r_m1_ulm", "r_m2b_viper_shadow", "r_m5_mm_static", "r_m9_determ"])
        prim = got[got["max_abs_blind"] >= 0.5]
        if len(prim) != n_primary:
            _fail(f"Replogle_{label} primary n={len(prim)} expected {n_primary}")
        compare_numeric(prim["r_m10_soft"], m10_ref.loc[prim.index, "r_m10_soft"], f"Replogle_{label}.r_m10_soft")
        got.to_csv(SCRATCH / f"replogle_{label.lower()}_ranks.csv")


def stage_r7() -> None:
    print("== R7 forest summary from committed experiment ranks ==", flush=True)
    long = pd.read_csv(AUDIT / "r7_forest_all_cohorts_experiment_ranks.csv")
    rows = []
    for (cohort, role, unit), sub in long.groupby(["cohort", "cohort_role", "cluster_unit"], sort=False):
        for label, mcol, bcol in COMPARISONS:
            mean, lo, hi, se = cluster_boot(sub[mcol].values, sub[bcol].values, sub["cluster_id"].values)
            rows.append(
                {
                    "cohort": cohort,
                    "comparison": label,
                    "mean_dlog2_rank": mean,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "boot_se": se,
                    "n": len(sub),
                }
            )
    summ = pd.DataFrame(rows)
    re_cohorts = ["Replogle K562 Primary 267", "GEO bulk RNA-seq 51"]
    re_rows = []
    for label, _, _ in COMPARISONS:
        hit = summ[(summ.comparison == label) & (summ.cohort.isin(re_cohorts))].set_index("cohort").loc[re_cohorts]
        re, lo, hi, tau2 = dersimonian_laird(hit["mean_dlog2_rank"].tolist(), hit["boot_se"].tolist())
        re_rows.append({"cohort": "RE pool (K562 + GEO)", "comparison": label, "mean_dlog2_rank": re, "ci95_lo": lo, "ci95_hi": hi, "tau2": tau2})
    got = pd.concat([summ, pd.DataFrame(re_rows)], ignore_index=True)
    ref = pd.read_csv(AUDIT / "r7_forest_all_cohorts_summary.csv")
    got.to_csv(SCRATCH / "r7_forest_summary.csv", index=False)
    merged = got.merge(ref, on=["cohort", "comparison"], suffixes=("_g", "_r"))
    compare_numeric(merged["mean_dlog2_rank_g"], merged["mean_dlog2_rank_r"], "R7.mean_dlog2_rank")
    compare_numeric(merged["ci95_lo_g"], merged["ci95_lo_r"], "R7.ci95_lo")


def stage_stats_from_ranks() -> None:
    print("== study_level_stats_harmonised from committed ranks ==", flush=True)
    d = pd.read_csv(OUT_DIR / "knocktf401_ranks_with_study_ids.csv").set_index("exp_id")
    h = pd.read_csv(OUT_DIR / "holdout135_harmonised_ranks.csv").set_index("exp_id")
    cols = ["r_m1_ulm", "r_m2b_viper_shadow", "r_m5_mm_static", "r_m9_determ"]
    d.loc[h.index, cols] = h[cols].values
    d["in_recalled"] = np.nan
    d.loc[h.index, "in_recalled"] = h["in_recalled"].values
    cohorts = {
        "Unseen": d[d["split_new"] == "Unseen"],
        "Combined_401": d,
        "Holdout_135": d[d["split_new"] != "Dev"],
        "Seen": d[d["split_new"] == "Seen"],
    }
    rows = []
    for cname, sub in cohorts.items():
        for base in ["r_m1_ulm", "r_m2b_viper_shadow", "r_m5_mm_static", "r_prior_loso"]:
            for ep, stat, p, pc in endpoint_tests(sub, base, "study_id"):
                rows.append({"cohort": cname, "base": base, "endpoint": ep, "stat": stat, "p_raw": p, "p_clust": pc})
    t = pd.DataFrame(rows)
    prim = t["cohort"].isin(["Unseen", "Combined_401"]) & t["base"].isin(["r_m1_ulm", "r_m2b_viper_shadow"])
    t["p_clust_Holm12"] = np.nan
    t.loc[prim, "p_clust_Holm12"] = multipletests(t.loc[prim, "p_clust"], method="holm")[1]
    t.to_csv(SCRATCH / "study_level_stats_harmonised.csv", index=False)
    ref = pd.read_csv(OUT_DIR / "study_level_stats_harmonised.csv")
    compare_numeric(t["p_clust"], ref["p_clust"], "study_level_stats.p_clust")
    compare_numeric(t["p_raw"], ref["p_raw"], "study_level_stats.p_raw")

    print("== R1a top1/top5/median from committed ranks ==", flush=True)
    ref_r1a = pd.read_csv(EXT_OUT / "r1a_random_baseline_check.csv")
    k562 = pd.read_csv(EXT_OUT / "replogle2022_k562_gwps_ranks.csv")
    rpe = pd.read_csv(EXT_OUT / "replogle2022_rpe1_ranks.csv")
    cohorts_r1a = {
        "KnockTF_Dev_266": d[d["split_new"] == "Dev"],
        "KnockTF_Unseen_64": d[d["split_new"] == "Unseen"],
        "Replogle_K562_Primary_267": k562[k562["max_abs_blind"] >= 0.5],
        "Replogle_K562_All_KD0.5_514": k562,
        "Replogle_RPE1_Primary_76": rpe[rpe["max_abs_blind"] >= 0.5],
    }
    methods = [("M1_ULM", "r_m1_ulm"), ("M5_MultiModal_Static", "r_m5_mm_static"), ("M9_Selective", "r_m9_determ")]
    for cname, sub in cohorts_r1a.items():
        for mname, col in methods:
            r = sub[col].to_numpy()
            hit = ref_r1a[(ref_r1a["cohort"] == cname) & (ref_r1a["method"] == mname)]
            if hit.empty:
                _fail(f"R1a missing {cname} {mname}")
                continue
            top1 = round(float(np.mean(r == 1) * 100), 4)
            # r1a stores various precision; compare hits if present else pct
            if "top1_hits" in hit.columns:
                if int(np.sum(r == 1)) != int(hit.iloc[0]["top1_hits"]):
                    _fail(f"R1a {cname} {mname} top1_hits")
            elif "top1_pct" in hit.columns:
                compare_numeric(pd.Series([float(np.mean(r == 1) * 100)]), hit["top1_pct"], f"R1a.{cname}.{mname}.top1_pct")


def stage_r3_strict_counts(got_dev: pd.DataFrame, got_unseen: pd.DataFrame) -> None:
    print("== R3 StrictBlinded M1/M5/M9 counts vs audit CSV ==", flush=True)
    ref = pd.read_csv(AUDIT / "r3_method_performance_by_blinding_and_metric.csv")
    ref = ref[ref["blinding_regime"] == "3_StrictBlinded_1888TFs"]
    mapping = {
        ("KnockTF_Dev_266", "M1_ULM"): (got_dev, "r_m1_ulm"),
        ("KnockTF_Dev_266", "M5_MultiModal_Static"): (got_dev, "r_m5_mm_static"),
        ("KnockTF_Dev_266", "M9_Selective_Deconf"): (got_dev, "r_m9_determ"),
        ("KnockTF_Unseen_64", "M1_ULM"): (got_unseen, "r_m1_ulm"),
        ("KnockTF_Unseen_64", "M5_MultiModal_Static"): (got_unseen, "r_m5_mm_static"),
        ("KnockTF_Unseen_64", "M9_Selective_Deconf"): (got_unseen, "r_m9_determ"),
    }
    k562 = pd.read_csv(EXT_OUT / "replogle2022_k562_gwps_ranks.csv")
    k562 = k562[k562["max_abs_blind"] >= 0.5]
    rpe = pd.read_csv(EXT_OUT / "replogle2022_rpe1_ranks.csv")
    rpe = rpe[rpe["max_abs_blind"] >= 0.5]
    mapping[("Replogle_K562_Primary_267", "M1_ULM")] = (k562, "r_m1_ulm")
    mapping[("Replogle_K562_Primary_267", "M5_MultiModal_Static")] = (k562, "r_m5_mm_static")
    mapping[("Replogle_K562_Primary_267", "M9_Selective_Deconf")] = (k562, "r_m9_determ")
    mapping[("Replogle_RPE1_Primary_76", "M1_ULM")] = (rpe, "r_m1_ulm")
    mapping[("Replogle_RPE1_Primary_76", "M5_MultiModal_Static")] = (rpe, "r_m5_mm_static")
    mapping[("Replogle_RPE1_Primary_76", "M9_Selective_Deconf")] = (rpe, "r_m9_determ")
    for (dset, method), (df, col) in mapping.items():
        hit = ref[(ref["dataset"] == dset) & (ref["method"] == method)]
        if hit.empty:
            _fail(f"R3 missing row {dset} {method}")
            continue
        r = df[col].to_numpy()
        top1 = int(np.sum(r == 1))
        top5 = int(np.sum(r <= 5))
        if top1 != int(hit.iloc[0]["top1_hits"]):
            _fail(f"R3 {dset} {method} top1 {top1} vs {int(hit.iloc[0]['top1_hits'])}")
        if top5 != int(hit.iloc[0]["top5_hits"]):
            _fail(f"R3 {dset} {method} top5 {top5} vs {int(hit.iloc[0]['top5_hits'])}")


def main() -> None:
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)
    print(f"scratch {SCRATCH}", flush=True)
    print("GEO local files only; no download.", flush=True)
    got_dev, got_unseen = stage_dev_unseen()
    stage_geo()
    stage_replogle()
    stage_r7()
    stage_stats_from_ranks()
    stage_r3_strict_counts(got_dev, got_unseen)
    report = SCRATCH / "r8_diff_report.txt"
    report.write_text("\n".join(DIFFS) + ("\n" if DIFFS else "ALL MATCH within 1e-9\n"))
    print(report.read_text())
    if DIFFS:
        raise SystemExit(f"R8a FAILED: {len(DIFFS)} diffs")
    print("R8a PASSED: all compared CSVs match within 1e-9")


if __name__ == "__main__":
    main()
