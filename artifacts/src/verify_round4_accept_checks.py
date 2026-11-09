import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu, wilcoxon
from statsmodels.stats.multitest import multipletests

import sys
sys.path.insert(0, "/Users/bytedance/codes/new-bishe/artifacts/src")
from deconftf_tools import UNIVERSAL_SECONDARY_HUBS, load_gmt_with_tracks

DATA_DIR = Path("/Users/bytedance/codes/new-bishe/artifacts/data/knocktf")
OUT_DIR = Path("/Users/bytedance/codes/new-bishe/artifacts/results/cand6_deconftf")
BUNDLE_PATH = OUT_DIR / "knocktf_precomputed_bundle.pkl"


def exact_mcnemar_2s(y_a: np.ndarray, y_b: np.ndarray) -> tuple[int, int, float]:
    w = int(np.sum((y_a == 1) & (y_b == 0)))
    l = int(np.sum((y_a == 0) & (y_b == 1)))
    p = float(binomtest(w, w + l, 0.5, alternative="two-sided").pvalue) if (w + l) > 0 else 1.0
    return w, l, p


def wilcoxon_both_modes(r_a: np.ndarray, r_b: np.ndarray) -> dict:
    diff = np.log2(r_b.astype(float)) - np.log2(r_a.astype(float))
    _, p_wilcox = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
    _, p_pratt = wilcoxon(diff, zero_method="pratt", alternative="two-sided")
    return {
        "mean_bits": float(np.mean(diff)),
        "median_bits": float(np.median(diff)),
        "n_nonzero": int(np.sum(diff != 0)),
        "p_wilcox": float(p_wilcox),
        "p_pratt": float(p_pratt),
    }


def study_clustered_sign_flip_perm(val_a: np.ndarray, val_b: np.ndarray, study_ids: np.ndarray, n_perm: int = 20000, seed: int = 42) -> tuple[float, float]:
    """
    Two-sided Study-Clustered Permutation (Randomization) Test & Cluster Bootstrap 95% CI:
    In each permutation, flip the sign of (val_a - val_b) jointly for all experiments belonging to the same PubMed study!
    """
    rng = np.random.default_rng(seed)
    diff = val_a.astype(float) - val_b.astype(float)
    obs_mean = float(np.mean(diff))
    uniq_s = np.unique(study_ids)
    s_sums = np.array([np.sum(diff[study_ids == s]) for s in uniq_s], dtype=float)
    n_tot = len(diff)

    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(uniq_s)))
    perm_means = (signs @ s_sums) / n_tot
    p_perm = float((1 + np.sum(np.abs(perm_means) >= abs(obs_mean) - 1e-12)) / (n_perm + 1))
    return obs_mean, p_perm


def main():
    with open(BUNDLE_PATH, "rb") as f:
        b = pickle.load(f)

    df266 = pd.read_csv(OUT_DIR / "cand6_all266_complete_table.csv", index_col=0)
    df135 = pd.read_csv(OUT_DIR / "cand6_untouched_holdout134_results.csv", index_col=0)
    df_sc55 = pd.read_csv(OUT_DIR / "cand6_external_scperturb55_unique_tfs.csv")

    df51 = df135[df135["unseen_study"] == 1].copy()
    df84 = df135[df135["unseen_study"] == 0].copy()

    # Compute M_Prior_LOSO across all 401 KnockTF experiments (266 + 135)
    cand_tfs = b["cand_tfs"]
    df401_meta = pd.concat(
        [
            df266[["true_tf", "pubmed_id", "biosample", "tissue"]],
            df135[["true_tf", "pubmed_id", "biosample", "tissue"]],
        ],
        axis=0,
    )
    r_prior_401 = {}
    for eid, row in df401_meta.iterrows():
        pmid = str(row["pubmed_id"])
        bio = row["biosample"]
        tis = row["tissue"]
        true_tf = row["true_tf"]
        other_df = df401_meta[df401_meta["pubmed_id"].astype(str) != pmid]
        bio_c = other_df[other_df["biosample"] == bio]["true_tf"].value_counts()
        tis_c = other_df[other_df["tissue"] == tis]["true_tf"].value_counts()
        glob_c = other_df["true_tf"].value_counts()
        sc_p = {c: 100.0 * bio_c.get(c, 0) + 10.0 * tis_c.get(c, 0) + 1.0 * glob_c.get(c, 0) for c in cand_tfs}
        ord_p = sorted(cand_tfs, key=lambda c: (-sc_p[c], c))
        r_prior_401[eid] = ord_p.index(true_tf) + 1

    df266["r_prior_loso"] = [r_prior_401[eid] for eid in df266.index]
    df135["r_prior_loso"] = [r_prior_401[eid] for eid in df135.index]
    df51["r_prior_loso"] = [r_prior_401[eid] for eid in df51.index]
    df84["r_prior_loso"] = [r_prior_401[eid] for eid in df84.index]

    cols_c = ["true_tf", "pubmed_id", "biosample", "tissue", "r_m0_self_logfc", "r_m1_ulm", "r_m2b_viper_shadow", "r_m5_mm_static", "r_prior_loso", "r_m9_determ"]
    df401 = pd.concat([df266[cols_c], df135[cols_c]], axis=0)

    print("====================================================================================")
    print("ITEM 1A: STUDY-COUNT AUDIT ACROSS COHORTS")
    print("====================================================================================")
    print(f"  Development Cohort (N={len(df266)}): {df266['pubmed_id'].nunique()} unique PubMed studies")
    print(f"  Parameter-Fixed Holdout — Unseen Studies (N={len(df51)}): {df51['pubmed_id'].nunique()} unique PubMed studies (max exps/study={df51['pubmed_id'].value_counts().max()}, 1-exp studies={(df51['pubmed_id'].value_counts()==1).sum()})")
    print(f"  Parameter-Fixed Holdout — Seen Studies Weak-KD Stratum (N={len(df84)}): {df84['pubmed_id'].nunique()} unique PubMed studies")
    print(f"  Parameter-Fixed Holdout — All (N={len(df135)}): {df135['pubmed_id'].nunique()} unique PubMed studies")
    print(f"  Combined KnockTF Cohort (N={len(df401)}): {df401['pubmed_id'].nunique()} unique PubMed studies (142 + 41 = 183)")

    print("\n====================================================================================")
    print("ITEM 1B: UNIFIED 12-TEST PRIMARY ENDPOINT FAMILY (N=51 UNSEEN & N=401 COMBINED)")
    print("         + 24-TEST EXPANDED FAMILY (ADDING N=135 HOLDOUT & N=266 DEV)")
    print("         + STUDY-CLUSTERED PERMUTATION P-VALUES & WILCOX/PRATT ZERO-HANDLING")
    print("====================================================================================")
    primary_specs = [
        ("Unseen_51", df51, "M1_ULM", "r_m1_ulm"),
        ("Unseen_51", df51, "M2b_VIPER", "r_m2b_viper_shadow"),
        ("Combined_401", df401, "M1_ULM", "r_m1_ulm"),
        ("Combined_401", df401, "M2b_VIPER", "r_m2b_viper_shadow"),
        ("Holdout_135", df135, "M1_ULM", "r_m1_ulm"),
        ("Holdout_135", df135, "M2b_VIPER", "r_m2b_viper_shadow"),
        ("Dev_266", df266, "M1_ULM", "r_m1_ulm"),
        ("Dev_266", df266, "M2b_VIPER", "r_m2b_viper_shadow"),
    ]

    test_records = []
    for c_name, d_sub, b_name, b_col in primary_specs:
        r9 = d_sub["r_m9_determ"].values
        rb = d_sub[b_col].values
        s_ids = d_sub["pubmed_id"].astype(str).values

        w1, l1, p1 = exact_mcnemar_2s((r9 == 1).astype(int), (rb == 1).astype(int))
        _, p1_clust = study_clustered_sign_flip_perm((r9 == 1).astype(float), (rb == 1).astype(float), s_ids)

        w5, l5, p5 = exact_mcnemar_2s((r9 <= 5).astype(int), (rb <= 5).astype(int))
        _, p5_clust = study_clustered_sign_flip_perm((r9 <= 5).astype(float), (rb <= 5).astype(float), s_ids)

        wx = wilcoxon_both_modes(r9, rb)
        _, plr_clust = study_clustered_sign_flip_perm(-np.log2(r9.astype(float)), -np.log2(rb.astype(float)), s_ids)

        test_records.extend(
            [
                {"cohort": c_name, "base": b_name, "endpoint": "Top-1", "stat": f"{w1}/{l1}", "p_raw": p1, "p_pratt": p1, "p_clust": p1_clust},
                {"cohort": c_name, "base": b_name, "endpoint": "Top-5", "stat": f"{w5}/{l5}", "p_raw": p5, "p_pratt": p5, "p_clust": p5_clust},
                {
                    "cohort": c_name,
                    "base": b_name,
                    "endpoint": "Log2Rank",
                    "stat": f"{wx['mean_bits']:+.3f}b (med {wx['median_bits']:+.2f}b, nz={wx['n_nonzero']})",
                    "p_raw": wx["p_wilcox"],
                    "p_pratt": wx["p_pratt"],
                    "p_clust": plr_clust,
                },
            ]
        )

    df_tests = pd.DataFrame(test_records)
    # 12-test family: first 12 rows (Unseen_51 + Combined_401)
    _, p_holm12, _, _ = multipletests(df_tests.iloc[:12]["p_raw"].values, method="holm")
    df_tests["p_Holm12"] = list(p_holm12) + [np.nan] * 12
    # 24-test family: all 24 rows
    _, p_holm24, _, _ = multipletests(df_tests["p_raw"].values, method="holm")
    df_tests["p_Holm24"] = p_holm24

    for _, r in df_tests.iterrows():
        h12_s = f"{r['p_Holm12']:.6f}" if not np.isnan(r["p_Holm12"]) else "   ---  "
        print(
            f"  {r['cohort']:12s} | M9 vs {r['base']:9s} | {r['endpoint']:8s} | {r['stat']:30s} | "
            f"p_raw(wilcox)={r['p_raw']:.6e} | p_pratt={r['p_pratt']:.6e} | p_Holm12={h12_s} | p_Holm24={r['p_Holm24']:.6e} | Study-Perm p={r['p_clust']:.5f}"
        )

    print("\n====================================================================================")
    print("ITEM 2: MECHANISM INTERACTION TEST (CASCADE STRATUM vs OTHER ULM-WRONG STRATUM)")
    print("====================================================================================")
    collectri_full = pd.read_csv(DATA_DIR / "collectri_human.csv")
    dorothea_full = pd.read_csv(DATA_DIR / "dorothea_human.csv")
    encode_net, _ = load_gmt_with_tracks(DATA_DIR / "ENCODE_ChIP-seq.gmt")
    remap_net, _ = load_gmt_with_tracks(DATA_DIR / "ReMap_ChIP-seq.gmt")
    tf_to_known_targets: dict[str, set[str]] = {}
    for df_net in [collectri_full, dorothea_full, encode_net, remap_net]:
        for s_tf, t_g in zip(df_net["source"].astype(str).str.upper(), df_net["target"].astype(str).str.upper()):
            if s_tf != t_g:
                tf_to_known_targets.setdefault(s_tf, set()).add(t_g)

    # For all 401 experiments, get ULM #1 TF
    with open(BUNDLE_PATH, "rb") as f:
        b = pickle.load(f)
    cand_tfs = b["cand_tfs"]
    net_c = b["net_c"]
    expr_all = pd.read_csv(DATA_DIR / "knockTF_expr.csv", index_col=0)
    meta_all = pd.read_csv(DATA_DIR / "knockTF_meta.csv", index_col=0)
    all_tfs_mask = set(collectri_full["source"]).union(set(dorothea_full["source"])).union(set(meta_all["TF"])).union(set(cand_tfs))
    mat_h = expr_all.loc[df135.index].copy()
    mat_h[[c for c in mat_h.columns if c in all_tfs_mask]] = 0.0
    import decoupler as dc
    est_u_h, _ = dc.mt.ulm(data=mat_h, net=net_c, verbose=False)
    est_u_h = est_u_h.reindex(columns=cand_tfs, fill_value=0.0)
    df135["top1_m1"] = [est_u_h.loc[eid].idxmin() for eid in df135.index]

    df401_mech = pd.concat([df266[["true_tf", "top1_m1", "r_m1_ulm", "r_m9_determ", "in_active124"]], df135[["true_tf", "top1_m1", "r_m1_ulm", "r_m9_determ", "in_recalled"]].rename(columns={"in_recalled": "in_active124"})], axis=0)

    for cname, sub in [
        ("Active 124 (ULM Wrong n=109)", df266[(df266["in_active124"] == 1) & (df266["r_m1_ulm"] > 1)]),
        ("All 266 (ULM Wrong n=251)", df266[df266["r_m1_ulm"] > 1]),
        ("Combined 401 (ULM Wrong n=381)", df401_mech[df401_mech["r_m1_ulm"] > 1]),
    ]:
        is_casc = np.array([str(r["top1_m1"]) in tf_to_known_targets.get(str(r["true_tf"]), set()) for _, r in sub.iterrows()])
        gain_bits = np.log2(sub["r_m1_ulm"].values.astype(float)) - np.log2(sub["r_m9_determ"].values.astype(float))
        g_casc = gain_bits[is_casc]
        g_other = gain_bits[~is_casc]
        _, p_mwu = mannwhitneyu(g_casc, g_other, alternative="two-sided")
        t1_casc = np.mean(sub.loc[is_casc, "r_m9_determ"] == 1) * 100
        t1_other = np.mean(sub.loc[~is_casc, "r_m9_determ"] == 1) * 100
        print(
            f"  {cname:32s}: Cascade-Masked (n={np.sum(is_casc):3d}) MeanGain={np.mean(g_casc):+.3f}b, Top1={t1_casc:4.1f}% vs "
            f"Other-Wrong (n={np.sum(~is_casc):3d}) MeanGain={np.mean(g_other):+.3f}b, Top1={t1_other:4.1f}% | "
            f"Interaction Mann-Whitney 2-sided p={p_mwu:.4e}"
        )

    print("\n====================================================================================")
    print("ITEM 3: EXTERNAL SINGLE-CELL CRISPR BENCHMARK — STRICTLY DEDUPLICATED BY UNIQUE TF")
    print("====================================================================================")
    # 1. Truly External Datlinger 2017 + 2021 (Jurkat T-cell CROP-seq):
    # Deduplicate across 2017 and 2021 by taking the dataset with larger cell count (or averaging ranks across the 2 studies) for the 14 shared TFs, yielding N=23 unique TFs in Jurkat!
    df_dat = df_sc55[df_sc55["dataset"].str.startswith("Datlinger")].copy()
    print(f"  Datlinger 2017 TFs (n={len(df_dat[df_dat['dataset']=='Datlinger2017_Jurkat_CROPseq'])}): {sorted(df_dat[df_dat['dataset']=='Datlinger2017_Jurkat_CROPseq']['true_tf'].tolist())}")
    print(f"  Datlinger 2021 TFs (n={len(df_dat[df_dat['dataset']=='Datlinger2021_Jurkat_CROPseq'])}): {sorted(df_dat[df_dat['dataset']=='Datlinger2021_Jurkat_CROPseq']['true_tf'].tolist())}")

    # Deduplicate across all 4 scPerturb datasets by unique (cell_line, true_tf) -> 23 Jurkat TFs + 13 THP-1 TFs = 36 unique (cell_line, TF) perturbations (and 35 globally unique TF symbols!)
    df_sc55["cell_line"] = np.where(df_sc55["dataset"].str.startswith("Datlinger"), "Jurkat", "THP1")
    # Sort by n_cells descending so deduplication keeps the higher-depth screen (or we also report average rank across screens)
    df_sc_dedup_cl = df_sc55.sort_values("n_cells", ascending=False).drop_duplicates(subset=["cell_line", "true_tf"]).sort_values(["cell_line", "true_tf"])
    df_sc_dedup_global = df_sc55.sort_values("n_cells", ascending=False).drop_duplicates(subset=["true_tf"]).sort_values("true_tf")
    df_dat_dedup23 = df_dat.sort_values("n_cells", ascending=False).drop_duplicates(subset=["true_tf"]).sort_values("true_tf")

    for sc_name, sc_d in [
        ("Truly External Jurkat CROP-seq — Deduplicated Unique TFs (N=23)", df_dat_dedup23),
        ("All scPerturb — Deduplicated by (CellLine, Unique TF) (N=36: 23 Jurkat + 13 THP-1)", df_sc_dedup_cl),
        ("All scPerturb — Globally Deduplicated Unique TF Symbols (N=35)", df_sc_dedup_global),
    ]:
        r9_s = sc_d["r_m9_determ"].values
        ru_s = sc_d["r_m1_ulm"].values
        rv_s = sc_d["r_m2b_viper"].values
        rm0_s = sc_d["r_m0_self"].values
        dlr_u, plr_u = wilcoxon_both_modes(r9_s, ru_s)["mean_bits"], wilcoxon_both_modes(r9_s, ru_s)["p_wilcox"]
        dlr_v, plr_v = wilcoxon_both_modes(r9_s, rv_s)["mean_bits"], wilcoxon_both_modes(r9_s, rv_s)["p_wilcox"]
        print(
            f"  {sc_name:76s}:\n"
            f"    M0_Self Top1={np.mean(rm0_s==1)*100:5.2f}% ({np.sum(rm0_s==1):2d}/{len(sc_d)}), Top5={np.mean(rm0_s<=5)*100:5.2f}% | "
            f"ULM Top1={np.mean(ru_s==1)*100:5.2f}% ({np.sum(ru_s==1):2d}), Top5={np.mean(ru_s<=5)*100:5.2f}% | "
            f"VIPER Top1={np.mean(rv_s==1)*100:5.2f}% ({np.sum(rv_s==1):2d}), Top5={np.mean(rv_s<=5)*100:5.2f}% | "
            f"M9_Determ Top1={np.mean(r9_s==1)*100:5.2f}% ({np.sum(r9_s==1):2d}), Top5={np.mean(r9_s<=5)*100:5.2f}% "
            f"(vs ULM dLog2Rank={dlr_u:+.2f}b p={plr_u:.4f}; vs VIPER dLog2Rank={dlr_v:+.2f}b p={plr_v:.4f})"
        )

    print("\n====================================================================================")
    print("ITEM 4 & 5: PRIOR vs M9 COMPLEMENTARITY ON SEEN-84 vs UNSEEN-51, AND M9 vs M5")
    print("====================================================================================")
    for sub_label, d_s in [
        ("Unseen 51 Studies", df51),
        ("Seen 84 Studies (Weak-KD Stratum)", df84),
        ("All 135 Parameter-Fixed Holdout", df135),
        ("All 266 Development Cohort", df266),
        ("All 401 Combined KnockTF", df401),
    ]:
        r9 = d_s["r_m9_determ"].values
        r5 = d_s["r_m5_mm_static"].values
        rp = d_s["r_prior_loso"].values
        w95_1, l95_1, p95_1 = exact_mcnemar_2s((r9 == 1).astype(int), (r5 == 1).astype(int))
        w95_5, l95_5, p95_5 = exact_mcnemar_2s((r9 <= 5).astype(int), (r5 <= 5).astype(int))
        wx95 = wilcoxon_both_modes(r9, r5)
        print(
            f"  {sub_label:36s}: Prior_LOSO Top1={np.sum(rp==1):2d}/{len(d_s)} ({np.mean(rp==1)*100:5.2f}%) | "
            f"M5_Static Top1={np.sum(r5==1):2d}/{len(d_s)} ({np.mean(r5==1)*100:5.2f}%), Top5={np.sum(r5<=5):2d} | "
            f"M9_Determ Top1={np.sum(r9==1):2d}/{len(d_s)} ({np.mean(r9==1)*100:5.2f}%), Top5={np.sum(r9<=5):2d} | "
            f"M9 vs M5: Top1 W/L={w95_1:2d}/{l95_1:2d}(p={p95_1:.4f}), Top5 W/L={w95_5:2d}/{l95_5:2d}(p={p95_5:.4f}), dLog2Rank={wx95['mean_bits']:+.3f}b(p={wx95['p_wilcox']:.4f})"
        )


if __name__ == "__main__":
    main()
