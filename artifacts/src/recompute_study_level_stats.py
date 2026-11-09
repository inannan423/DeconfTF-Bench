"""Recompute study-level statistics after recovering study IDs for the 63
KnockTF experiments whose Pubmed.ID was '-'.

Previously every '-' experiment shared one pseudo-study, which (i) collapsed 45
Dev experiments into one LOSO fold / permutation cluster and (ii) labelled all
18 '-' holdout experiments as "seen" because '-' also occurred in Dev.

Inputs : supplement_knocktf401_per_sample_ranks.csv, knocktf401_study_ids.csv
Outputs: knocktf401_ranks_with_study_ids.csv, study_level_stats_corrected.csv
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, "/Users/bytedance/codes/new-bishe/artifacts/src")
from verify_round4_accept_checks import exact_mcnemar_2s, study_clustered_sign_flip_perm, wilcoxon_both_modes

DATA_DIR = Path("/Users/bytedance/codes/new-bishe/artifacts/data/knocktf")
OUT_DIR = Path("/Users/bytedance/codes/new-bishe/artifacts/results/cand6_deconftf")

# Sub-series of one KLF4/KLF7 corneal-epithelium study (all cited by PMID 28916725 in PMC)
SAME_STUDY_GSES = {"GSE94468": "GSE94468-70", "GSE94469": "GSE94468-70", "GSE94470": "GSE94468-70"}


def load():
    d = pd.read_csv(OUT_DIR / "supplement_knocktf401_per_sample_ranks.csv", dtype={"pubmed_id": str})
    s = pd.read_csv(DATA_DIR / "knocktf401_study_ids.csv", dtype=str)
    s["study_id"] = [SAME_STUDY_GSES.get(g, sid) for g, sid in zip(s["profile_id"], s["study_id"])]
    d = d.merge(s[["exp_id", "profile_id", "study_id", "study_source"]], on="exp_id", how="left")
    assert d["study_id"].notna().all()

    dev_studies = set(d.loc[d["cohort"].str.startswith("Dev"), "study_id"])
    d["split_old"] = d["cohort"].map({"Dev_266_StrongKD": "Dev", "Holdout_51_UnseenStudies": "Unseen",
                                      "Holdout_84_SeenStudies_WeakKD": "Seen"})
    d["split_new"] = np.where(d["split_old"] == "Dev", "Dev",
                              np.where(d["study_id"].isin(dev_studies), "Seen", "Unseen"))
    return d


def prior_loso(d, cand_tfs):
    """Cell-line/tissue frequency prior, leaving out the experiment's own study."""
    ranks = []
    for _, row in d.iterrows():
        other = d[d["study_id"] != row["study_id"]]
        bio_c = other.loc[other["biosample"] == row["biosample"], "true_tf"].value_counts()
        tis_c = other.loc[other["tissue"] == row["tissue"], "true_tf"].value_counts()
        glob_c = other["true_tf"].value_counts()
        sc = {c: 100.0 * bio_c.get(c, 0) + 10.0 * tis_c.get(c, 0) + 1.0 * glob_c.get(c, 0) for c in cand_tfs}
        order = sorted(cand_tfs, key=lambda c: (-sc[c], c))
        ranks.append(order.index(row["true_tf"]) + 1)
    return np.array(ranks)


def endpoint_tests(sub, base_col, cluster_col):
    r9 = sub["r_m9_determ"].values
    rb = sub[base_col].values
    sid = sub[cluster_col].astype(str).values
    w1, l1, p1 = exact_mcnemar_2s((r9 == 1).astype(int), (rb == 1).astype(int))
    w5, l5, p5 = exact_mcnemar_2s((r9 <= 5).astype(int), (rb <= 5).astype(int))
    wx = wilcoxon_both_modes(r9, rb)
    _, c1 = study_clustered_sign_flip_perm((r9 == 1).astype(float), (rb == 1).astype(float), sid)
    _, c5 = study_clustered_sign_flip_perm((r9 <= 5).astype(float), (rb <= 5).astype(float), sid)
    _, cl = study_clustered_sign_flip_perm(-np.log2(r9.astype(float)), -np.log2(rb.astype(float)), sid)
    return [
        ("Top-1", f"{w1}/{l1}", p1, c1),
        ("Top-5", f"{w5}/{l5}", p5, c5),
        ("Log2Rank", f"{wx['mean_bits']:+.3f}b (med {wx['median_bits']:+.2f}b)", wx["p_wilcox"], cl),
    ]


def main():
    d = load()
    with open(OUT_DIR / "knocktf_precomputed_bundle.pkl", "rb") as f:
        cand_tfs = pickle.load(f)["cand_tfs"]

    d["r_prior_loso_old"] = d["r_prior_loso"]
    d["r_prior_loso"] = prior_loso(d, cand_tfs)
    d.to_csv(OUT_DIR / "knocktf401_ranks_with_study_ids.csv", index=False)

    print("=== Study recovery ===")
    print(d.groupby(["cohort", "study_source"]).size().to_string())
    print("\n=== Split reassignment (old -> new) ===")
    print(pd.crosstab(d["split_old"], d["split_new"]).to_string())
    moved = d[(d["split_old"] == "Seen") & (d["split_new"] == "Unseen")]
    print("\nMoved Seen -> Unseen:", moved[["exp_id", "true_tf", "profile_id", "study_id"]].to_string(index=False), sep="\n")

    cohorts = {
        "Dev_266": d[d["split_new"] == "Dev"],
        "Unseen_old51": d[d["split_old"] == "Unseen"],
        "Unseen_new": d[d["split_new"] == "Unseen"],
        "Seen_new": d[d["split_new"] == "Seen"],
        "Holdout_135": d[d["split_new"] != "Dev"],
        "Combined_401": d,
    }
    print("\n=== Cohort sizes, study counts and Top-1/Top-5 ===")
    for name, sub in cohorts.items():
        vc = sub["study_id"].value_counts()
        line = f"{name:13s} N={len(sub):3d} studies={sub['study_id'].nunique():3d} (1-exp={int((vc == 1).sum())}, max={vc.max()})"
        for col in ["r_m0_self_logfc", "r_m1_ulm", "r_m2b_viper_shadow", "r_m5_mm_static", "r_prior_loso", "r_m9_determ"]:
            line += f" | {col.replace('r_', '')} {int((sub[col] == 1).sum())}/{int((sub[col] <= 5).sum())}"
        print(line)

    rows = []
    for cname in ["Unseen_new", "Combined_401", "Unseen_old51", "Holdout_135", "Dev_266", "Seen_new"]:
        for base in ["r_m1_ulm", "r_m2b_viper_shadow"]:
            for ep, stat, p, pc in endpoint_tests(cohorts[cname], base, "study_id"):
                rows.append({"cohort": cname, "base": base, "endpoint": ep, "stat": stat, "p_raw": p, "p_clust": pc})
        for ep, stat, p, pc in endpoint_tests(cohorts[cname], "r_m5_mm_static", "study_id"):
            rows.append({"cohort": cname, "base": "r_m5_mm_static", "endpoint": ep, "stat": stat, "p_raw": p, "p_clust": pc})
        for ep, stat, p, pc in endpoint_tests(cohorts[cname], "r_prior_loso", "study_id"):
            rows.append({"cohort": cname, "base": "r_prior_loso", "endpoint": ep, "stat": stat, "p_raw": p, "p_clust": pc})
    t = pd.DataFrame(rows)

    # Primary 12-test family: M9 vs ULM/VIPER x 3 endpoints on Unseen_new + Combined_401,
    # using the study-clustered permutation p (the conservative choice adopted in round 5).
    prim = t["cohort"].isin(["Unseen_new", "Combined_401"]) & t["base"].isin(["r_m1_ulm", "r_m2b_viper_shadow"])
    t["p_clust_Holm12"] = np.nan
    t.loc[prim, "p_clust_Holm12"] = multipletests(t.loc[prim, "p_clust"], method="holm")[1]
    t["p_raw_Holm12"] = np.nan
    t.loc[prim, "p_raw_Holm12"] = multipletests(t.loc[prim, "p_raw"], method="holm")[1]
    t.to_csv(OUT_DIR / "study_level_stats_corrected.csv", index=False)

    pd.set_option("display.width", 250)
    print("\n=== Paired tests (M9 vs base), study-clustered permutation n=20000 ===")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
