"""Primary and secondary statistics on harmonised holdout ranks.

Dev ranks are unchanged (the refactored pipeline reproduces them exactly); holdout ranks
for M1, M2b, M5 and M9 come from rescore_holdout_harmonised.py. Same tests, study IDs and
12-test primary family as recompute_study_level_stats.py.

Output: results/cand6_deconftf/study_level_stats_harmonised.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, "/Users/bytedance/codes/new-bishe/artifacts/src")
from recompute_study_level_stats import endpoint_tests
from verify_round4_accept_checks import study_clustered_sign_flip_perm, wilcoxon_both_modes

OUT_DIR = Path("/Users/bytedance/codes/new-bishe/artifacts/results/cand6_deconftf")
COLS = ["r_m1_ulm", "r_m2b_viper_shadow", "r_m5_mm_static", "r_m9_determ"]


def main():
    d = pd.read_csv(OUT_DIR / "knocktf401_ranks_with_study_ids.csv").set_index("exp_id")
    h = pd.read_csv(OUT_DIR / "holdout135_harmonised_ranks.csv").set_index("exp_id")
    d.loc[h.index, COLS] = h[COLS].values
    d["in_recalled"] = np.nan
    d.loc[h.index, "in_recalled"] = h["in_recalled"].values

    cohorts = {
        "Unseen": d[d["split_new"] == "Unseen"],
        "Combined_401": d,
        "Holdout_135": d[d["split_new"] != "Dev"],
        "Seen": d[d["split_new"] == "Seen"],
    }
    print("=== Top-1 / Top-5 / Top-10 / MRR (harmonised holdout) ===")
    for name, sub in cohorts.items():
        line = f"{name:13s} N={len(sub):3d}"
        for col in ["r_m0_self_logfc", "r_prior_loso"] + COLS:
            r = sub[col]
            line += (f" | {col.replace('r_', '')} {int((r == 1).sum())}/{int((r <= 5).sum())}"
                     f"/{int((r <= 10).sum())} mrr={np.mean(1 / r):.3f}")
        print(line)

    rows = []
    for cname, sub in cohorts.items():
        for base in ["r_m1_ulm", "r_m2b_viper_shadow", "r_m5_mm_static", "r_prior_loso"]:
            for ep, stat, p, pc in endpoint_tests(sub, base, "study_id"):
                rows.append({"cohort": cname, "base": base, "endpoint": ep, "stat": stat, "p_raw": p, "p_clust": pc})
    t = pd.DataFrame(rows)
    prim = t["cohort"].isin(["Unseen", "Combined_401"]) & t["base"].isin(["r_m1_ulm", "r_m2b_viper_shadow"])
    t["p_clust_Holm12"] = np.nan
    t.loc[prim, "p_clust_Holm12"] = multipletests(t.loc[prim, "p_clust"], method="holm")[1]
    t.to_csv(OUT_DIR / "study_level_stats_harmonised.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n=== Paired tests, M9 vs base (study-clustered permutation, n=20000) ===")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    print("\n=== Recalled-only rank sensitivity ===")
    for cname in ["Unseen", "Holdout_135"]:
        sub = cohorts[cname]
        sub = sub[sub["in_recalled"] == 1]
        for base in ["r_m1_ulm", "r_m2b_viper_shadow"]:
            w = wilcoxon_both_modes(sub["r_m9_determ"].values, sub[base].values)
            _, pc = study_clustered_sign_flip_perm(-np.log2(sub["r_m9_determ"].values.astype(float)),
                                                   -np.log2(sub[base].values.astype(float)),
                                                   sub["study_id"].astype(str).values)
            print(f"  {cname} recalled n={len(sub)} vs {base}: mean {w['mean_bits']:+.2f}b "
                  f"median {w['median_bits']:+.2f}b p={w['p_wilcox']:.2g} p_clust={pc:.2g}")


if __name__ == "__main__":
    main()
