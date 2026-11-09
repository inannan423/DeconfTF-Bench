"""Score the 135 holdout experiments with the exact development pipeline.

The original holdout run (run_nar_hard_gates.py) re-implemented the evidence layers:
a smaller blinding set, a different cell-cycle residualisation, a 4-term M5 and no
enhanceosome rule. This script reuses `score_knocktf_experiments` and `m9_order` from
deconftf_tools so Dev and holdout share one code path.

Steps
1. Rebuild the Dev bundle with the refactored function and check it reproduces the cached
   bundle and the published Dev M9 ranks exactly.
2. Score the holdout with the Dev candidate set and recompute M1, M2b (VIPER shadow), M5, M9.
3. Write per-experiment ranks; LLM-based M8 is not re-run (it would need new API calls).

Output: results/cand6_deconftf/holdout135_harmonised_ranks.csv
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/bytedance/codes/new-bishe/artifacts/src")
from audit_reviewer_controls import compute_viper_shadow_scores
from deconftf_tools import m9_order, score_knocktf_experiments

DATA_DIR = Path("/Users/bytedance/codes/new-bishe/artifacts/data/knocktf")
OUT_DIR = Path("/Users/bytedance/codes/new-bishe/artifacts/results/cand6_deconftf")


def rank_of(order, tf):
    return order.index(tf) + 1


def method_ranks(b, eids):
    viper = compute_viper_shadow_scores(b["mat_blind"], b["est_ulm"], b["W_c"], b["gene_to_idx"],
                                        b["tf_to_idx"], b["cand_tfs"])
    rows = []
    for eid in eids:
        tf = b["obs_sub"].loc[eid, "TF"]
        r_ulm = rank_of(list(b["est_ulm"].loc[eid].sort_values(ascending=True).index), tf)
        r_doro = rank_of(list(b["est_d"].loc[eid].sort_values(ascending=True).index), tf)
        r_chea = rank_of(list(b["chea3_toprank"].loc[eid].sort_values(ascending=True).index), tf)
        r_mm = rank_of(list(b["multi_modal_score"].loc[eid].sort_values(ascending=False).index), tf)
        rows.append({
            "exp_id": eid,
            "r_m1_ulm": r_ulm,
            "r_m2b_viper_shadow": rank_of(list(viper.loc[eid].sort_values(ascending=False).index), tf),
            "r_m5_mm_static": r_mm,
            "r_m9_determ": rank_of(m9_order(eid, b), tf),
            "r_m9_no_enh": rank_of(m9_order(eid, b, use_enhanceosome=False), tf),
            # same recall definition as the Dev "active 124" cohort
            "in_recalled": int(min(r_ulm, r_doro, r_chea, r_mm) <= 25),
        })
    return pd.DataFrame(rows).set_index("exp_id")


def check_dev_reproduction(mat, obs):
    cached = pickle.loads((OUT_DIR / "knocktf_precomputed_bundle.pkl").read_bytes())
    msk = (obs["logFC"] < -1.0) & (obs["TF"].isin(set(pd.read_csv(DATA_DIR / "collectri_human.csv")["source"])))
    fresh = score_knocktf_experiments(mat.loc[msk], obs.loc[msk])
    assert fresh["cand_tfs"] == cached["cand_tfs"]
    assert list(fresh["obs_sub"].index) == list(cached["obs_sub"].index)
    for key in ["est_ulm", "est_c_res", "multi_modal_score"]:
        diff = float(np.nanmax(np.abs(fresh[key].values - cached[key].values)))
        print(f"  Dev bundle {key}: max |fresh - cached| = {diff:.2e}")
        assert diff < 1e-4, key

    dev = pd.read_csv(OUT_DIR / "cand6_all266_complete_table.csv", index_col=0)
    r = method_ranks(fresh, list(dev.index))
    for col in ["r_m1_ulm", "r_m2b_viper_shadow", "r_m5_mm_static", "r_m9_determ"]:
        n_diff = int((r[col] != dev.loc[r.index, col]).sum())
        print(f"  Dev {col}: {n_diff}/266 ranks differ from published table")
        assert n_diff == 0, col
    return cached["cand_tfs"]


def main():
    mat = pd.read_csv(DATA_DIR / "knockTF_expr.csv", index_col=0)
    obs = pd.read_csv(DATA_DIR / "knockTF_meta.csv", index_col=0)

    print("[1] Checking that the refactored pipeline reproduces Dev exactly")
    cand_tfs = check_dev_reproduction(mat, obs)

    print("[2] Scoring the 135 holdout experiments with the Dev pipeline")
    ranks = pd.read_csv(OUT_DIR / "knocktf401_ranks_with_study_ids.csv").set_index("exp_id")
    hold_ids = list(ranks.index[ranks["split_new"] != "Dev"])
    hb = score_knocktf_experiments(mat.loc[hold_ids], obs.loc[hold_ids], cand_tfs=cand_tfs)
    assert list(hb["obs_sub"].index) == hold_ids, "a holdout TF dropped out of the candidate set"

    new = method_ranks(hb, hold_ids)
    old = ranks.loc[hold_ids, ["r_m1_ulm", "r_m2b_viper_shadow", "r_m5_mm_static", "r_m9_determ"]]
    out = new.join(old.add_suffix("_orig")).join(ranks.loc[hold_ids, ["split_new", "study_id", "true_tf"]])
    out.to_csv(OUT_DIR / "holdout135_harmonised_ranks.csv")

    for split in ["Unseen", "Seen"]:
        g = out[out["split_new"] == split]
        print(f"\n  {split} (N={len(g)})   Top-1 / Top-5   [original pipeline in brackets]")
        for col in ["r_m1_ulm", "r_m2b_viper_shadow", "r_m5_mm_static", "r_m9_determ"]:
            print(f"    {col:20s} {int((g[col] == 1).sum()):2d} / {int((g[col] <= 5).sum()):2d}"
                  f"   [{int((g[col + '_orig'] == 1).sum()):2d} / {int((g[col + '_orig'] <= 5).sum()):2d}]")
        print(f"    {'r_m9_no_enh':20s} {int((g['r_m9_no_enh'] == 1).sum()):2d} / {int((g['r_m9_no_enh'] <= 5).sum()):2d}")


if __name__ == "__main__":
    main()
