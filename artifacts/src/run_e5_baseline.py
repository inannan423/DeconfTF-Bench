#!/usr/bin/env python3
"""Rebuild GEO, holdout-135, and Replogle K562 with the frozen scorer.

Unseen is scored inside the 135-experiment holdout and then sliced.
M10 ranks must match the published tables with zero differences.
This script does not call a model. It does not overwrite the published ranks.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/Users/bytedance/codes/new-bishe/artifacts/src")
from deconftf_tools import score_knocktf_experiments
from develop_m10_soft_gate import compute_bundle_coverage, m10_soft_order
from eval_p0_2_external import load_replogle_cohort
from run_e5_common import BUNDLE_DIR, OUT, REPLOGLE_STUDY, ROOT, assert_frozen

EXT = ROOT / "artifacts" / "results" / "cand6_deconftf" / "external"
AUDIT = ROOT / "artifacts" / "results" / "cand6_deconftf" / "audit"
DATA = ROOT / "artifacts" / "data" / "knocktf"
REPORT = OUT / "e5_baseline_report.txt"


def _rank(bundle, cov, exp_id, tf) -> int:
    order, _score = m10_soft_order(exp_id, bundle, cov_frac=cov, lambda_bind=1.0, lambda_hub=0.3)
    return order.index(tf) + 1


def _diff(got: dict, ref: pd.Series, label: str) -> int:
    missing = [i for i in ref.index if i not in got]
    changed = [i for i in ref.index if i in got and int(got[i]) != int(ref.loc[i])]
    extra = [i for i in got if i not in set(ref.index)]
    print(
        f"{label}: missing {len(missing)}, changed {len(changed)}, extra {len(extra)}",
        flush=True,
    )
    if missing or changed or extra:
        print(" first mismatches:", (missing + changed + extra)[:8], flush=True)
    return len(missing) + len(changed) + len(extra)


def _slice(bundle: dict, ids: list[str]) -> dict:
    out = dict(bundle)
    frames = [
        "obs_sub", "mat_blind", "est_ulm", "est_mlm", "est_c_idf", "est_c_res",
        "est_d", "est_prog", "multi_modal_score", "chea3_meanrank", "chea3_toprank",
    ]
    for key in frames:
        out[key] = bundle[key].loc[list(ids)]
    out["z_libs"] = {name: frame.loc[list(ids)] for name, frame in bundle["z_libs"].items()}
    return out


def _save(name: str, bundle: dict, cov: float, sample: pd.DataFrame) -> None:
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    (BUNDLE_DIR / f"{name}.pkl").write_bytes(pickle.dumps(bundle))
    sample.to_csv(BUNDLE_DIR / f"{name}_sample.csv", index=False)
    (BUNDLE_DIR / f"{name}_cov.json").write_text(
        json.dumps({"cov": cov, "n": int(len(sample))}),
        encoding="utf-8",
    )


def geo(cand: list[str]) -> int:
    qc = pd.read_csv(EXT / "r2_geo_qc_pass.csv")
    lfc = pd.read_parquet(EXT / "r2_geo_lfc_matrix.parquet")
    published = pd.read_csv(EXT / "r2_geo_phase2_ranks.csv").set_index("exp_id")
    pass_qc = qc[qc["qc_reason"] == "PASS"].copy()
    ids = [c for c in pass_qc["contrast_id"].tolist() if c in lfc.columns]
    mat = lfc[ids].T.astype(float)
    mat.index = ids
    mat.columns = [str(c).upper() for c in mat.columns]
    if mat.columns.duplicated().any():
        mat = mat.T.groupby(level=0).sum().T
    mat = mat.fillna(0.0)
    obs = pass_qc.set_index("contrast_id").loc[ids].copy()
    obs.index.name = "exp_id"
    obs["TF"] = obs["target_tf"].astype(str).str.upper()
    obs["logFC"] = obs["target_log2fc"].astype(float)
    obs["Biosample.Name"] = obs["cell_line"].astype(str)
    obs["Tissue.Type"] = obs["cell_line"].astype(str)
    obs["study_id"] = obs["study_id"].astype(str)
    bundle = score_knocktf_experiments(mat, obs, cand_tfs=cand)
    cov = compute_bundle_coverage(bundle)
    got = {}
    for exp_id in bundle["obs_sub"].index:
        tf = str(bundle["obs_sub"].loc[exp_id, "TF"])
        got[str(exp_id)] = _rank(bundle, cov, exp_id, tf)
    n_diff = _diff(got, published["r_m10_soft"].astype(int), "GEO")
    sample = pd.DataFrame({
        "exp_id": [str(i) for i in bundle["obs_sub"].index],
        "study_id": [str(bundle["obs_sub"].loc[i, "study_id"]) for i in bundle["obs_sub"].index],
    })
    if len(sample) != 51:
        raise SystemExit(f"GEO bundle has {len(sample)} rows, not 51")
    _save("geo", bundle, cov, sample)
    return n_diff


def unseen(cand: list[str]) -> int:
    mat = pd.read_csv(DATA / "knockTF_expr.csv", index_col=0)
    obs = pd.read_csv(DATA / "knockTF_meta.csv", index_col=0)
    ranks = pd.read_csv(
        ROOT / "artifacts" / "results" / "cand6_deconftf" / "knocktf401_ranks_with_study_ids.csv",
        usecols=["exp_id", "study_id", "split_new"],
    ).set_index("exp_id")
    hold_ids = list(ranks.index[ranks["split_new"] != "Dev"])
    if len(hold_ids) != 135:
        raise SystemExit(f"holdout has {len(hold_ids)} rows, not 135")
    print(f"[Unseen] scoring holdout {len(hold_ids)}", flush=True)
    full = score_knocktf_experiments(mat.loc[hold_ids], obs.loc[hold_ids], cand_tfs=cand)
    if list(full["obs_sub"].index) != hold_ids:
        raise SystemExit("a holdout TF dropped out of the candidate set")
    cov_full = compute_bundle_coverage(full)
    ids = list(ranks.index[ranks["split_new"] == "Unseen"])
    if len(ids) != 64:
        raise SystemExit(f"Unseen has {len(ids)} rows, not 64")
    sliced = _slice(full, ids)
    cov = compute_bundle_coverage(sliced)
    if cov != cov_full:
        raise SystemExit(f"slicing changed coverage: {cov} vs {cov_full}")
    ref = pd.read_csv(AUDIT / "r7_m10_ranks_knocktf.csv")
    ref = ref[ref["cohort"] == "KnockTF_Unseen_64"].set_index("exp_id")["r_m10_soft"]
    got = {}
    for exp_id in ids:
        tf = str(sliced["obs_sub"].loc[exp_id, "TF"])
        got[str(exp_id)] = _rank(sliced, cov, exp_id, tf)
    n_diff = _diff(got, ref.astype(int), "Unseen")
    sample = pd.DataFrame({
        "exp_id": [str(i) for i in ids],
        "study_id": ranks.loc[ids, "study_id"].astype(str).tolist(),
    })
    _save("unseen", sliced, cov, sample)
    return n_diff


def replogle(cand: list[str]) -> int:
    published = pd.read_csv(EXT / "replogle2022_k562_gwps_ranks.csv")
    primary = published[
        (published["target_log2fc"] <= -0.5) & (published["max_abs_blind"] >= 0.5)
    ].copy()
    if len(primary) != 267:
        raise SystemExit(f"K562 primary has {len(primary)} rows, not 267")
    masked = set(
        pd.read_csv(
            ROOT / "artifacts" / "results" / "cand6_deconftf" / "supplement_masked_tfs_list.csv"
        ).iloc[:, 0].astype(str).str.upper()
    )
    meta = pd.read_csv(DATA / "knockTF_meta.csv", index_col=0)
    tf_to_class = meta.groupby("TF")["TF.Class"].first().to_dict()
    h5 = ROOT / "artifacts" / "data" / "external" / "K562_gwps_normalized_bulk_01.h5ad"
    print("[Replogle] loading K562", flush=True)
    mat, obs = load_replogle_cohort(h5, cand, masked, tf_to_class, "K562", "Haematopoietic")
    keep = [i for i in primary["exp_id"].tolist() if i in mat.index]
    if len(keep) != 267:
        raise SystemExit(f"K562 reload kept {len(keep)}, not 267")
    print("[Replogle] scoring K562", flush=True)
    bundle = score_knocktf_experiments(mat.loc[keep], obs.loc[keep], cand_tfs=cand)
    cov = compute_bundle_coverage(bundle)
    ref = pd.read_csv(AUDIT / "r7_m10_ranks_replogle.csv")
    ref = ref[ref["cohort"] == "K562"].set_index("exp_id")["r_m10_soft"]
    got = {}
    for exp_id in bundle["obs_sub"].index:
        tf = str(bundle["obs_sub"].loc[exp_id, "TF"])
        got[str(exp_id)] = _rank(bundle, cov, exp_id, tf)
    n_diff = _diff(got, ref.astype(int), "Replogle K562")
    sample = pd.DataFrame({
        "exp_id": [str(i) for i in bundle["obs_sub"].index],
        "study_id": REPLOGLE_STUDY,
    })
    _save("replogle", bundle, cov, sample)
    return n_diff


def main() -> None:
    assert_frozen()
    dev = pickle.loads(
        (ROOT / "artifacts" / "results" / "cand6_deconftf" / "knocktf_precomputed_bundle.pkl").read_bytes()
    )
    cand = list(dev["cand_tfs"])
    diffs = {"GEO": geo(cand), "Unseen": unseen(cand), "Replogle K562": replogle(cand)}
    total = sum(diffs.values())
    lines = [f"{name}: {n} M10 rank differences" for name, n in diffs.items()]
    lines.append(f"total differences: {total}")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    if total != 0:
        raise SystemExit("baseline M10 ranks differ; no model calls were made")


if __name__ == "__main__":
    main()
