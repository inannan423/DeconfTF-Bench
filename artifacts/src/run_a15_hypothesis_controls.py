#!/usr/bin/env python3
"""Non-LLM hypothesis controls on KnockTF Dev 266. No LLM calls.

Three ways of proposing k hypotheses (k = 5 and 10), then
``verify_hypothesis`` and ``merge_verified_into_m10``.

(a) Random k TFs that have a human cell-line ChIP track. One draw of 10 per
    experiment from ``numpy`` Generator seed 42, experiments in bundle order.
    k = 5 is the first 5 of that draw.
(b) The first k TFs among M10 ranks 26-60 that have a human cell-line ChIP track.
(c) The k best frozen ChEA3 TopRank values. That ranking already pools ChEA3
    and Enrichr libraries. A recall-fail true TF has ChEA3 rank > 25, so k <= 10
    cannot contain it.

Does not read Unseen ranks, Replogle, or GEO expression or ranks.
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
    load_track_catalog,
    merge_verified_into_m10,
    prepare_hypothesis_resources,
    score_candidates,
    ulm_t_weighted,
    verify_hypothesis,
)

ROOT = Path("/Users/bytedance/codes/new-bishe")
OUT = ROOT / "artifacts" / "results" / "cand6_deconftf" / "agent"
BUNDLE_PATH = ROOT / "artifacts" / "results" / "cand6_deconftf" / "knocktf_precomputed_bundle.pkl"
FROZEN = {
    ROOT / "artifacts" / "src" / "deconftf_tools.py":
        "45939716963851d980d5767d9285832efd5b4b3537ce014b0e44f7affa225490",
    ROOT / "artifacts" / "src" / "develop_m10_soft_gate.py":
        "1e513f5236e56842e5ff96de3263175498ba9c1e6662ab8f228de3a1d7500677",
}


def assert_frozen() -> None:
    for path, expected in FROZEN.items():
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != expected:
            raise SystemExit(f"frozen hash changed: {path.name} {got}")


def check_weighted_ulm(bundle) -> float:
    y = bundle["mat_blind"].iloc[0].to_numpy(dtype=np.float64)
    genes = list(bundle["mat_blind"].columns)
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(y), size=80, replace=False))
    weights = rng.choice([-1.0, 1.0], size=80)
    local = ulm_t_weighted(y, idx, weights, min_targets=5)
    net = pd.DataFrame({"source": "TFX", "target": [genes[i] for i in idx], "weight": weights})
    data = pd.DataFrame(y.reshape(1, -1), index=["e"], columns=genes)
    est, _ = dc.mt.ulm(data=data, net=net, empty=False)
    gap = abs(local - float(est.iloc[0, 0]))
    if gap > 1e-6:
        raise SystemExit(f"weighted ULM does not match decoupler (abs diff {gap})")
    return gap


def main() -> None:
    assert_frozen()
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
    ulm_gap = check_weighted_ulm(bundle)

    human = [t for t in load_track_catalog() if t.cell and t.species == "HUMAN"]
    resources = prepare_hypothesis_resources(bundle, human)
    cand = list(bundle["cand_tfs"])
    chip_pool = sorted(
        tf for tf in cand if len(resources["chip"].get(tf) or []) >= 5
    )
    cov = compute_bundle_coverage(bundle)
    rng = np.random.default_rng(42)
    cache: dict = {}

    def cached_verify(exp_id, tf, m10_order):
        key = (exp_id, tf)
        if key not in cache:
            cache[key] = verify_hypothesis(exp_id, tf, bundle, resources)
        row = dict(cache[key])
        merged = merge_verified_into_m10(m10_order, [row])
        row["rank_if_merged_alone"] = merged.index(tf) + 1
        return row

    rows = []
    n_exp = len(bundle["obs_sub"].index)
    for i_exp, exp_id in enumerate(bundle["obs_sub"].index, start=1):
        if i_exp == 1 or i_exp % 50 == 0 or i_exp == n_exp:
            print(f"controls {i_exp}/{n_exp}", flush=True)
        obs = bundle["obs_sub"].loc[exp_id]
        true = str(obs["TF"])
        base = score_candidates(exp_id, bundle, cov_frac=cov)
        m10_order = base["m10_order"]
        r_ulm = list(base["ulm"].sort_values(ascending=True).index).index(true) + 1
        r_doro = list(bundle["est_d"].loc[exp_id].sort_values(ascending=True).index).index(true) + 1
        chea = list(bundle["chea3_toprank"].loc[exp_id].sort_values(ascending=True).index)
        r_chea = chea.index(true) + 1
        r_m5 = list(base["m5"].sort_values(ascending=False).index).index(true) + 1
        r_m10 = m10_order.index(true) + 1
        recalled = int(min(r_ulm, r_doro, r_chea, r_m5) <= 25)
        draw = list(rng.choice(chip_pool, size=10, replace=False))
        window = [tf for tf in m10_order[25:60] if tf in set(chip_pool)]
        proposals = {
            "random_chip": draw,
            "m10_rank_26_60_chip": window,
            "chea3_toprank": chea,
        }
        rec = {
            "exp_id": exp_id,
            "study_id": study.loc[exp_id],
            "true_tf": true,
            "in_recalled": recalled,
            "r_m10": r_m10,
            "r_chea": r_chea,
        }
        for name, ordered in proposals.items():
            for k in (5, 10):
                hyps = list(ordered[:k])
                verified = [cached_verify(exp_id, tf, m10_order) for tf in hyps]
                merged = merge_verified_into_m10(m10_order, verified)
                rank = merged.index(true) + 1
                proposed = int(true in hyps)
                true_row = next((v for v in verified if v["tf"] == true), None)
                rec[f"{name}_k{k}_rank"] = rank
                rec[f"{name}_k{k}_n"] = len(hyps)
                rec[f"{name}_k{k}_proposed"] = proposed
                rec[f"{name}_k{k}_true_t"] = None if true_row is None else true_row["t_stat"]
                rec[f"{name}_k{k}_true_source"] = None if true_row is None else true_row["source"]
        rows.append(rec)

    df = pd.DataFrame(rows)
    if int(df["in_recalled"].sum()) != 124:
        raise SystemExit(f"recall count is {int(df['in_recalled'].sum())}, expected 124")
    df.to_csv(OUT / "a15_hypothesis_controls_per_experiment.csv", index=False)

    fail = df[df["in_recalled"] == 0]
    ok = df[df["in_recalled"] == 1]
    summary_rows = []
    lines = [
        f"Dev 266 hypothesis controls. Weighted ULM vs decoupler abs diff {ulm_gap:.3e}.",
        "Primary rescue for a later LLM comparison: recall-fail AND M10 rank > 25 AND merged rank <= 25.",
        "Seed 42. k=5 is a prefix of the k=10 proposal. Merge inserts only t < 0.",
    ]
    for name in ("random_chip", "m10_rank_26_60_chip", "chea3_toprank"):
        for k in (5, 10):
            col = f"{name}_k{k}_rank"
            prop = f"{name}_k{k}_proposed"
            rescue = int((fail[col] <= 25).sum())
            incremental = int(((fail[col] <= 25) & (fail["r_m10"] > 25)).sum())
            harm_rec = int((ok[col] > 25).sum())
            harm_m10 = int(((df["r_m10"] <= 25) & (df[col] > 25)).sum())
            n_prop_fail = int(fail[prop].sum())
            summary_rows.append({
                "method": name,
                "k": k,
                "recall_fail_top25": rescue,
                "recall_fail_top25_pct": round(100 * rescue / len(fail), 2),
                "incremental_outside_m10": incremental,
                "incremental_pct_of_142": round(100 * incremental / len(fail), 2),
                "recall_fail_true_tf_proposed": n_prop_fail,
                "harm_among_124_recalled": harm_rec,
                "harm_recalled_pct": round(100 * harm_rec / len(ok), 2),
                "left_m10_top25": harm_m10,
            })
            lines.append(
                f"{name} k={k}: recall-fail Top-25 {rescue}/142 ({100 * rescue / len(fail):.2f}%), "
                f"incremental {incremental}/142, proposed the true TF in {n_prop_fail}/142 fail cases, "
                f"left an M10 Top-25 in {harm_m10} experiments."
            )
    pd.DataFrame(summary_rows).to_csv(OUT / "a15_hypothesis_controls_summary.csv", index=False)
    text = "\n".join(lines) + "\n"
    (OUT / "a15_hypothesis_controls_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    assert_frozen()


if __name__ == "__main__":
    main()
