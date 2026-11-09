#!/usr/bin/env python3
"""E5 endpoints. Run only after geo, unseen, and replogle have all finished.

The Holm family is nine tests on GEO 51: for v3, V1, and V2, (a) condition-C
Δlog2 versus M10, (b) condition-A Δlog2 minus condition-C Δlog2, (c) the
paired Top-5 name hit of condition A versus P1. Two-sided study-clustered
sign flips, 20,000 permutations, seed 42.
"""
from __future__ import annotations

import pickle
import sys

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, "/Users/bytedance/codes/new-bishe/artifacts/src")
from develop_m10_soft_gate import compute_bundle_coverage
from regagent_tools import load_track_catalog, prepare_hypothesis_resources, score_candidates
from run_a2_pilot import _apply, _known_symbols, parse_hypotheses
from run_e4_agent_variants import _episode_top5, merged_rank, parse_top5
from run_e5_common import BUNDLE_DIR, DATE_CUTOFF, OUT, geo_pub_dates
from run_e_common import load_jsonl
from verify_round4_accept_checks import study_clustered_sign_flip_perm

METHODS = ("v3", "V1", "V2")


def _ready() -> None:
    for cohort in ("geo", "unseen", "replogle"):
        text = (OUT / f"e5_{cohort}_status.txt").read_text(encoding="utf-8")
        if "stop: none" not in text:
            raise SystemExit(f"{cohort} did not finish: {text}")


def _kappa(left: np.ndarray, right: np.ndarray) -> float:
    n = len(left)
    both = int(((left == 1) & (right == 1)).sum())
    only_l = int(((left == 1) & (right == 0)).sum())
    only_r = int(((left == 0) & (right == 1)).sum())
    neither = int(((left == 0) & (right == 0)).sum())
    po = (both + neither) / n
    pe = ((both + only_l) / n) * ((both + only_r) / n) + ((only_r + neither) / n) * ((only_l + neither) / n)
    if abs(1.0 - pe) < 1e-12:
        return 1.0 if po == 1.0 else 0.0
    return float((po - pe) / (1.0 - pe))


def _names(row, kind: str, allowed):
    if row is None or row.get("error"):
        return False, []
    if kind == "v3" or kind == "P1":
        hyps = parse_hypotheses(row.get("content") or "")
        return hyps is not None, hyps or []
    if kind == "V1":
        mapping = row.get("id_map") or {}
        if mapping:
            reverse = {value: key for key, value in mapping.items()}
            obj, ids = parse_top5(row.get("content") or "", set(reverse))
            if obj is None:
                return False, []
            return True, [reverse[item] for item in ids]
        obj, top = parse_top5(row.get("content") or "", allowed)
        return obj is not None, top
    return False, []


def _cohort_table(cohort: str) -> pd.DataFrame:
    bundle = pickle.loads((BUNDLE_DIR / f"{cohort}.pkl").read_bytes())
    sample = pd.read_csv(BUNDLE_DIR / f"{cohort}_sample.csv")
    cov = compute_bundle_coverage(bundle)
    human = [track for track in load_track_catalog() if track.cell and track.species == "HUMAN"]
    resources = prepare_hypothesis_resources(bundle, human)
    known = _known_symbols(bundle, resources)
    raw = load_jsonl(OUT / f"e5_{cohort}_raw.jsonl")
    one = {}
    v2 = {}
    for row in raw:
        key = (row.get("variant"), row.get("exp_id"), row.get("condition"))
        if row.get("variant") == "V2":
            v2.setdefault((row.get("exp_id"), row.get("condition")), []).append(row)
        elif row.get("content") or row.get("error"):
            one[key] = row
    columns = {str(column).upper() for column in bundle["mat_blind"].columns}
    rows = []
    for rec in sample.itertuples(index=False):
        exp_id = str(rec.exp_id)
        true = str(bundle["obs_sub"].loc[exp_id, "TF"]).upper()
        base = score_candidates(exp_id, bundle, cov_frac=cov)
        order = [str(item) for item in base["m10_order"]]
        r_m10 = order.index(true) + 1
        pooled = set(order[:25])
        record = {
            "exp_id": exp_id,
            "study_id": str(rec.study_id),
            "true_tf": true,
            "r_m10": r_m10,
            "tf_in_matrix": int(true in columns),
        }
        for condition in ("A", "C"):
            got = one.get(("v3", exp_id, condition))
            ok, hyps = _names(got, "v3", None)
            applied = _apply(hyps if ok else [], exp_id, order, true, bundle, resources, known)
            record[f"v3_{condition}_ok"] = int(ok and got is not None)
            record[f"v3_{condition}_rank"] = applied["k10_rank"]
            record[f"v3_{condition}_hit5"] = int(ok and true in hyps[:5])
            record[f"v3_{condition}_top5"] = "|".join(hyps[:5])
            got = one.get(("V1", exp_id, condition))
            ok, top = _names(got, "V1", pooled)
            record[f"V1_{condition}_ok"] = int(ok and got is not None)
            record[f"V1_{condition}_rank"] = merged_rank(top if ok else [], order, true)
            record[f"V1_{condition}_hit5"] = int(ok and true in top[:5])
            record[f"V1_{condition}_top5"] = "|".join(top[:5])
            ok, top = _episode_top5(v2.get((exp_id, condition), []), None)
            record[f"V2_{condition}_ok"] = int(ok)
            record[f"V2_{condition}_rank"] = merged_rank(top if ok else [], order, true)
            record[f"V2_{condition}_hit5"] = int(ok and true in top[:5])
            record[f"V2_{condition}_top5"] = "|".join(top[:5])
        if cohort == "geo":
            got = one.get(("V1", exp_id, "B"))
            ok, top = _names(got, "V1", pooled)
            record["V1_B_ok"] = int(ok and got is not None)
            record["V1_B_rank"] = merged_rank(top if ok else [], order, true)
            record["V1_B_hit5"] = int(ok and true in top[:5])
            record["V1_B_top5"] = "|".join(top[:5])
        if cohort != "replogle":
            got = one.get(("P1", exp_id, "P1"))
            ok, hyps = _names(got, "P1", None)
            record["P1_ok"] = int(ok and got is not None)
            record["P1_hit5"] = int(ok and true in hyps[:5])
            record["P1_top5"] = "|".join(hyps[:5])
        rows.append(record)
    return pd.DataFrame(rows)


def _perm(val_a, val_b, study) -> tuple[float, float]:
    mean, pvalue = study_clustered_sign_flip_perm(
        np.asarray(val_a, dtype=float),
        np.asarray(val_b, dtype=float),
        np.asarray(study),
        n_perm=20000,
        seed=42,
    )
    return float(mean), float(pvalue)


def _describe(df: pd.DataFrame, cohort: str) -> list[str]:
    lines = [f"## {cohort} descriptive, n={len(df)}, studies={df['study_id'].nunique()}"]
    conditions = ["A", "C"] + (["B"] if cohort == "geo" else [])
    for method in METHODS:
        for condition in conditions:
            if method != "V1" and condition == "B":
                continue
            rank = df[f"{method}_{condition}_rank"].to_numpy()
            base = df["r_m10"].to_numpy()
            mean, pvalue = _perm(-np.log2(rank), -np.log2(base), df["study_id"])
            ok = int(df[f"{method}_{condition}_ok"].sum())
            top1 = int((rank == 1).sum())
            top5 = int((rank <= 5).sum())
            lines.append(
                f"{method} {condition}: parse {ok}/{len(df)}, mean Δlog2 {mean:.4f}, "
                f"cluster p {pvalue:.4f}, Top-1 {top1}/{len(df)}, Top-5 {top5}/{len(df)}"
            )
    if "P1_hit5" in df.columns:
        lines.append(f"P1 Top-5 hits {int(df['P1_hit5'].sum())}/{len(df)}")
    return lines


def _replogle_strata(df: pd.DataFrame) -> list[str]:
    """Descriptive only. Not part of the Holm family."""
    n_in = int(df["tf_in_matrix"].sum())
    n_out = int((df["tf_in_matrix"] == 0).sum())
    if (n_in, n_out, len(df)) != (169, 98, 267):
        raise SystemExit(f"Replogle matrix split is {n_in}/{n_out} of {len(df)}, not 169/98 of 267")
    lines = [
        "Replogle sensitivity, descriptive, not in the Holm family. "
        "The stratum is whether the true TF is a column of mat_blind."
    ]
    for label, mask in (("in matrix", df["tf_in_matrix"] == 1), ("absent", df["tf_in_matrix"] == 0)):
        sub = df.loc[mask]
        for method in METHODS:
            for condition in ("A", "C"):
                rank = sub[f"{method}_{condition}_rank"].to_numpy()
                delta = np.log2(sub["r_m10"].to_numpy()) - np.log2(rank)
                ok = int(sub[f"{method}_{condition}_ok"].sum())
                lines.append(
                    f"{method} {condition} {label}: parse {ok}/{len(sub)}, "
                    f"mean Δlog2 {float(delta.mean()):.4f}, "
                    f"Top-1 {int((rank == 1).sum())}/{len(sub)}, Top-5 {int((rank <= 5).sum())}/{len(sub)}"
                )
    return lines


def main() -> None:
    _ready()
    tables = {cohort: _cohort_table(cohort) for cohort in ("geo", "unseen", "replogle")}
    geo = tables["geo"]
    if len(geo) != 51:
        raise SystemExit(f"GEO has {len(geo)} rows, not 51")
    dates = geo_pub_dates()
    family = []
    for method in METHODS:
        c_rank = geo[f"{method}_C_rank"].to_numpy()
        a_rank = geo[f"{method}_A_rank"].to_numpy()
        base = geo["r_m10"].to_numpy()
        study = geo["study_id"]
        mean_c, p_c = _perm(-np.log2(c_rank), -np.log2(base), study)
        mean_ac, p_ac = _perm(-np.log2(a_rank), -np.log2(c_rank), study)
        mean_h, p_h = _perm(geo[f"{method}_A_hit5"], geo["P1_hit5"], study)
        family.append((f"{method} C vs M10", mean_c, p_c))
        family.append((f"{method} A−C", mean_ac, p_ac))
        family.append((f"{method} A Top-5 hit vs P1", mean_h, p_h))
    if len(family) != 9:
        raise SystemExit("Holm family is not 9 tests")
    _reject, p_holm, _a, _b = multipletests([item[2] for item in family], method="holm")
    holm_rows = []
    lines = ["Holm family m=9 on GEO 51. Two-sided, 20000 permutations, seed 42."]
    for (label, mean, raw), adjusted in zip(family, p_holm):
        holm_rows.append({"test": label, "estimate": mean, "p_raw": raw, "p_holm": float(adjusted)})
        lines.append(f"{label}: estimate {mean:.4f}, p {raw:.4f}, Holm p {float(adjusted):.4f}")
    lines.append("")
    lines.append("Cohen's kappa is on the paired Top-5 hit indicators. Overlap is the mean of |A names ∩ P1 names| / 5.")
    for method in METHODS:
        kappa = _kappa(geo[f"{method}_A_hit5"].to_numpy(), geo["P1_hit5"].to_numpy())
        overlap = []
        for rec in geo.itertuples(index=False):
            left = set(str(rec.__getattribute__(f"{method}_A_top5")).split("|")) - {""}
            right = set(str(rec.P1_top5).split("|")) - {""}
            overlap.append(len(left & right) / 5)
        lines.append(f"{method}: kappa {kappa:.4f}, overlap {float(np.mean(overlap)):.4f}")
    merged = geo.merge(dates, left_on="exp_id", right_on="contrast_id", how="left")
    if merged["pub_date"].isna().any():
        raise SystemExit("a GEO experiment has no public date")
    lines.append("")
    lines.append(f"GEO public-date split at {DATE_CUTOFF}. Not in the Holm family.")
    cutoff = pd.Timestamp(DATE_CUTOFF)
    p1_before = int(merged.loc[merged["pub_date"] <= cutoff, "P1_hit5"].sum())
    p1_after = int(merged.loc[merged["pub_date"] > cutoff, "P1_hit5"].sum())
    n_before = int((merged["pub_date"] <= cutoff).sum())
    n_after = int((merged["pub_date"] > cutoff).sum())
    lines.append(f"P1 Top-5: on or before {p1_before}/{n_before}, after {p1_after}/{n_after}")
    for method in METHODS:
        for label, mask in (("on or before", merged["pub_date"] <= cutoff), ("after", merged["pub_date"] > cutoff)):
            sub = merged.loc[mask]
            delta = np.log2(sub["r_m10"]) - np.log2(sub[f"{method}_A_rank"])
            delta = delta - (np.log2(sub["r_m10"]) - np.log2(sub[f"{method}_C_rank"]))
            lines.append(f"{method} A−C {label}: mean {float(delta.mean()):.4f} on {len(sub)}")
    lines.append("")
    for cohort, df in tables.items():
        lines.extend(_describe(df, cohort))
        lines.append("")
    lines.extend(_replogle_strata(tables["replogle"]))
    lines.append("")
    text = "\n".join(lines).rstrip() + "\n"
    pd.DataFrame(holm_rows).to_csv(OUT / "e5_holm.csv", index=False)
    for cohort, df in tables.items():
        df.to_csv(OUT / f"e5_{cohort}_per_experiment.csv", index=False)
    (OUT / "e5_endpoints.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
