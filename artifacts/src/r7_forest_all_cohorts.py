"""R7 cross-cohort forest plot of mean Δlog2 rank (cluster bootstrap 95% CI).

Cohorts: KnockTF Dev 266, Unseen 64, Replogle K562 Primary 267, RPE1 76, GEO 51.
Comparisons: M5 vs ULM, M9 vs M5, M10 vs M5, M9 vs ULM.
Descriptive DerSimonian–Laird random-effects pooling is restricted to the two
pre-registered external cohorts Replogle K562 Primary and GEO.

Does not overwrite frozen rank CSVs. M10 per-experiment ranks for non-GEO
cohorts are written to audit/.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/bytedance/codes/new-bishe/artifacts/src")
from deconftf_tools import score_knocktf_experiments  # noqa: E402
from develop_m10_soft_gate import compute_bundle_coverage, m10_soft_order  # noqa: E402
from eval_p0_2_external import load_replogle_cohort  # noqa: E402

ROOT = Path("/Users/bytedance/codes/new-bishe")
DATA_DIR = ROOT / "artifacts" / "data" / "knocktf"
EXT_DATA = ROOT / "artifacts" / "data" / "external"
OUT_DIR = ROOT / "artifacts" / "results" / "cand6_deconftf"
EXT_OUT = OUT_DIR / "external"
AUDIT = OUT_DIR / "audit"
FIG_DIR = ROOT / "papers" / "deconf-regulon" / "figures"
AUDIT.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

COMPARISONS = [
    ("M5 vs ULM", "r_m5_mm_static", "r_m1_ulm"),
    ("M9 vs M5", "r_m9_determ", "r_m5_mm_static"),
    ("M10 vs M5", "r_m10_soft", "r_m5_mm_static"),
    ("M9 vs ULM", "r_m9_determ", "r_m1_ulm"),
]


def cluster_boot(
    r_method: np.ndarray,
    r_base: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 5000,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    y_a = -np.log2(np.asarray(r_method, dtype=float))
    y_b = -np.log2(np.asarray(r_base, dtype=float))
    observed = float(np.mean(y_a - y_b))
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    c_to_idx = [np.where(clusters == c)[0] for c in uniq]
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sampled = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx = np.concatenate([c_to_idx[j] for j in sampled])
        diffs[i] = np.mean(y_a[idx] - y_b[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    se = float(np.std(diffs, ddof=1))
    return observed, float(lo), float(hi), se


def dersimonian_laird(means: List[float], ses: List[float]) -> Tuple[float, float, float, float]:
    means_a = np.asarray(means, dtype=float)
    ses_a = np.asarray(ses, dtype=float)
    w = 1.0 / np.maximum(ses_a**2, 1e-12)
    fe = float(np.sum(w * means_a) / np.sum(w))
    q = float(np.sum(w * (means_a - fe) ** 2))
    k = len(means_a)
    c = float(np.sum(w) - np.sum(w**2) / np.sum(w))
    tau2 = max(0.0, (q - (k - 1)) / c) if c > 0 else 0.0
    w_star = 1.0 / (ses_a**2 + tau2)
    re = float(np.sum(w_star * means_a) / np.sum(w_star))
    se_re = float(np.sqrt(1.0 / np.sum(w_star)))
    return re, re - 1.96 * se_re, re + 1.96 * se_re, tau2


def m10_rank_map(bundle: dict, cov: float, ids: List[str], tfs: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for eid, tf in zip(ids, tfs):
        order, _ = m10_soft_order(eid, bundle, cov_frac=cov, lambda_bind=1.0, lambda_hub=0.3)
        out[str(eid)] = order.index(tf) + 1
    return out


def knocktf_table() -> Tuple[pd.DataFrame, pd.DataFrame]:
    ranks401 = pd.read_csv(OUT_DIR / "knocktf401_ranks_with_study_ids.csv").set_index("exp_id")
    h135 = pd.read_csv(OUT_DIR / "holdout135_harmonised_ranks.csv").set_index("exp_id")
    for col in ["r_m1_ulm", "r_m2b_viper_shadow", "r_m5_mm_static", "r_m9_determ"]:
        ranks401.loc[h135.index, col] = h135[col].values
    dev = ranks401[ranks401["split_new"] == "Dev"].copy()
    unseen = ranks401[ranks401["split_new"] == "Unseen"].copy()
    return dev, unseen


def attach_m10_knocktf(dev: pd.DataFrame, unseen: pd.DataFrame, cand_tfs: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cache = AUDIT / "r7_m10_ranks_knocktf.csv"
    bundle_dev = pickle.loads((OUT_DIR / "knocktf_precomputed_bundle.pkl").read_bytes())
    cov_dev = compute_bundle_coverage(bundle_dev)
    m10_dev = m10_rank_map(
        bundle_dev, cov_dev, list(dev.index), list(dev["true_tf"].astype(str))
    )
    print(f"[M10] Dev 266 from existing bundle cov={cov_dev:.4f}", flush=True)

    if cache.exists():
        cached = pd.read_csv(cache).set_index("exp_id")
        if set(unseen.index).issubset(set(cached.index)):
            m10_unseen = cached["r_m10_soft"].to_dict()
            print(f"[M10] Unseen 64 reused from {cache}", flush=True)
        else:
            m10_unseen = {}
    else:
        m10_unseen = {}

    if not m10_unseen:
        mat_ktf = pd.read_csv(DATA_DIR / "knockTF_expr.csv", index_col=0)
        obs_ktf = pd.read_csv(DATA_DIR / "knockTF_meta.csv", index_col=0)
        hold_ids = list(unseen.index)
        print(f"[M10] scoring Unseen {len(hold_ids)} experiments", flush=True)
        hold_b = score_knocktf_experiments(mat_ktf.loc[hold_ids], obs_ktf.loc[hold_ids], cand_tfs=cand_tfs)
        cov_u = compute_bundle_coverage(hold_b)
        tfs = [str(obs_ktf.loc[eid, "TF"]) for eid in hold_ids]
        m10_unseen = m10_rank_map(hold_b, cov_u, hold_ids, tfs)
        print(f"[M10] Unseen 64 cov={cov_u:.4f}", flush=True)
        rows = [{"exp_id": eid, "cohort": "KnockTF_Unseen_64", "r_m10_soft": r} for eid, r in m10_unseen.items()]
        rows += [{"exp_id": eid, "cohort": "KnockTF_Dev_266", "r_m10_soft": r} for eid, r in m10_dev.items()]
        pd.DataFrame(rows).to_csv(cache, index=False)

    dev = dev.copy()
    unseen = unseen.copy()
    dev["r_m10_soft"] = [m10_dev[str(i)] for i in dev.index]
    unseen["r_m10_soft"] = [m10_unseen[str(i)] for i in unseen.index]
    return dev, unseen


def attach_m10_replogle(cand_tfs: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cache = AUDIT / "r7_m10_ranks_replogle.csv"
    k562_all = pd.read_csv(EXT_OUT / "replogle2022_k562_gwps_ranks.csv")
    rpe_all = pd.read_csv(EXT_OUT / "replogle2022_rpe1_ranks.csv")
    k562 = k562_all[(k562_all["target_log2fc"] <= -0.5) & (k562_all["max_abs_blind"] >= 0.5)].copy()
    rpe = rpe_all[(rpe_all["target_log2fc"] <= -0.5) & (rpe_all["max_abs_blind"] >= 0.5)].copy()
    print(f"[M10] K562 primary {len(k562)}  RPE1 primary {len(rpe)}", flush=True)

    if cache.exists():
        cached = pd.read_csv(cache)
        have = set(cached["exp_id"].astype(str))
        need = set(k562["exp_id"].astype(str)) | set(rpe["exp_id"].astype(str))
        if need.issubset(have):
            m10 = cached.set_index("exp_id")["r_m10_soft"].to_dict()
            print(f"[M10] Replogle reused from {cache}", flush=True)
            k562["r_m10_soft"] = k562["exp_id"].map(m10)
            rpe["r_m10_soft"] = rpe["exp_id"].map(m10)
            return k562, rpe

    masked = set(pd.read_csv(OUT_DIR / "supplement_masked_tfs_list.csv").iloc[:, 0].astype(str).str.upper())
    obs_ktf = pd.read_csv(DATA_DIR / "knockTF_meta.csv", index_col=0)
    tf_to_class = obs_ktf.groupby("TF")["TF.Class"].first().to_dict()
    rows = []
    for label, h5, bio, tissue, df in [
        ("K562", EXT_DATA / "K562_gwps_normalized_bulk_01.h5ad", "K562", "Haematopoietic", k562),
        ("RPE1", EXT_DATA / "rpe1_normalized_bulk_01.h5ad", "RPE1", "RPE", rpe),
    ]:
        print(f"[M10] scoring Replogle {label}", flush=True)
        mat, obs = load_replogle_cohort(h5, cand_tfs, masked, tf_to_class, bio, tissue, mode="normalized_z")
        keep = [i for i in df["exp_id"].tolist() if i in mat.index]
        mat = mat.loc[keep]
        obs = obs.loc[keep]
        bundle = score_knocktf_experiments(mat, obs, cand_tfs=cand_tfs)
        cov = compute_bundle_coverage(bundle)
        tfs = [str(obs.loc[eid, "TF"]) for eid in keep]
        m10 = m10_rank_map(bundle, cov, keep, tfs)
        print(f"[M10] {label} n={len(keep)} cov={cov:.4f}", flush=True)
        for eid, r in m10.items():
            rows.append({"exp_id": eid, "cohort": label, "r_m10_soft": r})
    pd.DataFrame(rows).to_csv(cache, index=False)
    m10 = {r["exp_id"]: r["r_m10_soft"] for r in rows}
    k562["r_m10_soft"] = k562["exp_id"].map(m10)
    rpe["r_m10_soft"] = rpe["exp_id"].map(m10)
    return k562, rpe


def geo_table() -> pd.DataFrame:
    g = pd.read_csv(EXT_OUT / "r2_geo_phase2_ranks.csv")
    if "exp_id" not in g.columns:
        g = g.rename(columns={g.columns[0]: "exp_id"})
    return g


def pack_cohort(name: str, role: str, cluster_unit: str, df: pd.DataFrame, id_col: str, cluster_col: str, tf_col: str) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "cohort": name,
            "cohort_role": role,
            "cluster_unit": cluster_unit,
            "exp_id": df[id_col].astype(str).values if id_col in df.columns else df.index.astype(str).values,
            "cluster_id": df[cluster_col].astype(str).values,
            "true_tf": df[tf_col].astype(str).values,
            "r_m1_ulm": df["r_m1_ulm"].values,
            "r_m5_mm_static": df["r_m5_mm_static"].values,
            "r_m9_determ": df["r_m9_determ"].values,
            "r_m10_soft": df["r_m10_soft"].values,
        }
    )
    return out


def summarise(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cohort, role, unit), sub in long.groupby(["cohort", "cohort_role", "cluster_unit"], sort=False):
        for label, mcol, bcol in COMPARISONS:
            mean, lo, hi, se = cluster_boot(sub[mcol].values, sub[bcol].values, sub["cluster_id"].values)
            rows.append(
                {
                    "cohort": cohort,
                    "cohort_role": role,
                    "cluster_unit": unit,
                    "n": len(sub),
                    "n_clusters": int(sub["cluster_id"].nunique()),
                    "comparison": label,
                    "mean_dlog2_rank": mean,
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "boot_se": se,
                    "analysis_type": "cohort",
                    "is_descriptive_re": 0,
                }
            )
    summ = pd.DataFrame(rows)
    re_cohorts = {"Replogle K562 Primary 267", "GEO bulk RNA-seq 51"}
    re_rows = []
    for label, _, _ in COMPARISONS:
        hit = summ[(summ.comparison == label) & (summ.cohort.isin(re_cohorts))].copy()
        hit = hit.set_index("cohort").loc[["Replogle K562 Primary 267", "GEO bulk RNA-seq 51"]]
        re, lo, hi, tau2 = dersimonian_laird(hit["mean_dlog2_rank"].tolist(), hit["boot_se"].tolist())
        re_rows.append(
            {
                "cohort": "RE pool (K562 + GEO)",
                "cohort_role": "descriptive random-effects (two pre-registered external cohorts)",
                "cluster_unit": "study (GEO) / TF-Class (K562)",
                "n": int(hit["n"].sum()),
                "n_clusters": int(hit["n_clusters"].sum()),
                "comparison": label,
                "mean_dlog2_rank": re,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "boot_se": (hi - lo) / (2 * 1.96),
                "analysis_type": "descriptive_random_effects",
                "is_descriptive_re": 1,
                "tau2": tau2,
            }
        )
    return pd.concat([summ, pd.DataFrame(re_rows)], ignore_index=True)


CAPTION = (
    "Cross-cohort mean Δlog₂ rank with cluster-bootstrap 95% CI. "
    "The DerSimonian–Laird diamond pools only the two pre-registered external cohorts "
    "Replogle K562 Primary and GEO (k = 2); with two studies the τ² estimate is unreliable, "
    "so the diamond is descriptive only. RPE1 is shown but not pooled, because it comes from "
    "the same Replogle 2022 study as K562 and is not independent. "
    "Panel x-axis ranges differ (labelled on each panel). "
    "For GEO M9 vs ULM the bootstrap CI is 0.10–1.72 (excludes 0) while the pre-registered "
    "study-clustered sign-flip permutation has raw p = 0.051 (Holm m=5 p = 0.20); "
    "the primary test is the permutation, and the bootstrap CI is descriptive."
)


def draw(summ: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["axes.edgecolor"] = "#111827"
    plt.rcParams["xtick.color"] = "#111827"
    plt.rcParams["ytick.color"] = "#111827"

    cohort_order = [
        "KnockTF Dev 266",
        "KnockTF Unseen 64",
        "Replogle K562 Primary 267",
        "Replogle RPE1 76",
        "GEO bulk RNA-seq 51",
        "RE pool (K562 + GEO)",
    ]
    role_short = {
        "KnockTF Dev 266": "development",
        "KnockTF Unseen 64": "post-dev, inspected",
        "Replogle K562 Primary 267": "pre-registered external",
        "Replogle RPE1 76": "pre-registered external",
        "GEO bulk RNA-seq 51": "pre-registered external",
        "RE pool (K562 + GEO)": "descriptive RE",
    }
    colors = {
        "KnockTF Dev 266": "#64748b",
        "KnockTF Unseen 64": "#d97706",
        "Replogle K562 Primary 267": "#1d4ed8",
        "Replogle RPE1 76": "#1d4ed8",
        "GEO bulk RNA-seq 51": "#1d4ed8",
        "RE pool (K562 + GEO)": "#111827",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 9.6), dpi=220, sharex=False, sharey=True)
    axes = axes.ravel()
    # Bottom = RE diamond (summary); top = development. yticklabels must follow the same order.
    display = list(reversed(cohort_order))
    y_pos = {c: i for i, c in enumerate(display)}

    for ax, (label, _, _) in zip(axes, COMPARISONS):
        sub = summ[summ.comparison == label].set_index("cohort")
        ax.axhline(0.5, color="#e5e7eb", lw=0.6, zorder=0)
        for cohort in cohort_order:
            y = y_pos[cohort]
            row = sub.loc[cohort]
            mean, lo, hi = row["mean_dlog2_rank"], row["ci95_lo"], row["ci95_hi"]
            col = colors[cohort]
            if cohort == "RE pool (K562 + GEO)":
                ax.plot([lo, hi], [y, y], color=col, lw=1.4, solid_capstyle="butt")
                ax.scatter([mean], [y], marker="D", s=36, facecolors="white", edgecolors=col, linewidths=1.2, zorder=3)
            else:
                ax.plot([lo, hi], [y, y], color=col, lw=1.6, solid_capstyle="butt")
                ax.scatter([mean], [y], marker="o", s=28, color=col, zorder=3, edgecolors="none")
        ax.axvline(0.0, color="#9ca3af", lw=0.8, ls="--")
        ax.set_title(label, fontsize=11, pad=6, loc="left")
        ax.set_yticks([y_pos[c] for c in display])
        ax.set_yticklabels(
            [f"{c}\n{role_short[c]}" for c in display],
            fontsize=7.5,
            linespacing=1.15,
        )
        ax.set_xlabel("mean Δlog₂ rank (bits)", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        xs = sub.loc[cohort_order, ["ci95_lo", "ci95_hi"]].to_numpy()
        span = float(np.nanmax(np.abs(xs)))
        xmin, xmax = -span * 1.15, span * 1.15
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-0.6, len(cohort_order) - 0.4)
        ax.text(
            0.99,
            0.98,
            f"x ∈ [{xmin:.2f}, {xmax:.2f}]",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
            color="#4b5563",
        )

    fig.suptitle("Cross-cohort rank shift (cluster bootstrap 95% CI)", fontsize=10, y=0.995)
    import textwrap

    fig.text(
        0.02,
        0.012,
        textwrap.fill(CAPTION, width=168),
        ha="left",
        va="bottom",
        fontsize=6.5,
        color="#111827",
        linespacing=1.25,
    )
    fig.tight_layout(rect=[0, 0.14, 1, 0.97])
    pdf = FIG_DIR / "fig_forest_all_cohorts.pdf"
    png = FIG_DIR / "fig_forest_all_cohorts.png"
    fig.savefig(pdf)
    fig.savefig(png)
    cap_path = FIG_DIR / "fig_forest_all_cohorts.caption.txt"
    cap_path.write_text(CAPTION + "\n")
    plt.close(fig)
    print(f"wrote {pdf} and {png}", flush=True)
    print(f"wrote {cap_path}", flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="redraw the figure from the committed summary CSV without rewriting ranks",
    )
    args = parser.parse_args()
    if args.plot_only:
        summ = pd.read_csv(AUDIT / "r7_forest_all_cohorts_summary.csv")
        draw(summ)
        return

    bundle_dev = pickle.loads((OUT_DIR / "knocktf_precomputed_bundle.pkl").read_bytes())
    cand_tfs = list(bundle_dev["cand_tfs"])
    dev, unseen = knocktf_table()
    dev, unseen = attach_m10_knocktf(dev, unseen, cand_tfs)
    k562, rpe = attach_m10_replogle(cand_tfs)
    geo = geo_table()

    parts = [
        pack_cohort("KnockTF Dev 266", "development", "study_id", dev.reset_index(), "exp_id", "study_id", "true_tf"),
        pack_cohort("KnockTF Unseen 64", "post-development, already inspected", "study_id", unseen.reset_index(), "exp_id", "study_id", "true_tf"),
        pack_cohort("Replogle K562 Primary 267", "pre-registered external (P0-2)", "TF-Class", k562, "exp_id", "tf_class", "true_tf"),
        pack_cohort("Replogle RPE1 76", "pre-registered external (P0-2)", "TF-Class", rpe, "exp_id", "tf_class", "true_tf"),
        pack_cohort("GEO bulk RNA-seq 51", "pre-registered external (R2)", "study_id", geo, "exp_id", "study_id", "true_tf"),
    ]
    long = pd.concat(parts, ignore_index=True)
    long_path = AUDIT / "r7_forest_all_cohorts_experiment_ranks.csv"
    long.to_csv(long_path, index=False)
    summ = summarise(long)
    summ_path = AUDIT / "r7_forest_all_cohorts_summary.csv"
    summ.to_csv(summ_path, index=False)
    print(summ.to_string(index=False), flush=True)
    draw(summ)
    print(f"wrote {long_path}")
    print(f"wrote {summ_path}")


if __name__ == "__main__":
    main()
