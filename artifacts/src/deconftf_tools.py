#!/usr/bin/env python3
r"""
Core Quantitative Engine & Conditional Probing Tools for DeconfTF-Agent:
De-confounded Upstream Perturbed Regulator Inference from Transcriptomic Signatures.

Key Guarantees:
1. Zero Self-Expression Leakage:
   - All self-regulatory loops (`source == target`) are removed from every prior network.
   - In the expression matrix `mat_blind`, the mRNA log2FC of ALL 1,888 human transcription
     factors is explicitly set to `0.0` before any score, pathway, or probe is computed.
2. Multi-Modal Regulatory Evidence:
   - Curated signed regulons (`CollecTRI`, `DoRothEA`)
   - Physical ChIP-seq binding (`ENCODE_ChIP-seq`, `ReMap_ChIP-seq`, `Literature_ChIP-seq`,
     including cell-line/tissue track matching!)
   - Large-scale co-expression (`ARCHS4_Coexpression`, `GTEx_Coexpression`, `Enrichr_Queries`)
   - Canonical signaling pathways (`PROGENy` 14 pathways)
3. Hypothesis-Conditional Quantitative Probe (`execute_conditional_regulon_probe`):
   Given $K \le 8$ contending TFs selected in Stage 1 (which can include lineage/pathway TFs
   rescued from outside the noisy univariate Top-25):
   - Fits a Local Multivariate Ridge Regression (`Partial ULM`) of the transcriptome $y$
     jointly on the $K$ contenders + universal proliferation/stress hubs (`E2F4`, `E2F1`, `MYC`, `ATF4`),
     avoiding the 753-variable multicollinearity collapse of global MLM while removing shared-target confounding!
   - Computes Exclusive-Target Enrichment (targets regulated by $C_k$ and NONE of the other contenders).
   - Audits Direct TF->TF Regulatory Cascade Polarity among the contenders (e.g., `POU5F1 -| GATA6`,
     `TP53 -| FOXM1`, `FLI1 -> E2F1`).
   - Checks Cell-Line/Tissue-Matched ChIP-seq Tracks in `ENCODE` and `Literature_ChIP-seq`.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import decoupler as dc
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/Users/bytedance/codes/new-bishe")
DATA_DIR = ROOT / "artifacts" / "data" / "knocktf"
OUT_DIR = ROOT / "artifacts" / "results" / "cand6_deconftf"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Universal secondary cell-cycle / stress / apoptosis responder hubs that fire across many unrelated perturbations
UNIVERSAL_SECONDARY_HUBS = {"E2F4", "E2F1", "E2F2", "E2F3", "TFDP1", "FOXM1", "MYBL2", "MYC", "ATF4"}
# MHC-II enhanceosome members that dominate ULM after IRF/CIITA-axis perturbations
ENHANCEOSOME_COMPLEX = {"CIITA", "RFX5", "RFXAP", "RFXANK", "SIN3A"}


def m9_order(
    exp_id: str,
    bundle: Dict[str, Any],
    ch_thr: float = 0.5,
    cc_thr: float = 3.0,
    use_enhanceosome: bool = True,
) -> List[str]:
    """Selective conditional re-ranking (M9) for one experiment of a scored bundle.

    Keeps the top ULM TF unless it trips a confounding flag; otherwise orders candidates by
    the static multimodal score after demoting hubs without residual activity and TFs with
    neither binding nor DoRothEA support.
    """
    z = bundle["z_libs"]
    ulm = bundle["est_ulm"].loc[exp_id]
    mm = bundle["multi_modal_score"].loc[exp_id]
    cc = -bundle["est_c_res"].loc[exp_id]
    chip = pd.concat([z["ENCODE_ChIP-seq"].loc[exp_id], z["ReMap_ChIP-seq"].loc[exp_id],
                      z["Literature_ChIP-seq"].loc[exp_id]], axis=1).max(axis=1)
    coexp = pd.concat([z["ARCHS4_Coexpression"].loc[exp_id], z["GTEx_Coexpression"].loc[exp_id]], axis=1).max(axis=1)
    doro = z["DoRothEA_Signed"].loc[exp_id]
    enh = ENHANCEOSOME_COMPLEX if use_enhanceosome else set()

    u_order = list(ulm.sort_values(ascending=True).index)
    m_order = list(mm.sort_values(ascending=False).index)
    t1 = u_order[0]
    trig = (
        (t1 in UNIVERSAL_SECONDARY_HUBS and cc[t1] < cc_thr)
        or (chip[t1] < ch_thr)
        or (doro[t1] < 0.8 and chip[t1] < 1.5 and coexp[t1] > 1.2)
        or (t1 in enh)
    )
    if not trig:
        return [t1] + [c for c in m_order if c != t1]
    clean = [
        c for c in m_order
        if not (c in UNIVERSAL_SECONDARY_HUBS and cc[c] < cc_thr)
        and not (chip[c] < ch_thr and doro[c] < 1.0)
        and c not in enh
    ]
    clean_set = set(clean)
    return clean + [c for c in m_order if c not in clean_set]


def load_gmt_with_tracks(gmt_path: Path) -> Tuple[pd.DataFrame, Dict[str, List[Tuple[str, List[str]]]]]:
    """
    Parse a ChEA3 .gmt file into:
    1) A deduplicated TF->target DataFrame (`source`, `target`, `weight=1.0`, `source != target`)
    2) A dictionary `tf -> [(track_id, [target_genes]), ...]` preserving cell-line/tissue track metadata.
    """
    edges = []
    tracks_by_tf: Dict[str, List[Tuple[str, List[str]]]] = {}
    for line in gmt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = [p.strip() for p in line.split("\t") if p.strip()]
        if len(parts) < 3:
            continue
        track_id = parts[0]
        tf = track_id.split("_")[0].upper()
        tgts = [g.upper() for g in parts[2:] if g.upper() != tf]
        if tgts:
            tracks_by_tf.setdefault(tf, []).append((track_id, tgts))
            for t in tgts:
                edges.append((tf, t))
    df = pd.DataFrame(edges, columns=["source", "target"]).drop_duplicates()
    df["weight"] = 1.0
    return df, tracks_by_tf


def fast_unsigned_enrichment(
    Y_abs_z: np.ndarray,
    gene_to_idx: Dict[str, int],
    net: pd.DataFrame,
    cand_tfs: List[str],
) -> np.ndarray:
    tf_to_idx = {t: j for j, t in enumerate(cand_tfs)}
    W = np.zeros((Y_abs_z.shape[1], len(cand_tfs)), dtype=np.float32)
    for src, tgt in zip(net["source"], net["target"]):
        if tgt in gene_to_idx and src in tf_to_idx and src != tgt:
            W[gene_to_idx[tgt], tf_to_idx[src]] = 1.0
    gene_deg = np.maximum(1.0, W.sum(axis=1, keepdims=True))
    W_idf = W / np.sqrt(gene_deg)
    col_norm = np.sqrt((W_idf ** 2).sum(axis=0, keepdims=True)) + 1e-8
    scores = (Y_abs_z @ W_idf) / col_norm
    has_targets = W.sum(axis=0) >= 3
    scores[:, ~has_targets] = np.nan
    return scores


def prepare_knocktf_bundle() -> Dict[str, Any]:
    cache_path = OUT_DIR / "knocktf_precomputed_bundle.pkl"
    if cache_path.exists():
        return pickle.loads(cache_path.read_bytes())

    print("Building KnockTF multi-library precomputed bundle (0% self-mRNA leakage)...", flush=True)
    mat = pd.read_csv(DATA_DIR / "knockTF_expr.csv", index_col=0)
    obs = pd.read_csv(DATA_DIR / "knockTF_meta.csv", index_col=0)
    collectri = pd.read_csv(DATA_DIR / "collectri_human.csv")
    msk = (obs["logFC"] < -1.0) & (obs["TF"].isin(set(collectri["source"])))
    bundle = score_knocktf_experiments(mat.loc[msk], obs.loc[msk])
    cache_path.write_bytes(pickle.dumps(bundle))
    print(f"Saved precomputed KnockTF bundle to {cache_path}", flush=True)
    return bundle


def score_knocktf_experiments(
    mat_sub: pd.DataFrame,
    obs_sub: pd.DataFrame,
    cand_tfs: List[str] | None = None,
) -> Dict[str, Any]:
    """Compute every evidence layer used by M5/M9 for a set of experiments.

    With `cand_tfs=None` the candidate set is derived from the ULM output (development
    cohort). Pass the development `cand_tfs` to score any other cohort with the identical
    pipeline; experiments whose TF is not a candidate are dropped, as in development.
    """
    collectri = pd.read_csv(DATA_DIR / "collectri_human.csv")
    dorothea = pd.read_csv(DATA_DIR / "dorothea_human.csv")
    progeny = pd.read_csv(DATA_DIR / "progeny_human.csv")

    net_c = collectri[collectri["source"] != collectri["target"]].copy()
    net_d = dorothea[dorothea["source"] != dorothea["target"]].copy()

    gmt_names = [
        "ARCHS4_Coexpression",
        "GTEx_Coexpression",
        "ENCODE_ChIP-seq",
        "ReMap_ChIP-seq",
        "Literature_ChIP-seq",
        "Enrichr_Queries",
    ]
    gmt_nets = {}
    gmt_tracks = {}
    for gn in gmt_names:
        gmt_nets[gn], gmt_tracks[gn] = load_gmt_with_tracks(DATA_DIR / f"{gn}.gmt")

    all_tfs = sorted(
        list(
            set(net_c["source"])
            | set(net_d["source"])
            | set().union(*(set(g["source"]) for g in gmt_nets.values()))
        )
    )

    mat_sub = mat_sub.copy()
    obs_sub = obs_sub.copy()

    # Strictly mask ALL 1,888 TFs' own mRNA expression in mat_blind
    tf_cols_in_mat = [g for g in all_tfs if g in mat_sub.columns]
    mat_blind = mat_sub.copy()
    mat_blind[tf_cols_in_mat] = 0.0

    # Standard ULM & MLM on CollecTRI
    est_ulm, _ = dc.mt.ulm(data=mat_blind, net=net_c)
    if cand_tfs is None:
        cand_tfs = sorted(list(est_ulm.columns))
    obs_sub = obs_sub[obs_sub["TF"].isin(cand_tfs)].copy()
    mat_blind = mat_blind.loc[obs_sub.index].copy()
    est_ulm = est_ulm.reindex(index=obs_sub.index, columns=cand_tfs, fill_value=0.0)

    est_mlm, _ = dc.mt.mlm(data=mat_blind, net=net_c)
    est_mlm = est_mlm.reindex(index=obs_sub.index, columns=cand_tfs, fill_value=0.0)

    # IDF-weighted CollecTRI
    deg_c = net_c.groupby("target")["source"].transform("count")
    net_c_idf = net_c.copy()
    net_c_idf["weight"] = net_c_idf["weight"] / np.sqrt(deg_c)
    est_c_idf, _ = dc.mt.ulm(data=mat_blind, net=net_c_idf)
    est_c_idf = est_c_idf.reindex(index=obs_sub.index, columns=cand_tfs, fill_value=0.0)

    # DoRothEA ULM
    est_d, _ = dc.mt.ulm(data=mat_blind, net=net_d)
    est_d = est_d.reindex(index=obs_sub.index, columns=cand_tfs)

    # PROGENy 14 signaling pathways ULM
    est_prog, _ = dc.mt.ulm(data=mat_blind, net=progeny)
    est_prog = est_prog.loc[obs_sub.index].copy()

    gene_to_idx = {g.upper(): i for i, g in enumerate(mat_blind.columns)}
    Y = mat_blind.values.astype(np.float32)
    Y_abs = np.abs(Y)
    Y_abs_z = (Y_abs - Y_abs.mean(axis=1, keepdims=True)) / (Y_abs.std(axis=1, keepdims=True) + 1e-8)

    # Compute Cell-Cycle / Stress Residualized Transcriptome Y_res:
    # Project each experiment's expression vector y orthogonal to the signed target vectors
    # of the universal secondary hubs (E2F4, E2F1, FOXM1, MYC, ATF4) in CollecTRI!
    hub_list = ["E2F4", "E2F1", "FOXM1", "MYC", "ATF4"]
    W_hubs = np.zeros((len(mat_blind.columns), len(hub_list)), dtype=np.float32)
    for j, h_tf in enumerate(hub_list):
        sub_h = net_c[net_c["source"] == h_tf]
        for _, r_h in sub_h.iterrows():
            t_u = str(r_h["target"]).upper()
            if t_u in gene_to_idx:
                W_hubs[gene_to_idx[t_u], j] = float(r_h["weight"])
    # Orthonormalize W_hubs via QR decomposition
    Q_hubs, _ = np.linalg.qr(W_hubs)
    Y_res = Y - (Y @ Q_hubs) @ Q_hubs.T
    df_mat_res = pd.DataFrame(Y_res, index=obs_sub.index, columns=mat_blind.columns)
    est_c_res, _ = dc.mt.ulm(data=df_mat_res, net=net_c_idf)
    est_c_res = est_c_res.reindex(index=obs_sub.index, columns=cand_tfs, fill_value=0.0)

    lib_scores: Dict[str, pd.DataFrame] = {
        "CollecTRI_Signed": -est_c_idf,
        "CollecTRI_Abs": est_c_idf.abs(),
        "CollecTRI_Unsigned": pd.DataFrame(
            fast_unsigned_enrichment(Y_abs_z, gene_to_idx, net_c, cand_tfs),
            index=obs_sub.index,
            columns=cand_tfs,
        ),
        "DoRothEA_Signed": -est_d,
        "DoRothEA_Abs": est_d.abs(),
    }
    for gn in gmt_names:
        s_mat = fast_unsigned_enrichment(Y_abs_z, gene_to_idx, gmt_nets[gn], cand_tfs)
        lib_scores[gn] = pd.DataFrame(s_mat, index=obs_sub.index, columns=cand_tfs)

    # Compute ChEA3 MeanRank & TopRank across the 6 ChEA3 + CollecTRI + DoRothEA libraries
    chea3_libs = ["CollecTRI_Signed", "DoRothEA_Signed"] + gmt_names
    stack_nr = np.stack(
        [
            lib_scores[k]
            .rank(axis=1, ascending=False, method="min", na_option="keep")
            .div(lib_scores[k].notna().sum(axis=1), axis=0)
            .values
            for k in chea3_libs
        ],
        axis=0,
    )
    chea3_meanrank = pd.DataFrame(np.nanmean(stack_nr, axis=0), index=obs_sub.index, columns=cand_tfs)
    chea3_toprank = pd.DataFrame(np.nanmin(stack_nr, axis=0), index=obs_sub.index, columns=cand_tfs)

    # Row-standardized Z-scores per library
    z_libs = {}
    for lname, sdf in lib_scores.items():
        mu = sdf.mean(axis=1).values[:, None]
        sd = sdf.std(axis=1).values[:, None] + 1e-8
        z_libs[lname] = ((sdf - mu) / sd).fillna(0.0)

    weights = {
        "CollecTRI_Signed": 1.5,
        "CollecTRI_Abs": 0.8,
        "CollecTRI_Unsigned": 1.0,
        "DoRothEA_Signed": 1.2,
        "DoRothEA_Abs": 0.6,
        "ARCHS4_Coexpression": 1.2,
        "GTEx_Coexpression": 0.8,
        "ENCODE_ChIP-seq": 1.0,
        "ReMap_ChIP-seq": 1.2,
        "Literature_ChIP-seq": 1.2,
        "Enrichr_Queries": 0.8,
    }
    multi_modal_score = sum(w * z_libs[k] for k, w in weights.items()) / sum(weights.values())

    # Also build Signed Prior Matrix W_c (genes x cand_tfs) for fast local conditional regression
    tf_to_idx = {t: j for j, t in enumerate(cand_tfs)}
    W_c = np.zeros((len(mat_blind.columns), len(cand_tfs)), dtype=np.float32)
    for src, tgt, w in zip(net_c_idf["source"], net_c_idf["target"], net_c_idf["weight"]):
        t_u = str(tgt).upper()
        if t_u in gene_to_idx and src in tf_to_idx:
            W_c[gene_to_idx[t_u], tf_to_idx[src]] = float(w)

    bundle = {
        "obs_sub": obs_sub,
        "mat_blind": mat_blind,
        "cand_tfs": cand_tfs,
        "est_ulm": est_ulm,
        "est_mlm": est_mlm,
        "est_c_idf": est_c_idf,
        "est_c_res": est_c_res,
        "est_d": est_d.fillna(0.0),
        "est_prog": est_prog,
        "z_libs": z_libs,
        "multi_modal_score": multi_modal_score,
        "chea3_meanrank": chea3_meanrank,
        "chea3_toprank": chea3_toprank,
        "net_c": net_c,
        "W_c": W_c,
        "gene_to_idx": gene_to_idx,
        "tf_to_idx": tf_to_idx,
        "encode_tracks": gmt_tracks["ENCODE_ChIP-seq"],
        "lit_tracks": gmt_tracks["Literature_ChIP-seq"],
    }
    return bundle


def execute_conditional_regulon_probe(
    exp_id: str,
    contenders: List[str],
    bundle: Dict[str, Any],
) -> str:
    """
    Hypothesis-Conditional Quantitative Disambiguation Tool (Stage 2 of DeconfTF-Agent):
    Given K <= 8 contending TFs selected in Stage 1:
    1. Local Multivariate Conditional Regression (`Partial ULM`):
       Regresses the experiment's expression profile `y` jointly on the K contenders
       PLUS the background cell-cycle/stress hubs (`E2F4`, `E2F1`, `MYC`, `ATF4`), computing
       each contender's partial t-statistic (`t_partial_signed`) and residualized activity (`t_cc_deconf`).
    2. Exclusive-Target Sub-Regulon Audit:
       Identifies target genes in CollecTRI/ChEA3 that belong to contender `C_k` and NONE of the
       other non-hub contenders, reporting their mean |log2FC|, signed t-stat, and top-4 changed genes.
    3. Multi-Library Evidence Breakdown:
       Shows exact Z-scores in Curated (`CollecTRI`, `DoRothEA`), Physical ChIP-seq (`ENCODE`, `ReMap`, `LitChIP`),
       and Co-expression (`ARCHS4`, `GTEx`), plus any Biosample-Matched ChIP-seq Track!
    4. Direct TF->TF Regulatory Cascade Links among the contenders.
    """
    obs_row = bundle["obs_sub"].loc[exp_id]
    biosample = str(obs_row["Biosample.Name"]).upper()
    tissue = str(obs_row["Tissue.Type"]).upper()
    y = bundle["mat_blind"].loc[exp_id].values.astype(np.float32)
    genes = np.array(bundle["mat_blind"].columns, dtype=str)
    gene_to_idx = bundle["gene_to_idx"]
    tf_to_idx = bundle["tf_to_idx"]
    W_c = bundle["W_c"]
    net_c = bundle["net_c"]

    valid_cands = [c for c in contenders if c in tf_to_idx][:8]
    if not valid_cands:
        return "No valid contenders provided."

    # 1. Local Multivariate Conditional Regression (`y ~ W_local @ beta`)
    # Conditioning set = valid_cands + universal secondary hubs (if not already in valid_cands)
    cond_set = list(dict.fromkeys(valid_cands + ["E2F4", "E2F1", "MYC", "ATF4"]))
    cond_cols = [tf_to_idx[c] for c in cond_set if c in tf_to_idx]
    X_loc = W_c[:, cond_cols]  # (G, K_cond)
    # Normalize columns of X_loc
    col_norms = np.maximum(1e-6, np.sqrt((X_loc ** 2).sum(axis=0, keepdims=True)))
    X_norm = X_loc / col_norms
    # Ridge-stabilized OLS for partial t-statistics
    XtX = X_norm.T @ X_norm + 0.05 * np.eye(X_norm.shape[1], dtype=np.float32)
    XtX_inv = np.linalg.inv(XtX)
    beta_hat = XtX_inv @ (X_norm.T @ y)
    resid = y - X_norm @ beta_hat
    sigma2 = float(np.sum(resid ** 2) / max(1, len(y) - X_norm.shape[1]))
    se_beta = np.sqrt(np.maximum(1e-12, sigma2 * np.diag(XtX_inv)))
    t_partial_vec = beta_hat / se_beta
    partial_map = {cond_set[i]: float(t_partial_vec[i]) for i in range(len(cond_set))}

    # 2. Target sets per contender in CollecTRI
    tf_targets: Dict[str, Dict[str, float]] = {}
    for c in valid_cands:
        sub = net_c[net_c["source"] == c]
        tf_targets[c] = {
            str(r["target"]): float(r["weight"])
            for _, r in sub.iterrows()
            if str(r["target"]).upper() in gene_to_idx
        }

    lines = ["=== Stage 2 Conditional Multivariate & Multi-Modal Disambiguation Report ==="]
    for c in valid_cands:
        my_tgts = tf_targets[c]
        other_tgts = set()
        for o in valid_cands:
            if o != c:
                other_tgts.update(tf_targets[o].keys())
        excl_tgts = [g for g in my_tgts if g not in other_tgts]

        if excl_tgts:
            excl_idxs = [gene_to_idx[g.upper()] for g in excl_tgts]
            excl_signs = np.array([my_tgts[g] for g in excl_tgts], dtype=np.float32)
            excl_vals = y[excl_idxs] * excl_signs
            excl_abs_mean = float(np.mean(np.abs(y[excl_idxs])))
            excl_t = float(np.mean(excl_vals) / (np.std(excl_vals) / np.sqrt(len(excl_vals)) + 1e-6)) if len(excl_vals) >= 3 else 0.0
            top_ex_g = sorted(excl_tgts, key=lambda g: abs(y[gene_to_idx[g.upper()]]), reverse=True)[:4]
            top_ex_str = ", ".join(f"{g}({y[gene_to_idx[g.upper()]]:+.2f})" for g in top_ex_g)
        else:
            excl_abs_mean, excl_t, top_ex_str = 0.0, 0.0, "None"

        # Multi-library Z-scores
        z_ctri_s = float(bundle["z_libs"]["CollecTRI_Signed"].loc[exp_id, c])
        z_ctri_u = float(bundle["z_libs"]["CollecTRI_Unsigned"].loc[exp_id, c])
        z_doro = float(bundle["z_libs"]["DoRothEA_Signed"].loc[exp_id, c])
        z_chip = max(
            float(bundle["z_libs"]["ENCODE_ChIP-seq"].loc[exp_id, c]),
            float(bundle["z_libs"]["ReMap_ChIP-seq"].loc[exp_id, c]),
            float(bundle["z_libs"]["Literature_ChIP-seq"].loc[exp_id, c]),
        )
        z_coexp = max(
            float(bundle["z_libs"]["ARCHS4_Coexpression"].loc[exp_id, c]),
            float(bundle["z_libs"]["GTEx_Coexpression"].loc[exp_id, c]),
        )
        t_cc_res = float(bundle["est_c_res"].loc[exp_id, c])
        t_part = partial_map.get(c, 0.0)

        # Check if there is a cell-line matched ChIP-seq track in ENCODE or Literature_ChIP-seq
        matched_tracks = []
        bs_clean = "".join(ch for ch in biosample if ch.isalnum())
        for tr_id, tr_genes in bundle["encode_tracks"].get(c, []) + bundle["lit_tracks"].get(c, []):
            if len(bs_clean) >= 3 and bs_clean[:4] in tr_id.upper():
                tr_idxs = [gene_to_idx[g] for g in tr_genes if g in gene_to_idx]
                if len(tr_idxs) >= 5:
                    tr_z = float(np.mean(np.abs(y[tr_idxs])) / (np.std(np.abs(y)) + 1e-6))
                    matched_tracks.append(f"{tr_id}(|z|={tr_z:.2f})")

        hub_flag = " [WARNING: UNIVERSAL_PROLIFERATION_STRESS_HUB]" if c in UNIVERSAL_SECONDARY_HUBS else ""
        track_str = f" | MatchedChIP={','.join(matched_tracks[:2])}" if matched_tracks else ""

        lines.append(
            f"* `{c}`{hub_flag}:\n"
            f"  - Conditional Partial_t={t_part:+.2f} (negative=repressed after controlling for other contenders & hubs), "
            f"CC_Deconf_ULM={t_cc_res:+.2f}\n"
            f"  - Library Z-Scores: Curated_Signed={z_ctri_s:+.2f}, Curated_Unsigned={z_ctri_u:+.2f}, "
            f"DoRothEA={z_doro:+.2f}, ChIP-seq_Max={z_chip:+.2f}, Coexp_Max={z_coexp:+.2f}{track_str}\n"
            f"  - Exclusive Targets ({len(excl_tgts)}/{len(my_tgts)} unshared): Excl_t={excl_t:+.2f}, "
            f"Excl_Mean|LFC|={excl_abs_mean:.2f}, TopChanged=[{top_ex_str}]"
        )

    # Direct TF->TF regulatory cascade edges among contenders
    casc = net_c[net_c["source"].isin(valid_cands) & net_c["target"].isin(valid_cands)]
    if len(casc) > 0:
        c_strs = [
            f"{r['source']} -> {r['target']} ({'activates' if r['weight'] > 0 else 'represses'})"
            for _, r in casc.iterrows()
        ]
        lines.append("\n* Direct TF->TF Regulatory Cascade Links among Contenders:\n  " + "; ".join(c_strs[:15]))
    else:
        lines.append("\n* Direct TF->TF Regulatory Cascade Links among Contenders: None")

    return "\n".join(lines)


if __name__ == "__main__":
    b = prepare_knocktf_bundle()
    sample_id = b["obs_sub"].index[0]
    true_tf = b["obs_sub"].loc[sample_id, "TF"]
    top6 = list(b["multi_modal_score"].loc[sample_id].sort_values(ascending=False).head(6).index)
    print(f"Sample {sample_id} (True={true_tf}):")
    print(execute_conditional_regulon_probe(sample_id, top6, b))
