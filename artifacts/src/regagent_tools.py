#!/usr/bin/env python3
"""Deterministic RegAgent tools (A1). No LLM calls.

Tools call frozen `deconftf_tools` / `develop_m10_soft_gate` and do not modify them.
`m11_order` is a fixed fusion of those tools. Its weights were declared before any
Dev Top-1 was computed and are not a searched grid. Do not run this module on
Unseen, Replogle, or GEO in the A1 round.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/bytedance/codes/new-bishe/artifacts/src")
from deconftf_tools import load_gmt_with_tracks, m9_order
from develop_m10_soft_gate import m10_soft_order

ROOT = Path("/Users/bytedance/codes/new-bishe")
DATA_DIR = ROOT / "artifacts" / "data" / "knocktf"

MIN_TARGETS = 5
CONTEXT_Z_WEIGHT = 1.0
EXCLUSIVE_Z_WEIGHT = 0.25
CASCADE_PENALTY = 0.5
N_M5_CONTENDERS = 8
N_CTX_CONTENDERS = 8

ASSEMBLIES = {"HG19", "HG38", "MM9", "MM10", "RN4", "RN6"}
SPECIES = {"HUMAN", "MOUSE", "RAT"}
ASSAYS = {"CHIPSEQ", "CHIPCHIP", "CHIPPET", "CHIPPCR"}
CELL_GROUPS = (
    frozenset({"HELA", "HELAS3"}),
    frozenset({"HEK293", "HEK293T", "HEK293FT", "293T"}),
    frozenset({"H1", "H1HESC", "H1ESC"}),
    frozenset({"MCF7", "MCF7ATCC"}),
    frozenset({"MDAMB231", "MDA231"}),
    frozenset({"SHSY5Y", "SY5Y"}),
    frozenset({"RPE1", "HTERTRPE1"}),
    frozenset({"GM12878", "GM12891", "GM12892"}),
    # HUVEC is the biosample name of the ENCODE ontology token, not a result-tuned alias.
    frozenset({"HUVEC", "ENDOTHELIALCELLOFUMBILICALVEIN"}),
)


@dataclass(frozen=True)
class ChipTrack:
    library: str
    track_id: str
    tf: str
    cell: Optional[str]
    species: Optional[str]
    style: str
    genes: tuple


def parse_track_name(track_id: str) -> Dict[str, Optional[str]]:
    """Parse a ChEA3-style ChIP track id into TF, cell/tissue token, and species.

    ENCODE: ``STAT1_HELAS3_HG19`` -> cell ``HELAS3``, species from the assembly.
    Literature: ``VDR_24763502_CHIPSEQ_THP1_HUMAN`` -> cell ``THP1``.
    ReMap bare symbols (``ADNP``) have no cell token.
    """
    parts = [p for p in str(track_id).split("_") if p]
    tf = parts[0].upper() if parts else ""
    if len(parts) >= 3 and parts[-1].upper() in ASSEMBLIES:
        genome = parts[-1].upper()
        cell = "_".join(parts[1:-1]).upper() or None
        if genome.startswith("HG") or genome.startswith("GRCH"):
            species = "HUMAN"
        elif genome.startswith("MM"):
            species = "MOUSE"
        elif genome.startswith("RN"):
            species = "RAT"
        else:
            species = "OTHER"
        return {"tf": tf, "cell": cell, "species": species, "style": "encode_assembly", "genome": genome}
    assay_i = next((i for i, p in enumerate(parts) if p.upper() in ASSAYS), None)
    if assay_i is not None and assay_i < len(parts) - 1:
        rest = parts[assay_i + 1 :]
        if rest[-1].upper() in SPECIES:
            species = rest[-1].upper()
            cell = "_".join(rest[:-1]).upper() or None
        else:
            species = "UNKNOWN"
            cell = "_".join(rest).upper() or None
        return {"tf": tf, "cell": cell, "species": species, "style": "literature_assay", "genome": None}
    return {"tf": tf, "cell": None, "species": None, "style": "no_cell", "genome": None}


def load_track_catalog() -> List[ChipTrack]:
    """Load ENCODE, ReMap, and Literature ChIP tracks via frozen ``load_gmt_with_tracks``."""
    files = {
        "ENCODE_ChIP-seq": DATA_DIR / "ENCODE_ChIP-seq.gmt",
        "ReMap_ChIP-seq": DATA_DIR / "ReMap_ChIP-seq.gmt",
        "Literature_ChIP-seq": DATA_DIR / "Literature_ChIP-seq.gmt",
    }
    catalog: List[ChipTrack] = []
    for library, path in files.items():
        _net, tracks = load_gmt_with_tracks(path)
        for tf, items in tracks.items():
            for track_id, genes in items:
                meta = parse_track_name(track_id)
                catalog.append(
                    ChipTrack(
                        library=library,
                        track_id=track_id,
                        tf=str(tf).upper(),
                        cell=meta["cell"],
                        species=meta["species"],
                        style=str(meta["style"]),
                        genes=tuple(genes),
                    )
                )
    return catalog


def track_parse_table(catalog: Sequence[ChipTrack]) -> pd.DataFrame:
    rows = []
    for library in ("ENCODE_ChIP-seq", "ReMap_ChIP-seq", "Literature_ChIP-seq"):
        sub = [t for t in catalog if t.library == library]
        human = [t for t in sub if t.cell and t.species == "HUMAN"]
        rows.append(
            {
                "library": library,
                "n_tracks": len(sub),
                "n_with_cell_token": sum(t.cell is not None for t in sub),
                "n_human_with_cell": len(human),
                "n_mouse": sum(t.species == "MOUSE" for t in sub),
                "n_no_cell": sum(t.cell is None for t in sub),
                "n_unique_human_cell_tokens": len({t.cell for t in human}),
                "n_unique_tfs": len({t.tf for t in sub}),
            }
        )
    return pd.DataFrame(rows)


def _compact(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def _cell_key(cell: str) -> str:
    return _compact(cell)


def context_keys(biosample: str, tissue: str) -> set:
    """Exact cell-line keys taken from the biosample name only.

    ``SK-N-MC`` becomes ``SKNMC``. ``Primary lymphatic endothelial cells``
    stays one long string, so ``ENDOTHELIAL`` cannot hit
    ``BRAINMICROVASCULARENDOTHELIALCELL``. ``tissue`` is the organ of origin
    in KnockTF (Lung, Pancreas, Kidney). It is not a cell-line name, and a
    track token such as ``LUNG`` must not match every lung-derived line.
    """
    del tissue
    keys: set = set()
    token = _compact(biosample)
    if len(token) >= 3:
        keys.add(token)
    extra: set = set()
    for group in CELL_GROUPS:
        if keys & group:
            extra |= set(group)
    keys |= extra
    return keys


def track_matches_context(cell: str, keys: set, *, biosample_compact: str = "") -> bool:
    token = _cell_key(cell)
    if len(token) < 3 or not keys:
        return False
    if token in keys:
        return True
    for group in CELL_GROUPS:
        if token in group and (keys & group):
            return True
    # Containment uses the biosample's compact cell-line name only, not tissue
    # words and not split anatomy tokens. Length >= 4 on the shorter side.
    probes = {biosample_compact} if len(biosample_compact) >= 4 else set()
    for group in CELL_GROUPS:
        if biosample_compact in group:
            probes |= {p for p in group if len(p) >= 4}
    for probe in probes:
        shorter, longer = (probe, token) if len(probe) <= len(token) else (token, probe)
        if len(shorter) >= 4 and shorter in longer:
            return True
    return False


def match_context_tracks(
    biosample: str,
    tissue: str,
    catalog: Sequence[ChipTrack],
    *,
    human_only: bool = True,
) -> List[ChipTrack]:
    """Deterministic cell-line match. No LLM.

    A human track matches when its cell token equals the punctuation-stripped
    biosample, shares a fixed synonym group, or contains / is contained in
    that compact cell-line name (shorter side length >= 4). Tissue is ignored.
    ReMap tracks have no cell token and never match. A mixed track such as
    ``HELAANDSCP4`` matches HeLa because the cell-line name is contained in
    the token. A generic biosample such as ``Fibroblast`` likewise matches
    every track token that contains that word.
    """
    keys = context_keys(biosample, tissue)
    biosample_compact = _compact(biosample)
    matched = []
    for track in catalog:
        if track.cell is None:
            continue
        if human_only and track.species != "HUMAN":
            continue
        if track_matches_context(track.cell, keys, biosample_compact=biosample_compact):
            matched.append(track)
    return matched


def ulm_t_binary(y: np.ndarray, target_idx: Sequence[int], min_targets: int = MIN_TARGETS) -> float:
    """t-statistic of an intercept + binary-weight univariate linear model.

    Same estimand as decoupler ``mt.ulm`` on a 0/1 target indicator when empty
    features are kept (``empty=False``). On a one-row matrix the default
    ``empty=True`` drops genes whose log2FC is 0, which is not how the frozen
    multi-sample ULM was fit. Negative values mean the targets are lower than
    the rest of the transcriptome.
    """
    return _ulm_from_centered(y - y.mean(), float(np.dot(y - y.mean(), y - y.mean())), y.shape[0], target_idx, min_targets)


def _ulm_from_centered(
    yc: np.ndarray,
    ssy: float,
    n: int,
    target_idx: Sequence[int],
    min_targets: int,
) -> float:
    idx = np.asarray(list(target_idx), dtype=np.int32)
    k = int(idx.size)
    if k < min_targets or k >= n - 1:
        return float("nan")
    sum_t = float(yc[idx].sum())
    sxx = k * (1.0 - k / n)
    if sxx <= 1e-12:
        return float("nan")
    b1 = sum_t / sxx
    sse = max(0.0, ssy - (b1 * b1) * sxx)
    se = np.sqrt((sse / (n - 2)) / sxx)
    if se <= 1e-12:
        return float("nan")
    return float(b1 / se)


def score_target_groups(
    y: np.ndarray,
    grouped: Mapping[str, Sequence[int]],
    cand: Iterable[str],
    n_tracks: Optional[Mapping[str, int]] = None,
    min_targets: int = MIN_TARGETS,
) -> pd.DataFrame:
    """ULM t for precomputed TF target-index sets. Same statistic as ``ulm_t_binary``."""
    y = np.asarray(y, dtype=np.float64)
    yc = y - float(y.mean())
    ssy = float(np.dot(yc, yc))
    n = int(y.shape[0])
    allowed = set(cand)
    rows = []
    for tf, idxs in grouped.items():
        if tf not in allowed:
            continue
        t_stat = _ulm_from_centered(yc, ssy, n, idxs, min_targets)
        if np.isnan(t_stat):
            continue
        rows.append(
            {
                "tf": tf,
                "t_ulm": t_stat,
                "n_targets": len(idxs),
                "n_tracks": 0 if n_tracks is None else int(n_tracks.get(tf, 0)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["tf", "t_ulm", "n_targets", "n_tracks"]).set_index("tf")
    return pd.DataFrame(rows).set_index("tf").sort_values("t_ulm")


def _target_indices(tracks: Iterable[ChipTrack], gene_to_idx: Mapping[str, int]) -> Dict[str, List[int]]:
    buckets: Dict[str, set] = {}
    for track in tracks:
        bucket = buckets.setdefault(track.tf, set())
        for gene in track.genes:
            if gene == track.tf:
                continue
            i = gene_to_idx.get(gene)
            if i is not None:
                bucket.add(i)
    return {tf: sorted(idxs) for tf, idxs in buckets.items()}


def context_regulon_test(
    exp_id: str,
    tracks: Sequence[ChipTrack],
    bundle: Dict[str, Any],
    min_targets: int = MIN_TARGETS,
) -> pd.DataFrame:
    """ULM on the union of the supplied tracks, one row per TF that has targets.

    TFs outside the frozen candidate set are dropped. A TF with fewer than
    ``min_targets`` genes present in the matrix is omitted rather than scored.
    """
    y = bundle["mat_blind"].loc[exp_id].to_numpy(dtype=np.float64)
    grouped = _target_indices(tracks, bundle["gene_to_idx"])
    n_tracks: Dict[str, int] = {}
    for track in tracks:
        n_tracks[track.tf] = n_tracks.get(track.tf, 0) + 1
    return score_target_groups(y, grouped, bundle["cand_tfs"], n_tracks, min_targets)


def _collectri_targets(bundle: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    cached = bundle.get("_regagent_collectri_targets")
    if cached is not None:
        return cached
    net = bundle["net_c"]
    out: Dict[str, Dict[str, float]] = {}
    for src, tgt, weight in zip(net["source"], net["target"], net["weight"]):
        out.setdefault(str(src), {})[str(tgt).upper()] = float(weight)
    bundle["_regagent_collectri_targets"] = out
    return out


def exclusive_target_test(
    exp_id: str,
    contenders: Sequence[str],
    bundle: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Signed enrichment on CollecTRI targets not shared with the other contenders.

    ``excl_t`` is the one-sample t of (log2FC * prior weight) on exclusive targets.
    For a knockdown of an activator this is negative.
    """
    targets = _collectri_targets(bundle)
    y = bundle["mat_blind"].loc[exp_id].to_numpy(dtype=np.float64)
    g2i = bundle["gene_to_idx"]
    valid = [c for c in contenders if c in targets]
    rows = []
    for tf in valid:
        mine = targets[tf]
        shared: set = set()
        for other in valid:
            if other != tf:
                shared.update(targets.get(other, {}))
        exclusive = [g for g in mine if g not in shared and g in g2i]
        excl_t = float("nan")
        if len(exclusive) >= 3:
            idx = [g2i[g] for g in exclusive]
            vals = y[idx] * np.array([mine[g] for g in exclusive], dtype=np.float64)
            excl_t = float(vals.mean() / (vals.std(ddof=1) / np.sqrt(len(vals)) + 1e-8))
        rows.append(
            {
                "tf": tf,
                "n_targets": len(mine),
                "n_exclusive": len(exclusive),
                "excl_t": excl_t,
            }
        )
    return rows


def cascade_check(
    exp_id: str,
    contenders: Sequence[str],
    bundle: Dict[str, Any],
    matched_tracks: Optional[Sequence[ChipTrack]] = None,
) -> List[Dict[str, Any]]:
    """Whether each contender is a direct target of another contender.

    CollecTRI supplies a signed edge. A matched ChIP track supplies an unsigned
    binding edge when the contender symbol sits in the other TF's track.
    ``exp_id`` is accepted so the tool has the same experiment scope as the others;
    the edges do not depend on expression.
    """
    del exp_id
    targets = _collectri_targets(bundle)
    chip_genes: Dict[str, set] = {}
    if matched_tracks:
        for track in matched_tracks:
            if track.tf in contenders:
                chip_genes.setdefault(track.tf, set()).update(track.genes)
    rows = []
    wanted = set(contenders)
    for tf in contenders:
        upstream = []
        for other in contenders:
            if other == tf:
                continue
            weight = targets.get(other, {}).get(tf)
            if weight is not None:
                upstream.append({"upstream": other, "via": "CollecTRI", "weight": weight})
            if tf in chip_genes.get(other, ()):
                upstream.append({"upstream": other, "via": "ChIP", "weight": None})
        rows.append({"tf": tf, "is_downstream": bool(upstream), "upstream": upstream, "n_contenders": len(wanted)})
    return rows


def _m10_triggered(exp_id: str, bundle: Dict[str, Any]) -> bool:
    """Same trigger as ``m10_soft_order`` (no enhanceosome term)."""
    from deconftf_tools import UNIVERSAL_SECONDARY_HUBS

    z = bundle["z_libs"]
    ulm = bundle["est_ulm"].loc[exp_id]
    cc = -bundle["est_c_res"].loc[exp_id]
    chip = pd.concat(
        [
            z["ENCODE_ChIP-seq"].loc[exp_id],
            z["ReMap_ChIP-seq"].loc[exp_id],
            z["Literature_ChIP-seq"].loc[exp_id],
        ],
        axis=1,
    ).max(axis=1)
    coexp = pd.concat(
        [z["ARCHS4_Coexpression"].loc[exp_id], z["GTEx_Coexpression"].loc[exp_id]],
        axis=1,
    ).max(axis=1)
    doro = z["DoRothEA_Signed"].loc[exp_id]
    t1 = ulm.sort_values(ascending=True).index[0]
    return bool(
        (t1 in UNIVERSAL_SECONDARY_HUBS and cc[t1] < 3.0)
        or (chip[t1] < 0.5)
        or (doro[t1] < 0.8 and chip[t1] < 1.5 and coexp[t1] > 1.2)
    )


def score_candidates(exp_id: str, bundle: Dict[str, Any], cov_frac: Optional[float] = None) -> Dict[str, Any]:
    """Frozen ULM, M5, M9, and M10 for one experiment."""
    ulm = bundle["est_ulm"].loc[exp_id]
    m5 = bundle["multi_modal_score"].loc[exp_id]
    m9 = m9_order(exp_id, bundle)
    m10, m10_score = m10_soft_order(exp_id, bundle, cov_frac=cov_frac)
    triggered = _m10_triggered(exp_id, bundle)
    if not triggered and m10[0] != ulm.sort_values(ascending=True).index[0]:
        raise RuntimeError(f"M10 trigger disagrees with m10_soft_order on {exp_id}")
    return {
        "ulm": ulm,
        "m5": m5,
        "m9_order": m9,
        "m10_order": m10,
        "m10_score": m10_score,
        "triggered": triggered,
    }


def _zscore(values: pd.Series) -> pd.Series:
    sd = float(values.std(ddof=0))
    if not np.isfinite(sd) or sd < 1e-8:
        return values * 0.0
    return (values - float(values.mean())) / sd


def m11_order(
    exp_id: str,
    bundle: Dict[str, Any],
    catalog: Sequence[ChipTrack],
    cov_frac: Optional[float] = None,
) -> List[str]:
    """Fixed tool order, then a declared rule fusion. Not an agent.

    1. ``score_candidates`` (ULM, M5, M9, M10).
    2. ``match_context_tracks`` on the sample biosample (tissue is not a cell line).
    3. ``context_regulon_test`` on the matched tracks.
    4. ``exclusive_target_test`` on ULM top-1 plus M5 top-8 plus context top-8.
    5. ``cascade_check`` on that same contender set.
    6. Score = M5 + 1.0 * z(-context t) + 0.25 * z(-exclusive t) - 0.5 if the
       TF is downstream of a contender with a higher M5 score.
    7. If the frozen M10 trigger is off, pin the ULM top TF at rank 1.
    """
    base = score_candidates(exp_id, bundle, cov_frac=cov_frac)
    obs = bundle["obs_sub"].loc[exp_id]
    matched = match_context_tracks(str(obs["Biosample.Name"]), str(obs["Tissue.Type"]), catalog)
    ctx = context_regulon_test(exp_id, matched, bundle)
    m5 = base["m5"]
    ulm_order = list(base["ulm"].sort_values(ascending=True).index)
    m5_order = list(m5.sort_values(ascending=False).index)
    ctx_order = list(ctx.sort_values("t_ulm").index) if len(ctx) else []
    contenders: List[str] = []
    for tf in ulm_order[:1] + m5_order[:N_M5_CONTENDERS] + ctx_order[:N_CTX_CONTENDERS]:
        if tf not in contenders:
            contenders.append(tf)
    excl_rows = exclusive_target_test(exp_id, contenders, bundle)
    casc_rows = cascade_check(exp_id, contenders, bundle, matched_tracks=matched)

    score = m5.astype(float).copy()
    if len(ctx):
        score.loc[ctx.index] = score.loc[ctx.index] + CONTEXT_Z_WEIGHT * _zscore(-ctx["t_ulm"])
    excl = pd.Series({r["tf"]: r["excl_t"] for r in excl_rows}, dtype=float).dropna()
    if len(excl):
        score.loc[excl.index] = score.loc[excl.index] + EXCLUSIVE_Z_WEIGHT * _zscore(-excl)
    m5_map = m5.to_dict()
    for row in casc_rows:
        if not row["is_downstream"]:
            continue
        tf = row["tf"]
        if any(m5_map.get(u["upstream"], -np.inf) > m5_map.get(tf, -np.inf) for u in row["upstream"]):
            score.loc[tf] = score.loc[tf] - CASCADE_PENALTY
    order = list(score.sort_values(ascending=False).index)
    if not base["triggered"]:
        t1 = ulm_order[0]
        order = [t1] + [tf for tf in order if tf != t1]
    return order


def context_rank(t_ulm: pd.Series, m5: pd.Series) -> List[str]:
    """A1 ranking: context-scored TFs first, then the rest by M5.

    Kept so the committed A1 script still reproduces its table. It is the
    ranking that pushed unmatched true TFs out of Top-25. New work uses
    ``m10_plus_evidence_order``.
    """
    scored = list(t_ulm.dropna().sort_values(ascending=True).index)
    scored_set = set(scored)
    rest = [tf for tf in m5.sort_values(ascending=False).index if tf not in scored_set]
    return scored + rest


def m10_plus_evidence_order(
    m10_order: Sequence[str],
    m10_score: pd.Series,
    evidence_t: pd.Series,
    weight: float = CONTEXT_Z_WEIGHT,
) -> List[str]:
    """M10 score plus a bonus for TFs that have an enrichment t.

    The bonus is ``weight * z(-t)`` inside the set of TFs that have a finite t.
    A TF with no t keeps its M10 score. Ties break by the original M10 rank,
    not by symbol order. Declared as the A1.5 correction of ``context_rank``;
    the weight is the same 1.0 already declared for M11, not refit on rescue.
    """
    score = m10_score.astype(float).copy()
    if evidence_t is not None and len(evidence_t):
        bonus = _zscore(-evidence_t.dropna().astype(float))
        score.loc[bonus.index] = score.loc[bonus.index] + weight * bonus
    m10_rank = pd.Series(np.arange(len(m10_order)), index=list(m10_order), dtype=float)
    frame = pd.DataFrame({"score": score, "m10_rank": m10_rank.reindex(score.index)})
    frame["m10_rank"] = frame["m10_rank"].fillna(float(len(m10_order)))
    frame = frame.sort_values(["score", "m10_rank"], ascending=[False, True], kind="mergesort")
    return list(frame.index)


def m11_m10_bonus_order(
    exp_id: str,
    bundle: Dict[str, Any],
    catalog: Sequence[ChipTrack],
    cov_frac: Optional[float] = None,
) -> List[str]:
    """A1.5 correction of ``m11_order``. Same tools, bonus sits on the M10 score.

    Context, exclusive-target, and cascade terms are added only for the TFs
    those tools actually score. Every other TF keeps its M10 score. The M10
    trigger pin is applied after the bonuses, as in the original M11. Weights
    stay 1.0 / 0.25 / 0.5.
    """
    base = score_candidates(exp_id, bundle, cov_frac=cov_frac)
    obs = bundle["obs_sub"].loc[exp_id]
    matched = match_context_tracks(str(obs["Biosample.Name"]), str(obs["Tissue.Type"]), catalog)
    ctx = context_regulon_test(exp_id, matched, bundle)
    m5 = base["m5"]
    ulm_order = list(base["ulm"].sort_values(ascending=True).index)
    m5_order = list(m5.sort_values(ascending=False).index)
    ctx_order = list(ctx.sort_values("t_ulm").index) if len(ctx) else []
    contenders: List[str] = []
    for tf in ulm_order[:1] + m5_order[:N_M5_CONTENDERS] + ctx_order[:N_CTX_CONTENDERS]:
        if tf not in contenders:
            contenders.append(tf)
    excl_rows = exclusive_target_test(exp_id, contenders, bundle)
    casc_rows = cascade_check(exp_id, contenders, bundle, matched_tracks=matched)

    evidence = ctx["t_ulm"] if len(ctx) else pd.Series(dtype=float)
    score = base["m10_score"].astype(float).copy()
    if len(evidence):
        bonus = _zscore(-evidence.dropna().astype(float))
        score.loc[bonus.index] = score.loc[bonus.index] + CONTEXT_Z_WEIGHT * bonus
    excl = pd.Series({r["tf"]: r["excl_t"] for r in excl_rows}, dtype=float).dropna()
    if len(excl):
        score.loc[excl.index] = score.loc[excl.index] + EXCLUSIVE_Z_WEIGHT * _zscore(-excl)
    m5_map = m5.to_dict()
    for row in casc_rows:
        if not row["is_downstream"]:
            continue
        tf = row["tf"]
        if any(m5_map.get(u["upstream"], -np.inf) > m5_map.get(tf, -np.inf) for u in row["upstream"]):
            score.loc[tf] = float(score.loc[tf]) - CASCADE_PENALTY
    m10_rank = pd.Series(np.arange(len(base["m10_order"])), index=list(base["m10_order"]), dtype=float)
    frame = pd.DataFrame({"score": score, "m10_rank": m10_rank.reindex(score.index).fillna(len(score))})
    frame = frame.sort_values(["score", "m10_rank"], ascending=[False, True], kind="mergesort")
    order = list(frame.index)
    if not base["triggered"]:
        t1 = ulm_order[0]
        order = [t1] + [tf for tf in order if tf != t1]
    return order


def ulm_t_weighted(y: np.ndarray, target_idx: Sequence[int], weights: Sequence[float], min_targets: int = MIN_TARGETS) -> float:
    """Pearson-correlation t of ``y`` against a weight vector. Same estimand as ULM.

    Non-target genes stay at weight 0. Matches ``decoupler.mt.ulm(..., empty=False)``
    on the same weights.
    """
    idx = np.asarray(list(target_idx), dtype=np.int32)
    w = np.asarray(list(weights), dtype=np.float64)
    if idx.size < min_targets or idx.size != w.size:
        return float("nan")
    x = np.zeros(int(y.shape[0]), dtype=np.float64)
    x[idx] = w
    yc = y - float(y.mean())
    xc = x - float(x.mean())
    sxx = float(np.dot(xc, xc))
    syy = float(np.dot(yc, yc))
    if sxx <= 1e-12 or syy <= 1e-12:
        return float("nan")
    r = float(np.dot(xc, yc)) / np.sqrt(sxx * syy)
    r = float(np.clip(r, -1.0 + 1e-12, 1.0 - 1e-12))
    df = float(y.shape[0] - 2)
    return float(r * np.sqrt(df / ((1.0 - r) * (1.0 + r))))


def prepare_hypothesis_resources(bundle: Dict[str, Any], human_tracks: Sequence[ChipTrack]) -> Dict[str, Any]:
    """Target indices for any-cell human ChIP, CollecTRI, and DoRothEA.

    DoRothEA is read from ``dorothea_human.csv``. The frozen scoring code is not modified.
    """
    g2i = bundle["gene_to_idx"]
    chip = _target_indices(human_tracks, g2i)
    collectri: Dict[str, List[tuple]] = {}
    net = bundle["net_c"]
    for src, tgt, weight in zip(net["source"], net["target"], net["weight"]):
        i = g2i.get(str(tgt).upper())
        if i is None or str(src) == str(tgt).upper():
            continue
        collectri.setdefault(str(src), []).append((i, float(weight)))
    dorothea: Dict[str, List[tuple]] = {}
    doro = pd.read_csv(DATA_DIR / "dorothea_human.csv", usecols=["source", "target", "weight"])
    for src, tgt, weight in zip(doro["source"], doro["target"], doro["weight"]):
        src_u = str(src).upper()
        tgt_u = str(tgt).upper()
        if src_u == tgt_u:
            continue
        i = g2i.get(tgt_u)
        if i is None:
            continue
        dorothea.setdefault(src_u, []).append((i, float(weight)))
    return {"chip": chip, "collectri": collectri, "dorothea": dorothea}


def verify_hypothesis(
    exp_id: str,
    tf: str,
    bundle: Dict[str, Any],
    resources: Mapping[str, Any],
    m10_order: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Enrichment test for one hypothesized TF. No LLM.

    Prefer the union of that TF's ChIP tracks in any human cell line. If fewer
    than 5 target genes are in the matrix, fall back to signed CollecTRI, then
    signed DoRothEA. ``t_stat`` is the ULM correlation t (negative supports a
    knockdown of an activator). ``rank_if_merged_alone`` applies
    ``merge_verified_into_m10`` to this single hypothesis.
    """
    tf = str(tf).upper()
    y = bundle["mat_blind"].loc[exp_id].to_numpy(dtype=np.float64)
    source = "none"
    t_stat = float("nan")
    n_targets = 0
    chip = resources["chip"].get(tf) or []
    if len(chip) >= MIN_TARGETS:
        t_stat = ulm_t_weighted(y, chip, np.ones(len(chip), dtype=np.float64))
        source = "chip_any_human_cell"
        n_targets = len(chip)
    else:
        col = resources["collectri"].get(tf) or []
        if len(col) >= MIN_TARGETS:
            idx, w = zip(*col)
            t_stat = ulm_t_weighted(y, idx, w)
            source = "collectri"
            n_targets = len(col)
        else:
            doro = resources["dorothea"].get(tf) or []
            if len(doro) >= MIN_TARGETS:
                idx, w = zip(*doro)
                t_stat = ulm_t_weighted(y, idx, w)
                source = "dorothea"
                n_targets = len(doro)
    row = {"tf": tf, "t_stat": t_stat, "source": source, "n_targets": n_targets, "exp_id": exp_id}
    if m10_order is not None:
        merged = merge_verified_into_m10(m10_order, [row])
        row["rank_if_merged_alone"] = merged.index(tf) + 1 if tf in merged else None
    return row


def merge_verified_into_m10(m10_order: Sequence[str], verified: Sequence[Mapping[str, Any]]) -> List[str]:
    """Put knockdown-consistent hypotheses at the front of the M10 order.

    A hypothesis is knockdown-consistent when its ULM t is finite and negative.
    Those TFs are ordered by t ascending and placed ahead of the M10 list.
    A hypothesis with t >= 0 or no score stays where M10 put it. Every other
    TF keeps its relative M10 order.

    This is not the z-score bonus. On Dev that bonus left the any-cell oracle
    at 12/142, so it cannot move a distant TF into Top-25. The insertion rule
    was declared from that Dev result, before the non-LLM control rates were
    computed. It is not a searched threshold.
    """
    supported = []
    seen = set()
    for row in verified:
        tf = str(row["tf"])
        t_stat = row.get("t_stat")
        if tf in seen or t_stat is None or not np.isfinite(t_stat) or t_stat >= 0:
            continue
        if tf not in set(m10_order):
            continue
        supported.append((float(t_stat), tf))
        seen.add(tf)
    supported.sort()
    front = [tf for _t, tf in supported]
    front_set = set(front)
    rest = [tf for tf in m10_order if tf not in front_set]
    return front + rest
