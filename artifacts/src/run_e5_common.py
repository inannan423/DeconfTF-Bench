#!/usr/bin/env python3
"""Shared E5 prompt rules. Does not call a model.

Condition B replaces the V1 candidate symbols with IDs. V1's evidence lines
contain no differentially expressed gene symbols, only the 25 names and
numeric evidence, so the ID map is those 25 names. Adding a gene list would
change what V1 sees relative to conditions A and C.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from run_a2_pilot_design import redact_tf_tokens, symbol_in_text
from run_e4_agent_variants import _context, _evidence

ROOT = Path("/Users/bytedance/codes/new-bishe")
OUT = ROOT / "artifacts" / "results" / "cand6_deconftf" / "agent"
BUNDLE_DIR = OUT / "e5_bundles"
FROZEN = {
    ROOT / "artifacts" / "src" / "deconftf_tools.py":
        "45939716963851d980d5767d9285832efd5b4b3537ce014b0e44f7affa225490",
    ROOT / "artifacts" / "src" / "develop_m10_soft_gate.py":
        "1e513f5236e56842e5ff96de3263175498ba9c1e6662ab8f228de3a1d7500677",
}
B_SEED_LABEL = "e5-condition-b-v1:42"
REPLOGLE_STUDY = "Replogle 2022 K562 genome-wide Perturb-seq"
E5_CAP_CNY = 175.0
DATE_CUTOFF = "2025-05-31"


def assert_frozen() -> None:
    for path, expected in FROZEN.items():
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != expected:
            raise SystemExit(f"frozen hash changed: {path.name} {got}")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def blind_seed(exp_id: str) -> int:
    digest = hashlib.sha256(f"{B_SEED_LABEL}:{exp_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def id_map(symbols: list[str], exp_id: str) -> dict[str, str]:
    """One permutation per experiment. The same symbol gets one ID."""
    ordered = list(dict.fromkeys(str(item) for item in symbols))
    rng = np.random.default_rng(blind_seed(exp_id))
    shuffled = ordered.copy()
    rng.shuffle(shuffled)
    return {symbol: f"ID{i:02d}" for i, symbol in enumerate(shuffled, start=1)}


def anonymize_evidence(text: str, mapping: dict[str, str]) -> str:
    lines = []
    for line in text.splitlines():
        rank, rest = line.split(". ", 1)
        symbol, tail = rest.split(" ", 1)
        if symbol not in mapping:
            raise SystemExit(f"evidence symbol {symbol} is not in the ID map")
        lines.append(f"{rank}. {mapping[symbol]} {tail}")
    return "\n".join(lines)


def prompt_bundle(bundle: dict, cohort: str) -> dict:
    """Metadata the model is allowed to see. Scoring still uses the original bundle."""
    out = dict(bundle)
    obs = bundle["obs_sub"].copy()
    if cohort == "geo":
        obs["Tissue.Type"] = "not provided"
    elif cohort == "replogle":
        obs["Biosample.Name"] = "K562"
        obs["Tissue.Type"] = "not provided"
    elif cohort not in {"unseen", "dev"}:
        raise SystemExit(cohort)
    out["obs_sub"] = obs
    return out


def study_for(cohort: str, study_id: str, tf: str) -> tuple[str, int]:
    if cohort == "replogle":
        text = REPLOGLE_STUDY
    else:
        text = str(study_id)
    redacted, n = redact_tf_tokens(text, tf)
    if symbol_in_text(redacted, tf):
        raise SystemExit(f"study id still contains {tf}")
    return redacted, n


def context_line(obs, study_id: str, condition: str, tf: str, cohort: str) -> tuple[str, int]:
    study, n_study = study_for(cohort, study_id, tf)
    if condition == "B":
        condition = "A"
    text = _context(obs, study, condition, tf)
    if symbol_in_text(text, tf):
        raise SystemExit(f"context still contains {tf}")
    bio_n = redact_tf_tokens(str(obs["Biosample.Name"]), tf)[1]
    tissue_n = redact_tf_tokens(str(obs["Tissue.Type"]), tf)[1]
    return text, n_study + bio_n + tissue_n


def build_v1_prompt(exp_id, condition, study_id, bundle, base, cohort: str) -> tuple[str, int, dict]:
    view = prompt_bundle(bundle, cohort)
    obs = view["obs_sub"].loc[exp_id]
    tf = str(bundle["obs_sub"].loc[exp_id, "TF"]).upper()
    pool = [str(item) for item in base["m10_order"][:25]]
    evidence = _evidence(exp_id, base["m10_order"], bundle, pool)
    mapping: dict[str, str] = {}
    if condition == "B":
        mapping = id_map(pool, exp_id)
        evidence = anonymize_evidence(evidence, mapping)
        for symbol in pool:
            if symbol_in_text(evidence, symbol):
                raise SystemExit(f"{exp_id}: candidate {symbol} survived ID replacement")
        ask = (
            "Rerank the fixed-pipeline top 25. The candidate symbols have been replaced "
            "by random IDs. Return JSON only, no markdown. "
            'Schema: {"top5": ["ID", "..."]} with exactly 5 IDs from this list, '
            "most plausible primary knockdown first.\n"
        )
        shown = "A"
    else:
        ask = (
            "Rerank the fixed-pipeline top 25. Return JSON only, no markdown. "
            'Schema: {"top5": ["SYMBOL", "..."]} with exactly 5 symbols from this list, '
            "most plausible primary knockdown first.\n"
        )
        shown = condition
    ctx, n_redact = context_line(obs, study_id, shown, tf, cohort)
    prompt = ask + ctx + "\nCandidates, with deterministic evidence:\n" + evidence + "\n"
    if condition != "B" and symbol_in_text(ctx, tf):
        raise SystemExit(f"leakage: {exp_id} {condition}")
    if condition == "B" and symbol_in_text(prompt, tf):
        sealed, n_more = redact_tf_tokens(prompt, tf)
        if symbol_in_text(sealed, tf):
            raise SystemExit(f"{exp_id} condition B still contains {tf}")
        prompt, n_redact = sealed, n_redact + n_more
    return prompt, n_redact, mapping


def forbid_long_strings(prompt: str, banned: list[str], exp_id: str) -> None:
    for text in banned:
        if text and len(text) >= 12 and text in prompt:
            raise SystemExit(f"{exp_id}: a banned metadata field is in the prompt")


def e5_purpose_spent(entries: list[dict]) -> float:
    total = 0.0
    for entry in entries:
        purpose = str(entry.get("purpose") or "")
        if purpose.startswith("e5"):
            total += float(entry.get("cost_cny") or 0.0)
    return round(total, 6)


def v3_prompt(exp_id, condition, study_id, bundle, base, cohort: str) -> tuple[str, int]:
    from run_a2_pilot_design import build_prompt

    view = prompt_bundle(bundle, cohort)
    tf = str(bundle["obs_sub"].loc[exp_id, "TF"]).upper()
    study, n_study = study_for(cohort, study_id, tf)
    text, n_prompt = build_prompt(exp_id, condition, view, study, base["m10_order"])
    return text, n_study + n_prompt


def v2_prompt(exp_id, condition, study_id, bundle, base, cohort: str) -> tuple[str, int]:
    view = prompt_bundle(bundle, cohort)
    obs = view["obs_sub"].loc[exp_id]
    tf = str(bundle["obs_sub"].loc[exp_id, "TF"]).upper()
    pool = ", ".join(str(item) for item in base["m10_order"][:25])
    ctx, n_redact = context_line(obs, study_id, condition, tf, cohort)
    prompt = (
        "Identify the primary knocked-down transcription factor. "
        "Reply with JSON only. A tool request has the form "
        '{"tool": "verify_hypothesis" or "exclusive_target_test" or "cascade_check", "tfs": ["SYMBOL"]}. '
        "At most 8 symbols per request and at most 6 tool requests. "
        "When you are done, reply {\"top5\": [five symbols]} or {\"abstain\": true}. "
        "Do not invent numbers. Only a tool response contains numeric evidence. "
        "You may name a symbol outside the starting list after a tool has scored it.\n"
        f"{ctx}\n"
        f"Fixed-pipeline top 25, names only: {pool}\n"
    )
    return prompt, n_redact


def p1_prompt(obs_row, cohort: str, study_id: str) -> tuple[str, int]:
    """Metadata only. GEO uses the same three fields as condition A. Unseen uses the E1 P1 fields."""
    tf = str(obs_row["TF"]).upper()
    n_redact = 0
    if cohort == "geo":
        bio, n_bio = redact_tf_tokens(obs_row["Biosample.Name"], tf)
        study, n_study = study_for(cohort, study_id, tf)
        lines = [f"Biosample = {bio}.", "Tissue = not provided.", f"Study = {study}."]
        n_redact = n_bio + n_study
    elif cohort == "unseen":
        lines = []
        for label, column in (
            ("Biosample", "Biosample.Name"),
            ("Tissue", "Tissue.Type"),
            ("PubMed", "Pubmed.ID"),
            ("Knock.Method", "Knock.Method"),
            ("Biosample.Type", "Biosample.Type"),
            ("Platform", "Platform"),
            ("Data.Source", "Data.Source"),
            ("Profile.ID", "Profile.ID"),
        ):
            value, n_one = redact_tf_tokens(obs_row[column], tf)
            n_redact += n_one
            lines.append(f"{label} = {value}.")
    else:
        raise SystemExit(f"P1 is not run on {cohort}")
    prompt = (
        "Which transcription factor was knocked down in this experiment? "
        "You are given metadata only. No expression values are provided. "
        "Return JSON only, no markdown. "
        'Schema: {"hypotheses": ["SYMBOL", "..."]} with exactly 10 HGNC symbols, '
        "most plausible first.\n"
        + "\n".join(lines)
        + "\n"
    )
    if symbol_in_text(prompt, tf):
        prompt, n_more = redact_tf_tokens(prompt, tf)
        n_redact += n_more
    if symbol_in_text(prompt, tf):
        raise SystemExit(f"P1 prompt still contains {tf}")
    return prompt, n_redact


def geo_pub_dates() -> pd.DataFrame:
    path = ROOT / "papers" / "deconf-regulon" / "agent_cutoff_geo51_dates.csv"
    dates = pd.read_csv(path, usecols=["contrast_id", "gse", "pub_date"])
    dates["pub_date"] = pd.to_datetime(dates["pub_date"], format="%Y/%m/%d")
    before = int((dates["pub_date"] <= pd.Timestamp(DATE_CUTOFF)).sum())
    after = int((dates["pub_date"] > pd.Timestamp(DATE_CUTOFF)).sum())
    if (before, after, len(dates)) != (29, 22, 51):
        raise SystemExit(f"GEO date split is {before}/{after} on {len(dates)}, not 29/22 of 51")
    return dates
