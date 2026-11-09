# DeconfTF-Bench v1.0 — Pre-Registered Evaluation Protocol & Audit Suite

This release archive contains the frozen deterministic scoring code, language-model agent evaluation scripts, pre-registration manifests, per-experiment rank tables, and supplementary spreadsheets for:

> **A pre-registered evaluation of language-model agents for primary transcription-factor identification**

## Key Contents
- `artifacts/src/deconftf_tools.py` (SHA-256 `45939716963851d980d5767d9285832efd5b4b3537ce014b0e44f7affa225490`): Frozen deterministic scoring pipeline (`ULM`, `VIPER-Shadow`, `M5`, `M9`) and 1,888-regulator strict blinding.
- `artifacts/src/develop_m10_soft_gate.py` (SHA-256 `1e513f5236e56842e5ff96de3263175498ba9c1e6662ab8f228de3a1d7500677`): Frozen coverage-modulated soft gate (`M10`).
- `artifacts/src/regagent_tools.py`, `run_a*.py`, `run_e*.py`: Deterministic agent tool implementations, pre-flight leakage assertions (including divergent-transcript redaction D8), memory probes (E1), counterfactual metadata swaps (E2), in-pool/ReAct variants (E4), and frozen external evaluation (E5).
- `papers/deconf-regulon/FROZEN_MANIFEST.md` & `e5_preregistration.md`: Cryptographic pre-registration manifests and deviation register (D1–D8).
- `papers/deconf-regulon/Supplementary_Tables_S1_to_S5.xlsx` & `Supplementary_Material.pdf`: Multi-sheet Excel workbook and compiled PDF for Supplementary Tables S1–S5 and E5 per-experiment predictions.
- `scripts/reproduce_all.sh`: One-command local verification script.
