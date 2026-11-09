# Frozen Pipeline Manifest & Pre-Registration for External Validation (P0-1)

**Frozen Timestamp**: 2026-11-02 (prior to downloading or scoring any P0-2 external validation dataset)  
**Project Root**: `/Users/bytedance/codes/new-bishe`  
**Python Environment**: `/Users/bytedance/codes/new-bishe/.venv-bio/bin/python` (Python 3.12, `decoupler 2.3.1`)

---

## 1. Frozen Algorithm & Hyperparameters (`M9_Deterministic_Deconf`)

All external datasets in P0-2 must be scored by calling `score_knocktf_experiments(mat_sub, obs_sub, cand_tfs=cand_tfs)` and `m9_order(exp_id, bundle, ch_thr=0.5, cc_thr=3.0, use_enhanceosome=True)` in `artifacts/src/deconftf_tools.py` without modification:

1. **Candidate Set (`cand_tfs`)**: Fixed to the exact 647 transcription factors from the KnockTF development bundle (`knocktf_precomputed_bundle.pkl`).
2. **Self-Expression Blinding**:
   - All self-regulatory edges (`source == target`) are removed from all prior networks (`CollecTRI`, `DoRothEA`, and all 6 ChEA3 `.gmt` libraries).
   - Before any activity score is computed, the expression value of all 1,888 regulators in `CollecTRI ∪ DoRothEA ∪ ChEA3_GMT_sources` (`supplement_masked_tfs_list.csv`) is set to `0.0`.
3. **Evidence Layers & Static Multimodal Score (`M5`)**:
   - `ULM`: `decoupler.mt.ulm` on signed `CollecTRI`.
   - `Cell-Cycle-Residualised Activity (CC)`: QR-orthonormal projection orthogonal to signed `CollecTRI` target vectors of `["E2F4", "E2F1", "FOXM1", "MYC", "ATF4"]`, followed by `decoupler.mt.ulm` on IDF-weighted `CollecTRI`.
   - `M5` (`multi_modal_score`): Weighted average of 11 row-standardised library Z-scores with frozen weights:
     - `CollecTRI_Signed`: `1.5`, `CollecTRI_Abs`: `0.8`, `CollecTRI_Unsigned`: `1.0`
     - `DoRothEA_Signed`: `1.2`, `DoRothEA_Abs`: `0.6`
     - `ARCHS4_Coexpression`: `1.2`, `GTEx_Coexpression`: `0.8`
     - `ENCODE_ChIP-seq`: `1.0`, `ReMap_ChIP-seq`: `1.2`, `Literature_ChIP-seq`: `1.2`
     - `Enrichr_Queries`: `0.8`
4. **Selective Conditional Re-ranking (`M9`)**:
   - Gate parameters fixed at $\tau_{\text{ChIP}} = 0.5$, $\tau_{\text{CC}} = 3.0$.
   - Universal secondary hubs fixed at `{"E2F4", "E2F1", "E2F2", "E2F3", "TFDP1", "FOXM1", "MYBL2", "MYC", "ATF4"}`.
   - Enhanceosome set fixed at `{"CIITA", "RFX5", "RFXAP", "RFXANK", "SIN3A"}`.
   - Confounding flag triggers if $T_1 = \arg\min \text{ULM}$ satisfies:
     1. $T_1 \in \text{UNIVERSAL\_SECONDARY\_HUBS}$ and $\text{CC}(T_1) < 3.0$; or
     2. $\text{ChIP\_Max\_Z}(T_1) < 0.5$; or
     3. $\text{DoRothEA\_Z}(T_1) < 0.8$ and $\text{ChIP\_Max\_Z}(T_1) < 1.5$ and $\text{Coexp\_Max\_Z}(T_1) > 1.2$; or
     4. $T_1 \in \text{ENHANCEOSOME\_COMPLEX}$.
   - If untriggered: rank order is $[T_1] + [c \in \text{M5\_order} : c \neq T_1]$.
   - If triggered: rank order demotes candidates failing the hub/ChIP-DoRothEA/enhanceosome checks to the tail of `M5_order`.

---

## 2. Cryptographic SHA-256 Checksums of Frozen Code, Priors & KnockTF Tables

```text
45939716963851d980d5767d9285832efd5b4b3537ce014b0e44f7affa225490  artifacts/src/deconftf_tools.py
36a7387c3eec3b814589971d8fcf0318ba4a0f932083be5072d7dab196a910ec  artifacts/src/rescore_holdout_harmonised.py
56446cd5a6bd17367dc8be1151820cb62f43e0487241de296827a88dd161e69e  artifacts/src/stats_harmonised_holdout.py
e2959ffaa20f415b86a38d9e0661f166e9c6978f95fbdb3c2e88437942d38a58  artifacts/src/recompute_study_level_stats.py
2ee437c164d37f2e4c19e12d3288d86b65ff0ccb8400703418441e4e31b73d90  artifacts/src/recover_missing_pmids.py
6ca6fbc6272f5e4ef716136b045c23fa1c07497d0bea10e06fa12352d2f7ccd4  artifacts/src/audit_reviewer_controls.py
da028731928ab8aae8f948021b550773dd52dc694e0a4bfc5623ea53aaa71c60  artifacts/src/verify_round4_accept_checks.py
ea15e2d37ab9723823ff00cc8b411ff08562957a2feb3151859b16fa50ad4c7d  artifacts/src/compute_p0_0_manuscript_tables.py
f4706bde7185746f9263ff83ca2cc3a31078d2bb7580305cf53182fe08757f79  artifacts/data/knocktf/collectri_human.csv
5462a0cab477b1ade0a26c192ff5e8b1bec8bdcff2937090b1ef960d7b2c2f6f  artifacts/data/knocktf/dorothea_human.csv
2217d6753a080d420a2e2ea1102371bda895201d2c9f41f50972e3b2704e1243  artifacts/data/knocktf/progeny_human.csv
80660361602d5c10549a8b21f7ee2290c041e21bde461130aaa6db5333751715  artifacts/data/knocktf/ENCODE_ChIP-seq.gmt
4e548946fb50847f215739a6e68d97bb2b6334f91be33d1a5ea01a596993518a  artifacts/data/knocktf/ReMap_ChIP-seq.gmt
458d6a59386188ec0bd2925bbc38d412311179ff7b87f0b28881c55d88661659  artifacts/data/knocktf/Literature_ChIP-seq.gmt
d80648684c9e9813bba8f55c5f81bc1d0cef994e53c8955bddcc4c43feeceaba  artifacts/data/knocktf/ARCHS4_Coexpression.gmt
223d732d911f395fe7e7a2c906e62b38e3311c6d5cb51c8e89a942fce27e404d  artifacts/data/knocktf/GTEx_Coexpression.gmt
99011b8da78c743e2a93d6cf28f8ab82a47d5ac43d3e708b42cc896aa1a978f7  artifacts/data/knocktf/Enrichr_Queries.gmt
ab1b79bd2036af88d6ad5c43944a3e32d3f5e9c452aae0f7f799ec182383b647  artifacts/data/knocktf/knocktf401_study_ids.csv
b9e1b7333431a3810901cf1e1df25f5d6e4d589da92858daba719082efc99857  artifacts/results/cand6_deconftf/knocktf401_ranks_with_study_ids.csv
4d62e4801091b57a4493cbdf5a8886f73af6d3491b09df26e9b10a2c36649e8a  artifacts/results/cand6_deconftf/holdout135_harmonised_ranks.csv
21c17de6be275b60aea68aeaa6869c98b9f5bee55bacc7fd00a0e6a2df93e8ff  artifacts/results/cand6_deconftf/study_level_stats_harmonised.csv
affdd4b98115d12e2f930e9defc51d6064ad2079455db21502005d0835369e44  artifacts/results/cand6_deconftf/m9_failure_decomposition.csv
2f914145b991a0c010f857dd4287191655b387638fe4257b1799b5602c59a3cd  artifacts/results/cand6_deconftf/m9_risk_coverage.csv
9253e51942cef2fad37a577c6b75a3100ca7d1f630e6bd01b5e6e5a2cc4f1f84  artifacts/results/cand6_deconftf/supplement_masked_tfs_list.csv
```

---

## 3. Pre-Registered External Validation Protocol (P0-2)

### 3.1 External Datasets
1. **Primary External Cohort — Replogle et al. 2022 (*Cell*) K562 Genome-Wide Perturb-seq Pseudobulk**:
   - Source: Replogle et al. 2022 published pseudobulk `.h5ad` file (`K562_gwps_normalized_bulk_01.h5ad` from Figshare; plus `rpe1_normalized_bulk_01.h5ad` as an additional cell-line check if available).
   - Inclusion criteria:
     - Perturbed target gene symbol is in the 647 frozen candidate TFs (`cand_tfs`).
     - Primary cohort filter: effective on-target knockdown defined by target-gene expression shift $\le -0.5$ (in the dataset's control-normalized log2FC / z-score units, or `pct_knockdown >= 30%` if raw knockdown fraction is provided in `obs`), with at least one non-blinded gene having $|\text{shift}| \ge 0.5$.
     - If multiple perturbations target the same TF in the same cell line, collapse to **1 profile per unique TF** by averaging the pseudobulk profiles across on-target perturbations for that TF before scoring (so the unit of analysis is the unique perturbed TF).
     - Stratification by knockdown strength (addressing Reviewer Concern M3): report results across all unique candidate TFs with negative on-target shift, moderate knockdown ($-1.0 < \text{on-target shift} \le -0.5$), and strong knockdown ($\text{on-target shift} \le -1.0$).
2. **Secondary External Cohort — Non-KnockTF TF Perturbation RNA-seq (ENCODE / GEO)**:
   - Source: ENCODE human TF knockdown/CRISPRi RNA-seq (K562 / HepG2) or post-KnockTF GEO TF perturbation series, after strictly excluding any experiment whose ENCODE/GEO accession (or PubMed ID) appears in `artifacts/data/knocktf/knockTF_meta.csv` or `knocktf401_study_ids.csv`.
   - Inclusion criteria: Target TF in the 647 frozen `cand_tfs`, target-gene $\text{log2FC} \le -0.5$, and $\max |\text{log2FC}| \ge 0.5$ after blinding.

### 3.2 Endpoints & Statistical Tests
- **Unit of inference**: Unique perturbed TF ($N_{\text{TF}}$) within each external dataset (or study-clustered permutation if multiple studies are pooled).
- **Primary Endpoints**:
  1. **Gating increment**: `M9` vs `M5` on $\Delta \log_2 \text{rank} = \log_2 r_{\text{M5}} - \log_2 r_{\text{M9}}$ (two-sided Wilcoxon signed-rank test, `wilcox` mode with `pratt` reported alongside, plus 5,000-resample bootstrap 95% CI and TF-Class clustered permutation p-value).
  2. **Overall recovery vs standard baseline**: `M9` vs `ULM` on $\Delta \log_2 \text{rank}$ (two-sided Wilcoxon signed-rank test) and Top-5 recovery (exact two-sided McNemar test), with Holm-Bonferroni adjustment across the primary external endpoint family.
- **Secondary Endpoints**:
  - `M9` vs `VIPER shadow` ($\Delta \log_2 \text{rank}$, Top-1, Top-5), `M5` vs `ULM`, MRR, and knockdown-strength stratified results.

### 3.3 Pre-Registered Decision Matrix (Go / No-Go)

| Decision Outcome | Pre-Registered Criterion | Manuscript & Venue Action |
| :--- | :--- | :--- |
| **Outcome A: Selective Gating Supported** | In $\ge 1$ external dataset with $\ge 100$ valid TFs: `M9` vs `M5` has mean/median $\Delta \log_2 \text{rank} > 0$ with two-sided $p < 0.05$; **AND** `M9` vs `ULM` is significant on both $\Delta \log_2 \text{rank}$ and Top-5 ($p_{\text{Holm}} < 0.05$); and the second external dataset has a consistent positive direction. | Proceed with *Bioinformatics* Original Paper framing; keep selective conditioning in the title. |
| **Outcome B: Multimodal Fusion Supported, Gating Not Supported** | Both `M9` and `M5` significantly outperform `ULM` ($\Delta \log_2 \text{rank}$ $p < 0.05$), but `M9` vs `M5` is not statistically significant ($p \ge 0.05$). | Reframe manuscript around "Multi-modal de-confounded benchmark audit + bottleneck decomposition" for *Briefings in Bioinformatics*; revise title to reflect multimodal fusion and benchmark pitfalls rather than selective gating. |
| **Outcome C: Neither Supported** | `M9` does not significantly outperform `ULM` on external validation. | Report negative generalisation results transparently without re-tuning parameters; target *BMC Bioinformatics* as a benchmark audit and negative-result study. |

**Pre-registration defect (recorded after GEO scoring; wording not retroactively changed):** Outcome B in this P0-1 §3.3 table requires **both** `M9` and `M5` to outperform `ULM`. v2 §4.5 later defined B as "`M5` (and `M9`)" — i.e. M5 vs ULM is sufficient and M9 vs ULM is parenthetical. The two locked texts disagree. GEO labelling must state both readings rather than pick one silently.

---

## 4. Pre-Registration v2 (Revised): Non-KnockTF GEO Human Bulk RNA-seq TF Perturbation Cohort (R2) & Frozen `M10_Soft_Deconf` (R5)

**Pre-Registration v2 Revision Timestamp**: 2026-11-03 (committed **prior to downloading any GEO expression matrix or computing any log2FC/scores** for the R2 GEO studies).  
**R6 GSM-contrast revision**: 2026-11-03 (commit after `r2_geo_contrasts.csv`; still **prior to any expression-matrix download or scoring**).  
**R6.1 pre-download patches** (2026-11-03; committed **before any expression-matrix download, log2FC, or scoring**): single-replicate-arm retention + sensitivity; GSE290232 TMD14/TMD8 metadata typo; locked expression-source priority (NCBI raw counts → author counts → TPM/FPKM; exclude log2FC/DE-only series).

### 4.1 Search Protocol & Automated Expansion Rule (Rule e)
- **Primary Search Query (`esearch` on `db=gds`)**:
  `"Homo sapiens"[Organism] AND "expression profiling by high throughput sequencing"[DataSet Type] AND ("knockdown"[Title] OR "knockout"[Title] OR "CRISPR"[Title] OR "CRISPRi"[Title] OR "shRNA"[Title] OR "siRNA"[Title] OR "depletion"[Title] OR "silencing"[Title] OR "loss"[Title] OR "deficiency"[Title]) AND "gse"[Entry Type] AND "2018/01/01"[PDAT] : "2026/11/01"[PDAT]` (`retmax = 2500`).
- **Pre-scoring Audit Table (`r2_geo_inclusion_decisions.csv`, Rule f)**:
  Across the 119 non-KnockTF title-matched series returned by the query, each GSE was first audited on series-level metadata (`Title`, `Summary`, `Overall Design`, `Supplementary File types`, and `Taxon`) in `artifacts/src/build_r2_inclusion_decisions.py`. That **title-level first pass** (76 `INCLUDE*` / 43 `EXCLUDE`) is retained as a historical table and is **not** the scoring cohort.
- **GSM-level Contrast Table (`r2_geo_contrasts.csv`, R6; Rule f revised)**:
  Every former `INCLUDE*` series was re-audited from GSM `characteristics_ch1`, `title`, `source_name_ch1`, `treatment_protocol_ch1`, and `library_strategy` (`artifacts/src/build_r6_geo_contrasts.py`), **without downloading any expression matrix**. SuperSeries were dropped in favour of RNA-seq SubSeries (`GSE212686` → `GSE212684`; `GSE231951` → `GSE231949`). The resulting locked table contains **74 valid contrasts** (`49 KD` / `22 KO` / `3 Degron`) across **66 series, 65 studies, and 56 unique candidate TFs**, of which **58/74 (78.4%)** target Lambert 2018 sequence-specific TFs; every GSM ID appears in at most one valid contrast. Eleven former `INCLUDE*` series were excluded at the GSM level (confounded/stimulated-only arms, miRNA-seq SuperSeries, indirect/paracrine designs, unmatched cell types, or reporter knock-in; reasons in `r2_geo_contrasts.csv`). Title-level `EXCLUDE` series remain excluded.
- **Fixed Sample-Size Gate for Query Expansion (Rule e)**:
  Series are processed in chronological order (2020–2026 first; then 2018–2019 if needed). Expansion is gated on the **GSM-level valid contrast count**, not on any method rank or score. The locked table already contains **74 ≥ 40** valid contrasts across **65 ≥ 20** studies, so **no 2015–2017 query expansion is performed**. Only if this count had been `< 40` would the search window have been extended to `2015/01/01 : 2017/12/31` using the identical query and inclusion rules before any method evaluation.

### 4.2 Experimental Design & Biological Inclusion/Exclusion Rules (Rules a, b, c)
1. **Dual Non-Overlap with KnockTF**: Any series whose `GSE` accession or linked `PubMed ID` appears in `knockTF_meta.csv` (907 experiments) or `knocktf401_study_ids.csv` is strictly excluded.
2. **Treatment / Stimulus Arms (Rule a)**:
   - For series involving drug, agonist, cytokine, or environmental treatments (e.g., palbociclib, abemaciclib, erlotinib, enzalutamide, GDC-9545, CD1530/retinoic acid, erastin, LPS, TNF-alpha/IFN-gamma/TGF-beta, arsenic, JQ1, radiation, hypoxia), **only the unstimulated / untreated / vehicle (DMSO) baseline contrast** (`TF perturbation vs matched control` in the unstimulated/vehicle condition) is included (`INCLUDE_BASELINE_ONLY`).
   - Any series lacking an unstimulated/vehicle baseline contrast (where both perturbation and control samples are treated with a drug/stimulus) is **excluded in its entirety** (`EXCLUDE`).
3. **Combinatorial Perturbations, Fusion Oncogenes, Patient/Germline Models & Non-Human Samples (Rule a)**:
   - Double or combinatorial knockouts/knockdowns without a single-TF contrast (e.g., `JUN/ATF3`, `siTFEB/TFE3`, `EP300 and CREBBP`, `PAX2 and PAX8`, `DNMT1/L1TD1`), fusion oncogene knockdowns (`EWSR1::FLI1`), viral co-infections (`EBV`, `HPV16`, `SARS-CoV-2`), and patient-derived or germline syndrome/deficiency models (`GATA2 deficiency iPSC`, `PURA syndrome`, clonal lineage tracing) are **excluded**.
   - Every sample's organism is verified from GSM metadata; any non-human (`Mus musculus`, `Rattus norvegicus`, xenograft mouse host reads) series or sample is **excluded**.
4. **Separate Knockdown (`KD`) vs CRISPR Knockout (`KO`) vs Targeted Protein Degradation (`Degron`) Criteria and Stratification (Rule b, R6)**:
   - Stratum is assigned **only from GSM `characteristics_ch1` and treatment protocol**, not from the series title. Vague title words (`loss`, `deficiency`, `depletion`) are not sufficient; the GSM fields must name the method.
   - Because CRISPR-Cas9 frameshift knockout (`KO`) and targeted protein degradation (`Degron`) frequently act via catalytic loss or protein removal without necessarily depleting steady-state mRNA below `-0.5 log2FC`, applying `target log2FC <= -0.5` to `KO` or `Degron` is inappropriate.
   - **Pre-registered rule**:
     - **`KD` stratum (`siRNA`, `shRNA`, `CRISPRi` / dCas9-KRAB, `ASO`, `knockdown`/`silencing`)**: Requires verified on-target mRNA depletion `target TF log2FC <= -0.5` AND `max |log2FC|_blind >= 0.5`.
     - **`KO` stratum (`CRISPR-Cas9 knockout`, `CRISPR RNP KO`, `genetic KO`)**: Included based on the study authors' documented KO genotype in GEO/publication metadata plus transcriptional responsiveness `max |log2FC|_blind >= 0.5` (no `target TF log2FC <= -0.5` filter required; actual target mRNA `log2FC` is recorded).
     - **`Degron` stratum (`PROTAC`, `dTAG`, `AID` / auxin-inducible degron)**: Same quantitative gates as `KO` (no `target TF log2FC <= -0.5` requirement; `max |log2FC|_blind >= 0.5`). Inclusion additionally requires a GEO- or paper-documented degradation reagent/allele (named PROTAC, dTAG ligand, or auxin) pairing a degraded sample with a matched vehicle/uninduced control. For degron time courses, a single contrast is taken at the **longest profiled timepoint** (maximum intended depletion duration), without inspecting expression.
   - Results are reported on: (i) the **Combined Primary R2 Cohort (`KD + KO + Degron`)**, and (ii) **pre-registered `KD`, `KO`, and `Degron` strata separately** (with `KD` further stratified into strong `<= -1.0` vs moderate `-1.0 < log2FC <= -0.5`).
5. **Pre-Registered Sequence-Specific TF Subset Analysis (Rule c)**:
   - Using the official Lambert et al. 2018 (*Cell*) list of 1,639 human sequence-specific DNA-binding TFs (`artifacts/data/knocktf/lambert2018_human_tfs_1639.txt`), each candidate target is annotated as `is_lambert2018_tf = 1` (sequence-specific TF) or `0` (chromatin modifier / transcriptional cofactor / non-canonical regulator such as `EZH2`, `EP300`, `CREBBP`, `ARID1A`, `KDM5A`, `KDM5C`, `DNMT1`, `BRD4`, `RB1`, `ENO1`, `SMARCA4`, `TAF1`, `EWSR1`).
   - All primary and secondary endpoints are reported both on **all eligible regulators in the 647 candidate set** and on the **pre-registered Lambert 2018 sequence-specific TF subset**.

### 4.3 Quantitative Expression Processing Protocol (Rule d)
1. **Expression Matrix Source Priority** (locked before any download; each contrast records the source actually used in `r2_geo_qc_pass.csv`):
   - **Priority 1 — NCBI GEO-generated raw counts**: `GSE*_raw_counts_GRCh38.p13_NCBI.tsv.gz`. Convert to CPM (`CPM = count / sum(count) * 1e6`) then `log2FC = log2((mean_treated_CPM + 1.0) / (mean_control_CPM + 1.0))`.
   - **Priority 2 — author-provided sample-level gene counts**: series supplementary integer count tables (`.tsv/.csv/.txt/.xlsx` or a series_matrix of counts). Same CPM + log2FC as Priority 1.
   - **Priority 3 — author-provided sample-level TPM or FPKM**: `log2FC = log2((mean_treated + 1.0) / (mean_control + 1.0))`.
   - **Exclude**: if the only available files are precomputed log2FC tables, differential-expression result tables, or other summaries **without a sample-level expression matrix**, the contrast is excluded (`QC_FAIL_NO_SAMPLE_MATRIX`) and does not enter scoring. Sample-level log2-normalised matrices that are not counts/TPM/FPKM are not used.
   - **Download-size gate**: before downloading, estimate the total byte size of expression files that would be fetched. If the estimate exceeds **5 GB**, stop and report without downloading.
   - **Minimum Gene Coverage**: At least `15,000` quantified human gene symbols must be present in the matrix after identifier mapping.
2. **Single-replicate arms (R6.1; locked before download)**:
   - Three retained contrasts have an arm with `n = 1`: `GSE316716_U87MG_FLI1_KD` (`n_treated=1`, `n_control=1`), `GSE316716_U251MG_FLI1_KD` (`n_treated=1`, `n_control=1`), and `GSE179314_HCT116_SP3_KD` (`n_treated=1`, `n_control=2`). They **remain in the primary cohort**; log2FC is computed as usual (the mean of a singleton arm is that sample).
   - **Sensitivity analysis (pre-registered)**: all endpoints are also reported on the subset of QC-pass contrasts with `n_treated >= 2` **and** `n_control >= 2`.
3. **GSE290232 TMD14 / TMD8 metadata typo (R6.1; locked before download)**:
   - Treated GSMs (`GSM8808919`–`GSM8808924`) have `characteristics_ch1` `cell line: TMD14`, while sample titles and control GSMs (`GSM8808913`–`GSM8808918`) are TMD8. This is recorded in the contrast `notes` field and treated as a **GEO metadata typo**, not as a distinct cell line. The contrast is **retained** in the primary cohort.
   - **Sensitivity analysis (pre-registered)**: list this contrast and report all endpoints with it excluded.
4. **Phase split (locked before download)**:
   - **Phase 1 (QC only)**: download → CPM/TPM → log2FC → the QC metrics below. **Do not** call `score_knocktf_experiments`, `m9_order`, `m10_soft_order`, or any ranking function, and **do not** inspect true-TF ranks.
   - QC gates: `KD` requires `target TF log2FC <= -0.5`; every stratum requires `max |log2FC|_blind >= 0.5`; gene symbols after mapping `>= 15,000`; a permitted sample-level matrix source must be recorded. Output: `r2_geo_qc_pass.csv`. If fewer than 40 contrasts pass, stop and report (query expansion still gated on count only).
   - **Phase 2 (one-shot scoring)**: score QC-pass contrasts once with frozen `M1_ULM`, `M2b_VIPER_shadow`, `M5`, `M9`, and `M10`. Outcomes A/B/C from the `m = 5` family; M10 generalisation from `m = 7`. Report KD/KO/Degron strata, Lambert subset, and the R6.1 sensitivity analyses. No parameter re-run after seeing results.
5. **Gene Identifier Mapping & Duplicate Symbol Resolution**:
   - NCBI Entrez Gene IDs (`GeneID` in NCBI GEO raw count tables) and Ensembl Gene IDs (`ENSG*`, stripped of version suffixes) are mapped to official uppercase HGNC symbols using `artifacts/data/gwas/hgnc_protein_coding.txt` (augmented by `Homo_sapiens.GRCh38.p13.annot.tsv.gz` / KnockTF gene symbols for non-coding/TF symbols).
   - If multiple rows map to the same uppercase HGNC gene symbol, their expression values are **averaged prior to computing `log2FC`** (matching `eval_p0_2_external.py`).
6. **Multiple Control / Perturbation Group Pairing**:
   - Within a study/cell line, each distinct perturbed TF is paired with its **experiment-matched negative control** (`shNC`/`shCtrl`/`shScramble` for shRNA; `siNC`/`siCtrl` for siRNA; `sgNT`/`sgCtrl`/`WT`/`Parental` for CRISPR KO/CRISPRi).
   - If multiple negative control groups exist (e.g. `shNC1` and `shNC2`, or `WT` and `sgNT`), all negative control replicates for that cell line/batch are pooled by averaging their normalized expression profiles to form a single reference control profile.
   - If multiple independent hairpins/guides (`sh1`, `sh2` or `sg1`, `sg2`) target the **same TF in the same cell line and condition**, their normalized profiles are averaged into **one contrast per `(study_id, cell_line, target_tf)`** (preventing pseudo-replication within a cell line), whereas distinct cell lines or distinct TFs in the same study are retained as separate contrasts clustered by `study_id`.

### 4.4 Frozen Specification of `M10_Soft_Deconf` (Developed on KnockTF Dev 266 Only; R6 re-freeze)
- **Motivation**: R1c showed that `M9`'s binary tail-demotion rule (`ChIP_Max_Z < 0.5 & DoRothEA_Z < 1.0`) demotes 30.2% of true TFs in KnockTF Unseen and 72.2–78.9% in Replogle Perturb-seq.
- **R5 freeze vacated**: checksum `2d031d746d751ed84a902144bdb1b5fe11274161646fe92ccda945fc9edfd9bc` of `artifacts/src/develop_m10_soft_gate.py` is **no longer the frozen M10 implementation**. Vacated because, after R5 and **before any GEO scoring**, the `lambda_bind` grid on Dev 266 was expanded from `{0.25, 0.5, 0.75, 1.0}` to `{0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0}` (plus a `lambda=100` hard-demotion limit control). Unseen 64 and Replogle were not used to choose the new parameters.
- **Frozen `M10_Soft_Deconf` Formula (`m10_soft_order` in `artifacts/src/develop_m10_soft_gate.py`)**:
  - Uses the **exact same trigger condition** on `T_1 = argmin(ULM)` as `M9` (`tau_ChIP = 0.5, tau_CC = 3.0`, without the dead MHC-II rule).
  - Instead of binary tail-demotion, `M10` computes for every candidate `c in 1..647` a continuous coverage-scaled penalty subtracted from `M5` (`mm[c]`):
    - `def_bind(c) = max(0, 0.5 - ChIP_Max_Z(c)) * max(0, 1.0 - DoRothEA_Z(c))`
    - `def_hub(c) = 1(c in UNIVERSAL_SECONDARY_HUBS) * max(0, 3.0 - max(0, CC(c)))`
    - `S_M10(c) = mm(c) - cov_exp * (lambda_bind * def_bind(c) + lambda_hub * def_hub(c))`
    where `cov_exp in [0, 1]` is the mean fraction of `CollecTRI` targets present in the expression matrix across the 647 candidate TFs.
  - **R6 locked parameters** (170-fold LOSO-CV on Dev 266, expanded grid; `artifacts/src/r6_refreeze_m10_loso.py`):
    - `(lambda_bind, lambda_hub) = (1.0, 0.3)` selected in **168/170** folds; `(1.5, 0.3)` in 1/170; `(1.0, 0.15)` in 1/170.
    - Full-data objective on the same grid is also maximised at `(1.0, 0.3)` (interior, not on the expanded boundary). The `lambda=100` limit control converges toward `M9` (Dev Top-1 31/266, median rank 78.5 vs `M9` 80.5).
    - LOSO-CV operating point on Dev 266: Top-1 28/266 (10.53%), Top-5 50/266 (18.80%), median rank 58.5.
    - Fold-level votes: `artifacts/results/cand6_deconftf/external/r6_m10_dev266_loso_fold_votes.csv`.
    - Full expanded grid with LOSO vote counts: `artifacts/results/cand6_deconftf/external/r6_m10_dev266_grid_with_loso.csv`.
    - The earlier full-data grid `r5_m10_dev266_grid_search.csv` and post-dev descriptive `r5_m10_soft_gate_evaluation.csv` are retained and not overwritten.
  - If the trigger flag is NOT raised, `T_1` is placed at rank 1 followed by all remaining TFs in `S_M10` order; if the flag IS raised, all 647 TFs are ranked by `S_M10` in descending order.
- **`cov_exp` limitation (locked before GEO scoring)**: on Dev 266, `cov_exp ≈ 0.972` (essentially constant across experiments). GEO bulk RNA-seq matrices are expected to be full-transcriptome with similarly high coverage (`≈ 0.97`), so the coverage-scaling term is **untested on both Dev and the R2 GEO cohort**. The only descriptive observation of `cov_exp` away from 1 is on already-inspected Replogle pseudobulk (`cov ≈ 0.33–0.43`) and is not a validation result.

### 4.5 Pre-Registered Primary Endpoint Family, Holm Adjustment & Decision Matrix (Rule d)
- **Statistical Test**: Two-sided **study-clustered sign-flip permutation test** (20,000 permutations, flipping signs jointly for all contrasts sharing a `study_id`, seed = 42), alongside Wilcoxon signed-rank (`wilcox` and `pratt`) and exact two-sided McNemar tests.
- **Primary 5-Test Family for `M9` / `M5` Evaluation (Holm-Bonferroni adjusted over `m = 5` tests)**:
  1. `M9` vs `M5` on `dlog2_rank = log2(r_M5) - log2(r_M9)`
  2. `M9` vs `M1_ULM` on `dlog2_rank = log2(r_ULM) - log2(r_M9)`
  3. `M9` vs `M1_ULM` on Top-5 recovery
  4. `M5` vs `M1_ULM` on `dlog2_rank = log2(r_ULM) - log2(r_M5)`
  5. `M5` vs `M1_ULM` on Top-5 recovery
- **Pre-Registered Co-Primary 2-Test Family for `M10_Soft_Deconf` (Holm-Bonferroni adjusted over `m = 2` tests)**:
  1. `M10` vs `M5` on `dlog2_rank = log2(r_M5) - log2(r_M10)` (supported if study-clustered `p_Holm < 0.05` and mean/median `> 0`)
  2. `M10` vs `M1_ULM` on `dlog2_rank = log2(r_ULM) - log2(r_M10)` (supported if study-clustered `p_Holm < 0.05` and mean/median `> 0`)
- **Holm-family usage (R6; locked before any GEO scoring)**:
  - **Outcomes A / B / C are decided solely by the `m = 5` primary family** (`M9`/`M5` vs each other and vs `ULM`).
  - **Whether `M10` may be claimed to generalise is decided by the combined `m = 7` Holm adjustment** over the 5 primary tests plus the 2 `M10` co-primary tests. The `m = 2` `M10` family is reported but is not by itself sufficient for a generalisation claim.
- **Decision Matrix on the Primary R2 Cohort**:
  - **Outcome A (Selective Gating Supported)**: In the `m = 5` family, `M9 vs M5` has `dlog2_rank > 0` with study-clustered `p_Holm < 0.05`, AND `M9` vs `ULM` is significant on `dlog2_rank` (`p_Holm < 0.05`). A parallel `M10` generalisation claim additionally requires the corresponding `M10 vs M5` and `M10 vs ULM` tests to survive the **`m = 7` combined Holm** adjustment.
  - **Outcome B (Multimodal Fusion Supported, Gating Not Supported)**: In the `m = 5` family, `M5` (and `M9`) significantly outperforms `ULM` on `dlog2_rank` (`p_Holm < 0.05`), but `M9 vs M5` is not significant. `M10 vs M5` may still be described if it survives `m = 7` Holm; otherwise `M10` is not claimed to generalise.
  - **Outcome C (Neither Supported)**: In the `m = 5` family, neither `M5` nor `M9` significantly outperforms `ULM` on `dlog2_rank` (`p_Holm >= 0.05`).

**Conflict with P0-1 §3.3 (pre-registration defect, not a post-hoc redefinition):** §4.5 B is satisfied if **M5** vs ULM is significant, with M9 vs ULM in parentheses. P0-1 §3.3 B requires **both** M9 and M5 vs ULM. GEO results are therefore labelled: *under v2 wording this is B (only M5 vs ULM is significant, and only at m=5); M9 vs ULM is not significant; under the P0-1 original wording this is C*. The conflict is reported as a defect of the two pre-registrations, not resolved by rewriting either table.

### 4.6 Cryptographic SHA-256 Checksums of R1/R3/R5 Scripts, Pre-Scoring R2 Inclusion Table, and R6 Contrast Table

```text
0d216d217d1704b188407fb0a6822ef67995a34d4471c2ab1a237b9833715904  artifacts/src/run_r1_diagnostics.py
bb19fbaccc611d9585f94532ead4ccfcb379a9ad2621cb4a8a3eb9a3155be748  artifacts/src/run_r3_benchmark_audit.py
2559b3f4b994fe4fe78cef5ef28e738e298642ad01b5e402f225d5ae275932c5  artifacts/src/build_r2_inclusion_decisions.py
b7e8202cb6d91b0fd38ada67e2cdbad49a2faa295ba9cfd80f1082a9344d7b8b  artifacts/src/build_r6_geo_contrasts.py
e3b4ecb0643ecd8290f48c3c518c1d1dbfe5e38ee73436f19f6aa822c97d7876  artifacts/src/r2_geo_phase1_qc.py
1e513f5236e56842e5ff96de3263175498ba9c1e6662ab8f228de3a1d7500677  artifacts/src/develop_m10_soft_gate.py
93c6857be8c1b9e2a7c916e0a9893f5a6fe7e9e38e43cc27156afa7b8b45e4b0  artifacts/src/r6_refreeze_m10_loso.py
ddc47f22ec27f0df5613449b9b3d2d5860f50eb8f8f1bf9dcbf8dc5c94e9592b  artifacts/data/knocktf/lambert2018_human_tfs_1639.txt
32ea5628a6a9303cf15790d4e25ed4cd73cd15a8e82657e81c369433ccd26836  artifacts/results/cand6_deconftf/external/r1a_random_baseline_check.csv
279786178acfb89f1bb1ad64e6bb7d9226500fe1c0bb86a13a4774b02c36efe1  artifacts/results/cand6_deconftf/external/r1b_knocktf_restricted8k_summary.csv
bd8759293d80fa9d9ad87e6a016d22aab35f18d22efde3bc470c6fc7ae4e7a67  artifacts/results/cand6_deconftf/external/r1c_gate_mechanism_audit.csv
fa425483c76a1388b3fd52c973e79c780264bb0e6b7535d88397b426d1a73308  artifacts/results/cand6_deconftf/external/r2_geo_inclusion_decisions.csv
73128cca29ba26860cf25a210cc52af0a85ac497915adc68a682b45a16d12f86  artifacts/results/cand6_deconftf/external/r2_geo_contrasts.csv
60e0b8b7f79f329b25a9209828cec1f4863474b6608fd5a79330759164a07257  artifacts/results/cand6_deconftf/external/r2_geo_download_size_estimate.csv
521a51732f7a3b22c26946b628604b1450cce6aa19f9677a6f304e8d8e9aecc7  artifacts/results/cand6_deconftf/external/r2_geo_expression_sources.csv
4602ddd64af6af04f15b3ca7a862dd955f0b126b54bdc3da38e6dd0945589c31  artifacts/results/cand6_deconftf/external/r2_geo_qc_pass.csv
55e27a693fb395aa5b3cb4355d82f835f27f4d6471661df769dd7e00ed5e6d20  artifacts/results/cand6_deconftf/external/r2_geo_lfc_matrix.parquet
a54c28b3d98f4d2e4f0def04f1745a23d694e155fa9fe582faaf339683f35c6e  artifacts/results/cand6_deconftf/external/r5_m10_soft_gate_evaluation.csv
6bf197630a8027f83c77c759410fcf51b1fa4ed0354c7c3054638afcde0c58f7  artifacts/results/cand6_deconftf/external/r6_m10_dev266_grid_with_loso.csv
747e0189e2dcace7bdcef685d851b8fdd05f5f5d4fc7b6a51950874fb7585d9d  artifacts/results/cand6_deconftf/external/r6_m10_dev266_loso_fold_votes.csv
de8a1b8b3a34417064ba8761aa70de2acfde4a8cbb1423f510f819b8b7023f33  artifacts/results/cand6_deconftf/audit/r3_method_performance_by_blinding_and_metric.csv
4e6f693d2060e5e34f50c4484b592527020eb92438462e19d941b356b5c68080  artifacts/results/cand6_deconftf/audit/r3_ranking_inversion_and_significance_shift.csv
9ed48546038d12ad62dc77f1faddb3454f3ba5db6de4426dfcdf8f6fb8ff24e3  artifacts/results/cand6_deconftf/audit/r3_viper_native_vs_shadow_comparison.csv
95c44dd3761a7b2ac7c92dc92fbebca2c31fd4f4f66a89310b6676ef9fc2842a  artifacts/results/cand6_deconftf/audit/r3_decoupler_empty_col_drop_audit.csv
725e62948a036305f91b2e46b1414f50c717ea89bc8e82b7451fbf594814e9b5  artifacts/src/r2_geo_phase2_score.py
4f77c3a612b0e1403153dfd93f9945f6a120de4bcce3b1bbd13e698441f7a018  artifacts/results/cand6_deconftf/external/r2_geo_phase2_ranks.csv
da17797f8f3293fc2f961ce741a3c95f035cb9d5dbff6d0622b6de2d2c308879  artifacts/results/cand6_deconftf/external/r2_geo_phase2_endpoints.csv
fd217dc50d1c30b854ac208761f438bfd25c9e392afc8903c4dd05b6e16fba2c  artifacts/results/cand6_deconftf/external/r2_geo_phase2_holm.csv
5e762dda943b15a741b99dd4d4a8a131b5c542b378f20522e7f85976b56620f9  artifacts/results/cand6_deconftf/external/r2_geo_phase2_decision.txt
```

R5 freeze of `develop_m10_soft_gate.py` (`2d031d746d751ed84a902144bdb1b5fe11274161646fe92ccda945fc9edfd9bc`) is **vacated** (§4.4). The live R6 re-freeze checksum is `1e513f5236e56842e5ff96de3263175498ba9c1e6662ab8f228de3a1d7500677`. `deconftf_tools.py` remains `45939716963851d980d5767d9285832efd5b4b3537ce014b0e44f7affa225490`.

R6 contrast-table freeze `13af05662c4c7aa132214616ef4905039dfa32cbee5d604d8d35bcf0687f03a4` is superseded by R6.1 `73128cca29ba26860cf25a210cc52af0a85ac497915adc68a682b45a16d12f86` (added `notes` for three single-replicate-arm contrasts and the GSE290232 TMD14/TMD8 typo; **no GSM membership or inclusion decision changed**).

Phase 1 QC freeze R6.1 (2026-11-03; 25/74 PASS): `r2_geo_qc_pass.csv` SHA-256 `d8b1e310ce315e77092c75ad26134f80172c63e74c543ce0d7d3258b93c810f8` is **superseded** by R6.2 (parser-only re-QC, thresholds unchanged).

R6.2 QC freeze (2026-11-03; after ID/column/table parser fixes, **before any scoring**): `r2_geo_qc_pass.csv` SHA-256 `4602ddd64af6af04f15b3ca7a862dd955f0b126b54bdc3da38e6dd0945589c31`. **51/74 contrasts PASS** (`≥ 40`), so Phase 2 one-shot scoring proceeds. Query expansion remains gated on pre-QC contrast count only (74 ≥ 40; no 2015–2017 expansion). `deconftf_tools.py` remains `45939716963851d980d5767d9285832efd5b4b3537ce014b0e44f7affa225490`. M10 remains `1e513f5236e56842e5ff96de3263175498ba9c1e6662ab8f228de3a1d7500677`. QC script SHA-256 `e3b4ecb0643ecd8290f48c3c518c1d1dbfe5e38ee73436f19f6aa822c97d7876`.

Phase 2 one-shot scoring freeze (2026-11-03; single run, no re-run): ranks `4f77c3a612b0e1403153dfd93f9945f6a120de4bcce3b1bbd13e698441f7a018`. Label (not a raw "Outcome B"): **under v2 §4.5 wording this is B** (only M5 vs ULM is significant, and only at m=5: `p_Holm=0.0417`; m=7 Holm `0.058`); **M9 vs ULM is not significant** (`p_Holm=0.202`); **under P0-1 §3.3 original wording this is C**. M9 vs M5 is not significant (mean dlog2 −0.156). M10 does **not** generalise under m=7 Holm. Primary rank/QC hashes above are unchanged by the later deviation_sensitivity and forest-plot work.

The Phase 1 script listed above (`e3b4ecb0…`) is the freeze that produced the primary 51/74 QC. A later `--collapse/--tag/--overwrite` guard was added so default re-runs cannot overwrite those files; that live script is not a re-freeze of the primary QC. R7 forest: `papers/deconf-regulon/figures/fig_forest_all_cohorts.{pdf,png}`; per-experiment ranks `audit/r7_forest_all_cohorts_experiment_ranks.csv`; summary `audit/r7_forest_all_cohorts_summary.csv`.

---

## 5. Deviations from pre-registration

All departures from the locked P0-1 / v2 text, collected in one place. None of these rewrite `deconftf_tools.py` (`45939716…`) or the frozen M9 operator.

1. **R3 audit-script bug fix** (`run_r3_benchmark_audit.py`). *What:* the audit script applied `fillna(0)` to `est_d` before row-standardising DoRothEA Z-scores, and additionally masked lowercase `C11orf95`; frozen `deconftf_tools.py` does neither. *Why:* implementation bug, not a methods change. *Impact:* after the fix, R3 matched `knocktf401_ranks_with_study_ids.csv` / `holdout135_harmonised_ranks.csv` at 0/266 and 0/64 differences. Frozen scoring code was not edited.

2. **M10 re-freeze (R6), before any GEO download.** *What:* vacated R5 checksum `2d031d74…`; expanded `lambda_bind` grid on Dev 266 only; 170-fold LOSO locked `(λ_bind, λ_hub) = (1.0, 0.3)`. *Why:* the original R5 grid was too coarse; Unseen 64 and Replogle were not used to pick the new point. *Impact:* live M10 is `develop_m10_soft_gate.py` `1e513f52…`. Descriptive R5 CSVs were retained and not overwritten.

3. **Contrast-table `notes` column (R6.1), before any GEO download.** *What:* added notes for three single-replicate-arm contrasts and the GSE290232 TMD14/TMD8 metadata typo. *Why:* reviewer required the documentation. *Impact:* GSM membership and INCLUDE/EXCLUDE decisions unchanged. Hash `13af0566…` superseded by `73128cca…`.

4. **Parser edits after the first QC run (R6.2).** *What:* after seeing 25/74 PASS (and an uncommitted earlier ~4/74), the Phase 1 parser was changed (HGNC two-pass, official-symbol priority, column mapping, table readers, SUM collapse of duplicate symbols) and QC was re-run. Scoring functions, QC thresholds (`n_genes ≥ 15,000`, KD log2FC ≤ −0.5, blinded |log2FC| ≥ 0.5), and the scoring script were not changed. *Why:* 37 of 49 first-pass failures were engineering (unreadable matrices, column map, missing symbols), not biology. Pass *counts* were seen; method *ranks* were not. *Impact:* 25 → 51 PASS; one PASS→FAIL (GSE231959 MYCN, `SAMPLE_MAP`). Phase 2 ran once after 51 ≥ 40. Frozen primary QC hash `4602ddd6…`; ranks `4f77c3a6…`.

5. **Duplicate-symbol collapse: SUM vs the pre-registered average (R6.2, reviewer-requested).** *What:* §4.3.5 says duplicate uppercase HGNC symbols are **averaged** before log2FC (matching `eval_p0_2_external.py`). R6.2 primary QC used **SUM**, at reviewer request, because collapsing transcript-level IDs to genes should add counts. *Why:* reviewer instruction after the first QC, before scoring. *Impact:* the primary GEO result uses SUM. `deviation_sensitivity` re-collapsed with average: the same 51 contrasts PASS (`n_genes` identical; max |Δ target log2FC| = 0.13); 17/51 M5 ranks changed. m=5: M5 vs ULM +1.10 bits, `p_Holm=0.029` (still the only significant test); M9 vs ULM `p_Holm=0.15` (still ns). Under average, M5 vs ULM also survives m=7 Holm (`p=0.041`); the SUM primary does not (`p_Holm7=0.058`). **Does not change the primary conclusion.** Outputs: `r2_geo_qc_pass_average.csv`, `r2_geo_phase2_ranks_average.csv`, `r2_geo_phase2_holm_average.csv`, `audit/r2_geo_deviation_average_vs_sum.csv`. Primary hashes `4602ddd6…` / `4f77c3a6…` unchanged.

6. **M9 threshold LOSO script lost (R10).** *What:* the original 170-fold LOSO vote table cited as 165/170 for (τ_ChIP, τ_CC) = (0.5, 3.0) — and historically also as 141/142 and 167/170 — cannot be recovered. A documented recompute on frozen `m9_order` (`artifacts/src/r10_m9_threshold_loso.py`) shows full-Dev Top-1 is 30–31/266 across the nine-cell grid, and **every** leave-one-study fold is tied under train Top-1 (mean 5.9 of 9 cells). Main analysis (ties: first in grid order) selects (0.5, 2.0) in 165/170 folds and LOSO Top-1 26/266. Preferring (0.5, 3.0) among ties recovers 27/266 and 166/170 votes for that pair. The M10 `cv_objective` has no ties and yields 28/266. *Why:* Dev cannot uniquely identify the threshold pair; the historical vote counts were an artefact of an undocumented tie-break. *Impact:* (0.5, 3.0) remains the frozen operator used for holdout, Replogle and GEO; it was locked before those cohorts were scored. Manuscript LOSO Top-1 updated from 27/266 to 26/266 (main analysis). Frozen `deconftf_tools.py` (`45939716…`) and all M9 rank CSVs unchanged. **Does not change any external-validation conclusion.**

---

## 6. Errata

The citation lines and SHA-256 rows above are left unchanged so that frozen hashes remain reconstructible.

**Self-expression mask list.** §1 item 2 and the hash table cite `artifacts/results/cand6_deconftf/supplement_masked_tfs_list.csv` as the list of 1,888 regulators in `CollecTRI ∪ DoRothEA ∪ ChEA3_GMT_sources`. That file has 864 rows and is **not** the frozen union (862 of 864 symbols are in the union; `TCP1` and `ZC3H11A` are not). The union reconstruction is `artifacts/results/cand6_deconftf/supplement_masked_tfs_list_v2.csv` (1,888 symbols after dropping self-loops; 1,783 present in the KnockTF matrix; 779 Replogle K562; 754 Replogle RPE1; 1,867 GEO). Scoring always uses the in-code union in `deconftf_tools.py`, not the 864-row CSV.

---

## 7. Agent-evaluation pre-registration (v3.1, 2026-11-04)

This section freezes the external run. It does not replace §§1–6 and does not change `deconftf_tools.py` (`45939716…`) or `develop_m10_soft_gate.py` (`1e513f52…`). v3.1 replaces the v3 wording of this section. The operational rules are in `papers/deconf-regulon/e5_preregistration.md`.

Condition B is V1 only. On the locked v3 Dev 60 it returned parseable IDs for 52/60 calls (4.28923 CNY). Ranks were not computed. The implementation hash is `run_e5_common.py` below. V1 evidence lines contain no differentially expressed gene symbols, so the ID map is the 25 candidate names and the gene list is empty.

Entrants, chosen on Dev only:

| Entrant | What is frozen | SHA-256 |
|---|---|---|
| v3 propose–verify | `artifacts/src/run_a2_pilot_design.py` (`build_prompt`, conditions A and C) plus `merge_verified_into_m10` in `regagent_tools.py`. D8 hash. D7 hash was `f0e661b59fa321b89f438e3f0b8025504fbcf07f3dd9b5541a74502938d01214`; the pre-D7 hash was `6b5fbd84f4647d3ab7b9c0dd88a2e313e956c8d0a1a66b8a664c12e2b83e544c` | `4d18e2d8df48ef3915ca2b95d9b1515cd1b817ab17d133b9c3fd0c59f3179b18` |
| E4 V1 pool rerank | `artifacts/src/run_e4_agent_variants.py` | `384fa84a6828b3fd9b1bfeb867a7f1362ae0848a6100394e0b725c1ff894547d` |
| E4 V2 tool agent | same file, V2 branch | same hash |
| E1 probe P1 | `artifacts/src/run_e1_memory_probe.py` | `8ccb7347208700f0df26223db95de45a77ebdc8082b444553e2784266bfe4d87` |
| tool implementations | `artifacts/src/regagent_tools.py` | `de22c8b0d2a6c5934e3ab103b5b11a6abbe0a9f948d1fc77134b7a5fd446aa5b` |
| E5 prompts, condition B, endpoints | `artifacts/src/run_e5_common.py` | `32529ab3a91a67347bda11e70e5c1982ea1506eb9982071011d36d1a4a42d245` |
| E5 cohort runner | `artifacts/src/run_e5_external.py` | `f366ae83328f94bf9a27af1656c70cb4a382ef10ef253410d723f6eeaf339a93` |
| E5 endpoint script, not yet run | `artifacts/src/run_e5_endpoints.py`. D7 adds the Replogle matrix-presence stratum; previous hash `561d7411839ab19bbe6554f642fe2121f608b270c50b3b10d85dbcd83d316104` | `1b12ab88de18ae1de7e0e9d2d7af89f4fa56f0db9d7e0273326d07b70930fa55` |
| E5 baseline reproduce | `artifacts/src/run_e5_baseline.py` | `85460403121d2b621eabbc187f854e759c39cd9b4775922dedc37d42c06d99fb` |

Model for these entrants is `deepseek-v4.1-flash`, temperature 0, `reasoning_effort=low`, `max_completion_tokens=16384`. `deepseek-v4-pro` is not an entrant.

Dev fact that the external test is allowed to contradict: under condition C, V1 mean Δlog2 versus M10 is +0.004 and V2 is −0.171. The selection rule still carries both, plus v3 propose–verify.

Cohorts, in this order, with no code change between them: GEO 51, Unseen 64 (already used for the old M8 call; say so), Replogle K562 Primary 267. Conditions A and C on all three. Condition B on GEO, V1 only. P1 on GEO and Unseen only. GEO metadata is cell line, no tissue, and the GSE id. `series_title`, `method_detail`, and evidence fields are not used. The stage cap is 175 CNY. The session soft cap stays 450 CNY.

Holm family m = 9, GEO only, two-sided, 20,000 permutations, seed 42: for v3, V1, and V2, condition-C Δlog2 versus M10; the A−C difference of that Δlog2; and the paired 0/1 Top-5 name hit of condition A versus P1. Cohen's κ and the name-overlap rate are descriptive. A public-date split of GEO at 2025-05-31 (29 on or before, 22 after) is descriptive and not in the family. Endpoints are computed only after all three cohorts finish.

**Deviation D7 (2026-11-04), before any E5 endpoint was computed.** Replogle stopped before its first call. `build_prompt` treated a missing expression column as a failed mask. Of the 267 K562 profiles, 169 true TFs are columns and are already 0; 98 are absent, so their expression cannot enter the prompt; none is a nonzero column. D7 changes that one assert to "the column is absent, or it is present and equals 0". A present nonzero column still stops the run. The DE-list check and the true-TF text assert are unchanged. V1 and V2 do not have this assert, so those builders were not edited. Regenerating the 856 GEO and Unseen episodes already sent (408 + 448; V2 compared on the first prompt only) matched the stored prompts with 0 byte differences. Replogle is then run on all 267. A descriptive split of those 267 into 169 present and 98 absent is reported with the endpoints and is not in the Holm family.

**Deviation D8 (2026-11-04), still before any E5 endpoint was computed.** After D7, prompt construction stopped on `K562_normalized_z_CTBP1` because the downregulated gene `CTBP1-DT` contains the true TF as a hyphen-separated token. That gene is the divergent transcript of CTBP1, so the hit is a real label leak and the assert was right to catch it. v3.1 already said a hit is replaced with `[PERT]` and counted; the code aborted instead. D8 replaces such a DE gene name with `[PERT]`, keeps the log2FC, and counts the replacement. The value counts, the exact-TF DE check, the mask assert, and the final text assert are unchanged. After replacement the text assert passes. Two Replogle prompts are affected, condition A and condition C of that one experiment. V2 does not list DE genes, so it is not edited. Regenerating the same 856 first prompts again matched the stored text with 0 byte differences. A later descriptive count, not in the Holm family, will record how many experiments in each cohort have such a neighbor transcript in the 30 down or 30 up genes.
