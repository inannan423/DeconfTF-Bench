#!/usr/bin/env python3
"""Describe why condition A beat condition C on the finished v3 pilot.

Uses the saved v3 table and Dev 266 labels only. Does not call a model and
does not read Unseen, Replogle, or GEO.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

ROOT = Path("/Users/bytedance/codes/new-bishe")
OUT = ROOT / "artifacts" / "results" / "cand6_deconftf" / "agent"


def main() -> None:
    df = pd.read_csv(OUT / "a2v3_pilot_per_experiment.csv")
    labels = pd.read_csv(
        ROOT / "artifacts" / "results" / "cand6_deconftf" / "knocktf401_ranks_with_study_ids.csv",
        usecols=["exp_id", "study_id", "split_new"],
    )
    bundle = pickle.loads(
        (ROOT / "artifacts" / "results" / "cand6_deconftf" / "knocktf_precomputed_bundle.pkl").read_bytes()
    )
    obs = bundle["obs_sub"]
    study = labels.set_index("exp_id")["study_id"].astype(str)
    dev_ids = set(labels.loc[labels["split_new"] == "Dev", "exp_id"].astype(str))
    meta = pd.DataFrame(
        {
            "true_tf": obs["TF"].astype(str).str.upper(),
            "biosample": obs["Biosample.Name"].astype(str),
            "tissue": obs["Tissue.Type"].astype(str),
        },
        index=obs.index.astype(str),
    )
    meta["study_id"] = study.reindex(meta.index).astype(str)
    meta = meta.loc[meta.index.isin(dev_ids)]

    def biosample_modes(study_id: str, bio: str):
        other = meta[(meta["study_id"] != str(study_id)) & (meta["biosample"] == bio)]
        if other.empty:
            return set(), 0
        counts = other["true_tf"].value_counts()
        top = int(counts.max())
        return set(counts[counts == top].index), int(len(other))

    rows = []
    for rec in df.itertuples(index=False):
        m = meta.loc[rec.exp_id]
        siblings = meta[(meta["study_id"] == str(rec.study_id)) & (meta.index != rec.exp_id)]
        sib_tfs = set(siblings["true_tf"])
        modes, n_bio = biosample_modes(rec.study_id, m["biosample"])
        hyps = []
        if isinstance(rec.A_hypotheses, str) and rec.A_hypotheses not in ("", "nan"):
            hyps = rec.A_hypotheses.split("|")
        first = hyps[0] if hyps else ""
        hit = int(rec.A_first_hit) == 1
        rows.append({
            "exp_id": rec.exp_id,
            "study_id": rec.study_id,
            "biosample": m["biosample"],
            "true_tf": str(rec.true_tf),
            "in_recalled": int(rec.in_recalled),
            "r_m10": int(rec.r_m10),
            "A_first": first,
            "A_hit": int(hit),
            "C_hit": int(rec.C_first_hit),
            "A_k10": int(rec.A_k10_rank),
            "C_k10": int(rec.C_k10_rank),
            "A_proposed_k10": int(rec.A_k10_proposed),
            "A_better": int(rec.A_k10_rank < rec.C_k10_rank),
            "n_siblings": int(len(siblings)),
            "sibling_tfs": "|".join(sorted(sib_tfs)),
            "hit_in_same_study": int(hit and str(rec.true_tf) in sib_tfs),
            "n_biosample_other_studies": n_bio,
            "biosample_modes": "|".join(sorted(modes)),
            "hit_is_biosample_mode": int(hit and str(rec.true_tf) in modes),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "a2v3_ac_decomposition.csv", index=False)
    better = out[out["A_better"] == 1]
    hits = out[out["A_hit"] == 1]
    explained = (hits["hit_in_same_study"] == 1) | (hits["hit_is_biosample_mode"] == 1)
    lines = [
        "v3 条件 A 相对条件 C 的描述性分解。不调用模型，不改判定。",
        "A 更好：k=10 合并名次严格小于 C。同研究：Dev 266 里同一 study_id 的其他实验的真实 TF。",
        "细胞系常用 TF：离开本实验所属研究后，同一 Biosample.Name 在 Dev 里出现次数最多的真实 TF；并列都算。没有其他研究的同名生物样本时，这一项为空。",
        f"60 个里 A 更好 {len(better)}，持平 {int((out['A_k10'] == out['C_k10']).sum())}，C 更好 {int((out['A_k10'] > out['C_k10']).sum())}。",
        f"A 更好的 {len(better)} 个里，A 的 k=10 提出了真实 TF 并且名次优于 M10 的有 {int(((better['A_proposed_k10'] == 1) & (better['A_k10'] < better['r_m10'])).sum())} 个；A 的名次仍等于 M10、只是 C 更差的有 {int((better['A_k10'] == better['r_m10']).sum())} 个。",
        f"A 更好且第 1 个假设命中的有 {int(better['A_hit'].sum())} 个，其中 C 没有命中的有 {int(((better['A_hit'] == 1) & (better['C_hit'] == 0)).sum())} 个。",
        f"条件 A 的第 1 个假设一共命中 {len(hits)}/60。其中可以用同研究其他实验解释的 {int(hits['hit_in_same_study'].sum())} 个，可以用细胞系众数解释的 {int(hits['hit_is_biosample_mode'].sum())} 个，两者至少占一条的 {int(explained.sum())} 个。",
        "同研究能解释的两个是 DataSet_01_198（SCC-1，TP63；该研究在 Dev 里的其他实验也是 TP63）和 DataSet_01_015（STA-ET-1，FLI1；其他实验也是 FLI1）。后者是 28 个召回失败里条件 A 唯一的第 1 假设命中，条件 C 没有命中。",
        "细胞系众数只解释 DataSet_01_150（MCF7，TFAP2C）。离开该研究后，Dev 里还有 16 个 MCF7 实验，众数是 HIF1A、PBX1、SPDEF、TFAP2C 四个并列，不是单一的常用 TF。条件 C 的第 1 个假设也命中了这一个。",
        f"其余 {len(hits) - int(explained.sum())} 个第 1 假设命中对不上这两条。其中 {int((hits['n_siblings'] == 0).sum())} 个在 Dev 里没有同研究的其他实验，{int((hits['n_biosample_other_studies'] == 0).sum())} 个没有其他研究的同名生物样本。",
    ]
    text = "\n".join(lines) + "\n"
    (OUT / "a2v3_ac_decomposition.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
