# -*- coding: utf-8 -*-
"""Recall@1 of every setting of Table 1, with the gate thresholds of lib.thresholds.

Rows (columns: the two diagnosis tools, retrieval, and each generator LLM):
  (1) Baseline (R / T / G)                       (2) Baseline (RAG)
  (3) Retrieval-based reranked (T / G)           (4) Retrieval-based reranked (RUT / RUG)
  (5) Retrieval-based reranked (RAG)
  (6) LLM reranked (R / G)                       (7) LLM reranked (R / G, phenotype-grounded)
  (8) LLM reranked (RUG)                         (9) LLM reranked (RUG, phenotype-grounded)
  (10) LLM reranked (RAG)                        (11) LLM reranked (RAG, phenotype-grounded)
Pool R is reranked by config.R_RERANKER. Results go to results/scores/.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from lib import config as C
from lib import thresholds as TH
from lib.evaluation import gold_of, hit
from lib.pools import build_pool
from lib.rarebench import cases

ROWS = [
    ("(1) Baseline (R / T / G)", "base"),
    ("(2) Baseline (RAG)", "base-RAG"),
    ("(3) Retrieval-based reranked (T / G)", "A|G"),
    ("(4) Retrieval-based reranked (RUT / RUG)", "A|RUG"),
    ("(5) Retrieval-based reranked (RAG)", "A|RAG"),
    ("(6) LLM reranked (R / G)", "B|G/pure"),
    ("(7) LLM reranked (R / G, phenotype-grounded)", "B|G/pheno"),
    ("(8) LLM reranked (RUG)", "B|RUG/pure"),
    ("(9) LLM reranked (RUG, phenotype-grounded)", "B|RUG/pheno"),
    ("(10) LLM reranked (RAG)", "B|RAG/pure"),
    ("(11) LLM reranked (RAG, phenotype-grounded)", "B|RAG/pheno"),
]


def _reranked(pool, evidence, model):
    d = C.rerank_dir(pool, evidence, model)
    return d.exists() and any(d.rglob("*.json"))


def cells_for(llms, tools):
    cells = []
    for m in llms:
        if C.generation_dir("G", m).exists():
            cells += [f"A|G/{m}", f"A|RUG/{m}"]
        if C.generation_dir("RAG", m).exists():
            cells.append(f"A|RAG/{m}")
        for p in ("G", "RUG", "RAG"):
            for e in C.EVIDENCES:
                if _reranked(p, e, m):
                    cells.append(f"B|{p}/{e}/{m}")
    for e in C.EVIDENCES:
        if _reranked("R", e, C.R_RERANKER):
            cells.append(f"B|R/{e}/{C.R_RERANKER}")
    for t in tools:
        if (C.RES_TOOLS / t).exists():
            cells += [f"A|T/{t}", f"A|RUT/{t}"]
    return cells


def baseline_hits(pool, source, datasets):
    """Top-1 hits of a pool's own order, per case."""
    out = []
    for ds in datasets:
        for c in cases(ds):
            p = build_pool(source, ds, c.cid, pool)
            out.append(hit(p[:1], gold_of(c)) if p else 0)
    return np.array(out, dtype=np.int8)


def table1(llms=C.STUDY_LLMS, tools=C.TOOLS, datasets=C.DATASETS, out_dir: Path | None = None):
    llms = [m for m in llms if C.generation_dir("G", m).exists()]
    cells = cells_for(llms, tools)
    print(f"[score] {len(cells)} gate cells over {sum(len(cases(d)) for d in datasets)} cases")
    HM = {c: TH.matrix(c, datasets).astype(np.int32) for c in cells}
    ths, hits = TH.loo(HM, llms, [t for t in tools if f"A|T/{t}" in HM])
    n = next(iter(HM.values())).shape[1]
    r1 = lambda h: round(100.0 * float(np.sum(h)) / n, 1)

    cols = [*(("tool", t) for t in tools), ("retrieval", "ICTO"), *(("llm", m) for m in llms)]
    table = {}
    for label, key in ROWS:
        row = {}
        for kind, name in cols:
            v = None
            if kind == "tool":
                if key == "base" and f"A|T/{name}" in HM:
                    v = r1(HM[f"A|T/{name}"][-1])
                elif key == "A|G" and f"A|T/{name}" in hits:
                    v = r1(hits[f"A|T/{name}"])
                elif key == "A|RUG" and f"A|RUT/{name}" in hits:
                    v = r1(hits[f"A|RUT/{name}"])
            elif kind == "retrieval":
                if key == "base":
                    v = r1(baseline_hits("R", "-", datasets))
                elif key in ("B|G/pure", "B|G/pheno"):
                    c = f"B|R/{key.split('/')[1]}/{C.R_RERANKER}"
                    v = r1(hits[c]) if c in hits else None
            else:
                if key == "base" and f"A|G/{name}" in HM:
                    v = r1(HM[f"A|G/{name}"][-1])
                elif key == "base-RAG" and f"A|RAG/{name}" in HM:
                    v = r1(HM[f"A|RAG/{name}"][-1])
                elif key.startswith(("A|", "B|")) and f"{key}/{name}" in hits:
                    v = r1(hits[f"{key}/{name}"])
            row[name] = v
        table[label] = row

    names = [n for _, n in cols]
    print(f"\nRecall@1 (%) by setting (pool R reranked by {C.R_RERANKER}; '-' = not run or not applicable)")
    print(f"{'':46}" + "".join(f"{x:>14}" for x in names))
    for label, row in table.items():
        print(f"{label:46}" + "".join(f"{('-' if row[x] is None else row[x]):>14}" for x in names))

    out_dir = out_dir or C.RES_SCORES
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "table1.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["setting"] + names)
        for label, row in table.items():
            w.writerow([label] + ["" if row[x] is None else row[x] for x in names])
    (out_dir / "table1.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    np.savez_compressed(out_dir / "gate_thresholds.npz",
                        **{f"th::{c}": ths[c] for c in cells}, **{f"hit::{c}": hits[c] for c in cells})
    print(f"\n[score] written to {out_dir}: table1.csv, table1.json, gate_thresholds.npz "
          f"(per-case thresholds and gated top-1 hits)")
    return table, ths, hits
