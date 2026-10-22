# -*- coding: utf-8 -*-
"""Retrieval-based diagnosis: each case's ICTO ranking of the disease corpus.

The top `topk` diseases of every case go to results/retrieval/<dataset>.json; the first 5
form pool R, which also enters RUG, RUT and the RAG prompt. Recall@1/3/5 are recorded as a
check.
"""
from __future__ import annotations

import json
import time

from lib import config as C
from lib.evaluation import gold_of, hit
from lib.ontology import get_ontology
from lib.rarebench import cases, mondo_xref

KS = (1, 3, 5)


def table_path(ds: str):
    return C.RES_RETRIEVAL / f"{ds}.json"


def retrieve(datasets=None, topk: int = 10, seed: int = C.TIE_SEED) -> None:
    onto = get_ontology()
    names = mondo_xref().primary_name
    C.RES_RETRIEVAL.mkdir(parents=True, exist_ok=True)
    for ds in (datasets or C.DATASETS):
        table, hits, t0 = {}, {k: 0 for k in KS}, time.time()
        todo = cases(ds)
        for case in todo:
            ranked = onto.rank(case.hpo, topk=topk, tie_seed=seed)
            table[case.cid] = [{"rank": r + 1, "mondo_id": mid, "name": names.get(mid, ""),
                                "score": round(score, 6)} for r, (mid, score) in enumerate(ranked)]
            ids = [mid for mid, _ in ranked]
            for k in KS:
                hits[k] += hit(ids[:k], gold_of(case))
        recall = {f"R@{k}": round(100 * hits[k] / len(todo), 2) for k in KS}
        table_path(ds).write_text(json.dumps({
            "dataset": ds, "n_cases": len(todo), "topk": topk, "tie_seed": seed,
            "corpus_size": onto.n_diseases, "recall": recall, "candidates": table,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[retrieve] {ds}: {len(todo)} cases | " + "  ".join(f"{k}={v:.1f}" for k, v in recall.items())
              + f" ({time.time() - t0:.0f}s) -> {table_path(ds).relative_to(C.RESULTS)}")
