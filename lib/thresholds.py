# -*- coding: utf-8 -*-
"""Gate thresholds read from capability relations, leave-one-case-out and leave-one-LLM-out.

A gate cell is one gate on one ranking: the margin gate on a pool's incoming order
('A|<pool>/<llm>', and 'A|T/<tool>', 'A|RUT/<tool>' for the tools) or the routing gate on
an LLM reranker's order ('B|<pool>/<evidence>/<llm>'). For every cell, `matrix` records
whether the gold-standard diagnosis ranks first at each threshold of a grid (0 to 1 in
steps of 0.005), plus a last row with the incoming order.

The threshold that scores case i is estimated without case i (`loo`):
  capability    the Recall@1 of the cell's incoming ranking on the other cases
  optimum       the grid threshold with the highest Recall@1 on the other cases
  boundary      a cell is a boundary cell when Recall@1 at threshold 0 is within 0.5
                points of its optimum; when every cell of an LLM in a family is a
                boundary cell, those cells take threshold 0
  relation      a line fitted to (capability, optimum) over the cells of the other LLMs
                of the same family (boundary cells left out, except those of the most
                capable boundary LLM, which anchor the fit); a cell's threshold is the
                line read at its own capability, clipped to [0, 1]
Families: the margin gate on G; on the pools RAG and RUG; the routing gate on G; on RAG
and RUG. The tool cells read the pooled RAG/RUG margin relation of all LLMs, and the
reranking of R reads the pooled RAG/RUG routing relation of all LLMs.
"""
from __future__ import annotations

import numpy as np

from lib import config as C
from lib import gates as GT
from lib.evaluation import gold_of, hit
from lib.pools import build_pool
from lib.rarebench import cases

GRID = [round(i * 0.005, 3) for i in range(201)]
TAUS = GRID + [10.0]                       # last row: the incoming order (no gating)
NT = len(GRID)
EVS = C.EVIDENCES
FAMILIES = {
    "tau-G": lambda m: [f"A|G/{m}"],
    "tau-pool": lambda m: [f"A|RAG/{m}", f"A|RUG/{m}"],
    "theta-G": lambda m: [f"B|G/{e}/{m}" for e in EVS],
    "theta-pool": lambda m: [f"B|{p}/{e}/{m}" for p in ("RAG", "RUG") for e in EVS],
}


def _rows(cell, datasets):
    """(incoming order, scores, gold) of every case, in dataset and case order; (None,
    None, gold) for a case whose pool is empty."""
    kind, rest = cell.split("|")
    parts = rest.split("/")
    pool = parts[0]
    out = []
    for ds in datasets:
        for c in cases(ds):
            gold = gold_of(c)
            if kind == "A":
                order = build_pool(parts[1], ds, c.cid, pool)
            else:
                ev, llm = parts[1], parts[2]
                base = build_pool("-" if pool == "R" else llm, ds, c.cid, pool)
                order = (GT.reranked_order(pool, ev, llm, ds, c.cid) or base) if base else None
            if not order:
                out.append((None, None, gold))
                continue
            hpo = tuple(sorted(set(c.hpo)))
            out.append((order, GT.scores(hpo, order), gold))
    return out


def matrix(cell, datasets=C.DATASETS) -> np.ndarray:
    """H[t, case] = 1 when the gated order ranks the gold-standard diagnosis first."""
    fn = GT.margin_gate if cell.startswith("A|") else GT.router
    rows = _rows(cell, datasets)
    H = np.zeros((len(TAUS), len(rows)), dtype=np.int8)
    for ci, (order, sc, gold) in enumerate(rows):
        if order is None:
            continue
        for ti, t in enumerate(TAUS):
            H[ti, ci] = hit(fn(order, sc, float(t))[:1], gold)
    return H


def loo(HM: dict, llms, tools=()) -> tuple[dict, dict]:
    """Per-case thresholds and gated top-1 hits of every cell in HM (see the module text)."""
    grid = np.array(GRID)
    cells = list(HM)
    N = next(iter(HM.values())).shape[1]
    SUM = {c: HM[c].sum(1) for c in cells}
    ths = {c: np.zeros(N) for c in cells}
    hits = {c: np.zeros(N, np.int8) for c in cells}
    ti = lambda t: int(np.argmin(np.abs(grid - t)))
    fams = {f: of for f, of in FAMILIES.items() if all(c in HM for m in llms for c in of(m))}
    for i in range(N):
        cur = {c: 100.0 * (SUM[c] - HM[c][:, i]) / (N - 1) for c in cells}
        inc = {c: cur[c][-1] for c in cells}
        opt = {c: float(grid[int(cur[c][:NT].argmax())]) for c in cells}
        flat = {c: cur[c][:NT].max() - cur[c][0] <= 0.5 for c in cells}

        def fit(cs):
            if len(cs) < 2:                             # nothing to relate: no gating
                return lambda x: 0.0
            b, a = np.polyfit([inc[c] for c in cs], [opt[c] for c in cs], 1)
            return lambda x: float(np.clip(a + b * x, 0, 1))

        th = {}
        for f, of in fams.items():
            plateau = {m for m in llms if all(flat[c] for c in of(m))}
            for held in llms:
                if held in plateau:
                    for c in of(held):
                        th[c] = 0.0
                    continue
                cs = [c for m in llms if m != held and m not in plateau for c in of(m) if not flat[c]]
                rest = [m for m in llms if m != held and m in plateau]
                if rest:                                # the most capable boundary LLM anchors the fit
                    anchor = max(rest, key=lambda m: np.mean([inc[c] for c in of(m)]))
                    cs += of(anchor)
                g = fit(cs)
                for c in of(held):
                    th[c] = g(inc[c])
        pooled_a = [f"A|{p}/{m}" for m in llms for p in ("RAG", "RUG") if f"A|{p}/{m}" in HM]
        g_tools = fit([c for c in pooled_a if not flat[c]])
        for t in tools:
            cs = [c for c in (f"A|T/{t}", f"A|RUT/{t}") if c in HM]
            z = all(flat[c] for c in cs)
            for c in cs:
                th[c] = 0.0 if z else g_tools(inc[c])
        pooled_b = [f"B|{p}/{e}/{m}" for m in llms for p in ("RAG", "RUG") for e in EVS
                    if f"B|{p}/{e}/{m}" in HM]
        g_r = fit([c for c in pooled_b if not flat[c]])
        for e in EVS:
            c = f"B|R/{e}/{C.R_RERANKER}"
            if c in HM:
                th[c] = g_r(inc[c])
        for c in cells:
            t = th.get(c, 0.0)
            ths[c][i] = t
            hits[c][i] = HM[c][ti(t), i]
    return ths, hits
