# -*- coding: utf-8 -*-
"""The phenotypic evidence of a pool and the two gates that weigh it against the order in
which its candidates enter.

The normalized score of a candidate is its ICTO divided by the total IC of the case's
terms, which puts it in [0, 1] for every case. A candidate without a profile has no score
(None) and keeps its position under both gates.

  margin gate (retrieval-based reranking)  from the incoming order, a candidate moves ahead
      of the one in front of it only when its score exceeds that candidate's by more than
      tau; tau = 0 gives the score order.
  routing gate (LLM reranking)  the score order replaces the reranker's order when the two
      highest scores of the pool differ by at least theta.
"""
from __future__ import annotations

import json

from lib import config as C
from lib.ontology import get_ontology


def scores(hpo, pool):
    """The normalized ICTO of each candidate, in pool order (None when it cannot be scored)."""
    onto = get_ontology()
    mass = onto.ic_mass(hpo)
    return [(onto.icto(hpo, m) / mass if (mass > 0 and onto.profile(m)) else None) for m in pool]


def margin_gate(order, sc, tau):
    """Adjacent passes over the scored candidates, repeated until nothing moves: a candidate
    passes the one ahead of it when its score leads by more than tau."""
    tau = max(0.0, tau)
    slots = [i for i, s in enumerate(sc) if s is not None]
    seq = [order[i] for i in slots]
    val = [sc[i] for i in slots]
    moved = True
    while moved:
        moved = False
        for j in range(len(seq) - 1):
            if val[j + 1] - val[j] > tau:
                seq[j], seq[j + 1] = seq[j + 1], seq[j]
                val[j], val[j + 1] = val[j + 1], val[j]
                moved = True
    out = list(order)
    for slot, m in zip(slots, seq):
        out[slot] = m
    return out


def score_order(order, sc):
    """The scored candidates sorted by score (ties keep their order)."""
    slots = [i for i, s in enumerate(sc) if s is not None]
    ranked = sorted(slots, key=lambda i: -sc[i])
    out = list(order)
    for slot, src in zip(slots, ranked):
        out[slot] = order[src]
    return out


def router(order, sc, theta):
    """The score order when the two highest scores differ by at least theta, else `order`."""
    vals = sorted((s for s in sc if s is not None), reverse=True)
    gap = (vals[0] - vals[1]) if len(vals) >= 2 else float("inf")
    return score_order(order, sc) if gap >= theta else list(order)


def reranked_order(pool, evidence, model, ds, cid):
    """An LLM reranker's order for one case, or None when there is no usable answer (none
    recorded, or one that failed the closed-set check)."""
    p = C.rerank_dir(pool, evidence, model) / ds / f"{cid}.json"
    if not p.exists():
        return None
    rec = json.loads(p.read_text(encoding="utf-8"))
    if rec["closed_set"] != "ok":
        return None
    return [frozenset(s) for s in rec["reranked"]]
