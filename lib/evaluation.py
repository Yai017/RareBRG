# -*- coding: utf-8 -*-
"""Ontology-grounded matching: a candidate matches the gold-standard diagnosis when one of
its MONDO ids is a gold-standard concept or lies within `config.MATCH_HOPS` is_a hops of
one, in either direction. A candidate is a set of MONDO ids; any member can match."""
from __future__ import annotations

import json
from functools import lru_cache

from lib import config as C
from lib.rarebench import Case, mondo_xref


@lru_cache(maxsize=1)
def _isa() -> tuple[dict, dict]:
    graph = json.loads(C.MONDO_GRAPH.read_text(encoding="utf-8"))
    return graph["parents"], graph["children"]


@lru_cache(maxsize=None)
def within_hops(mid: str, hops: int) -> frozenset[str]:
    """The concepts at most `hops` is_a steps from `mid`, up or down."""
    parents, children = _isa()
    seen, frontier = {mid}, {mid}
    for _ in range(hops):
        nxt = set()
        for node in frontier:
            nxt.update(parents.get(node, []))
            nxt.update(children.get(node, []))
        nxt -= seen
        if not nxt:
            break
        seen |= nxt
        frontier = nxt
    return frozenset(seen)


@lru_cache(maxsize=None)
def _accepted(gold: frozenset[str], hops: int) -> frozenset[str]:
    out: set[str] = set()
    for g in gold:
        out |= within_hops(g, hops)
    return frozenset(out)


def hit(predictions, gold, hops: int = C.MATCH_HOPS) -> int:
    """1 if any prediction matches the gold standard. `predictions` holds MONDO ids or
    sets of ids."""
    accepted = _accepted(gold if isinstance(gold, frozenset) else frozenset(gold), hops)
    for pred in predictions:
        ids = (pred,) if isinstance(pred, str) else tuple(pred)
        if any(p in accepted for p in ids):
            return 1
    return 0


def gold_of(case: Case) -> frozenset[str]:
    """The MONDO concepts of a case's gold-standard diagnosis."""
    return frozenset(mondo_xref().gold_to_mondo(case.gold_codes))
