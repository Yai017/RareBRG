# -*- coding: utf-8 -*-
"""Candidate pools, built the same way for every LLM and every tool:

    R    retrieval: the top 5 of the ICTO ranking
    G    the LLM's 5 diagnoses
    RUG  G followed by the candidates of R not already in it
    RAG  the LLM's 5 diagnoses from retrieval-augmented generation
    T    a diagnosis tool's top 5 (PhenoBrain or PubCaseFinder)
    RUT  T followed by the candidates of R not already in it

A candidate is a set of MONDO ids. A name that resolves to no concept is dropped, and
candidates that share an id are merged into the slot of the first. An LLM sees each
candidate under the MONDO primary name of one of its ids.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache

from lib import config as C
from lib.normalize import is_decided, name_to_mondo
from lib.rarebench import cases, mondo_xref
from lib.retrieve import table_path
from lib.tools import name_key, names_without_xref

ALL_POOLS = C.LLM_POOLS + C.TOOL_POOLS


def merge_pool(candidates) -> list[frozenset]:
    """A pool of MONDO id-sets from candidates given as iterables of ids, in order.
    Candidates whose id-sets intersect are merged into the slot of the first; a candidate
    without ids is dropped."""
    merged: list[set] = []
    for cand in candidates:
        ids = {c for c in cand if c}
        if not ids:
            continue
        target = next((s for s in merged if s & ids), None)
        if target is None:
            merged.append(set(ids))
        else:
            target |= ids
    return [frozenset(s) for s in merged]


def pick_id(idset: frozenset) -> str:
    """The id a candidate is shown under: its only id, or a fixed-seed choice."""
    ids = sorted(idset)
    if len(ids) == 1:
        return ids[0]
    h = int(hashlib.md5(f"{C.DISPLAY_SEED}|{'|'.join(ids)}".encode()).hexdigest(), 16)
    return ids[h % len(ids)]


def display(idset: frozenset) -> str:
    rep = pick_id(idset)
    return mondo_xref().primary_name.get(rep) or rep


def displayed(pool):
    """[(id-set, shown name)] of a pool, as an LLM reranker sees it."""
    return [(s, display(s)) for s in pool]


# ---- sources ---------------------------------------------------------------------------
def generated_names(model: str, pool: str, ds: str, cid: str):
    """The diagnosis names an LLM generated for a case (pool G or RAG), or None when the
    case has no record."""
    p = C.generation_dir(pool, model) / ds / f"{cid}.json"
    if not p.exists():
        return None
    rec = json.loads(p.read_text(encoding="utf-8"))
    return [d["name"] for d in rec["diagnoses"] if d.get("name", "").strip()]


@lru_cache(maxsize=None)
def _retrieved(ds: str) -> dict:
    return json.loads(table_path(ds).read_text(encoding="utf-8"))["candidates"]


def retrieved_ids(ds: str, cid: str) -> list[str]:
    return [c["mondo_id"] for c in _retrieved(ds).get(cid, [])[:C.POOL_SIZE]]


def tool_candidates(tool: str, ds: str, cid: str):
    """A tool's candidates for a case as lists of MONDO ids, or None when the case has no
    tool record (see lib.tools). A candidate that MONDO's cross-references do not cover is
    resolved by name."""
    p = C.RES_TOOLS / tool / ds / f"{cid}.json"
    if not p.exists():
        return None
    out = []
    for c in json.loads(p.read_text(encoding="utf-8"))["candidates"][:C.POOL_SIZE]:
        ids = c["mondo_ids"] or [m for m in [name_to_mondo(name_key(c["name"]))] if m]
        if ids:
            out.append(ids)
    return out


def _resolved(names) -> list[frozenset]:
    return merge_pool([[m] for m in (name_to_mondo(n) for n in names) if m])


# ---- pools -----------------------------------------------------------------------------
def build_pool(source: str, ds: str, cid: str, pool: str):
    """The pool of one case as a list of MONDO id-sets, or None when its source has no
    record. `source` is the LLM for G, RUG and RAG, the tool for T and RUT, and unused
    for R."""
    assert pool in ALL_POOLS, pool
    if pool == "RAG":
        names = generated_names(source, "RAG", ds, cid)
        return None if names is None else _resolved(names)
    ret_ids = retrieved_ids(ds, cid)
    if pool == "R":
        return merge_pool([[i] for i in ret_ids])
    if pool in C.TOOL_POOLS:
        ids = tool_candidates(source, ds, cid)
        t_pool = merge_pool(ids) if ids else None
        if pool == "T":
            return t_pool or None
        r_pool = merge_pool([[i] for i in ret_ids])
        return merge_pool(([list(s) for s in t_pool] if t_pool else []) + [list(s) for s in r_pool]) or None
    names = generated_names(source, "G", ds, cid)
    if names is None:
        return None
    g_pool = _resolved(names)
    if pool == "G":
        return g_pool
    return merge_pool([list(s) for s in g_pool] + [[i] for i in ret_ids])          # RUG


# ---- names to normalize ----------------------------------------------------------------
def generated_name_list(model: str, pools=C.GENERATION_POOLS, datasets=None) -> list[str]:
    """Every name an LLM generated in the given pools."""
    out = set()
    for pool in pools:
        for ds in (datasets or C.DATASETS):
            for c in cases(ds):
                out.update(generated_names(model, pool, ds, c.cid) or [])
    return sorted(out)


def guard_normalized(model: str | None, pools=C.GENERATION_POOLS, datasets=None) -> None:
    """Stop unless every name of an LLM's generations (or, with model None, every tool
    candidate name left to resolution by name) is decided in the resolution map: an
    undecided name would silently leave its pool."""
    names = generated_name_list(model, pools, datasets) if model else names_without_xref(datasets=datasets)
    missing = [n for n in names if not is_decided(n)]
    if missing:
        what, flag = (model, f"-m {model}") if model else ("tool candidates", "--tools")
        raise SystemExit(f"[normalize] {what}: {len(missing)} name(s) not yet normalized, e.g. {missing[:5]}.\n"
                         f"            Run `python run.py normalize {flag}` (see README).")
