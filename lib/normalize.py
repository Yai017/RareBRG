# -*- coding: utf-8 -*-
"""Concept resolution: every diagnosis name is normalized to a MONDO concept or marked invalid.

  1  resolution map   a name already decided is taken from config.RESOLUTION_MAP
  2  literal matching a new name that equals a MONDO primary name or synonym, as written or
                      without a trailing parenthetical, is resolved and added to the map
  3  agent matching   the names left over get their SapBERT top-5 MONDO candidates and are
                      written in batches for the diagnosis-normalize skill
                      (skills/diagnosis-normalize), in which a coding agent chooses among them
  4  cross-check      each agent decision is checked against MONDO before it is added to the map

A map entry is keyed by the cleaned name: mondo_id (or null), primary_name, source
(literal_match, agent_match or invalid), note, and for agent decisions id_from and the SapBERT
candidates the agent was shown. Encoded identifiers of the diagnosis tools are resolved through
MONDO's cross-references in lib.tools; their names enter here only when those miss.
"""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache

from lib import config as C

RESOLVED = ("literal_match", "agent_match")
DECIDED = RESOLVED + ("invalid",)


def clean_disease_name(raw: str) -> str:
    """The map key of a name: NFKC-normalized, surrounding quotes and spaces removed, inner
    whitespace collapsed."""
    text = str(raw or "").strip()
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'^["\'‘’“”\s]+', "", text)
    text = re.sub(r'["\'‘’“”\s]+$', "", text)
    return re.sub(r"\s+", " ", text).strip()


# ---- 1  resolution map ---------------------------------------------------------------------
@lru_cache(maxsize=1)
def resolution_map() -> dict:
    return json.loads(C.RESOLUTION_MAP.read_text(encoding="utf-8"))


def _save_map(name_map: dict) -> None:
    C.RESOLUTION_MAP.write_text(json.dumps(name_map, ensure_ascii=False, indent=2), encoding="utf-8")


def name_to_mondo(name: str) -> str | None:
    """The MONDO id a name resolves to, or None (invalid, or not decided yet)."""
    entry = resolution_map().get(clean_disease_name(name))
    if entry and entry.get("mondo_id") and entry.get("source") in RESOLVED:
        return entry["mondo_id"]
    return None


def is_decided(name: str) -> bool:
    """Whether the map holds a decision for the name (a concept, or invalid)."""
    entry = resolution_map().get(clean_disease_name(name))
    return bool(entry) and entry.get("source") in DECIDED


# ---- 2  literal matching -------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s or "")).lower().strip())


_TRAILING_PAREN = re.compile(r"\s*\([^()]*\)\s*$")


def strip_trailing_parens(clean: str) -> str:
    """'Behcet disease (BD)' -> 'Behcet disease'; '' when there is no trailing parenthetical."""
    base = _TRAILING_PAREN.sub("", str(clean or "")).strip()
    return base if base and base != clean else ""


@lru_cache(maxsize=1)
def _mondo_names() -> tuple[dict, dict]:
    """(normalized MONDO name -> MONDO id, MONDO id -> primary name). Primary names take
    precedence over synonyms; a synonym of several concepts goes to the one with the shortest
    primary name, the most general of them."""
    concepts = json.loads(C.MONDO_CONCEPTS.read_text(encoding="utf-8"))
    primary = {c["mondo_id"]: c["primary_name"] for c in concepts}
    str2mondo: dict[str, str] = {}
    for c in concepts:
        k = _norm(c["primary_name"])
        if k and k not in str2mondo:
            str2mondo[k] = c["mondo_id"]
    owners: dict[str, list[str]] = {}
    for c in concepts:
        for nm in c.get("synonyms") or []:
            k = _norm(nm)
            if k and k not in str2mondo:
                owners.setdefault(k, []).append(c["mondo_id"])
    for k, ids in owners.items():
        str2mondo[k] = min(ids, key=lambda m: (len(primary.get(m, "")), m))
    return str2mondo, primary


def literal_match(clean: str):
    """(MONDO id, note) of a name that equals a MONDO primary name or synonym, as written or
    without a trailing parenthetical; None otherwise."""
    str2mondo, _ = _mondo_names()
    hit = str2mondo.get(_norm(clean))
    if hit:
        return hit, "exact MONDO name or synonym"
    base = strip_trailing_parens(clean)
    if base and (hit := str2mondo.get(_norm(base))):
        return hit, f"exact MONDO name or synonym after removing the parenthetical: '{base}'"
    return None


# ---- 3  agent matching: batches for the skill ---------------------------------------------
BATCH_SIZE = 50


def batch_files(prefix: str) -> list:
    return sorted((C.NORMALIZE_WORKDIR / "batches").glob(f"{prefix}_batch*.json"),
                  key=lambda p: int(p.stem.rsplit("batch", 1)[1]))


def verdict_files(prefix: str) -> list:
    return sorted(C.NORMALIZE_WORKDIR.glob(f"{prefix}_verdict*.json"))


def batch_prefixes() -> list[str]:
    """The prefixes (model names, or `tools`) that have batches waiting for agent matching."""
    return sorted({p.stem.rsplit("_batch", 1)[0] for p in (C.NORMALIZE_WORKDIR / "batches").glob("*_batch*.json")})


def _write_batches(prefix: str, items: list[dict]) -> int:
    """Replace the batches of `prefix` with `items` ({clean, candidates_top5}), BATCH_SIZE per file."""
    for f in batch_files(prefix):
        f.unlink()
    folder = C.NORMALIZE_WORKDIR / "batches"
    if items:
        folder.mkdir(parents=True, exist_ok=True)
    else:                                                 # round finished: drop empty scratch folders
        for d in (folder, C.NORMALIZE_WORKDIR):
            if d.exists() and not any(d.iterdir()):
                d.rmdir()
    n = (len(items) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(n):
        (folder / f"{prefix}_batch{i}.json").write_text(
            json.dumps(items[i * BATCH_SIZE:(i + 1) * BATCH_SIZE], ensure_ascii=False, indent=2), encoding="utf-8")
    return n


def resolve(names, prefix: str) -> dict:
    """Steps 1 and 2 for a list of names, and the input of step 3: literal matches are added
    to the map, and every name still undecided gets its SapBERT top-5 candidates in the batches
    of `prefix` (workdir/batches/<prefix>_batch{i}.json) for agent matching."""
    if verdict_files(prefix):
        raise SystemExit(f"agent verdicts of {prefix} are waiting: run `python run.py normalize --apply` first")
    name_map = resolution_map()
    _, primary = _mondo_names()
    cleaned = sorted({c for c in map(clean_disease_name, names) if c})
    in_map = sum(1 for c in cleaned if name_map.get(c, {}).get("source") in DECIDED)
    literal, pending = 0, []
    for clean in cleaned:
        if name_map.get(clean, {}).get("source") in DECIDED:
            continue
        hit = literal_match(clean)
        if hit:
            mid, note = hit
            name_map[clean] = {"mondo_id": mid, "primary_name": primary[mid], "source": "literal_match",
                               "note": note}
            literal += 1
        else:
            pending.append(clean)
    if literal:
        _save_map(name_map)
    items = []
    if pending:
        from lib.sapbert import Matcher
        items = [{"clean": c, "candidates_top5": cands} for c, cands in zip(pending, Matcher().top_k(pending, 5))]
    batches = _write_batches(prefix, items)
    return {"names": len(cleaned), "in_map": in_map, "literal": literal, "pending": len(pending),
            "batches": batches}


# ---- 4  cross-check ------------------------------------------------------------------------
def apply_verdicts(prefix: str) -> dict:
    """Add the agent verdicts of `prefix` (workdir/<prefix>_verdict{i}.json) to the map after the
    cross-check: a MONDO id is accepted only when it exists and its primary name matches the one
    the agent reported; a null verdict marks the name invalid. The names of the batches still
    undecided afterwards (cross-check failed, or no verdict) are written as new batches for
    another round, and the applied verdict files are removed."""
    name_map = resolution_map()
    _, primary = _mondo_names()
    shown = {}
    for bp in batch_files(prefix):
        for item in json.loads(bp.read_text(encoding="utf-8")):
            shown[item["clean"]] = item.get("candidates_top5", [])
    added, rejected = {"agent_match": 0, "invalid": 0}, []
    for vp in verdict_files(prefix):
        for v in json.loads(vp.read_text(encoding="utf-8")):
            clean, mid = v["clean"], v.get("mondo_id")
            if clean not in shown or name_map.get(clean, {}).get("source") in DECIDED:
                continue
            if mid:
                real, reported = primary.get(mid), v.get("primary_name", "")
                if real is None or (reported and real.lower() != reported.lower()):
                    rejected.append((clean, mid, reported, real))
                    continue
                name_map[clean] = {"mondo_id": mid, "primary_name": real, "source": "agent_match",
                                   "note": v.get("rationale", ""), "id_from": v.get("source_of_id", ""),
                                   "candidates_top5": shown[clean]}
            else:
                name_map[clean] = {"mondo_id": None, "primary_name": "", "source": "invalid",
                                   "note": v.get("rationale", ""), "candidates_top5": shown[clean]}
            added[name_map[clean]["source"]] += 1
    _save_map(name_map)
    for vp in verdict_files(prefix):
        vp.unlink()
    left = [{"clean": c, "candidates_top5": cands} for c, cands in shown.items()
            if name_map.get(c, {}).get("source") not in DECIDED]
    batches = _write_batches(prefix, left)
    return {"added": added, "rejected": rejected, "pending": len(left), "batches": batches}
