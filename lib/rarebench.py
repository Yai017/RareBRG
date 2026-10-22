# -*- coding: utf-8 -*-
"""RareBench cases, the phenotype text of a case, and gold-standard codes -> MONDO."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

from lib import config as C


@dataclass(frozen=True)
class Case:
    dataset: str
    index: int                    # line number in the dataset file
    hpo: tuple[str, ...]          # the patient's HPO term ids
    gold_codes: tuple[str, ...]   # gold-standard OMIM / Orphanet identifiers

    @property
    def cid(self) -> str:
        return f"{self.dataset}_{self.index}"


@lru_cache(maxsize=None)
def cases(dataset: str) -> tuple[Case, ...]:
    """All cases of one dataset, in file order."""
    out: list[Case] = []
    with open(C.DATASETS_DIR / f"{dataset}.jsonl", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            rec = json.loads(line)
            out.append(Case(dataset=dataset, index=i,
                            hpo=tuple(rec.get("Phenotype") or []),
                            gold_codes=tuple(rec.get("RareDisease") or [])))
    return tuple(out)


def case(dataset: str, index: int) -> Case:
    return next(c for c in cases(dataset) if c.index == index)


# Six obsolete HPO terms of LIRICAL cases are missing from RareBench's phenotype naming. The
# study's prompts name them by their HPO label without the "obsolete " prefix; any other
# term that RareBench does not name appears under its HPO label as it is.
EXTRA_NAMES = {
    "HP:0001006": "Hypotrichosis",
    "HP:0003116": "Abnormal echocardiogram",
    "HP:0006187": "Fusion of midphalangeal joints",
    "HP:0006957": "Loss of ability to walk",
    "HP:0040083": "Toe walking",
    "HP:0040180": "Hyperkeratosis pilaris",
}


@lru_cache(maxsize=1)
def _hpo_names() -> dict[str, str]:
    """HPO id -> the name used in prompts: RareBench's own naming of the term, else
    EXTRA_NAMES, else the HPO label."""
    names = dict(json.loads(C.RAREBENCH_PHENOTYPE_NAMES.read_text(encoding="utf-8")))
    for source in (EXTRA_NAMES, json.loads(C.HPO_GRAPH.read_text(encoding="utf-8"))["labels"]):
        for hid, name in source.items():
            names.setdefault(hid, name)
    return names


def phenotype_text(c: Case) -> str:
    """The case's HPO terms by name, comma-separated, in the order given."""
    names = _hpo_names()
    return ", ".join(names.get(h, h) for h in c.hpo)


# ---- gold-standard codes -> MONDO ----------------------------------------------------
@dataclass
class MondoXref:
    """OMIM / Orphanet code -> MONDO ids, and MONDO id -> primary name."""
    code2mondo: dict[str, set[str]] = field(default_factory=dict)
    primary_name: dict[str, str] = field(default_factory=dict)

    def gold_to_mondo(self, codes) -> list[str]:
        out: set[str] = set()
        for code in codes or []:
            out |= self.code2mondo.get(code, set())
        return sorted(out)


@lru_cache(maxsize=1)
def mondo_xref() -> MondoXref:
    """MONDO's cross-references to OMIM and Orphanet (Orphanet and OMIM codes are not
    bridged to each other)."""
    concepts = json.loads(C.MONDO_CONCEPTS.read_text(encoding="utf-8"))
    code2mondo: dict[str, set[str]] = defaultdict(set)
    primary: dict[str, str] = {}
    for c in concepts:
        mid = c["mondo_id"]
        primary[mid] = c.get("primary_name", "")
        for src, vals in (c.get("xrefs") or {}).items():
            if src.upper() not in ("OMIM", "ORPHANET", "ORPHA"):
                continue
            for v in (vals if isinstance(vals, list) else [vals]):
                code = str(v).replace("Orphanet:", "ORPHA:").replace("ORPHANET:", "ORPHA:")
                if code.startswith(("OMIM:", "ORPHA:")):
                    code2mondo[code].add(mid)
    return MondoXref(code2mondo=dict(code2mondo), primary_name=primary)
