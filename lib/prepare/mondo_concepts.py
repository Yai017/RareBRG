# -*- coding: utf-8 -*-
"""MONDO concepts (not obsolete) with their primary name, exact and related synonyms, and
cross-references, from mondo.json -> config.MONDO_CONCEPTS:
[{"mondo_id", "primary_name", "synonyms": [...], "xrefs": {DB: [ids]}}]."""
from __future__ import annotations

import json

from lib import config as C

PREFIX = "http://purl.obolibrary.org/obo/MONDO_"


def build() -> None:
    nodes = json.loads(C.MONDO_JSON.read_text(encoding="utf-8"))["graphs"][0].get("nodes", [])
    concepts = []
    for n in nodes:
        nid = n.get("id", "")
        primary = n.get("lbl")
        meta = n.get("meta") or {}
        if not nid.startswith(PREFIX) or not primary or meta.get("deprecated"):
            continue
        synonyms = []
        for s in meta.get("synonyms") or []:
            val = (s.get("val") or "").strip()
            if val and s.get("pred", "") in ("hasExactSynonym", "hasRelatedSynonym") \
                    and val not in synonyms and val != primary:
                synonyms.append(val)
        xrefs: dict[str, list[str]] = {}
        for x in meta.get("xrefs") or []:
            v = (x.get("val") or "").strip()
            if ":" in v:
                xrefs.setdefault(v.partition(":")[0].strip().upper(), []).append(v)
        concepts.append({"mondo_id": "MONDO:" + nid.rsplit("_", 1)[-1], "primary_name": primary.strip(),
                         "synonyms": synonyms, "xrefs": xrefs})
    C.MONDO_CONCEPTS.parent.mkdir(parents=True, exist_ok=True)
    C.MONDO_CONCEPTS.write_text(json.dumps(concepts, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[prepare] MONDO: {len(concepts)} concepts -> {C.MONDO_CONCEPTS.relative_to(C.ROOT)}")
