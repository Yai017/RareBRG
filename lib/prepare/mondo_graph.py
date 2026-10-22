# -*- coding: utf-8 -*-
"""MONDO is_a graph and OMIM / Orphanet cross-references from mondo.json -> config.MONDO_GRAPH:
{"parents": {concept: [parents]}, "children": {concept: [children]}, "xrefs": {concept: [ids]}}."""
from __future__ import annotations

import json
from collections import defaultdict

from lib import config as C
from lib.prepare.hpo_graph import short_id

XREF_PREFIXES = ("OMIM:", "Orphanet:", "ORPHA:")


def build() -> None:
    graphs = json.loads(C.MONDO_JSON.read_text(encoding="utf-8")).get("graphs", [])
    parents: dict[str, list[str]] = defaultdict(list)
    children: dict[str, list[str]] = defaultdict(list)
    xrefs: dict[str, list[str]] = defaultdict(list)
    for g in graphs:
        for e in g.get("edges", []):
            sub, obj = short_id(e.get("sub", "")), short_id(e.get("obj", ""))
            if not sub or not obj or not sub.startswith("MONDO:") or not obj.startswith("MONDO:"):
                continue
            if e.get("pred", "") in ("is_a", "subClassOf", "rdfs:subClassOf"):
                if obj not in parents[sub]:
                    parents[sub].append(obj)
                if sub not in children[obj]:
                    children[obj].append(sub)
        for node in g.get("nodes", []):
            nid = short_id(node.get("id", ""))
            if not nid or not nid.startswith("MONDO:"):
                continue
            for x in (node.get("meta") or {}).get("xrefs") or []:
                val = x.get("val") or ""
                if val.startswith(XREF_PREFIXES) and val not in xrefs[nid]:
                    xrefs[nid].append(val)
    C.MONDO_GRAPH.parent.mkdir(parents=True, exist_ok=True)
    C.MONDO_GRAPH.write_text(json.dumps({"parents": dict(parents), "children": dict(children),
                                         "xrefs": dict(xrefs)}, ensure_ascii=False), encoding="utf-8")
    print(f"[prepare] MONDO: {sum(map(len, parents.values()))} is_a edges, {len(xrefs)} concepts with "
          f"OMIM/Orphanet xrefs -> {C.MONDO_GRAPH.relative_to(C.ROOT)}")
