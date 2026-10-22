# -*- coding: utf-8 -*-
"""HPO is_a graph and term labels from hp.json -> config.HPO_GRAPH:
{"parents": {term: [parents]}, "children": {term: [children]}, "labels": {term: label}}."""
from __future__ import annotations

import json
from collections import defaultdict

from lib import config as C


def short_id(uri: str) -> str | None:
    """'http://purl.obolibrary.org/obo/HP_0000118' -> 'HP:0000118'."""
    last = (uri or "").rsplit("/", 1)[-1]
    if "_" not in last:
        return None
    prefix, num = last.split("_", 1)
    return f"{prefix}:{num}"


def build() -> None:
    graphs = json.loads(C.HP_JSON.read_text(encoding="utf-8")).get("graphs", [])
    parents: dict[str, list[str]] = defaultdict(list)
    children: dict[str, list[str]] = defaultdict(list)
    labels: dict[str, str] = {}
    for g in graphs:
        for e in g.get("edges", []):
            sub, obj = short_id(e.get("sub", "")), short_id(e.get("obj", ""))
            if not sub or not obj or not sub.startswith("HP:") or not obj.startswith("HP:"):
                continue
            if e.get("pred", "") in ("is_a", "subClassOf", "rdfs:subClassOf"):
                if obj not in parents[sub]:
                    parents[sub].append(obj)
                if sub not in children[obj]:
                    children[obj].append(sub)
        for n in g.get("nodes", []):
            nid = short_id(n.get("id", ""))
            if nid and nid.startswith("HP:") and n.get("lbl"):
                labels[nid] = n["lbl"]
    C.HPO_GRAPH.parent.mkdir(parents=True, exist_ok=True)
    C.HPO_GRAPH.write_text(json.dumps({"parents": dict(parents), "children": dict(children),
                                       "labels": labels}, ensure_ascii=False), encoding="utf-8")
    print(f"[prepare] HPO: {len(labels)} terms, {sum(map(len, parents.values()))} is_a edges "
          f"-> {C.HPO_GRAPH.relative_to(C.ROOT)}")
