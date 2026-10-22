# -*- coding: utf-8 -*-
"""HPO, HPOA and MONDO, and the ICTO (information content term overlap) score.

Disease corpus
    The MONDO concepts with at least one HPOA annotation of aspect P (phenotypic
    abnormality), reached through MONDO's OMIM and Orphanet cross-references; negated
    annotations are dropped. The corpus is both what retrieval ranks and the reference
    for information content.

Score
    IC(h) = -ln(|corpus diseases annotated with h or a descendant of h| / |corpus|).
    ICTO of a candidate = sum of IC(q) * w(q) over the patient terms q that the candidate
    explains, i.e. q is in its profile: its annotations with their HPO ancestors. The
    weight w is 1 for an annotated disease; a grouping concept without annotations of
    its own takes the profile of its annotated subtypes, w(q) = k/n when k of its n
    subtypes explain q.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from functools import lru_cache

from lib import config as C


class Ontology:
    def __init__(self):
        graph = json.loads(C.HPO_GRAPH.read_text(encoding="utf-8"))
        self.hpo_parents: dict[str, list[str]] = graph["parents"]
        self.hpo_children: dict[str, list[str]] = graph["children"]
        self.hpo_labels: dict[str, str] = graph["labels"]

        mondo = json.loads(C.MONDO_GRAPH.read_text(encoding="utf-8"))
        self.xrefs: dict[str, list[str]] = mondo["xrefs"]
        self.disease_children: dict[str, list[str]] = mondo["children"]

        # HPOA annotations, keyed by the OMIM / Orphanet ids MONDO cross-references
        annot: dict[str, set[str]] = defaultdict(set)
        with open(C.HPOA, encoding="utf-8") as fh:
            for line in fh:
                if not line or line[0] == "#" or line.startswith("database_id"):
                    continue
                row = line.rstrip("\n").split("\t")
                if len(row) < 11 or row[2] == "NOT" or row[10] != C.HPOA_ASPECT:
                    continue
                annot[row[0]].add(row[3])

        self.disease_hpo: dict[str, set[str]] = {}
        for mid in self.xrefs:
            hpos: set[str] = set()
            for key in self.hpoa_ids(mid):
                hpos |= annot.get(key, set())
            if hpos:
                self.disease_hpo[mid] = hpos
        self.corpus: list[str] = sorted(self.disease_hpo)
        self.n_diseases = len(self.disease_hpo)

        # information content over the corpus
        directly: dict[str, set[str]] = defaultdict(set)
        for mid, hpos in self.disease_hpo.items():
            for h in hpos:
                directly[h].add(mid)
        self.ic: dict[str, float] = {}
        for h in set(directly) | set(self.hpo_parents) | set(self.hpo_children):
            stack, seen, covered = [h], set(), set()
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x)
                covered |= directly.get(x, set())
                for child in self.hpo_children.get(x, []):
                    if child not in seen:
                        stack.append(child)
            if covered:
                self.ic[h] = -math.log(len(covered) / self.n_diseases)

        self._closure: dict[str, set[str]] = {}

    # ---- ontology helpers ------------------------------------------------------------
    def hpoa_ids(self, mid: str) -> set[str]:
        """The HPOA disease ids (OMIM:..., ORPHA:...) of a MONDO concept."""
        keys = set()
        for x in self.xrefs.get(mid, ()):
            if x.startswith("OMIM:"):
                keys.add(x)
            elif x.lower().startswith("orphanet:"):
                keys.add("ORPHA:" + x.split(":", 1)[1])
        return keys

    @lru_cache(maxsize=None)
    def ancestors(self, h: str) -> frozenset[str]:
        """The HPO ancestors of h, without h."""
        seen, frontier = set(), {h}
        while frontier:
            nxt = set()
            for node in frontier:
                for parent in self.hpo_parents.get(node, []):
                    if parent not in seen and parent != h:
                        seen.add(parent)
                        nxt.add(parent)
            frontier = nxt
        return frozenset(seen)

    def closure(self, mid: str) -> set[str]:
        """An annotated disease's annotations with all their HPO ancestors."""
        if mid not in self._closure:
            own = self.disease_hpo[mid]
            clo = set(own)
            for h in own:
                clo |= self.ancestors(h)
            self._closure[mid] = clo
        return self._closure[mid]

    @lru_cache(maxsize=None)
    def annotated_descendants(self, mid: str) -> tuple[str, ...]:
        """The corpus diseases below `mid` in the MONDO hierarchy."""
        seen, stack, out = {mid}, [mid], set()
        while stack:
            node = stack.pop()
            for child in self.disease_children.get(node, []):
                if child in seen:
                    continue
                seen.add(child)
                stack.append(child)
                if child in self.disease_hpo:
                    out.add(child)
        return tuple(sorted(out))

    # ---- scores ----------------------------------------------------------------------
    @lru_cache(maxsize=None)
    def profile(self, mid) -> tuple[tuple[str, float], ...]:
        """A candidate's profile as (HPO term, weight) pairs; empty when it cannot be scored.

        `mid` is a MONDO id or a frozenset of ids; a set takes each term at its highest
        weight among its members."""
        if not isinstance(mid, str):
            ids = frozenset(mid)
            if not ids:
                return ()
            if len(ids) == 1:
                return self.profile(next(iter(ids)))
            w: dict[str, float] = {}
            for m in ids:
                for h, wt in self.profile(m):
                    if wt > w.get(h, 0.0):
                        w[h] = wt
            return tuple(sorted(w.items()))
        if mid in self.disease_hpo:
            return tuple((h, 1.0) for h in sorted(self.closure(mid)))
        subtypes = self.annotated_descendants(mid)
        if not subtypes:
            return ()
        counts: dict[str, int] = defaultdict(int)
        for sub in subtypes:
            for h in self.closure(sub):
                counts[h] += 1
        n = len(subtypes)
        return tuple((h, k / n) for h, k in sorted(counts.items()))

    def icto(self, patient_hpo, mid) -> float:
        """ICTO of a candidate (a MONDO id or a set of ids) for the patient's terms."""
        prof = dict(self.profile(mid))
        total = 0.0
        for q in patient_hpo:
            w = prof.get(q, 0.0)
            if w > 0:
                total += self.ic.get(q, 0.0) * w
        return total

    def ic_mass(self, patient_hpo) -> float:
        """The total IC of the patient's terms, the most any candidate can explain."""
        return sum(self.ic.get(q, 0.0) for q in patient_hpo)

    # ---- retrieval -------------------------------------------------------------------
    def _icto_annotated(self, patient_hpo: tuple, mid: str) -> float:
        """ICTO of a corpus disease (weight 1). `patient_hpo` is a sorted tuple, so that the
        floating-point sum, and with it which scores tie, does not depend on set order."""
        clo = self.closure(mid)
        return sum(self.ic.get(q, 0.0) for q in patient_hpo if q in clo)

    def rank(self, patient_hpo, topk: int, tie_seed: int) -> list[tuple[str, float]]:
        """The `topk` corpus diseases with the highest ICTO, ties broken at random under
        `tie_seed`."""
        terms = tuple(sorted(set(patient_hpo)))
        scored = [(mid, s) for mid in self.corpus if (s := self._icto_annotated(terms, mid)) > 0]
        rng = random.Random(tie_seed)
        jitter = {mid: rng.random() for mid, _ in scored}
        scored.sort(key=lambda t: (-t[1], -jitter[t[0]]))
        return scored[:topk]


@lru_cache(maxsize=1)
def get_ontology() -> Ontology:
    return Ontology()
