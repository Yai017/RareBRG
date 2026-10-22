# -*- coding: utf-8 -*-
"""LLM reranking of a pool.

The reranker is given the case's phenotypes and the candidates of the pool and asked to
return all of them, and only them, from most to least likely. It sees either the candidate
names alone ('pure', the candidate-only setting) or each candidate with up to 20 of its
most characteristic phenotypes ('pheno', the phenotype-grounded setting): its HPOA
annotations ranked by within-disease frequency x IC.

An answer is a list of names. It passes the closed-set check when it names exactly the
candidates shown; its names are then mapped back to the candidates. An answer that fails
the check is kept as a failure and the case is scored at the pool's incoming order. One
answer is drawn per case. Prompts go to results/prompts/rerank/ and answers to
results/rerank/<pool>/<evidence>/<model>/<dataset>/<case_id>.json.
"""
from __future__ import annotations

import collections
import json
import time
from functools import lru_cache

from lib import config as C
from lib import generate
from lib import llm
from lib import pools as P
from lib.normalize import clean_disease_name
from lib.ontology import get_ontology
from lib.rarebench import cases

# ---- prompt ------------------------------------------------------------------------------
PHENO_TOPK = 20
RERANK_HEADER = ("The following candidate rare diagnoses have already been proposed for this "
                 "patient (in no particular order):")
RERANK_INSTRUCTION = ("Re-rank ALL and ONLY these candidates for THIS patient. Do not add, drop, "
                      "or rename any candidate — output the same diagnoses, reordered.")
PHENO_HEADER = ("The following candidate rare diagnoses have already been proposed for this patient "
                "(in no particular order).\n"
                f"For each candidate below, up to {PHENO_TOPK} of the phenotypes most characteristic "
                "of that disease are listed (fewer where fewer are on record), to inform your "
                "clinical judgement of how well it fits this patient:")

# HPO frequency terms -> within-disease frequency
_FREQ_TERMS = {"HP:0040280": 1.0, "HP:0040281": 0.9, "HP:0040282": 0.55,
               "HP:0040283": 0.17, "HP:0040284": 0.025, "HP:0040285": 0.0}


def _frequency(raw: str) -> float:
    """An HPOA frequency (HPO term, 'n/m' or 'x%') as a number; 0.5 when not given."""
    s = (raw or "").strip()
    if not s:
        return 0.5
    if s in _FREQ_TERMS:
        return _FREQ_TERMS[s]
    if s.endswith("%"):
        try:
            return max(0.0, min(1.0, float(s[:-1]) / 100))
        except ValueError:
            return 0.5
    if "/" in s:
        try:
            n, m = s.split("/")
            return float(n) / float(m) if float(m) else 0.5
        except (ValueError, ZeroDivisionError):
            return 0.5
    return 0.5


@lru_cache(maxsize=1)
def _hpoa_frequencies() -> dict:
    """HPOA disease id -> {HPO term: within-disease frequency}, aspect P, negated rows dropped."""
    out: dict = {}
    with open(C.HPOA, encoding="utf-8") as fh:
        for line in fh:
            if not line or line[0] == "#" or line.startswith("database_id"):
                continue
            r = line.rstrip("\n").split("\t")
            if len(r) < 11 or r[2] == "NOT" or r[10] != C.HPOA_ASPECT:
                continue
            v = _frequency(r[7] if len(r) > 7 else "")
            d = out.setdefault(r[0], {})
            if v > d.get(r[3], 0.0):
                d[r[3]] = v
    return out


def _characteristic(onto, idset) -> dict:
    """{HPO term: frequency} of a candidate; a concept without annotations of its own takes
    those of its annotated subtypes."""
    freq = _hpoa_frequencies()
    out: dict = {}
    for mid in idset:
        keys = onto.hpoa_ids(mid)
        if not any(k in freq for k in keys):
            keys = {k for sub in onto.annotated_descendants(mid) for k in onto.hpoa_ids(sub)}
        for k in keys:
            for h, v in freq.get(k, {}).items():
                if v > out.get(h, 0.0):
                    out[h] = v
    return out


def _pheno_lines(cands) -> str:
    onto = get_ontology()
    lab = onto.hpo_labels
    out = []
    for rank, (idset, shown) in enumerate(cands, 1):
        freqs = _characteristic(onto, idset)
        if not freqs:
            out.append(f"{rank}. {shown}: no phenotype annotations are currently on record "
                       f"for this disease")
            continue
        top = sorted(freqs, key=lambda q: (-(freqs[q] * onto.ic.get(q, 0.0)), lab.get(q, q)))[:PHENO_TOPK]
        out.append(f"{rank}. {shown}: {', '.join(lab.get(q, q) for q in top)}")
    return "\n".join(out)


def _answer_format(n: int) -> str:
    slots = "\n".join('    {"disease_name": "...", "rationale": "..."}' + ("," if i < n - 1 else "")
                      for i in range(n))
    return (f"Enumerate all {n} candidates listed above.\nEach diagnosis should be a rare disease.\n\n"
            "For each diagnosis, provide the disease name and a brief rationale.\n"
            "Order the diagnoses from most likely (first) to least likely (last).\n\n"
            "Output a JSON object with this exact structure:\n{\n  \"ranked_diagnoses\": [\n"
            + slots + "\n  ]\n}")


def _render(ds: str, idx: int, cands, evidence: str) -> str:
    head = generate.prompt("G", ds, idx).split("\n", 1)[0]       # "Patient's phenotype: ..."
    if evidence == "pheno":
        block = f"{PHENO_HEADER}\n{_pheno_lines(cands)}"
    else:
        block = RERANK_HEADER + "\n" + "\n".join(f"{i}. {shown}" for i, (_, shown) in enumerate(cands, 1))
    return f"{head}\n\n{block}\n\n{RERANK_INSTRUCTION}\n\n{_answer_format(len(cands))}"


def prompt(pool: str, evidence: str, generator: str, ds: str, idx: int, cands) -> str:
    """The rerank prompt of a case; `cands` = [(id-set, shown name)]. Rendered once and
    shared by every reranker of the pool."""
    f = C.rerank_prompt_path(pool, evidence, generator, ds, f"{ds}_{idx}")
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(_render(ds, idx, cands, evidence), encoding="utf-8")
    return f.read_text(encoding="utf-8")


# ---- answer --------------------------------------------------------------------------------
def answer_names(obj) -> list[str]:
    """The disease names of a rerank answer (`ranked_diagnoses`, or the first list of
    entries under another key), in order."""
    obj = obj or {}
    items = obj.get("ranked_diagnoses")
    if not items:
        for v in obj.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                items = v
                break
    out = []
    for it in items or []:
        nm = it.get("disease_name") if isinstance(it, dict) else (it if isinstance(it, str) else "")
        if nm and nm.strip():
            out.append(nm.strip())
    return out


def _full_key(name: str) -> str:
    return clean_disease_name(name).casefold().strip()


def _compact_key(name: str) -> str:
    """The name without punctuation or spacing, so that a candidate written back with its
    separators moved ('spondylodysplastic type 2' for 'spondylodysplastic type, 2') still
    matches; every letter and digit is kept."""
    return "".join(ch for ch in _full_key(name) if ch.isalnum())


def _base_key(name: str) -> str:
    k = clean_disease_name(name).casefold()
    return k.split(" (", 1)[0].strip() or k


def closed_set(names, cands) -> str | None:
    """None when the answer names exactly the candidates shown (compared on
    `_compact_key`), otherwise the reason it does not."""
    want = sorted(_compact_key(shown) for _, shown in cands)
    got = sorted(_compact_key(n) for n in names)
    if got == want:
        return None
    if len(got) != len(want):
        return f"returned {len(got)} names for a pool of {len(want)}"
    return "same count but different names"


def align(names, cands):
    """The candidates in the answer's order. Names are matched to candidates on the whole
    name, then without separators, then without a trailing parenthetical; each candidate
    is used once, and candidates the answer does not name follow in pool order."""
    taken = [False] * len(cands)
    slot = [None] * len(names)

    def claim(order, key):
        for oi in order:
            k = key(names[oi])
            for ci, (_, shown) in enumerate(cands):
                if not taken[ci] and key(shown) == k:
                    slot[oi] = ci
                    taken[ci] = True
                    break

    order = range(len(names))
    claim(order, _full_key)
    claim([oi for oi in order if slot[oi] is None], _compact_key)
    claim([oi for oi in order if slot[oi] is None], _base_key)
    out = [cands[slot[oi]] for oi in order if slot[oi] is not None]
    out.extend(c for ci, c in enumerate(cands) if not taken[ci])
    return out


# ---- run -----------------------------------------------------------------------------------
def run(model: str, pool: str, evidence: str, datasets=None, limit=None) -> None:
    """Rerank pool `pool` of every case with `model`: its own pool for G, RUG and RAG, pool R
    otherwise. Cases already answered are skipped, so an interrupted run resumes where it
    stopped."""
    generator = "-" if pool == "R" else model
    outroot = C.rerank_dir(pool, evidence, model)
    ran = done = failed = 0
    reasons = collections.Counter()
    t0 = time.time()
    for ds in (datasets or C.DATASETS):
        todo = cases(ds)[:limit] if limit else cases(ds)
        for c in todo:
            out = outroot / ds / f"{c.cid}.json"
            if out.exists():
                done += 1
                continue
            pool_ids = P.build_pool(generator, ds, c.cid, pool)
            if not pool_ids:
                continue
            cands = P.displayed(pool_ids)
            user = prompt(pool, evidence, generator, ds, c.index, cands)
            try:
                raw, obj = llm.chat(model, user, system=C.RERANK_SYSTEM_PROMPT)
            except Exception as exc:                  # network or parse failure: retried on resume
                failed += 1
                print(f"  ! {c.cid}: {type(exc).__name__}: {str(exc)[:100]}")
                continue
            names = answer_names(obj)
            verdict = closed_set(names, cands)
            if verdict:
                reasons[verdict.split(" for ")[0]] += 1
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({
                "case_id": c.cid, "pool": pool, "evidence": evidence, "model": model,
                "candidates": [sorted(s) for s in pool_ids],
                "reranked": [sorted(s) for s, _ in align(names, cands)],
                "closed_set": verdict or "ok",
                "raw_response": raw,
            }, ensure_ascii=False, indent=1), encoding="utf-8")
            ran += 1
        print(f"[rerank {pool}/{evidence}] {model} {ds}: {ran} new, {done} already done, "
              f"{failed} failed ({time.time() - t0:.0f}s)")
    bad = sum(reasons.values())
    print(f"[rerank {pool}/{evidence}] closed set: {ran - bad}/{ran} answers named exactly the "
          f"candidates shown" + ("" if not bad else "; " + ", ".join(f"{k} {why}" for why, k in reasons.items())))
    llm.unload(model)
