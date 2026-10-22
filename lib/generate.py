# -*- coding: utf-8 -*-
"""LLM generation: pools G and RAG.

    G    the LLM's 5 most likely diagnoses for the case's phenotypes
    RAG  the same prompt with the 5 candidates of R listed by MONDO primary name, which
         the LLM may use, reorder or replace, and add to

A prompt is rendered once to results/prompts/generate/<pool>/ and read back from there.
Each answer is written to results/generate/<pool>/<model>/<dataset>/<case_id>.json.
"""
from __future__ import annotations

import json
import time
from functools import lru_cache

from lib import config as C
from lib import llm
from lib.rarebench import case, cases, phenotype_text
from lib.retrieve import table_path

CANDIDATE_HEADER = ("A phenotype-based retrieval tool has proposed the following candidate rare "
                    "diseases for this patient (ranked most to least likely):")
CANDIDATE_INSTRUCTION = ("You may use, reorder, or replace these candidates, and you may add "
                         "diagnoses of your own.")


@lru_cache(maxsize=None)
def _retrieved(ds: str) -> dict:
    p = table_path(ds)
    if not p.exists():
        raise SystemExit(f"no retrieval results for {ds}; run `python run.py retrieve` first")
    return json.loads(p.read_text(encoding="utf-8"))["candidates"]


def _render(pool: str, ds: str, idx: int) -> str:
    c = case(ds, idx)
    base = C.GENERATION_PROMPT.replace("[[patient_info]]", phenotype_text(c))
    if pool == "G":
        return base
    head, _, tail = base.partition("\n")
    listing = "\n".join(f"{r['rank']}. {r['name']}" for r in _retrieved(ds)[c.cid][:C.POOL_SIZE])
    return f"{head}\n\n{CANDIDATE_HEADER}\n{listing}\n{CANDIDATE_INSTRUCTION}\n\n{tail}"


def prompt(pool: str, ds: str, idx: int) -> str:
    """The generation prompt of a case for pool G or RAG."""
    f = C.generation_prompt_path(pool, ds, f"{ds}_{idx}")
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(_render(pool, ds, idx), encoding="utf-8")
    return f.read_text(encoding="utf-8")


def run(model: str, pool: str, datasets=None, limit=None) -> None:
    """Generate every case of `datasets` with `model`. Cases already answered are skipped,
    so an interrupted run resumes where it stopped."""
    outroot = C.generation_dir(pool, model)
    for ds in (datasets or C.DATASETS):
        todo = cases(ds)[:limit] if limit else cases(ds)
        ran = done = failed = 0
        t0 = time.time()
        for c in todo:
            out = outroot / ds / f"{c.cid}.json"
            if out.exists():
                done += 1
                continue
            try:
                raw, obj = llm.chat(model, prompt(pool, ds, c.index), system=C.GENERATION_SYSTEM_PROMPT)
            except Exception as exc:                  # network or parse failure: retried on resume
                failed += 1
                print(f"  ! {c.cid}: {type(exc).__name__}: {str(exc)[:100]}")
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"case_id": c.cid, "pool": pool, "model": model,
                                       "diagnoses": llm.parse_diagnoses(obj), "raw_response": raw},
                                      ensure_ascii=False, indent=1), encoding="utf-8")
            ran += 1
        print(f"[generate {pool}] {model} {ds}: {ran} new, {done} already done, {failed} failed "
              f"({time.time() - t0:.0f}s)")
    llm.unload(model)
