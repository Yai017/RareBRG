# -*- coding: utf-8 -*-
"""RareBRG command line, one command per step of the study.

  download   fetch RareBench and the HPO, HPOA and MONDO releases of the study
  prepare    build the derived resources (or render every prompt without calling a model)
  retrieve   retrieval-based diagnosis: ICTO over the disease corpus           -> pool R
  tools      query PhenoBrain and PubCaseFinder (fetch), map them to MONDO (map) -> pool T
  generate   LLM generation, plain or retrieval-augmented                      -> pools G, RAG
  normalize  concept resolution of diagnosis names: map lookup, literal matching and SapBERT
             candidates for agent matching (--apply, run by the agent, cross-checks its verdicts)
  rerank     LLM reranking of a pool, candidate-only or phenotype-grounded
  score      Recall@1 of every setting, with the gate thresholds (Table 1)

Typical order:
  python run.py download
  python run.py prepare all
  python run.py retrieve
  python run.py tools fetch
  python run.py tools map
  python run.py normalize --tools
  python run.py generate  -m qwen3.5-9b
  python run.py normalize -m qwen3.5-9b
  python run.py rerank    -m qwen3.5-9b
  python run.py rerank    -m opus-4.8 -p R
  python run.py score
"""
from __future__ import annotations

import argparse

from lib import config as C

PREPARE_STEPS = ("hpo-graph", "mondo-graph", "mondo-concepts", "sapbert-index")


def _download(a):
    from lib.download import download_all
    download_all()


def _render_prompts():
    """Every generation prompt, the rerank prompts of pool R, and the rerank prompts of the
    pools of every LLM with generation outputs."""
    from lib import generate, pools, rerank
    from lib.rarebench import cases
    n = 0
    for ds in C.DATASETS:
        for c in cases(ds):
            for pool in C.GENERATION_POOLS:
                generate.prompt(pool, ds, c.index)
                n += 1
    print(f"[prepare prompts] {n} generation prompts")
    jobs = [("-", "R")] + [(m, p) for m in C.MODELS for p in ("G", "RUG", "RAG")
                           if C.generation_dir("RAG" if p == "RAG" else "G", m).exists()]
    for gen, pool in jobs:
        k = 0
        for ev in C.EVIDENCES:
            for ds in C.DATASETS:
                for c in cases(ds):
                    p = pools.build_pool(gen, ds, c.cid, pool)
                    if p:
                        rerank.prompt(pool, ev, gen, ds, c.index, pools.displayed(p))
                        k += 1
        print(f"[prepare prompts] {k} rerank prompts for pool {pool}" + ("" if gen == "-" else f" of {gen}"))


def _prepare(a):
    from lib.prepare import hpo_graph, mondo_concepts, mondo_graph, sapbert_index
    builders = {"hpo-graph": hpo_graph.build, "mondo-graph": mondo_graph.build,
                "mondo-concepts": mondo_concepts.build, "sapbert-index": sapbert_index.build}
    if a.step == "prompts":
        _render_prompts()
    elif a.step == "all":
        for step in PREPARE_STEPS[:3]:
            builders[step]()
    else:
        builders[a.step]()


def _retrieve(a):
    from lib.retrieve import retrieve
    retrieve(datasets=C.resolve_datasets(a.dataset), topk=a.topk)


def _tools(a):
    from lib import tools
    which = C.TOOLS if a.tool == "all" else (a.tool,)
    if a.step == "fetch":
        tools.fetch(which, datasets=C.resolve_datasets(a.dataset), limit=a.limit)
    else:
        tools.map_replies(which, datasets=C.resolve_datasets(a.dataset))


def _pools(arg, choices):
    return list(choices) if arg == "all" else [arg]


def _generate(a):
    from lib import generate
    for pool in _pools(a.pool, C.GENERATION_POOLS):
        generate.run(a.model, pool, datasets=C.resolve_datasets(a.dataset), limit=a.limit)


def _normalize(a):
    from lib import normalize as N
    if a.model and a.tools:
        raise SystemExit("normalize: give -m MODEL or --tools, not both")
    if a.apply:                          # run by the agent at the end of the skill
        prefixes = [a.model or "tools"] if (a.model or a.tools) else N.batch_prefixes()
        if not prefixes:
            print("[normalize] no batches are waiting")
        for prefix in prefixes:
            r = N.apply_verdicts(prefix)
            print(f"[normalize] {prefix}: cross-checked agent verdicts added to the resolution map: {r['added']}")
            for clean, mid, reported, real in r["rejected"]:
                why = "not a MONDO concept" if real is None else f"its primary name is '{real}', not '{reported}'"
                print(f"            rejected {clean!r} -> {mid}: {why}")
            print(f"            {r['pending']} name(s) still pending in {r['batches']} new batch(es): run Agent "
                  f"Matching and Agent Audit on them, then apply again" if r["pending"] else
                  "            every name is decided")
        return
    if not (a.model or a.tools):
        raise SystemExit("normalize: give -m MODEL or --tools")
    from lib.pools import generated_name_list
    from lib.tools import names_without_xref
    prefix = a.model or "tools"
    datasets = C.resolve_datasets(a.dataset)
    names = (generated_name_list(a.model, _pools(a.pool, C.GENERATION_POOLS), datasets) if a.model
             else names_without_xref(datasets=datasets))
    r = N.resolve(names, prefix)
    print(f"[normalize] {prefix}: {r['names']} names: {r['in_map']} in the resolution map, "
          f"{r['literal']} resolved by literal matching, {r['pending']} left for agent matching")
    if r["pending"]:
        batches = (C.NORMALIZE_WORKDIR / "batches" / f"{prefix}_batch*.json").relative_to(C.ROOT)
        print(f"            SapBERT candidates in {r['batches']} batch(es) {batches}\n"
              f"            Ask a coding agent to run the diagnosis-normalize skill on them; it applies its "
              f"verdicts itself (`python run.py normalize --apply`).")


def _rerank(a):
    from lib import rerank
    from lib.pools import guard_normalized
    pools = _pools(a.pool, ("G", "RUG", "RAG"))
    datasets = C.resolve_datasets(a.dataset)
    needed = sorted({"RAG" if p == "RAG" else "G" for p in pools if p != "R"})
    if needed:
        guard_normalized(a.model, needed, datasets)
    for pool in pools:
        for ev in (C.EVIDENCES if a.evidence == "all" else (a.evidence,)):
            rerank.run(a.model, pool, ev, datasets=datasets, limit=a.limit)


def _score(a):
    from lib.pools import guard_normalized
    from lib.score import table1
    llms = a.llms.split(",") if a.llms else list(C.STUDY_LLMS)
    for m in llms:
        guard_normalized(m, [p for p in C.GENERATION_POOLS if C.generation_dir(p, m).exists()])
    guard_normalized(None)
    table1(llms=llms)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="run.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    models = list(C.MODELS)

    def datasets(p):
        p.add_argument("-d", "--dataset", default=None,
                       help="one dataset under data/datasets/; omit for the four RareBench subsets, "
                            "'all' for every dataset there")

    def limit(p):
        p.add_argument("--limit", type=int, default=None, help="only the first N cases of each dataset")

    p = sub.add_parser("download", help="fetch the datasets and ontology releases")
    p.set_defaults(fn=_download)

    p = sub.add_parser("prepare", help="build the derived resources, or render the prompts")
    p.add_argument("step", choices=["all", *PREPARE_STEPS, "prompts"],
                   help="all = hpo-graph, mondo-graph and mondo-concepts; sapbert-index is needed "
                        "only by the diagnosis-normalize skill")
    p.set_defaults(fn=_prepare)

    p = sub.add_parser("retrieve", help="retrieval-based diagnosis (pool R)")
    p.add_argument("--topk", type=int, default=10, help="diseases kept per case (pool R takes the top 5)")
    datasets(p)
    p.set_defaults(fn=_retrieve)

    p = sub.add_parser("tools", help="query the diagnosis tools (fetch) and map their answers to MONDO (map)")
    p.add_argument("step", choices=["fetch", "map"])
    p.add_argument("--tool", default="all", choices=[*C.TOOLS, "all"])
    limit(p)
    datasets(p)
    p.set_defaults(fn=_tools)

    p = sub.add_parser("generate", help="LLM generation (pools G and RAG)")
    p.add_argument("-m", "--model", required=True, choices=models)
    p.add_argument("-p", "--pool", default="all", choices=[*C.GENERATION_POOLS, "all"])
    limit(p)
    datasets(p)
    p.set_defaults(fn=_generate)

    p = sub.add_parser("normalize", help="concept resolution of an LLM's or the tools' diagnosis names")
    p.add_argument("-m", "--model", choices=models, help="the LLM whose generated names are normalized")
    p.add_argument("--tools", action="store_true",
                   help="the names of the tool candidates that MONDO's cross-references do not cover")
    p.add_argument("-p", "--pool", default="all", choices=[*C.GENERATION_POOLS, "all"])
    p.add_argument("--apply", action="store_true",
                   help="cross-check the agents' verdicts and add them to the resolution map (run by the "
                        "agent at the end of the skill; without -m/--tools, every waiting batch)")
    datasets(p)
    p.set_defaults(fn=_normalize)

    p = sub.add_parser("rerank", help="LLM reranking")
    p.add_argument("-m", "--model", required=True, choices=models)
    p.add_argument("-p", "--pool", default="all", choices=[*C.LLM_POOLS, "all"],
                   help="all = the LLM's own pools G, RUG and RAG; R must be named")
    p.add_argument("-e", "--evidence", default="all", choices=[*C.EVIDENCES, "all"],
                   help="pure = candidate names only; pheno = with characteristic phenotypes")
    limit(p)
    datasets(p)
    p.set_defaults(fn=_rerank)

    p = sub.add_parser("score", help="Recall@1 of every setting (Table 1)")
    p.add_argument("--llms", default=None,
                   help=f"comma-separated LLMs (default: {','.join(C.STUDY_LLMS)})")
    p.set_defaults(fn=_score)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
