# -*- coding: utf-8 -*-
"""Paths, pinned releases, models and the constants shared by every step.

Paths are resolved from the repository root. Model outputs, rendered prompts and scores
go under results/, which the environment variable RAREBRG_RESULTS can redirect.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---- data ------------------------------------------------------------------------------
DATA = ROOT / "data"
DATASETS_DIR = DATA / "datasets"                      # RareBench (run.py download)
ONTOLOGIES = DATA / "ontologies"                      # HPO, HPOA, MONDO (run.py download)
DERIVED = DATA / "derived"                            # built from them (run.py prepare)
# Free-text diagnosis name -> MONDO concept, or invalid, as decided by concept resolution
# (lib/normalize.py). The map of the study ships with the repository and covers the names that recur.
RESOLUTION_MAP = DATA / "resolution_map.json"
# Batches and verdicts of agent matching go to the skill's scratch folder.
NORMALIZE_WORKDIR = ROOT / "skills" / "diagnosis-normalize" / "workdir"

RAREBENCH_PHENOTYPE_NAMES = DATASETS_DIR / "phenotype_mapping.json"
HP_JSON = ONTOLOGIES / "hp.json"
HPOA = ONTOLOGIES / "phenotype.hpoa"
MONDO_JSON = ONTOLOGIES / "mondo.json"

HPO_GRAPH = DERIVED / "hpo_isa_graph.json"
MONDO_GRAPH = DERIVED / "mondo_isa_graph.json"
MONDO_CONCEPTS = DERIVED / "mondo_concepts.json"
SAPBERT_EMBEDDINGS = DERIVED / "mondo_embeddings_sapbert.npy"
SAPBERT_NAMES = DERIVED / "mondo_string_index_sapbert.json"

# ---- results ---------------------------------------------------------------------------
RESULTS = Path(os.environ.get("RAREBRG_RESULTS", ROOT / "results"))
RES_RETRIEVAL = RESULTS / "retrieval"
RES_TOOL_REPLIES = RESULTS / "tool_replies"
RES_TOOLS = RESULTS / "tools"
RES_PROMPTS = RESULTS / "prompts"
RES_GENERATE = RESULTS / "generate"
RES_RERANK = RESULTS / "rerank"
RES_SCORES = RESULTS / "scores"

# ---- benchmark -------------------------------------------------------------------------
# The RareBench subsets, in the order of every table.
DATASETS = ("MME", "LIRICAL", "HMS", "RAMEDIS")


def available_datasets():
    """Every dataset present under data/datasets/ (one <name>.jsonl each), sorted."""
    return tuple(sorted(p.stem for p in DATASETS_DIR.glob("*.jsonl")))


def resolve_datasets(arg=None):
    """The dataset selection of every command: None gives the four RareBench subsets,
    'all' every dataset under data/datasets/ (RareBench first), a name that dataset."""
    if arg is None:
        return DATASETS
    if arg == "all":
        return DATASETS + tuple(d for d in available_datasets() if d not in DATASETS)
    return (arg,)


# ---- pinned releases -------------------------------------------------------------------
# (url, md5, destination) of each input; `run.py download` fetches and checks them. HPO and
# HPOA come from two releases of the HPO project: the ontology from 2026-06-06 and the
# annotation file from 2026-06-23.
RAREBENCH_REVISION = "6f054e04071953ef2c1779b279074245f2ab398c"
_HF = f"https://huggingface.co/datasets/chenxz/RareBench/resolve/{RAREBENCH_REVISION}"
SOURCES = {
    "hpo": ("http://purl.obolibrary.org/obo/hp/releases/2026-06-06/hp.json",
            "7adc36570708788a061b28849e48357b", HP_JSON),
    "hpoa": ("https://github.com/obophenotype/human-phenotype-ontology/releases/download/"
             "v2026-06-23/phenotype.hpoa", "c226fcfb352159682b089f99423d26df", HPOA),
    "mondo": ("http://purl.obolibrary.org/obo/mondo/releases/2026-06-02/mondo.json",
              "d856c79cccef196f210fdf8447dfe935", MONDO_JSON),
    "rarebench-names": (f"{_HF}/mapping/phenotype_mapping.json",
                        "3247cc2a7c654aea38eab8f6f2aec34a", RAREBENCH_PHENOTYPE_NAMES),
}
RAREBENCH_ZIP = (f"{_HF}/data.zip", "10bbfd5648f1de02a54010f53e92f85d")
RAREBENCH_FILES = {"MME": "2011d7c6e6058acc8359424d5d5ea13a",
                   "LIRICAL": "e6e8281ec5597c20f12feea54f928ffc",
                   "HMS": "9650ec6dad75e18c1f3c69cf72d90dde",
                   "RAMEDIS": "2898f073b3d1b0c095c9be1fe2cacf70"}

# ---- method ----------------------------------------------------------------------------
# Disease profiles use HPOA's phenotypic-abnormality annotations (aspect P); negated rows
# are dropped.
HPOA_ASPECT = "P"
# Ontology-grounded matching: a candidate matches when one of its MONDO ids is a
# gold-standard concept or lies within this many is_a hops of one (0 = exact match).
MATCH_HOPS = 1
# Size of pools R, G, RAG and T, and of the retrieved list shown in the RAG prompt.
POOL_SIZE = 5
# Retrieval breaks score ties by fixed-seed randomization.
TIE_SEED = 777
# A candidate with several MONDO ids is shown to an LLM under the primary name of one of
# them, chosen with this seed.
DISPLAY_SEED = 42

# ---- LLMs ------------------------------------------------------------------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
# Decoding of the models served by Ollama.
OLLAMA_OPTIONS = {"temperature": 0.2, "top_p": 0.9, "seed": 42}

GENERATION_SYSTEM_PROMPT = (
    "You are a specialist in the field of rare diseases. You will be provided and asked about a "
    "complicated clinical case; read it carefully and then provide a diverse and comprehensive "
    "differential diagnosis."
)
# Reranking sends no system prompt: the rendered prompt is the whole instruction.
RERANK_SYSTEM_PROMPT = ""

# The generation prompt; [[patient_info]] takes the names of the case's HPO terms.
GENERATION_PROMPT = """\
Patient's phenotype: [[patient_info]]
Enumerate the top 5 most likely diagnoses. Be precise, and try to cover many unique possibilities.
Each diagnosis should be a rare disease.

For each diagnosis, provide the disease name and a brief rationale.
Order the diagnoses from most likely (first) to least likely (last).

Output a JSON object with this exact structure:
{
  "top_5_diagnoses": [
    {"disease_name": "...", "rationale": "..."},
    {"disease_name": "...", "rationale": "..."},
    {"disease_name": "...", "rationale": "..."},
    {"disease_name": "...", "rationale": "..."},
    {"disease_name": "...", "rationale": "..."}
  ]
}
"""

# name -> provider and model id (see lib/llm.py). API keys are read from OPENAI_API_KEY,
# ANTHROPIC_API_KEY and DEEPSEEK_API_KEY. An optional `options` dict sets the decoding
# parameters; by default Ollama models use OLLAMA_OPTIONS and API models their provider's
# defaults.
MODELS = {
    "llama3.1-8b": {"provider": "ollama", "model": "llama3.1:8b"},
    "qwen3.5-9b": {"provider": "ollama", "model": "qwen3.5:9b"},
    "gemma4-12b": {"provider": "ollama", "model": "gemma4:12b"},
    "phi4-14b": {"provider": "ollama", "model": "phi4:14b"},
    "opus-4.8": {"provider": "anthropic", "model": "claude-opus-4-8"},
    "deepseek-flash": {"provider": "deepseek", "model": "deepseek-flash"},
    "deepseek-v4-pro": {"provider": "deepseek", "model": "deepseek-v4-pro"},
}
# The five LLMs of the study, in the order of its tables.
STUDY_LLMS = ("llama3.1-8b", "qwen3.5-9b", "phi4-14b", "gemma4-12b", "opus-4.8")
# The LLM that reranks pool R, which no LLM generated.
R_RERANKER = "opus-4.8"

# ---- pools -----------------------------------------------------------------------------
#   R    retrieval                         G    LLM generation
#   RUG  G followed by the rest of R       RAG  retrieval-augmented generation
#   T    a diagnosis tool                  RUT  T followed by the rest of R
GENERATION_POOLS = ("G", "RAG")
LLM_POOLS = ("R", "G", "RUG", "RAG")
TOOL_POOLS = ("T", "RUT")
# Evidence given to an LLM reranker: the candidate names alone ('pure') or each candidate
# with its most characteristic phenotypes ('pheno').
EVIDENCES = ("pure", "pheno")
TOOLS = ("pubcasefinder", "phenobrain")


def generation_dir(pool, model):
    """results/generate/<pool>/<model>/, pool G or RAG."""
    assert pool in GENERATION_POOLS, pool
    return RES_GENERATE / pool / model


def rerank_dir(pool, evidence, model):
    """results/rerank/<pool>/<evidence>/<model>/."""
    assert pool in LLM_POOLS and evidence in EVIDENCES, (pool, evidence)
    return RES_RERANK / pool / evidence / model


def generation_prompt_path(pool, ds, cid):
    """results/prompts/generate/<pool>/<dataset>/<case_id>.txt."""
    assert pool in GENERATION_POOLS, pool
    return RES_PROMPTS / "generate" / pool / ds / f"{cid}.txt"


def rerank_prompt_path(pool, evidence, generator, ds, cid):
    """results/prompts/rerank/<pool>/<evidence>/<generator>/<dataset>/<case_id>.txt; pool R,
    which no LLM generated, has no <generator> level. Every reranker of a pool reads the
    same prompt."""
    assert pool in LLM_POOLS and evidence in EVIDENCES, (pool, evidence)
    base = RES_PROMPTS / "rerank" / pool / evidence
    return (base if pool == "R" else base / generator) / ds / f"{cid}.txt"
