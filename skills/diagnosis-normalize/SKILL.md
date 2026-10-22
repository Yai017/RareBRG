---
name: diagnosis-normalize
description: Agent matching of rare-disease diagnosis names that the resolution map and literal matching could not resolve. Batches prepared by `python run.py normalize` give each name its SapBERT Top-5 MONDO candidates; a no-memory agent selects the right concept among them, else searches the full local MONDO table with queries it builds (verifying online for hard cases), else marks the name invalid, and a second agent audits every decision. Every ID is copied verbatim from tool output, never from memory.
---

# Diagnosis Name Normalization: Agent Matching

This skill is the agent matching step of the concept resolution pipeline. The pipeline code,
`python run.py normalize`, has already looked every name up in the resolution map, tried literal
matching against MONDO labels and synonyms, and retrieved the SapBERT Top-5 MONDO candidates of each
name left over. Those names are in `skills/diagnosis-normalize/workdir/batches/<prefix>_batch{i}.json`
(~50 per file), where `<prefix>` is the model name, or `tools` for diagnosis tool candidates; each
item is `{clean, candidates_top5}`, with `candidates_top5` =
`[{mondo_id, primary_name, sim, synonyms_preview}]`. Agents decide each name, and the verdicts go
back to the pipeline, whose cross-check adds them to the resolution map (step 3).

> **Why two retrievers (candidate-retrieval semantic + full-MONDO-search lexical).** SapBERT is semantic (embeddings):
> it matches acronyms/eponyms that share almost no words with the concept (e.g. "CdLS" → *Cornelia
> de Lange syndrome*) — which a word-overlap search cannot. The lexical `mondo_search_cli.py` matches
> on shared words, catching concepts SapBERT happened to rank outside its Top-5. They fail on
> different cases, so the agent is given both: the pre-computed semantic Top-5 in the batches, and the lexical
> search tool on demand in Agent Matching. Neither alone is enough.

## Rules

- **Every name in the batches gets a verdict**, a concept or `invalid`; none is skipped.
- **IDs only from tool output** (the IRON RULE below) and **`invalid` only after a real search**
  (the full-MONDO search / online-verification steps) — the two rules that keep the map free of
  hallucinated IDs and false-nulls.
- The matching and audit agents write verdict files only. The resolution map is written by the
  pipeline's cross-check, which you run at the end (step 3).

## 1. Agent Matching
Dispatch fresh **no-memory** agents in the **background**, one per batch (≤16 concurrent; each agent
**writes its verdicts to `skills/diagnosis-normalize/workdir/<prefix>_verdict{i}.json`** for batch
`<prefix>_batch{i}.json` — output is too long to return inline). For every name and its Top-5
candidates, the agent decides in this order (on a re-dispatch the item carries an `audit` opinion from
the previous round — read it and act on it first, then follow the order):

1. One of the 5 candidates is the right concept → select its `mondo_id` → **`agent_match`**.
2. None of the 5 fit → call the **full-MONDO search** code (run from the repo root):
   `python skills/diagnosis-normalize/scripts/mondo_search_cli.py "<query>" --limit 15`
   (prints real `{mondo_id, primary_name, synonyms, match_score}`; ~1s, no GPU). Search
   **comprehensively** — try the target's OWN name in a few phrasings AND queries you construct from
   what the candidates reveal (the disease family, naming pattern, word root). The one thing to avoid
   is querying a *candidate's* name (that searches a different disease). **To reach a PARENT/grouping
   concept, query the STRIPPED generic form** — drop the subtype and qualifiers (e.g. search "X" or
   "inherited X", not "X, autosomal recessive"): a qualified query is dominated by the specific
   subtypes and buries the umbrella concept past the result limit, so the parent only surfaces when you
   search the bare generic term. **Stopping rule (so `invalid`
   is reproducible):** keep querying until either a convincing concept is found, OR at least **3
   substantially-different** query formulations have been tried and failed — never conclude `invalid`
   after a single search. A real match → **`agent_match`**.
3. Still none and the case is hard (gene-named, ambiguous eponym) → **online verification, narrowly
   scoped** — use it ONLY to settle a gene–disease relationship, disambiguate an eponym, or expand an
   acronym; do NOT do a broad literature review. Then reconstruct a better query and search the full
   MONDO table again. A real match → **`agent_match`**.
4. Nothing matches after all of the above → **`invalid`**.

**Match criteria (the agent's judgement).** A candidate or search hit matches when it is the same
disease, an accepted synonym, or the clearly-intended concept. Two principles are easy to violate —
apply them as GENERAL rules (the patterns below are signals, not a fixed checklist):

- **Concept level — be faithful to the name's CORE claim; the result must not assert more, or other,
  than the name does (`normalize ⊆ diagnose`), and never a wrong sibling.** Separate a qualifier the
  name **asserts** (a "type N", a gene/locus given AS the diagnosis — these BIND the level) from one
  that is merely **illustrative** (a parenthetical example signalled by "e.g.", "such as", "due to",
  "-related", or a bare example gene — these are droppable):
  - An **asserted** subtype that maps to a real concept → use that level.
  - An **asserted** subtype that is **fabricated / non-existent** (e.g. a "type N" that no ontology
    concept supports) → the name is **`invalid`**; do NOT generalise it up to the parent — the
    diagnosis claimed a specific thing that does not exist, and mapping to the broader real disease
    would assert something the diagnosis did not validly claim.
  - An **illustrative** qualifier → drop it entirely and resolve the MAIN name (its own concept, or
    its parent when the main name is generic/uncoded). This holds EVEN IF you recognise that the
    example gene/term has its own specific subtype — an "e.g."/"such as" gene must NOT pull the result
    down to that gene's subtype; the MAIN name fixes the level (e.g. a name whose core is the generic
    umbrella term, with a gene offered only as an example, resolves to the umbrella, not the subtype).
  - A **gene named alone does not imply a subtype**: a gene-centred name ("GENE-related disorder") →
    the **gene-level** disease concept, not a phenotype-specific subtype, unless the name states it.
  - A real but separately-uncoded subtype/form falls back to its parent.
  - **When the candidates are ALL specific subtypes/SIBLINGS** (e.g. every candidate is an "X type N")
    but the name sits at a HIGHER level — just "X", or "X" with a qualifier that no single coded
    concept captures (an inheritance mode, a generic grouping) — do NOT pick any one sibling (that
    asserts a subtype the name never claimed, and is likely the wrong sibling). Resolve to the
    **PARENT / grouping concept** (the umbrella "X", or an "inherited X" grouping). If that parent is
    not among the candidates, you MUST search for it — see the full-MONDO search step.
- An **"A or B"** of subtypes of one parent disease maps to the shared parent.
- **A self-contradictory or fabricated name is `invalid`, not its nearest concept** — never
  force-resolve a name to a concept it partly denies (reach `invalid` only after the Agent Matching search has
  genuinely failed). General signals of contradiction: a named gene that is not a known cause of the
  named disease, or an inheritance/mode inconsistent with that gene (verify the gene–disease link
  online whenever a gene is named); an explicit negation or exclusion ("non-", "excluding", "other
  than"); mutually inconsistent descriptors within the name. Also invalid: an invented or garbled
  eponym, and a composite of distinct diseases with no single ontology entry.
- **A wrong or garbled name is `invalid` — do not reconstruct the intended disease from an apparent
  typo or error.** If the name as written denotes the wrong disease, a non-existent entity, or a
  garbling of a real name, it is invalid even when you can guess what was probably meant.
- **A name that denotes a COMMON (non-rare) disease is `invalid`.** This pipeline normalizes
  rare-disease diagnoses — the diagnosing model was required to output rare diseases — so a clearly
  common/ordinary disease is out of scope, even if a MONDO entry exists for it.

**IRON RULE — IDs come only from tool output.** Never author a MONDO ID from memory; a remembered ID
is almost always wrong. Copy each `mondo_id` verbatim from a candidate or from a `mondo_search_cli.py`
result the agent actually ran, and copy the `primary_name` printed beside it.

**Verdict the agent writes** (one record per name, same order and count as the input batch):
```json
{"clean": "<echo verbatim>",
 "mondo_id": "MONDO:0001234" | null,
 "primary_name": "<chosen id's primary_name, verbatim; empty if null>",
 "source_of_id": "candidates_top5" | "mondo_search_cli",
 "rationale": "<=15 words"}
```
`source_of_id` is the **agent-phase provenance only** (where the agent got the id) — it is NOT a map
`source` mark. The pipeline's apply step sets the map `source` (`agent_match` / `invalid`) independently.

## 2. Agent Audit — a second agent per batch (REQUIRED)

A single disambiguation pass does not always follow the Match criteria, and its misses follow the
same few patterns (illustrative-gene narrowing, "A or B" picking one alternative, over-narrow
to a sibling, contradictory/non-rare names resolved instead of `invalid`, a `primary_name` that does not
match its id). So each batch is checked by a **second fresh no-memory audit agent** before its verdicts
are handed back.

The audit agent **judges only — it does not edit the verdict.** It reads the batch
(`<prefix>_batch{i}.json`, the candidates) and the matching agent's `<prefix>_verdict{i}.json` (the
same names in the same order), and for every verdict decides **pass** or **flag**, checking:
- an **illustrative** ("e.g." / "such as" / example-gene) qualifier narrowed to that gene's subtype;
- an **"A or B"** of subtypes mapped to ONE alternative instead of the shared parent;
- a name at a **higher level** mapped to a specific sibling subtype;
- an **internally-contradictory / fabricated / non-rare / garbled** name resolved instead of `invalid`, AND
  the reverse — a **recoverable** name wrongly marked `invalid`;
- any `mondo_id` whose `primary_name` does not match its concept.
For each flagged verdict it records a short **audit opinion** — which criterion is violated and the
direction of the fix (e.g. "illustrative gene → resolve to the umbrella"; "A-or-B → resolve to the shared
parent"; "sibling subtype → search the stripped generic term"; "contradictory → should be `invalid`";
"recoverable → re-run the full search"). It writes its judgement back into `<prefix>_verdict{i}.json` as an
`audit` field on each item — `{"ok": true}` or `{"ok": false, "opinion": "..."}` — and leaves `mondo_id`
untouched.

**The fix happens back in Agent Matching, not in the audit.** Flagged items are re-dispatched to a fresh
Agent Matching agent **with the audit opinion attached**; guided by it, the matching agent re-selects from
the candidates, and if none fit reconstructs a better query and searches the full MONDO table (verifying
online when needed), writing a new verdict. That verdict is audited again. **This Matching ↔ Audit loop
runs at most 3 iterations per batch**; whatever a verdict is after the third audit is what stands.

**Audit each batch as its verdicts land** — do not defer to the end.

## 3. Apply the verdicts (call code)
When every batch has its final verdicts, run `python run.py normalize --apply` from the repository
root yourself; the user does not run it. Its cross-check accepts a `mondo_id` only when it EXISTS in
the MONDO concepts AND its real `primary_name` matches the one the agent reported, and adds the verdicts
to the resolution map (`agent_match` / `invalid`). The names whose verdicts fail are written as new
batches: if it reports names still pending, run steps 1 and 2 on the new batches and apply again,
until it reports that every name is decided.

## Commands

Both are run from the repository root.

- `python skills/diagnosis-normalize/scripts/mondo_search_cli.py "<query>" --limit 15`: the full-MONDO
  search of step 1 (no GPU needed).
- `python run.py normalize --apply`: the cross-check and write-back of step 3.
