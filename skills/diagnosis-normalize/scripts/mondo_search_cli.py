# -*- coding: utf-8 -*-
"""Full-MONDO search for Agent Matching: real MONDO concepts whose primary name or a
synonym overlaps the query by words, for when the SapBERT top 5 lacks the right concept.
The agent takes mondo_id values only from this output, never from memory.

Usage:
  python skills/diagnosis-normalize/scripts/mondo_search_cli.py "autosomal recessive cutis laxa" [--limit 15]

Prints a JSON list of {mondo_id, primary_name, synonyms, match_score}; no GPU needed.
"""
import json
import re
import sys
import unicodedata
from pathlib import Path


def _concepts_path() -> Path:
    """data/derived/mondo_concepts.json of the RareBRG repository (the folder holding run.py)."""
    for d in [Path.cwd(), *Path(__file__).resolve().parents]:
        if (d / "run.py").exists() and (d / "lib").is_dir():
            return d / "data" / "derived" / "mondo_concepts.json"
    raise SystemExit("run this script from the RareBRG repository root")


CONCEPTS_PATH = _concepts_path()


def _norm(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s or "")).lower().strip())


def search(query, limit=15):
    q = _norm(query)
    toks = set(q.split())
    scored = []
    for c in json.loads(CONCEPTS_PATH.read_text(encoding="utf-8")):
        best = 0.0
        for nm in [c.get("primary_name", "")] + (c.get("synonyms") or []):
            nn = _norm(nm)
            if nn == q:
                best = 1.0
                break
            ntoks = set(nn.split())
            if toks and ntoks:
                best = max(best, len(toks & ntoks) / len(toks | ntoks))
        if best > 0.25:
            scored.append((best, c))
    scored.sort(key=lambda x: -x[0])
    return [{"mondo_id": c["mondo_id"], "primary_name": c["primary_name"],
             "synonyms": (c.get("synonyms") or [])[:6], "match_score": round(s, 3)}
            for s, c in scored[:limit]]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        raise SystemExit(__doc__)
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 15
    print(json.dumps(search(args[0], limit=limit), ensure_ascii=False, indent=2))
