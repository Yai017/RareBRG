# -*- coding: utf-8 -*-
"""SapBERT candidate retrieval for agent matching: the top-k MONDO concepts of a name by
embedding similarity to every MONDO primary name and synonym (SapBERT-from-PubMedBERT-fulltext;
the index is built by `python run.py prepare sapbert-index`). Needs torch and transformers."""
from __future__ import annotations

import json

import numpy as np

from lib import config as C

MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"


class Matcher:
    def __init__(self):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise SystemExit("SapBERT candidate retrieval needs torch and transformers "
                             "(pip install torch transformers)")
        if not C.SAPBERT_EMBEDDINGS.exists():
            raise SystemExit("no SapBERT index; run `python run.py prepare sapbert-index` first")
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device).eval()
        self.index = json.loads(C.SAPBERT_NAMES.read_text(encoding="utf-8"))
        self.concepts = {c["mondo_id"]: c for c in json.loads(C.MONDO_CONCEPTS.read_text(encoding="utf-8"))}
        self.embeddings = torch.from_numpy(np.load(C.SAPBERT_EMBEDDINGS)).to(self.device)

    def _embed(self, texts: list[str], max_len: int = 32):
        with self.torch.no_grad():
            enc = self.tokenizer(texts, padding=True, truncation=True, max_length=max_len,
                                 return_tensors="pt").to(self.device)
            embs = self.model(**enc).last_hidden_state[:, 0, :]               # [CLS] embedding
            return embs / embs.norm(dim=1, keepdim=True).clamp(min=1e-12)

    def top_k(self, names: list[str], k: int = 5, batch_size: int = 128) -> list[list[dict]]:
        """For each name, its k most similar distinct MONDO concepts:
        [{mondo_id, primary_name, sim, synonyms_preview}]."""
        out = []
        for i in range(0, len(names), batch_size):
            sims = (self._embed(names[i:i + batch_size]) @ self.embeddings.T).cpu().numpy()
            for row in sims:
                seen: dict[str, dict] = {}
                for idx in np.argsort(-row):
                    mid = self.index[idx]["mondo_id"]
                    if mid not in seen:
                        c = self.concepts[mid]
                        seen[mid] = {"mondo_id": mid, "primary_name": c["primary_name"],
                                     "sim": round(float(row[idx]), 4),
                                     "synonyms_preview": (c.get("synonyms") or [])[:3]}
                        if len(seen) >= k:
                            break
                out.append(list(seen.values()))
        return out
