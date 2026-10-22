# -*- coding: utf-8 -*-
"""SapBERT embeddings of every MONDO primary name and synonym, used only by the
diagnosis-normalize skill to retrieve candidate concepts for new names
(SapBERT-from-PubMedBERT-fulltext; Liu et al., NAACL 2021). Needs torch and transformers;
a GPU is recommended. Writes config.SAPBERT_EMBEDDINGS and config.SAPBERT_NAMES."""
from __future__ import annotations

import json
import time

import numpy as np

from lib import config as C

MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
BATCH = 256        # lower it if the GPU runs out of memory
MAX_LEN = 32       # disease names are short


def build() -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

    rows = []
    for c in json.loads(C.MONDO_CONCEPTS.read_text(encoding="utf-8")):
        if c["primary_name"]:
            rows.append({"mondo_id": c["mondo_id"], "name_type": "primary", "name": c["primary_name"]})
        for s in c.get("synonyms", []):
            if s:
                rows.append({"mondo_id": c["mondo_id"], "name_type": "synonym", "name": s})
    print(f"[prepare] SapBERT on {device}: embedding {len(rows)} MONDO names")

    embs = np.zeros((len(rows), 768), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(rows), BATCH):
            enc = tokenizer([r["name"] for r in rows[i:i + BATCH]], padding=True, truncation=True,
                            max_length=MAX_LEN, return_tensors="pt").to(device)
            out = model(**enc).last_hidden_state[:, 0, :]          # [CLS] embedding
            embs[i:i + len(out)] = out.cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms

    C.SAPBERT_EMBEDDINGS.parent.mkdir(parents=True, exist_ok=True)
    np.save(C.SAPBERT_EMBEDDINGS, embs)
    C.SAPBERT_NAMES.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[prepare] SapBERT index: {len(rows)} names ({time.time() - t0:.0f}s) "
          f"-> {C.SAPBERT_EMBEDDINGS.relative_to(C.ROOT)}")
