# -*- coding: utf-8 -*-
"""Fetch the public inputs at the versions used in the study and check their MD5:
the HPO ontology (2026-06-06 release), the HPOA annotations (2026-06-23 release),
MONDO (2026-06-02 release), and RareBench (Hugging Face, pinned revision): its four
subsets and its phenotype naming. Files already present with the right checksum are
kept.
"""
from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import requests

from lib import config as C


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _get(url: str) -> bytes:
    r = requests.get(url, timeout=600)
    r.raise_for_status()
    return r.content


def _fetch(name: str, url: str, md5: str, dest: Path) -> None:
    if dest.exists() and _md5(dest.read_bytes()) == md5:
        print(f"[download] {name}: present ({dest.relative_to(C.ROOT)})")
        return
    print(f"[download] {name}: {url}")
    data = _get(url)
    if _md5(data) != md5:
        raise SystemExit(f"[download] {name}: checksum mismatch (expected {md5}, got {_md5(data)})")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    print(f"[download] {name}: saved {dest.relative_to(C.ROOT)} ({len(data) / 1e6:.1f} MB)")


def rarebench() -> None:
    want = {ds: C.DATASETS_DIR / f"{ds}.jsonl" for ds in C.RAREBENCH_FILES}
    if all(p.exists() and _md5(p.read_bytes()) == C.RAREBENCH_FILES[ds] for ds, p in want.items()):
        print("[download] RareBench subsets: present")
        return
    url, md5 = C.RAREBENCH_ZIP
    print(f"[download] RareBench subsets: {url}")
    data = _get(url)
    if _md5(data) != md5:
        raise SystemExit(f"[download] RareBench: checksum mismatch (expected {md5}, got {_md5(data)})")
    C.DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for ds, p in want.items():
            content = z.read(f"data/{ds}.jsonl")
            if _md5(content) != C.RAREBENCH_FILES[ds]:
                raise SystemExit(f"[download] RareBench {ds}: checksum mismatch")
            p.write_bytes(content)
            print(f"[download] RareBench {ds}: saved {p.relative_to(C.ROOT)}")


def download_all() -> None:
    rarebench()
    for name, (url, md5, dest) in C.SOURCES.items():
        _fetch(name, url, md5, dest)
