# -*- coding: utf-8 -*-
"""Phenotype-driven diagnosis tools, the source of pools T and RUT: PubCaseFinder and
PhenoBrain, queried through their public web services.

    fetch  send each case's HPO term ids to the service and keep the reply, with its
           date, in results/tool_replies/<tool>/<dataset>/<case_id>.json. A case whose
           reply is kept is not queried again, since the services change over time.
    map    resolve the encoded identifiers of each kept reply's top 5 through MONDO's
           cross-references, into results/tools/<tool>/<dataset>/<case_id>.json.

A candidate takes every MONDO id that its OMIM / Orphanet identifiers resolve to. A
candidate whose identifiers resolve to none keeps an empty id list and is resolved by name
through the concept resolution of lib.normalize (`run.py normalize --tools`).
"""
from __future__ import annotations

import io
import json
import time

import requests

from lib import config as C
from lib.rarebench import cases, mondo_xref

HEADERS = {"User-Agent": "RareBRG (https://github.com/Yai017/RareBRG)"}
PCF_URL = "https://pubcasefinder.dbcls.jp/api/pcf_get_ranked_list"
PB_URL = "https://www.phenobrain.cs.tsinghua.edu.cn"


def pubcasefinder(hpos: list[str], topk: int = C.POOL_SIZE) -> list[dict]:
    import pandas as pd
    resp = requests.get(f"{PCF_URL}?target=orphanet&format=tsv&hpo_id=" + ",".join(hpos),
                        headers=HEADERS, timeout=90)
    resp.raise_for_status()
    text = resp.text
    if not text.strip() or "not found" in text.lower():
        return []
    df = pd.read_csv(io.StringIO(text), sep="\t").head(topk)

    def col(keys):
        for key in keys:
            for c in df.columns:
                if key in str(c).lower():
                    return c
        return None

    idc, namec, scorec = col(["orpha", "id"]), col(["disease_name", "name", "label"]), col(["score", "rank"])
    out = []
    for _, row in df.iterrows():
        oid = str(row[idc]) if idc else ""
        if oid.isdigit():
            oid = "ORPHA:" + oid
        score = None
        if scorec is not None and pd.notna(row[scorec]):
            try:
                score = float(row[scorec])
            except ValueError:
                score = str(row[scorec])
        out.append({"orpha_id": oid, "name": str(row[namec]).strip() if namec else "", "score": score})
    return out


def phenobrain(hpos: list[str], topk: int = C.POOL_SIZE, polls: int = 45) -> list[dict]:
    resp = requests.get(f"{PB_URL}/predict?model=Ensemble" + "".join(f"&hpoList[]={h}" for h in hpos)
                        + f"&topk={topk}", headers=HEADERS, timeout=90)
    resp.raise_for_status()
    task_id = resp.json()["TASK_ID"]
    result = None
    for _ in range(polls):
        rr = requests.get(f"{PB_URL}/query-predict-result?taskId={task_id}", headers=HEADERS, timeout=90)
        if rr.status_code == 200:
            state = str(rr.json().get("state", "")).lower()
            if state == "success":
                result = rr.json()["result"]
                break
            if state == "failure":
                break
        time.sleep(2)
    if not result:
        return []
    codes = [r["CODE"] for r in result]
    detail = requests.post(f"{PB_URL}/disease-list-detail", json={"diseaseList": codes},
                           headers=HEADERS, timeout=90).json()
    return [{"rd": c, "eng": detail[c].get("ENG_NAME", ""), "source_codes": detail[c].get("SOURCE_CODES", [])}
            for c in codes if c in detail]


QUERY = {"pubcasefinder": pubcasefinder, "phenobrain": phenobrain}


def reply_path(tool: str, ds: str, cid: str):
    return C.RES_TOOL_REPLIES / tool / ds / f"{cid}.json"


def fetch_one(tool: str, ds: str, cid: str, hpos: list[str], retries: int = 3):
    """The kept reply of a case, querying the service only when none is kept."""
    path = reply_path(tool, ds, cid)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    for attempt in range(retries):
        try:
            reply = {"case_id": cid, "fetched": time.strftime("%Y-%m-%d"), "n_hpo": len(hpos),
                     "top5": QUERY[tool](list(hpos))}
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(reply, ensure_ascii=False, indent=1), encoding="utf-8")
            time.sleep(1)
            return reply
        except Exception as exc:
            if attempt == retries - 1:
                print(f"  {cid} [{tool}] failed: {str(exc)[:140]}", flush=True)
                return None
            time.sleep(3)
    return None


def candidates(tool: str, reply: dict, xref) -> list[dict]:
    """The reply's candidates with the MONDO ids of their identifiers (empty when MONDO's
    cross-references do not cover them)."""
    out = []
    for item in reply.get("top5") or []:
        if tool == "pubcasefinder":
            codes, name = ([item["orpha_id"]] if item.get("orpha_id") else []), item.get("name", "")
        else:
            codes, name = (item.get("source_codes") or []), item.get("eng", "")
        codes = [c for c in codes if c]
        mondo = set()
        for code in codes:
            mondo |= xref.code2mondo.get(code, set())
        out.append({"name": name, "codes": codes, "mondo_ids": sorted(mondo)})
    return out


def name_key(name: str) -> str:
    """The part of a candidate name that concept resolution looks up: PhenoBrain names read
    "Name; ABBREVIATION"."""
    return name.split(";")[0].strip()


def fetch(tools=C.TOOLS, datasets=None, limit=None) -> None:
    for ds in (datasets or C.DATASETS):
        todo = cases(ds)[:limit] if limit else cases(ds)
        for k, case in enumerate(todo):
            for tool in tools:
                fetch_one(tool, ds, case.cid, list(case.hpo))
            if k % 50 == 0:
                print(f"  [tools fetch] {ds} {k}/{len(todo)}", flush=True)
        print(f"[tools fetch] {ds} done", flush=True)


def map_replies(tools=C.TOOLS, datasets=None) -> None:
    xref = mondo_xref()
    for tool in tools:
        for ds in (datasets or C.DATASETS):
            built = missing = by_name = 0
            for case in cases(ds):
                path = reply_path(tool, ds, case.cid)
                if not path.exists():
                    missing += 1
                    continue
                ranked = candidates(tool, json.loads(path.read_text(encoding="utf-8")), xref)
                by_name += sum(1 for c in ranked if not c["mondo_ids"])
                out = C.RES_TOOLS / tool / ds / f"{case.cid}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps({"case_id": case.cid, "tool": tool, "candidates": ranked},
                                          ensure_ascii=False, indent=1), encoding="utf-8")
                built += 1
            print(f"[tools map] {tool}/{ds}: {built} cases, {missing} without a reply, "
                  f"{by_name} candidate(s) left to resolution by name", flush=True)


def names_without_xref(tools=C.TOOLS, datasets=None) -> list[str]:
    """The names of the tool candidates whose identifiers MONDO's cross-references do not cover."""
    out = set()
    for tool in tools:
        for ds in (datasets or C.DATASETS):
            for f in (C.RES_TOOLS / tool / ds).glob("*.json"):
                for c in json.loads(f.read_text(encoding="utf-8"))["candidates"]:
                    if not c["mondo_ids"] and name_key(c["name"]):
                        out.add(name_key(c["name"]))
    return sorted(out)
