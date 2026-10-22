# -*- coding: utf-8 -*-
"""One chat call for every LLM, local or behind an API.

Providers, set per model in config.MODELS:
    ollama     a local Ollama server (config.OLLAMA_URL), with JSON output and thinking off
    openai     the OpenAI Chat Completions API or an OpenAI-compatible server (key in
               OPENAI_API_KEY; base URL in OPENAI_BASE_URL or the model's `base_url`)
    anthropic  the Anthropic Messages API (key in ANTHROPIC_API_KEY)
    deepseek   the DeepSeek API (key in DEEPSEEK_API_KEY)

An answer that cannot be parsed as JSON is requested again with the same prompt, up to
three requests in all.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

import requests

from lib import config as C

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)
MAX_TOKENS = int(os.environ.get("RAREBRG_MAX_TOKENS", "4096"))


def extract_json_object(raw: str) -> dict[str, Any]:
    """The first JSON object in an answer, inside a code fence or not."""
    text = (raw or "").strip()
    fence = _FENCE.search(text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in: {raw[:200]!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError(f"unbalanced braces in: {raw[:200]!r}")


def _spec(model: str) -> dict:
    if model not in C.MODELS:
        raise SystemExit(f"unknown model {model!r}; add it to MODELS in lib/config.py")
    return C.MODELS[model]


def _options(spec: dict) -> dict:
    if "options" in spec:
        return dict(spec["options"] or {})
    return dict(C.OLLAMA_OPTIONS) if spec["provider"] == "ollama" else {}


def _messages(system: str, user: str) -> list[dict]:
    # an empty system prompt sends no system message
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]


def _ollama(spec, system, user, opts, timeout):
    payload = {"model": spec["model"], "stream": False, "think": False, "format": "json",
               "messages": _messages(system, user), "options": opts}
    r = requests.post(f"{C.OLLAMA_URL}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    return str((r.json().get("message") or {}).get("content") or "")


def _chat_completions(base, key, spec, system, user, opts, timeout):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    payload = {"model": spec["model"], "messages": _messages(system, user),
               "response_format": {"type": "json_object"}, **opts}
    r = requests.post(f"{base.rstrip('/')}/chat/completions", json=payload, headers=headers,
                      timeout=timeout)
    r.raise_for_status()
    return str(r.json()["choices"][0]["message"].get("content") or "")


def _openai(spec, system, user, opts, timeout):
    base = spec.get("base_url") or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return _chat_completions(base, os.environ.get("OPENAI_API_KEY"), spec, system, user, opts, timeout)


def _deepseek(spec, system, user, opts, timeout):
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY is not set")
    return _chat_completions("https://api.deepseek.com", key, spec, system, user, opts, timeout)


def _anthropic(spec, system, user, opts, timeout):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY is not set")
    payload = {"model": spec["model"], "max_tokens": spec.get("max_tokens", MAX_TOKENS),
               "messages": [{"role": "user", "content": user}], **opts}
    if system:
        payload["system"] = system
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    r = requests.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers,
                      timeout=timeout)
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text")


_CALL = {"ollama": _ollama, "openai": _openai, "anthropic": _anthropic, "deepseek": _deepseek}


def chat(model: str, user: str, system: str, retries: int = 3, timeout: int = 600) -> tuple[str, dict]:
    """(answer text, parsed JSON object). Raises after `retries` unparseable answers."""
    spec = _spec(model)
    call, opts = _CALL[spec["provider"]], _options(spec)
    err: Exception | None = None
    for _ in range(retries):
        raw = call(spec, system, user, opts, timeout)
        try:
            return raw, extract_json_object(raw)
        except ValueError as exc:
            err = exc
    raise ValueError(f"no parseable JSON after {retries} answers: {err}")


def unload(model: str) -> None:
    """Free the GPU after a local model's run, so the next model loads alone."""
    spec = _spec(model)
    if spec["provider"] != "ollama":
        return
    try:
        requests.post(f"{C.OLLAMA_URL}/api/generate", json={"model": spec["model"], "keep_alive": 0},
                      timeout=120)
    except requests.RequestException as exc:
        print(f"  (unloading {model} failed: {exc})", flush=True)


# ---- answer parsing ----------------------------------------------------------------------
def _entry_name(entry: dict) -> str:
    """The diagnosis name of one answer entry; a mistyped key (`disease__name`,
    `disease_de`, ...) is accepted."""
    if not isinstance(entry, dict):
        return ""
    v = entry.get("disease_name")
    if isinstance(v, str) and v:
        return v
    for k, val in entry.items():
        kl = str(k).lower()
        if isinstance(val, str) and val and ("name" in kl or "disease" in kl) and "ratio" not in kl:
            return val
    for k, val in entry.items():                    # otherwise the string that is not a rationale
        if isinstance(val, str) and val and "ratio" not in str(k).lower():
            return val
    return ""


def _entry_rationale(entry: dict) -> str:
    if not isinstance(entry, dict):
        return ""
    v = entry.get("rationale")
    if isinstance(v, str) and v:
        return v
    for k, val in entry.items():                    # mistyped: ratione, ration, ...
        if isinstance(val, str) and "ratio" in str(k).lower():
            return val
    return ""


def parse_diagnoses(obj: dict) -> list[dict]:
    """[{name, rationale}] of a generation answer (`top_5_diagnoses`, or the first list of
    entries under a mistyped key), at most five; entries without a name are skipped."""
    obj = obj or {}
    items = obj.get("top_5_diagnoses")
    if not items:
        for value in obj.values():
            if isinstance(value, list) and value and isinstance(value[0], dict) \
                    and any("name" in str(k).lower() for k in value[0]):
                items = value
                break
    out = [{"name": _entry_name(d), "rationale": _entry_rationale(d)} for d in (items or [])]
    return [d for d in out if d["name"].strip()][:C.POOL_SIZE]
