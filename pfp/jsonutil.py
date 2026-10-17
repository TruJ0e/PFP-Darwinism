"""
Tolerant JSON extraction for live model output.

Live Nemotron / OpenAI-compatible models rarely return bare JSON: replies arrive
wrapped in ```json fences, prefixed with prose ("Sure! Here is..."), or preceded
by a <think>...</think> reasoning block. Every PFP component that parses a model
reply goes through these helpers so a formatting quirk never silently flips a
verdict (e.g. `bool("false") is True`).
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, Optional

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)

_TRUE_WORDS = {"true", "yes", "y", "1", "survived", "pass", "passed"}
_FALSE_WORDS = {"false", "no", "n", "0", "failed", "fail", "killed", "falsified"}


def _candidate_texts(raw: str) -> list[str]:
    text = _THINK_RE.sub("", raw or "")
    # An unterminated <think> block (truncated reply) — keep only what follows it, if anything
    if "<think>" in text.lower() and "</think>" not in text.lower():
        text = text[text.lower().rfind("<think>") + len("<think>"):]
    candidates = [m.strip() for m in _FENCE_RE.findall(text)]
    candidates.append(text.strip())
    return [c for c in candidates if c]


def extract_json(raw: Optional[str]) -> Optional[Any]:
    """
    Return the first JSON object (or array) found in `raw`, or None.
    Tries, in order: fenced code blocks, the whole reply, then the first
    decodable `{...}` / `[...]` span anywhere in the text.
    """
    if raw is None:
        return None
    decoder = json.JSONDecoder()
    for text in _candidate_texts(str(raw)):
        try:
            return json.loads(text)
        except (ValueError, TypeError):
            pass
        for idx, ch in enumerate(text):
            if ch not in "{[":
                continue
            try:
                obj, _ = decoder.raw_decode(text[idx:])
                return obj
            except ValueError:
                continue
    return None


def parse_json_object(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    """Like `extract_json`, but only accepts a JSON object (dict)."""
    obj = extract_json(raw)
    return obj if isinstance(obj, dict) else None


def coerce_bool(value: Any, default: Optional[bool] = None) -> Optional[bool]:
    """
    Strict boolean coercion for model verdicts. Strings like "false" / "no"
    map to False (plain `bool("false")` would be True), and non-finite floats
    (`nan`, `inf`) return `default` instead of `True`. Unknown values return
    `default` so callers can treat them as inconclusive.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            return default
        return bool(value)
    if isinstance(value, str):
        word = value.strip().lower()
        if word in _TRUE_WORDS:
            return True
        if word in _FALSE_WORDS:
            return False
    return default


def as_str_list(value: Any) -> list[str]:
    """Normalize a model-provided list field to a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, (str, int, float)):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [str(x).strip() for x in value if str(x).strip()]
