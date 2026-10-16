"""
Tunable dials and their safe ranges (Piece 1 weights, Piece 4/6 thresholds).

The Piece 8 self-rewrite loop lets a model propose dial changes. A proposition
is untrusted input: a typo'd key, a string, a sign flip ("reward open
objections") or `gauntlet_rounds: 0` would silently disable selection. Every
proposed change is validated here before it can reach the sandbox, and every
version's dials are re-validated at the run boundary.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

from .issue_detector import CertaintyWeights

# name -> (min, max, kind). Signs are part of the semantics: survival/merges are
# pluses, open objections and untested age are minuses (DESIGN.md Piece 1).
DIAL_BOUNDS: Dict[str, Tuple[float, float, str]] = {
    "survived_falsifications_weight": (0.0, 10.0, "float"),
    "merges_attracted_weight": (0.0, 10.0, "float"),
    "open_critic_objections_weight": (-10.0, 0.0, "float"),
    "age_without_testing_weight": (-2.0, 0.0, "float"),
    "prefilter_threshold": (0.0, 0.95, "float"),
    "gauntlet_rounds": (1, 9, "int"),
}

DEFAULT_DIALS: Dict[str, Any] = {
    "survived_falsifications_weight": 3.0,
    "merges_attracted_weight": 1.5,
    "open_critic_objections_weight": -1.25,
    "age_without_testing_weight": -0.15,
    "prefilter_threshold": 0.20,
    "gauntlet_rounds": 3,
}


def sanitize_dials(changes: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Validate proposed dial changes.
    Returns (accepted, rejected) where `rejected` maps key -> reason.
    Out-of-range numbers are rejected (not clamped): a proposition that wants
    a value outside the safe envelope is a signal for the anti-model, not
    something to quietly approximate.
    """
    accepted: Dict[str, Any] = {}
    rejected: Dict[str, str] = {}
    for key, value in (changes or {}).items():
        if key not in DIAL_BOUNDS:
            rejected[str(key)] = "unknown_dial"
            continue
        lo, hi, kind = DIAL_BOUNDS[key]
        if isinstance(value, bool):
            rejected[key] = "not_numeric"
            continue
        try:
            num = float(value)
        except (TypeError, ValueError):
            rejected[key] = "not_numeric"
            continue
        if math.isnan(num) or math.isinf(num):
            rejected[key] = "not_finite"
            continue
        if kind == "int":
            if num != int(num):
                rejected[key] = "not_integer"
                continue
            num = int(num)
        if not (lo <= num <= hi):
            rejected[key] = f"out_of_range[{lo},{hi}]"
            continue
        accepted[key] = num
    return accepted, rejected


def resolve_dials(dials: Dict[str, Any]) -> Dict[str, Any]:
    """Defaults overlaid with the valid subset of `dials` (invalid entries fall back to defaults)."""
    accepted, _ = sanitize_dials(dials)
    merged = dict(DEFAULT_DIALS)
    merged.update(accepted)
    return merged


def weights_from_dials(dials: Dict[str, Any]) -> CertaintyWeights:
    d = resolve_dials(dials)
    return CertaintyWeights(
        survived_falsifications=float(d["survived_falsifications_weight"]),
        merges_attracted=float(d["merges_attracted_weight"]),
        open_critic_objections=float(d["open_critic_objections_weight"]),
        age_without_testing=float(d["age_without_testing_weight"]),
    )
