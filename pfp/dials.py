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
import re
from typing import Any, Dict, Mapping, Tuple

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


# The sandbox yardstick (v0.31). The certainty weights are dials a proposition may rewrite, so a
# winner's `certainty_score` is measured in units the candidate itself chose: raising
# `survived_falsifications_weight` 3.0 -> 3.25 lifted the mock sandbox winner from 22.5 to 24.75
# with the identical run shape, and the paired A/B adopted it as an improvement (lowering a weight
# got an unchanged candidate auto-reverted). Both sides of the A/B, and the live-run fallback, are
# therefore scored on the factory weights, which no proposition can reach. Only falsification
# evidence counts: untested age, merges attracted (v0.34, see `evidence_score`) and the
# probe-cluster support bonus are ranking signals, not proof.
REFERENCE_WEIGHTS = CertaintyWeights(
    survived_falsifications=float(DEFAULT_DIALS["survived_falsifications_weight"]),
    merges_attracted=float(DEFAULT_DIALS["merges_attracted_weight"]),
    open_critic_objections=float(DEFAULT_DIALS["open_critic_objections_weight"]),
    age_without_testing=float(DEFAULT_DIALS["age_without_testing_weight"]),
)


def _evidence_count(value: Any) -> int:
    """A non-negative whole count, else 0 (journal/summary data is untrusted)."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float) and math.isfinite(value) and value == int(value):
        return max(0, int(value))
    return 0


_GAUNTLET_ROUND = re.compile(r"pass_gauntlet_round_(\d+)(?::|$)")
_SURVIVAL_SCHEDULES = frozenset(
    {"post_merge", "contradiction", "background_sample", "pre_exit", "dependent_reverify"}
)


def _corroborated_survivals(record: Mapping[str, Any]) -> int:
    """Return a fail-closed, schedule-normalized survival count.

    Modern branch records carry a provenance entry for every survival. The raw counter is a
    cache and must not be stronger evidence than that audit trail. Each fixed schedule earns at
    most one unit and each reference gauntlet round earns at most one unit, so a configuration
    cannot improve its A/B score merely by arranging repeated calls from the same schedule.

    Records without the key are legacy inputs from before provenance existed; retain their old
    counter semantics for replay compatibility. A present but malformed provenance field is
    modern corrupt data and therefore scores zero rather than falling back to the counter.
    """
    claimed = _evidence_count(record.get("survived_falsifications"))
    if "pass_provenance" not in record:
        return claimed
    provenance = record.get("pass_provenance")
    if not isinstance(provenance, list):
        return 0
    credited: set[str] = set()
    reference_rounds = int(DEFAULT_DIALS["gauntlet_rounds"])
    for entry in provenance:
        what = entry.get("survived_what") if isinstance(entry, Mapping) else None
        if not isinstance(what, str):
            continue
        trigger = what.split(":", 1)[0]
        gauntlet = _GAUNTLET_ROUND.fullmatch(trigger)
        if gauntlet:
            round_number = int(gauntlet.group(1))
            if 1 <= round_number <= reference_rounds:
                credited.add(f"pass_gauntlet_round_{round_number}")
        elif trigger in _SURVIVAL_SCHEDULES:
            credited.add(trigger)
    return min(claimed, len(credited))


def evidence_score(record: Mapping[str, Any]) -> float:
    """
    Version-invariant score of a branch record (a winners-box entry or a mined winner shape):
    its survived falsifications and open objections on `REFERENCE_WEIGHTS`, floored at 0. Use it
    whenever scores from runs on different dials are compared.

    `merges_attracted` is not counted (v0.34). An `overlap` verdict is agreement, not a survived
    attack, and how many merges a winner accumulates is set by dials: `prefilter_threshold` gates
    the contradiction check as well as the alignment check, so raising it hides clashes that would
    have killed a branch, which then merges again instead. On the mock sandbox, threshold 0.95
    skipped a 0.71-similarity clash, the doomed branch merged into the winner, and the candidate
    out-scored the incumbent 22.5 to 19.5 (merges 5 vs 3, survivals 5 vs 5); the reverse rewrite,
    restoring the clash check, read as a 13% degradation. Merges still reach the winner's own
    certainty and ranking; they just can't move the A/B.

    `gauntlet_rounds` is a dial too: a candidate that raises it to 9 buys six more survivals per
    winner and would out-score the incumbent by testing more, not by reasoning better. Survivals
    recorded (in `pass_provenance`) from gauntlet rounds past the reference count are not
    counted; a candidate that runs fewer rounds still loses the evidence it skipped.
    """
    if not isinstance(record, Mapping):
        return 0.0
    w = REFERENCE_WEIGHTS
    survived = _corroborated_survivals(record)
    score = (
        survived * w.survived_falsifications
        + _evidence_count(record.get("open_critic_objections")) * w.open_critic_objections
    )
    return round(max(0.0, score), 4)
