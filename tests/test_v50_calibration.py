"""v0.50 calibration, regression test for the v0.51 one-line fix.

The calibration block originally rewrote the rationale but never assigned
survived = True, so a converted branch still died (run 13, branch_0012: the
fail event carried the calibration rationale). These tests pin the conversion.
"""

from __future__ import annotations

import json

from pfp import FAST_JUDGE_MODEL, SUPER_MODEL, BranchState
from pfp.fail_judge import _is_possibility_only_objection

from test_v41_judge_ladder import _judge


def _kill(objection: str, rationale: str = "") -> str:
    return json.dumps({"survived": False, "objection": objection, "rationale": rationale})


_POSS_PRIMARY = _kill("relies on an unsupported causal claim without any logical or empirical justification")
_POSS_APPEAL = _kill(
    "relies on an unverified causal claim that scent cues improve regulation",
    "the claim assumes the mechanism works without showing how",
)
_CONTRADICTION = _kill(
    "branch contradicts itself by claiming no peer prompts while proposing a peer mentor",
    "contradiction with its own proposal",
)


def test_possibility_only_kill_converts_to_survival_after_upheld_appeal() -> None:
    judge, tracker, _ = _judge({FAST_JUDGE_MODEL: [_POSS_PRIMARY], SUPER_MODEL: [_POSS_APPEAL]})
    out = judge.falsify_branch("b1", schedule_trigger="pass_gauntlet_round_1")
    assert out.survived
    assert not out.appeal_overturned  # appeal upheld the kill; calibration converted it
    assert "Calibration:" in out.rationale
    assert tracker.get("b1").state == BranchState.POSSIBLE
    assert tracker.get("b1").survived_falsifications == 1


def test_concrete_break_kill_still_kills() -> None:
    judge, tracker, _ = _judge({FAST_JUDGE_MODEL: [_CONTRADICTION], SUPER_MODEL: [_CONTRADICTION]})
    out = judge.falsify_branch("b1", schedule_trigger="pass_gauntlet_round_1")
    assert not out.survived
    assert tracker.get("b1").state == BranchState.FAIL


def test_mock_canned_objections_never_convert() -> None:
    canned = [
        "Contradiction with established LAN reachability remains fatal under independent review.",
        "Branch hit the verified line and failed falsification.",
        "Falsified: contradicts established main-line observation (laptop->desktop SSH already succeeds without fail2ban/proxy block).",
    ]
    for objection in canned:
        assert not _is_possibility_only_objection(objection, "Appeal judge confirmed fatal contradiction.")


def test_run10_style_objections_convert() -> None:
    assert _is_possibility_only_objection(
        "relies on an unjustified causal leap", ""
    )
    assert _is_possibility_only_objection(
        "asserts the intervention will reduce disruption but this causal claim lacks logical support and ignores plausible countervailing mechanisms",
        "",
    )
    assert _is_possibility_only_objection(
        "assumes layering cues will uniformly reduce overload without showing how", ""
    )
