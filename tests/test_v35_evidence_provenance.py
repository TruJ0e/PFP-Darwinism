"""v0.35: the sandbox yardstick scores corroborated evidence, not counters/call volume."""

from __future__ import annotations

from pfp.dials import evidence_score
from pfp.self_improvement import SCORE_YARDSTICK


def _proof(trigger: str, angle: str = "logic") -> dict[str, object]:
    return {"survived_what": f"{trigger}:{angle}", "judge_model": "judge", "timestamp": 1.0}


def test_yardstick_id_marks_the_provenance_custody_boundary():
    assert SCORE_YARDSTICK == "reference_evidence_v3"


def test_raw_survival_counter_cannot_manufacture_modern_evidence():
    assert evidence_score({"survived_falsifications": 999, "pass_provenance": []}) == 0.0
    assert evidence_score({"survived_falsifications": 999, "pass_provenance": "forged"}) == 0.0


def test_counter_and_provenance_cross_check_in_both_directions():
    proofs = [_proof("pre_exit"), _proof("post_merge")]
    assert evidence_score({"survived_falsifications": 1, "pass_provenance": proofs}) == 3.0
    assert evidence_score({"survived_falsifications": 20, "pass_provenance": proofs}) == 6.0


def test_repeating_a_dial_controlled_schedule_cannot_buy_score():
    once = [_proof("contradiction")]
    repeated = [_proof("contradiction", angle) for angle in ("logic", "assumption", "counterexample")]
    assert evidence_score({"survived_falsifications": 1, "pass_provenance": once}) == 3.0
    assert evidence_score({"survived_falsifications": 3, "pass_provenance": repeated}) == 3.0


def test_only_reference_gauntlet_rounds_are_unique_evidence():
    proofs = [
        _proof("pass_gauntlet_round_1"),
        _proof("pass_gauntlet_round_1", "assumption"),
        _proof("pass_gauntlet_round_2"),
        _proof("pass_gauntlet_round_3"),
        _proof("pass_gauntlet_round_4"),
        _proof("pass_gauntlet_round_9"),
    ]
    assert evidence_score({"survived_falsifications": len(proofs), "pass_provenance": proofs}) == 9.0


def test_unknown_or_malformed_provenance_never_earns_credit():
    proofs = [_proof("invented_schedule"), {}, None, "pre_exit:logic"]
    assert evidence_score({"survived_falsifications": 4, "pass_provenance": proofs}) == 0.0


def test_legacy_record_without_provenance_key_remains_replayable():
    assert evidence_score({"survived_falsifications": 5, "open_critic_objections": 1}) == 13.75
