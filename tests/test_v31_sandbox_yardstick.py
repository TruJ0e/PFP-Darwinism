"""v0.31: the sandbox A/B is scored on a yardstick no proposition can rewrite.

Certainty weights and `gauntlet_rounds` are dials. Before v0.31 each sandbox side was scored by
its winner's `certainty_score` in its *own* weights, so a candidate that only raised a weight (or
ran more gauntlet rounds) out-scored the incumbent on an identical run and was adopted, and one
that lowered a weight was auto-reverted for doing nothing.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from pfp import MainHarness, MockOpenAIClient
from pfp.dials import DEFAULT_DIALS, REFERENCE_WEIGHTS, evidence_score
from pfp.self_improvement import (
    SCORE_YARDSTICK,
    RewriteProposition,
    SandboxResult,
    SelfImprovementLoop,
    VersionRegistry,
)

PROBLEM = "A small business's SSH works laptop->desktop but not desktop->laptop."


def _events() -> List[Any]:
    harness = MainHarness(client=MockOpenAIClient(), starter_agents=2, hard_cap_usd=2.0)
    harness.run(PROBLEM)
    return harness.arena_feed.events


def _scripted_loop(dial_changes: Dict[str, Any], paired: bool = True):
    registry = VersionRegistry()
    loop = SelfImprovementLoop(client=MockOpenAIClient(), registry=registry, paired_sandbox=paired)
    loop.synthesize_proposition = lambda m, a, c: RewriteProposition(  # type: ignore[assignment]
        summary="scripted", dial_changes=dict(dial_changes), prompt_changes={},
        self_assessment=a, anti_model_critique=c,
    )
    return loop, registry


def _gauntlet(rounds: List[int]) -> List[Dict[str, Any]]:
    return [{"survived_what": f"pass_gauntlet_round_{r}:logic", "falsifier_angle": "logic"} for r in rounds]


# ---- the yardstick itself ----------------------------------------------------------------------


def test_reference_weights_are_the_factory_dials():
    assert REFERENCE_WEIGHTS.survived_falsifications == DEFAULT_DIALS["survived_falsifications_weight"]
    assert REFERENCE_WEIGHTS.merges_attracted == DEFAULT_DIALS["merges_attracted_weight"]
    assert REFERENCE_WEIGHTS.open_critic_objections == DEFAULT_DIALS["open_critic_objections_weight"]


def test_evidence_score_ignores_the_recorded_certainty():
    rec = {"survived_falsifications": 6, "merges_attracted": 3, "open_critic_objections": 0}
    assert evidence_score(rec) == 18.0  # merges don't count since v0.34
    assert evidence_score({**rec, "certainty_score": 999.0}) == 18.0


def test_open_objections_subtract_and_the_score_floors_at_zero():
    assert evidence_score({"survived_falsifications": 2, "open_critic_objections": 1}) == 4.75
    assert evidence_score({"survived_falsifications": 0, "open_critic_objections": 5}) == 0.0


@pytest.mark.parametrize("bad", [True, -3, 2.5, float("nan"), float("inf"), "7", None, [4]])
def test_malformed_counts_count_for_nothing(bad):
    assert evidence_score({"survived_falsifications": bad, "open_critic_objections": 0}) == 0.0
    assert evidence_score({"survived_falsifications": 2, "open_critic_objections": bad}) == 6.0


def test_non_mapping_records_score_zero():
    assert evidence_score(None) == 0.0  # type: ignore[arg-type]
    assert evidence_score(["survived_falsifications"]) == 0.0  # type: ignore[arg-type]


def test_gauntlet_rounds_past_the_reference_count_earn_nothing():
    rounds = list(range(1, 10))  # a 9-round gauntlet + a pre-exit check
    rec = {
        "survived_falsifications": 10,
        "pass_provenance": _gauntlet(rounds) + [{"survived_what": "pre_exit:logic"}],
    }
    assert evidence_score(rec) == 4 * REFERENCE_WEIGHTS.survived_falsifications


def test_malformed_provenance_is_ignored_not_trusted():
    rec = {
        "survived_falsifications": 3,
        "pass_provenance": ["pass_gauntlet_round_7:logic", {"survived_what": 7},
                            {"survived_what": "pass_gauntlet_round_12x"}, None],
    }
    assert evidence_score(rec) == 9.0
    assert evidence_score({"survived_falsifications": 3, "pass_provenance": "junk"}) == 9.0


# ---- the sandbox A/B on the real mock ---------------------------------------------------------


@pytest.fixture(scope="module")
def sandbox_scores() -> Dict[str, Dict[str, Any]]:
    loop = SelfImprovementLoop(client=MockOpenAIClient(), registry=VersionRegistry())
    cases = {
        "incumbent": {},
        "inflated_weights": {"survived_falsifications_weight": 3.25, "merges_attracted_weight": 1.75},
        "deflated_weights": {"survived_falsifications_weight": 1.0},
        "gauntlet_9": {"gauntlet_rounds": 9},
        "gauntlet_1": {"gauntlet_rounds": 1},
    }
    return {name: loop.run_sandbox_test(d).to_dict() for name, d in cases.items()}


def test_reweighting_alone_neither_wins_nor_loses_the_ab(sandbox_scores):
    base = sandbox_scores["incumbent"]
    for name in ("inflated_weights", "deflated_weights"):
        res = sandbox_scores[name]
        assert res["score"] == base["score"] > 0
        assert res["certainty_score"] != base["certainty_score"]  # the old score moved


def test_more_gauntlet_rounds_cannot_buy_a_higher_score(sandbox_scores):
    res = sandbox_scores["gauntlet_9"]
    assert res["certainty_score"] > sandbox_scores["incumbent"]["certainty_score"]
    assert res["score"] == sandbox_scores["incumbent"]["score"]


def test_fewer_gauntlet_rounds_still_lose_the_evidence_they_skipped(sandbox_scores):
    assert sandbox_scores["gauntlet_1"]["score"] < sandbox_scores["incumbent"]["score"]


def test_sandbox_records_name_their_yardstick(sandbox_scores):
    for res in sandbox_scores.values():
        assert res["score_yardstick"] == SCORE_YARDSTICK
        json.dumps(res)


def test_inflating_weights_is_not_adopted_as_an_improvement():
    # Before v0.31 the candidate scored 24.75 against a 22.5 baseline on the identical run shape
    loop, registry = _scripted_loop({"survived_falsifications_weight": 3.25, "merges_attracted_weight": 1.75})
    res = loop.run_post_run_improvement(_events(), problem=PROBLEM, run_sandbox=True)
    sb = res["sandbox_result"]
    assert sb["baseline_source"] == "paired_sandbox"
    assert sb["score"] == sb["baseline_score"] == sb["baseline_sandbox"]["score"]
    assert sb["certainty_score"] > sb["baseline_sandbox"]["certainty_score"]


def test_lowering_a_weight_is_not_auto_reverted_for_doing_nothing():
    # Before v0.31: 10.5 vs 22.5 = a 53% "degradation" from a pure unit change
    loop, registry = _scripted_loop({"survived_falsifications_weight": 1.0})
    res = loop.run_post_run_improvement(_events(), problem=PROBLEM, run_sandbox=True)
    assert res["auto_revert_check"]["reverted"] is False
    assert registry.active_version_id == res["registered_version"]["version_id"]


def test_dropping_the_gauntlet_is_still_caught_as_degradation():
    loop, registry = _scripted_loop({"gauntlet_rounds": 1})
    initial = registry.active_version_id
    res = loop.run_post_run_improvement(_events(), problem=PROBLEM, run_sandbox=True)
    sb = res["sandbox_result"]
    assert sb["score"] < sb["baseline_score"]
    assert res["auto_revert_check"]["reverted"] is True
    assert registry.active_version_id == initial


# ---- the live-run baseline uses the same yardstick ---------------------------------------------


def test_mined_winners_carry_their_evidence_score():
    loop, _ = _scripted_loop({})
    metrics = loop.miner.mine(_events())
    shapes = metrics.winning_branch_shapes
    assert shapes
    for shape in shapes:
        assert shape["evidence_score"] == evidence_score(shape)
    assert metrics.spend_efficiency["max_winner_evidence"] == max(s["evidence_score"] for s in shapes)


def test_unpaired_baseline_is_the_live_runs_evidence_not_its_certainty():
    loop, _ = _scripted_loop({"gauntlet_rounds": 2}, paired=False)
    seen: List[float] = []

    def fake_sandbox(candidate_dials, test_problem="p", baseline_score=5.0, candidate_prompts=None):
        seen.append(baseline_score)
        return SandboxResult(True, test_problem, 1.0, baseline_score, 1, 0.01)

    loop.run_sandbox_test = fake_sandbox  # type: ignore[assignment]
    res = loop.run_post_run_improvement(_events(), problem=PROBLEM, run_sandbox=True)
    assert res["sandbox_result"]["baseline_source"] == "run_metrics"
    expected = res["mined_metrics"]["spend_efficiency"]["max_winner_evidence"]
    assert seen == [expected] and res["sandbox_result"]["baseline_score"] == expected
