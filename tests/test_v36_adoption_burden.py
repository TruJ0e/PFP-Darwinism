"""v0.36: the burden of proof is on the rewrite.

Before v0.36 a coherent candidate became active unless it measured more than
`degradation_threshold` (15%) below its paired baseline. Ties and losses of up to 15% were
adopted, and the adopted loser produced the next cycle's baseline, so drift compounded with no
floor (0.85^n). On the mock, `prefilter_threshold: 0.95`, the dial v0.34 shows hiding a clash
check, ties the incumbent 15.0 to 15.0 and was adopted. A changed candidate measured against a
paired baseline now has to score strictly higher; otherwise it is registered as "unproven",
logged as a failed experiment, and the incumbent stays active.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from pfp import MainHarness, MockOpenAIClient
from pfp.self_improvement import RewriteProposition, SandboxResult, SelfImprovementLoop, VersionRegistry

from test_v11_paired_sandbox import _loop_with

PROBLEM = "A small business's SSH works laptop->desktop but not desktop->laptop."


@pytest.fixture(scope="module")
def events() -> List[Dict[str, Any]]:
    harness = MainHarness(client=MockOpenAIClient(), starter_agents=2, hard_cap_usd=2.0)
    harness.run(PROBLEM)
    return [e.to_dict() for e in harness.arena_feed.events]


def _real_loop(dial_changes: Dict[str, Any], registry: VersionRegistry | None = None):
    registry = registry or VersionRegistry()
    loop = SelfImprovementLoop(client=MockOpenAIClient(), registry=registry, paired_sandbox=True)
    loop.synthesize_proposition = lambda m, a, c: RewriteProposition(  # type: ignore[assignment]
        summary="scripted", dial_changes=dict(dial_changes), prompt_changes={},
        self_assessment=a, anti_model_critique=c,
    )
    return loop, registry


# ---- the defect, on the real mock sandbox --------------------------------------------------------


def test_clash_hiding_prefilter_that_only_ties_is_not_adopted(events):
    loop, registry = _real_loop({"prefilter_threshold": 0.95})
    initial = registry.active_version_id
    res = loop.run_post_run_improvement(events, problem=PROBLEM, run_sandbox=True)
    sb = res["sandbox_result"]
    assert sb["baseline_source"] == "paired_sandbox"
    assert sb["score"] == sb["baseline_score"]  # a tie: no evidence the rewrite helps
    assert sb["adoption_rule"] == "strict_gain_over_paired_baseline"
    cand = res["registered_version"]["version_id"]
    assert res["registered_version"]["status"] == "unproven"
    assert registry.active_version_id == initial
    assert res["active_version_id"] == initial
    assert res["auto_revert_check"]["reverted"] is False
    failed = registry.failed_experiments
    assert [f["version_id"] for f in failed] == [cand]
    assert failed[0]["reason"] == "unproven_no_gain_over_paired_baseline"


# ---- scripted scores ------------------------------------------------------------------------------


def test_sub_threshold_losses_no_longer_compound(events):
    # 10% below baseline: under the 15% revert threshold, so it used to be adopted, and the next
    # cycle then measured against the loser
    loop, registry, _ = _loop_with({"gauntlet_rounds": 2}, {"2": 9.0, "3": 10.0})
    initial = registry.active_version_id
    for _ in range(3):
        res = loop.run_post_run_improvement(events, problem=PROBLEM, run_sandbox=True)
        assert res["registered_version"]["status"] == "unproven"
        assert res["sandbox_result"]["baseline_score"] == 10.0
        assert registry.active_version_id == initial
    assert len(registry.failed_experiments) == 3


def test_a_measured_gain_is_adopted(events):
    loop, registry, _ = _loop_with({"gauntlet_rounds": 2}, {"2": 10.5, "3": 10.0})
    res = loop.run_post_run_improvement(events, problem=PROBLEM, run_sandbox=True)
    assert res["registered_version"]["status"] == "adopted"
    assert registry.active_version_id == res["registered_version"]["version_id"]
    assert registry.failed_experiments == []


def test_a_large_loss_is_still_reverted_and_logged_once(events):
    loop, registry, _ = _loop_with({"gauntlet_rounds": 2}, {"2": 2.0, "3": 10.0})
    initial = registry.active_version_id
    res = loop.run_post_run_improvement(events, problem=PROBLEM, run_sandbox=True)
    assert res["auto_revert_check"]["reverted"] is True
    assert res["registered_version"]["status"] == "reverted"
    assert registry.active_version_id == initial
    assert len(registry.failed_experiments) == 1
    assert registry.failed_experiments[0]["reason"].startswith("auto_reverted_degradation_")


def test_identical_candidate_is_not_held_to_a_gain_over_itself(events):
    loop, registry, calls = _loop_with({}, {"3": 4.0})
    res = loop.run_post_run_improvement(events, problem=PROBLEM, run_sandbox=True)
    assert len(calls) == 1
    assert res["sandbox_result"]["baseline_source"] == "identical_candidate"
    assert res["sandbox_result"]["adoption_rule"] == "degradation_threshold"
    assert res["registered_version"]["status"] == "adopted"
    assert registry.failed_experiments == []


def test_unpaired_baselines_keep_the_threshold_rule(events):
    # A live-run baseline is a different problem: only the degradation threshold applies
    loop, registry, _ = _loop_with({"gauntlet_rounds": 2}, {"2": 3.0, "3": 3.0}, paired=False)
    res = loop.run_post_run_improvement(events, problem=PROBLEM, run_sandbox=True)
    assert res["sandbox_result"]["baseline_source"] == "run_metrics"
    assert res["sandbox_result"]["adoption_rule"] == "degradation_threshold"
    assert res["registered_version"]["status"] != "unproven"


def test_halted_baseline_falls_back_to_the_threshold_rule(events):
    loop, registry, _ = _loop_with({"gauntlet_rounds": 2}, {"2": 9.0, "3": 10.0})
    scripted = loop.run_sandbox_test

    def halted_baseline(candidate_dials, *args, **kwargs):
        res = scripted(candidate_dials, *args, **kwargs)
        if candidate_dials.get("gauntlet_rounds", 3) == 3:
            res = SandboxResult(
                coherent=False, test_problem=res.test_problem, score=0.0,
                baseline_score=res.baseline_score, passed_branches=0, total_spent_usd=0.01,
                halted_reason="hard_cap",
            )
        return res

    loop.run_sandbox_test = halted_baseline  # type: ignore[assignment]
    res = loop.run_post_run_improvement(events, problem=PROBLEM, run_sandbox=True)
    assert res["sandbox_result"]["baseline_source"] == "run_metrics_fallback"
    assert res["sandbox_result"]["adoption_rule"] == "degradation_threshold"


def test_a_tie_under_a_pin_is_unproven_and_leaves_the_pin_alone(events):
    loop, registry, _ = _loop_with({"gauntlet_rounds": 2}, {"2": 10.0, "3": 10.0})
    pinned = registry.get_version()
    registry.pin_version(pinned.version_id)
    res = loop.run_post_run_improvement(events, problem=PROBLEM, run_sandbox=True)
    assert res["registered_version"]["status"] == "unproven"
    assert registry.pinned_version_id == pinned.version_id
    assert registry.get_version().version_id == pinned.version_id


# ---- the registry -----------------------------------------------------------------------------------


@pytest.mark.parametrize("choose", ["pin_version", "restore_version"])
def test_an_operator_can_still_choose_an_unproven_version(events, choose):
    loop, registry, _ = _loop_with({"gauntlet_rounds": 2}, {"2": 10.0, "3": 10.0})
    res = loop.run_post_run_improvement(events, problem=PROBLEM, run_sandbox=True)
    cand = res["registered_version"]["version_id"]
    getattr(registry, choose)(cand)
    assert registry.active_version_id == cand
    assert registry.get_version(cand).status == "adopted"


def test_unproven_status_survives_save_and_load(events):
    loop, registry, _ = _loop_with({"gauntlet_rounds": 2}, {"2": 10.0, "3": 10.0})
    res = loop.run_post_run_improvement(events, problem=PROBLEM, run_sandbox=True)
    cand = res["registered_version"]["version_id"]
    reloaded = VersionRegistry._from_raw(json.loads(json.dumps(registry.to_dict())))
    assert reloaded.get_version(cand).status == "unproven"
    assert reloaded.active_version_id == registry.active_version_id
    assert reloaded.get_version(cand).sandbox_result["adoption_rule"] == "strict_gain_over_paired_baseline"
