"""v0.32: auto-revert judges the version that was measured, and a pin is never collateral damage.

Before v0.32, under a manual pin the post-run cycle registered the candidate (held: the pin kept it
inactive, yet it was labelled "adopted"), sandboxed it, then called `evaluate_and_auto_revert`
on the *current* version, which is the pinned one. A degraded candidate therefore got the
operator's pinned version marked "reverted", unpinned, and replaced by its parent, while the bad
candidate stayed "adopted" and never reached the failed-experiment log the anti-model learns from.

The event-log miner also crashed on a malformed snapshot field (a replayed or legacy v1 journal
is untrusted), killing the cycle before anything was recorded.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from pfp import MainHarness, MockOpenAIClient
from pfp.self_improvement import (
    EventLogMiner,
    RewriteProposition,
    SelfImprovementLoop,
    VersionRegistry,
)

PROBLEM = "A small business's SSH works laptop->desktop but not desktop->laptop."


@pytest.fixture(scope="module")
def events() -> List[Dict[str, Any]]:
    harness = MainHarness(client=MockOpenAIClient(), starter_agents=2, hard_cap_usd=2.0)
    harness.run(PROBLEM)
    return [e.to_dict() for e in harness.arena_feed.events]


def _pinned_loop(dial_changes: Dict[str, Any]):
    registry = VersionRegistry()
    pinned = registry.register_rewrite(
        config_dials={"merges_attracted_weight": 2.0},
        prompts=dict(registry.get_version().prompts),
        proposition={"summary": "operator's choice"},
        sandbox_result={"coherent": True},
    )
    registry.pin_version(pinned.version_id)
    loop = SelfImprovementLoop(client=MockOpenAIClient(), registry=registry, paired_sandbox=True)
    loop.synthesize_proposition = lambda m, a, c: RewriteProposition(  # type: ignore[assignment]
        summary="scripted", dial_changes=dict(dial_changes), prompt_changes={},
        self_assessment=a, anti_model_critique=c,
    )
    return loop, registry, pinned


# ---- the post-run cycle under a pin --------------------------------------------------------------


def test_degraded_candidate_under_a_pin_reverts_itself_not_the_pin(events):
    loop, registry, pinned = _pinned_loop({"gauntlet_rounds": 1})
    res = loop.run_post_run_improvement(events, problem=PROBLEM, run_sandbox=True)
    cand = res["registered_version"]["version_id"]
    check = res["auto_revert_check"]
    assert check["reverted"] is True
    assert check["reverted_from"] == cand
    assert check["reverted_to"] == pinned.version_id
    assert registry.pinned_version_id == pinned.version_id
    assert registry.get_version().version_id == pinned.version_id
    assert registry.get_version(pinned.version_id).status == "adopted"
    assert registry.get_version(cand).status == "reverted"
    assert res["registered_version"]["status"] == "reverted"
    assert [f["version_id"] for f in registry.failed_experiments] == [cand]
    assert res["active_version_id"] == pinned.version_id


def test_passing_candidate_under_a_pin_is_held_not_adopted(events):
    loop, registry, pinned = _pinned_loop({"survived_falsifications_weight": 1.0})
    res = loop.run_post_run_improvement(events, problem=PROBLEM, run_sandbox=True)
    cand = res["registered_version"]["version_id"]
    assert res["auto_revert_check"]["reverted"] is False
    assert res["auto_revert_check"]["active_version_id"] == pinned.version_id
    assert registry.get_version(cand).status == "held_by_pin"
    assert registry.get_version().version_id == pinned.version_id
    assert registry.failed_experiments == []


def test_without_a_pin_a_degraded_candidate_still_falls_back_to_its_parent(events):
    registry = VersionRegistry()
    initial = registry.active_version_id
    loop = SelfImprovementLoop(client=MockOpenAIClient(), registry=registry, paired_sandbox=True)
    loop.synthesize_proposition = lambda m, a, c: RewriteProposition(  # type: ignore[assignment]
        summary="scripted", dial_changes={"gauntlet_rounds": 1}, prompt_changes={},
        self_assessment=a, anti_model_critique=c,
    )
    res = loop.run_post_run_improvement(events, problem=PROBLEM, run_sandbox=True)
    assert res["auto_revert_check"]["reverted"] is True
    assert res["auto_revert_check"]["reverted_to"] == initial
    assert registry.active_version_id == initial


# ---- the registry primitives ---------------------------------------------------------------------


def _registry_with_pinned_and_held():
    reg = VersionRegistry()
    prompts = dict(reg.get_version().prompts)
    pinned = reg.register_rewrite(
        config_dials={}, prompts=prompts, proposition={}, sandbox_result={"coherent": True}
    )
    reg.pin_version(pinned.version_id)
    held = reg.register_rewrite(
        config_dials={"gauntlet_rounds": 4}, prompts=prompts, proposition={},
        sandbox_result={"coherent": True},
    )
    return reg, pinned, held


def test_explicit_version_id_reverts_only_that_version():
    reg, pinned, held = _registry_with_pinned_and_held()
    assert held.status == "held_by_pin"
    out = reg.evaluate_and_auto_revert(1.0, 10.0, version_id=held.version_id)
    assert out["reverted"] is True and out["reverted_from"] == held.version_id
    assert reg.pinned_version_id == pinned.version_id
    assert reg.active_version_id == pinned.version_id
    assert pinned.status == "adopted" and held.status == "reverted"


def test_reverted_version_is_dropped_from_the_next_run_queue():
    reg, _, held = _registry_with_pinned_and_held()
    reg.pending_next_run_version_id = held.version_id
    reg.evaluate_and_auto_revert(1.0, 10.0, version_id=held.version_id)
    assert reg.pending_next_run_version_id is None


def test_default_still_judges_the_current_version():
    reg, pinned, _ = _registry_with_pinned_and_held()
    out = reg.evaluate_and_auto_revert(1.0, 10.0)
    assert out["reverted_from"] == pinned.version_id
    assert reg.pinned_version_id is None


def test_unknown_version_id_is_refused():
    reg, _, _ = _registry_with_pinned_and_held()
    with pytest.raises(KeyError):
        reg.evaluate_and_auto_revert(1.0, 10.0, version_id="v999_nope")


@pytest.mark.parametrize("how", ["pin", "restore"])
def test_choosing_a_held_version_adopts_it(how):
    reg, _, held = _registry_with_pinned_and_held()
    if how == "pin":
        reg.pin_version(held.version_id)
    else:
        reg.restore_version(held.version_id)
    assert held.status == "adopted"
    assert reg.get_version().version_id == held.version_id


def test_held_status_survives_save_and_load(tmp_path):
    reg, pinned, held = _registry_with_pinned_and_held()
    path = reg.save(tmp_path / "registry.json")
    back = VersionRegistry.load(path)
    assert back.get_version(held.version_id).status == "held_by_pin"
    assert back.pinned_version_id == pinned.version_id


# ---- the miner treats journal snapshots as untrusted ---------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("merges_attracted", "many"),
        ("survived_falsifications", [1]),
        ("open_critic_objections", None),
        ("depth", None),
        ("certainty_score", "high"),
        ("certainty_score", float("nan")),
        ("merge_parents", 5),
    ],
)
def test_malformed_winner_snapshot_fields_count_for_nothing(events, field, value):
    bad = []
    for e in events:
        e = dict(e, snapshot=dict(e["snapshot"]))
        if e["snapshot"].get("state") == "Pass":
            e["snapshot"][field] = value
        bad.append(e)
    metrics = EventLogMiner().mine(bad)
    shapes = metrics.winning_branch_shapes
    assert shapes, "the mock run must produce winners"
    key = {"merge_parents": "merge_parents"}.get(field, field)
    expected = [] if field == "merge_parents" else 0
    assert all(s[key] == expected for s in shapes)


@pytest.mark.parametrize("value", ["lots", None, float("inf"), True])
def test_malformed_spend_details_count_for_nothing(events, value):
    bad = []
    for e in events:
        e = dict(e, details=dict(e["details"]))
        if e["event_type"] == "budget_change":
            e["details"]["spent_usd"] = value
        if e["event_type"] == "merge":
            e["details"]["transferred_budget_usd"] = value
            e["details"]["synthesis_cost_usd"] = value
        bad.append(e)
    metrics = EventLogMiner().mine(bad)
    assert metrics.spend_efficiency["total_spent_usd"] == 0.0
    assert metrics.merge_patterns["avg_transferred_budget_usd"] == 0.0
