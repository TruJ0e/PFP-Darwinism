from __future__ import annotations

import pytest

from pfp import (
    FailJudge,
    MainHarness,
    MergeSynthesizer,
    MockOpenAIClient,
    PFPTracker,
)

PROBLEM = "Cannot connect to remote server over SSH on port 22"


def test_race_mode_defaults_and_budget_cut():
    client = MockOpenAIClient()
    harness = MainHarness(client=client, race_mode=True)
    assert harness.race_mode is True
    assert harness.call_budget == 90
    assert harness.time_budget_seconds == 60.0
    assert harness.depth_cap == 2
    assert harness.piggyback_briefs is True

    summary = harness.run(PROBLEM)
    assert summary["race"]["race_mode"] is True
    assert summary["race"]["calls_budget"] == 90
    assert summary["race"]["calls_used"] <= 90
    assert summary["race"]["anytime_winner"] is not None
    assert summary["race"]["cut_by_budget"] is True
    assert summary["halted_reason"] == "race_budget_expired"
    assert len(summary["winners_box"]) == 1
    winner = summary["winners_box"][0]
    assert winner["id"] == summary["race"]["anytime_winner"]["branch_id"]


def test_race_mode_mock_determinism_and_reproducibility():
    s1 = MainHarness(client=MockOpenAIClient(), race_mode=True).run(PROBLEM)
    s2 = MainHarness(client=MockOpenAIClient(), race_mode=True).run(PROBLEM)
    assert s1["race"]["calls_used"] == s2["race"]["calls_used"]
    assert s1["race"]["anytime_winner"]["branch_id"] == s2["race"]["anytime_winner"]["branch_id"]
    assert s1["winners_box"][0]["id"] == s2["winners_box"][0]["id"]


def test_test_before_merge_deferral_and_attacks_faced():
    tracker = PFPTracker()
    b1 = tracker.register_branch("b1", None, 1.0, "Path A", origin="starter")
    b2 = tracker.register_branch("b2", None, 1.0, "Path B", origin="starter")
    assert not b1.is_merge_eligible
    assert not b2.is_merge_eligible
    assert b1.attacks_faced == 0
    assert b2.attacks_faced == 0

    client = MockOpenAIClient()
    synth = MergeSynthesizer(client=client, tracker=tracker, budget_ledger=None, require_tested=True)
    with pytest.raises(ValueError, match="not eligible for merge: 0 falsification attacks faced"):
        synth.merge_branches(b1.id, b2.id, child_branch_id="child_1")


def test_falsifier_increments_attacks_faced_and_prioritizes_untested():
    tracker = PFPTracker()
    b1 = tracker.register_branch("b1", None, 1.0, "Fork 1", origin="starter")
    b2 = tracker.register_branch("b2", None, 1.0, "Fork 2", origin="starter")
    client = MockOpenAIClient()
    judge = FailJudge(client=client, tracker=tracker)

    out = judge.sample_background(force=True)
    assert out is not None
    target = out.branch_id
    assert target in (b1.id, b2.id)

    rec = tracker.get(target)
    assert rec.attacks_faced == 1

    other_id = b2.id if target == b1.id else b1.id
    out2 = judge.sample_background(force=True)
    assert out2 is not None
    assert out2.branch_id == other_id
