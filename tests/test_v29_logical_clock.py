"""v0.29: untested age runs on the arena's logical clock, and similarity never hashes branch IDs."""

from __future__ import annotations

import math
import time

import pytest

from pfp import MainHarness, MockOpenAIClient
from pfp.alignment_detector import AlignmentDetector
from pfp.fail_judge import FailJudge
from pfp.issue_detector import IssueDetector, untested_round_units
from pfp.tracker import BranchState, PFPTracker

SANDBOX_PROBLEM = "Sandbox verification: SSH works laptop->desktop but not desktop->laptop."


@pytest.mark.parametrize(
    "rounds, units",
    [(0, 0.0), (1, 1.0), (3, 3.0), (2.9, 2.0), (-2, 0.0), (True, 0.0), (False, 0.0),
     (math.nan, 0.0), (math.inf, 0.0), ("junk", 0.0), (None, 0.0)],
)
def test_untested_round_units_count_whole_rounds_and_ignore_bad_values(rounds, units) -> None:
    assert untested_round_units(rounds) == units


def test_round_age_takes_precedence_over_seconds() -> None:
    detector = IssueDetector(MockOpenAIClient())
    kw = dict(survived_falsifications=1, merges_attracted=0, open_critic_objections=0)
    # Wall-clock seconds are ignored once the arena supplies its round clock
    assert detector.compute_certainty(untested_rounds=0, seconds_since_last_test=9_999.0, **kw) == 3.0
    assert detector.compute_certainty(untested_rounds=2, **kw) == round(3.0 - 2 * 0.15, 4)
    # Legacy direct callers keep the whole-second path
    assert detector.compute_certainty(seconds_since_last_test=2.5, **kw) == round(3.0 - 2 * 0.15, 4)


def test_tracker_round_clock_tracks_birth_and_tests() -> None:
    tracker = PFPTracker()
    a = tracker.register_branch("branch_1", parent=None, budget_remaining=0.1, fork_under_exploration="a")
    assert tracker.arena_round == 0 and a.last_tested_round == 0
    assert tracker.advance_round() == 1
    b = tracker.register_branch("branch_2", parent=None, budget_remaining=0.1, fork_under_exploration="b")
    assert b.last_tested_round == 1
    tracker.advance_round()
    assert (tracker.untested_rounds(a), tracker.untested_rounds(b)) == (2, 1)
    tracker.update_branch("branch_1", tested=True)
    assert tracker.untested_rounds(a) == 0


def test_certainty_refresh_ignores_wall_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    # Before v0.29 a branch refreshed after a slow call lost 0.15 per elapsed second: call latency,
    # not evidence, set sibling budgets and the background target.
    tracker = PFPTracker()
    tracker.register_branch("branch_1", parent=None, budget_remaining=0.1, fork_under_exploration="a")
    judge = FailJudge(MockOpenAIClient(), tracker, detector=IssueDetector(MockOpenAIClient()))
    before = judge._refresh_branch_certainty("branch_1")
    real_time = time.time
    monkeypatch.setattr(time, "time", lambda: real_time() + 3_600.0)
    assert judge._refresh_branch_certainty("branch_1") == before == 0.0


def test_similarity_is_blind_to_branch_ids() -> None:
    # Merge traces name their parents, and spawn-phase IDs follow thread scheduling. Hashed into
    # the embedding they shifted cosine similarity in the 4th decimal and flipped which pair merged.
    def similarity(ids_a: tuple, ids_b: tuple) -> float:
        tracker = PFPTracker()
        for bid, (pa, pb) in (("branch_1", ids_a), ("branch_2", ids_b)):
            tracker.register_branch(bid, parent=None, budget_remaining=0.1, fork_under_exploration="Unified fix")
            tracker.update_branch(bid, reasoning_step=f"[Merged {pa} + {pb}] open TCP/22 and start sshd")
        detector = AlignmentDetector(MockOpenAIClient(), prefilter_threshold=0.0)
        (pair,) = detector.embedding_prefilter(tracker.active_branches())
        return pair.cosine_similarity

    assert similarity(("branch_3", "branch_4"), ("branch_5", "branch_6")) == 1.0
    assert similarity(("branch_3", "branch_4"), ("branch_17", "branch_123")) == 1.0


def test_selection_round_decays_only_untested_live_branches() -> None:
    harness = MainHarness(client=MockOpenAIClient(), starter_agents=1, enable_selection=True)
    tracker = harness.tracker
    for bid in ("branch_1", "branch_2", "branch_3"):
        tracker.register_branch(bid, parent=None, budget_remaining=0.1, fork_under_exploration=bid,
                                certainty_score=1.0)
    tracker.advance_round()  # spawn phase closes; selection round 1 opens
    tracker.update_branch("branch_2", tested=True)
    tracker.transition("branch_3", BranchState.FAIL, reason="test")
    assert harness._age_untested_branches() == 1
    weight = harness.detector.weights.age_without_testing
    assert tracker.get("branch_1").certainty_score == round(1.0 + weight, 4)
    assert tracker.get("branch_2").certainty_score == 1.0  # tested this round
    assert tracker.get("branch_3").certainty_score == 1.0  # dead branches don't age
    assert tracker.arena_round == 2
    # Tested last round is untested this round: it decays on the next tick
    assert harness._age_untested_branches() == 2


def _arena_outcome(starters: int) -> tuple:
    harness = MainHarness(
        client=MockOpenAIClient(),
        starter_agents=starters,
        hard_cap_usd=2.0,
        depth_dial=3,
        k_probes=3,
        gauntlet_rounds=3,
        enable_selection=True,
    )
    summary = harness.run(SANDBOX_PROBLEM)
    winners = tuple(
        sorted(
            (w["fork_under_exploration"], w["certainty_score"], w["survived_falsifications"])
            for w in summary["winners_box"]
        )
    )
    states = tuple(sorted((r.fork_under_exploration, r.state.value) for r in harness.tracker.all_records()))
    return winners, states, round(harness.budget_ledger.total_spent_usd, 6)


def test_repeated_mock_arenas_reach_one_outcome() -> None:
    # Before v0.29 ~1 in 60 identical 3-starter mock runs reached a different winner: untested age
    # was wall-clock seconds (a run straddling a second boundary aged one twin and not the other)
    # and alignment embeddings hashed scheduling-drawn parent IDs out of merge traces.
    outcomes = {_arena_outcome(3) for _ in range(12)}
    assert len(outcomes) == 1, outcomes
    ((winners, states, _),) = outcomes
    assert winners, "mock arena produced no winner"
