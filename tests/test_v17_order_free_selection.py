"""v0.17: selection decisions don't depend on the order branches were registered in."""

from __future__ import annotations

import random

import pytest

from pfp import MainHarness, MockOpenAIClient
from pfp.alignment_detector import AlignmentDetector
from pfp.fail_judge import FailJudge
from pfp.tracker import BranchRecord, BranchState, PFPTracker, content_order_key, id_free

SANDBOX_PROBLEM = "Sandbox verification: SSH works laptop->desktop but not desktop->laptop."

_FORKS = (
    "Inspect inbound firewall rules for TCP port 22 on the laptop host",
    "Inspect inbound firewall rules for TCP port 22 on the laptop host",
    "Check whether sshd is enabled and listening on port 22 on the laptop",
    "Audit public key permissions and sshd_config auth modes on the laptop",
)


def _tracker(order: list) -> PFPTracker:
    tracker = PFPTracker()
    for idx in order:
        tracker.register_branch(f"branch_{idx}", parent=None, budget_remaining=0.1, fork_under_exploration=_FORKS[idx])
    return tracker


def test_content_order_key_blanks_branch_ids() -> None:
    assert id_free("merge of branch_3 and branch_12") == "merge of branch_# and branch_#"
    a = BranchRecord(id="branch_7", parent=None, budget_remaining=0.1, fork_under_exploration="see branch_2")
    b = BranchRecord(id="branch_1", parent=None, budget_remaining=0.1, fork_under_exploration="see branch_9")
    assert content_order_key(a)[0] == content_order_key(b)[0]
    assert content_order_key(b) < content_order_key(a)  # full content tie -> ID decides


def test_lineage_digest_separates_ancestry_not_ids() -> None:
    tracker = PFPTracker()
    for sid in ("starter_agent_1", "starter_agent_2"):
        tracker.register_branch(sid, parent=None, budget_remaining=0.1, fork_under_exploration="root")
    # Same text, same ancestry, different (scheduling-drawn) IDs -> same digest
    a = tracker.register_branch("branch_4", parent="starter_agent_1", budget_remaining=0.1, fork_under_exploration="f")
    b = tracker.register_branch("branch_9", parent="starter_agent_1", budget_remaining=0.1, fork_under_exploration="f")
    assert a.lineage_digest == b.lineage_digest
    # Same text, different ancestry -> different digest, so content ties don't fall to the ID
    c = tracker.register_branch("branch_5", parent="starter_agent_2", budget_remaining=0.1, fork_under_exploration="f")
    assert c.lineage_digest != a.lineage_digest
    # Merge children hash both parents, in either orientation
    m1 = tracker.register_branch("branch_10", parent="branch_4", budget_remaining=0.1, fork_under_exploration="u",
                                 origin="merge", merge_parents=("branch_4", "branch_5"))
    m2 = tracker.register_branch("branch_11", parent="branch_5", budget_remaining=0.1, fork_under_exploration="u",
                                 origin="merge", merge_parents=("branch_5", "branch_4"))
    assert m1.merge_parents == ["branch_4", "branch_5"]
    assert m1.lineage_digest == m2.lineage_digest != a.lineage_digest


def test_prefilter_pairs_do_not_follow_registration_order() -> None:
    def pairs(order: list) -> list:
        tracker = _tracker(order)
        detector = AlignmentDetector(MockOpenAIClient(), prefilter_threshold=0.0)
        out = detector.embedding_prefilter(tracker.active_branches())
        by_id = {r.id: r.fork_under_exploration for r in tracker.all_records()}
        return [(by_id[p.branch_a_id], by_id[p.branch_b_id], p.cosine_similarity) for p in out]

    baseline = pairs([0, 1, 2, 3])
    for order in ([3, 2, 1, 0], [2, 0, 3, 1], [1, 3, 0, 2]):
        assert pairs(order) == baseline


@pytest.mark.parametrize("swap", [False, True])
def test_contradiction_full_tie_loser_ignores_argument_order(swap: bool) -> None:
    tracker = PFPTracker()
    tracker.register_branch("branch_1", parent=None, budget_remaining=0.1, fork_under_exploration="Zeta: port 22 is open")
    tracker.register_branch("branch_2", parent=None, budget_remaining=0.1, fork_under_exploration="Alpha: port 22 is closed")
    a, b = (tracker.get("branch_2"), tracker.get("branch_1")) if swap else (tracker.get("branch_1"), tracker.get("branch_2"))
    loser = min((a, b), key=FailJudge._contradiction_strength)
    # Equal certainty, falsifications and objections: content, not position, picks the loser
    assert loser.fork_under_exploration.startswith("Alpha")


def test_pre_exit_order_ignores_caller_order(monkeypatch: pytest.MonkeyPatch) -> None:
    def tested_order(order: list) -> list:
        tracker = _tracker(order)
        judge = FailJudge(MockOpenAIClient(), tracker)
        seen: list = []
        monkeypatch.setattr(judge, "run_pass_readiness_gauntlet", lambda bid: seen.append(bid) or [])
        judge.verify_pre_exit([r.id for r in tracker.all_records()])
        return [id_free(tracker.get(bid).fork_under_exploration) for bid in seen]

    baseline = tested_order([0, 1, 2, 3])
    assert baseline == sorted(baseline)
    assert tested_order([3, 1, 2, 0]) == baseline


def _arena_outcome(starters: int, shuffle_seed) -> tuple:
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


@pytest.mark.parametrize("starters", [3, 9])
def test_arena_outcome_is_invariant_to_record_order(monkeypatch: pytest.MonkeyPatch, starters: int) -> None:
    # Parallel subtrees register branches in thread-scheduling order. Before v0.17 that order
    # decided similarity-tie pairs, merge orientation, contradiction losers, and who made the
    # pre-exit cut: a shuffled record order changed the winners and the bill (7 orderings of
    # the 9-starter mock arena gave 7 different outcomes). Shuffle it and demand one outcome.
    real_all_records = PFPTracker.all_records
    outcomes = set()
    for seed in (None, 0, 1, 2):
        rng = random.Random(seed)

        def shuffled(self, _rng=rng, _seed=seed):
            recs = real_all_records(self)
            if _seed is not None:
                _rng.shuffle(recs)
            return recs

        monkeypatch.setattr(PFPTracker, "all_records", shuffled)
        outcomes.add(_arena_outcome(starters, seed))
    assert len(outcomes) == 1, outcomes
    ((winners, states, spend),) = outcomes
    assert winners, "mock arena produced no winner"
    assert any(state == BranchState.ABSORBED.value for _, state in states)
