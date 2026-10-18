"""v0.26: a refused merge spends nothing, absorbs nothing, and never mints budget."""

from __future__ import annotations

import pytest

from pfp import AlignmentVerdict, BranchState, BudgetLedger, MergeSynthesizer, MockOpenAIClient, PFPTracker
from pfp.alignment_detector import AlignmentJudgeResult


class _CountingClient:
    def __init__(self) -> None:
        self.inner = MockOpenAIClient(force_cost_scale=1.0)
        self.calls = 0
        self.on_call = None
        outer = self

        class _Completions:
            @staticmethod
            def create(*args, **kwargs):
                outer.calls += 1
                if outer.on_call is not None:
                    outer.on_call()
                return outer.inner.chat.completions.create(*args, **kwargs)

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def _setup():
    client = _CountingClient()
    tracker = PFPTracker()
    ledger = BudgetLedger()
    tracker.register_branch("p1", parent="root", budget_remaining=0.025, fork_under_exploration="firewall TCP/22")
    tracker.register_branch("p2", parent="root", budget_remaining=0.030, fork_under_exploration="sshd listening")
    synth = MergeSynthesizer(client=client, tracker=tracker, budget_ledger=ledger)
    return client, tracker, ledger, synth


def _assert_untouched(client, tracker, ledger) -> None:
    assert client.calls == 0
    assert ledger.total_spent_usd == 0.0
    assert tracker.get("p1").state == BranchState.POSSIBLE
    assert tracker.get("p1").budget_remaining == 0.025
    assert tracker.get("p2").state == BranchState.POSSIBLE
    assert tracker.get("p2").budget_remaining == 0.030


def test_self_merge_is_refused_instead_of_doubling_budget() -> None:
    # Absorbed -> Absorbed is a no-op transition, so p1+p1 used to mint a 2x budget child
    client, tracker, ledger, synth = _setup()
    with pytest.raises(ValueError, match="itself"):
        synth.merge_branches("p1", "p1", "m1")
    _assert_untouched(client, tracker, ledger)
    assert not tracker.has("m1")


@pytest.mark.parametrize("child_id", ["p1", "p2", "taken"])
def test_colliding_child_id_is_refused_before_absorbing_parents(child_id: str) -> None:
    # Child registration used to run last: a collision raised after both parents were zeroed
    client, tracker, ledger, synth = _setup()
    tracker.register_branch("taken", parent=None, budget_remaining=0.0, fork_under_exploration="x")
    with pytest.raises(ValueError, match="already registered"):
        synth.merge_branches("p1", "p2", child_id)
    _assert_untouched(client, tracker, ledger)


def test_alignment_for_another_pair_is_refused() -> None:
    client, tracker, ledger, synth = _setup()
    tracker.register_branch("p3", parent="root", budget_remaining=0.01, fork_under_exploration="fail2ban")
    stale = AlignmentJudgeResult("p1", "p3", 0.9, AlignmentVerdict.OVERLAP, overlapping_claims=["TCP/22"])
    with pytest.raises(ValueError, match="does not cover"):
        synth.merge_branches("p1", "p2", "m1", alignment=stale)
    _assert_untouched(client, tracker, ledger)


def test_alignment_in_either_orientation_is_accepted() -> None:
    _, tracker, _, synth = _setup()
    al = AlignmentJudgeResult("p2", "p1", 0.9, AlignmentVerdict.OVERLAP, overlapping_claims=["TCP/22"])
    res = synth.merge_branches("p1", "p2", "m1", alignment=al)
    assert res.transferred_budget_usd <= 0.055
    assert tracker.get("m1").state == BranchState.POSSIBLE


def test_parent_dying_during_synthesis_books_spend_and_absorbs_nothing() -> None:
    client, tracker, ledger, synth = _setup()
    client.on_call = lambda: tracker.transition("p2", BranchState.FAIL, reason="killed_concurrently")
    with pytest.raises(ValueError, match="non-alive"):
        synth.merge_branches("p1", "p2", "m1")
    assert client.calls == 1
    # The Super call was really paid for: it is booked, as overdraft, never hidden
    assert ledger.total_spent_usd > 0.0
    assert ledger.overdraft_usd == ledger.total_spent_usd
    assert tracker.get("p1").state == BranchState.POSSIBLE
    assert tracker.get("p1").budget_remaining == 0.025
    assert tracker.get("p2").budget_remaining == 0.030
    assert not tracker.has("m1")
