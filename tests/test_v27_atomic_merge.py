"""v0.27: a merge commits whole (both parents absorbed + child registered) or not at all."""

from __future__ import annotations

import threading

import pytest

from pfp import BranchState, BudgetLedger, MergeSynthesizer, PFPTracker
from pfp.tracker import InvalidStateTransitionError

from test_v26_merge_custody import _CountingClient


class _KillerTracker(PFPTracker):
    """Starts a worker that tries to kill p2 the moment the merge absorbs p1."""

    def __init__(self) -> None:
        super().__init__()
        self.killer_errors: list = []
        self.killer: threading.Thread | None = None

    def _kill_p2(self) -> None:
        try:
            self.transition("p2", BranchState.FAIL, reason="killed_concurrently")
        except InvalidStateTransitionError as exc:
            self.killer_errors.append(exc)

    def transition(self, branch_id, new_state, reason=""):
        rec = super().transition(branch_id, new_state, reason)
        if branch_id == "p1" and new_state == BranchState.ABSORBED and self.killer is None:
            self.killer = threading.Thread(target=self._kill_p2)
            self.killer.start()
            # Give the worker every chance to land between the two absorptions
            self.killer.join(timeout=0.3)
        return rec


def _setup(tracker: PFPTracker):
    client = _CountingClient()
    ledger = BudgetLedger()
    tracker.register_branch("p1", parent="root", budget_remaining=0.025, fork_under_exploration="firewall TCP/22")
    tracker.register_branch("p2", parent="root", budget_remaining=0.030, fork_under_exploration="sshd listening")
    return client, ledger, MergeSynthesizer(client=client, tracker=tracker, budget_ledger=ledger)


def test_parent_kill_cannot_land_between_the_two_absorptions() -> None:
    # Unlocked, p2 died after p1 was absorbed and zeroed: Fail -> Absorbed then raised,
    # leaving p1's budget destroyed and no child to carry it
    tracker = _KillerTracker()
    _, _, synth = _setup(tracker)
    res = synth.merge_branches("p1", "p2", "m1")
    tracker.killer.join(timeout=5)
    assert not tracker.killer.is_alive()
    assert tracker.get("p1").state == BranchState.ABSORBED
    assert tracker.get("p2").state == BranchState.ABSORBED
    assert tracker.get("m1").state == BranchState.POSSIBLE
    assert tracker.get("m1").budget_remaining == res.transferred_budget_usd
    # The late kill hit an already-absorbed branch and was refused, not the merge
    assert len(tracker.killer_errors) == 1


def test_child_id_taken_during_synthesis_absorbs_nothing() -> None:
    tracker = PFPTracker()
    client, ledger, synth = _setup(tracker)
    client.on_call = lambda: tracker.register_branch(
        "m1", parent=None, budget_remaining=0.0, fork_under_exploration="raced"
    )
    with pytest.raises(ValueError, match="already registered"):
        synth.merge_branches("p1", "p2", "m1")
    assert client.calls == 1
    assert ledger.total_spent_usd > 0.0
    assert ledger.overdraft_usd == ledger.total_spent_usd
    for pid, budget in (("p1", 0.025), ("p2", 0.030)):
        assert tracker.get(pid).state == BranchState.POSSIBLE
        assert tracker.get(pid).budget_remaining == budget
    assert tracker.get("m1").fork_under_exploration == "raced"


def test_successful_merge_conserves_money() -> None:
    tracker = PFPTracker()
    _, ledger, synth = _setup(tracker)
    res = synth.merge_branches("p1", "p2", "m1")
    pool = 0.025 + 0.030
    deducted = min(pool, res.synthesis_cost_usd)
    assert res.transferred_budget_usd == pytest.approx(pool - deducted, abs=1e-6)
    assert ledger.total_spent_usd == pytest.approx(res.synthesis_cost_usd, abs=1e-6)
    assert ledger.overdraft_usd == pytest.approx(0.0, abs=1e-6)


def test_atomic_blocks_other_workers_until_released() -> None:
    tracker = PFPTracker()
    tracker.register_branch("p1", parent=None, budget_remaining=0.01, fork_under_exploration="x")
    done = threading.Event()

    def worker() -> None:
        tracker.transition("p1", BranchState.FAIL)
        done.set()

    with tracker.atomic():
        t = threading.Thread(target=worker)
        t.start()
        assert not done.wait(timeout=0.2)
        assert tracker.get("p1").state == BranchState.POSSIBLE
    t.join(timeout=5)
    assert done.is_set()
    assert tracker.get("p1").state == BranchState.FAIL
