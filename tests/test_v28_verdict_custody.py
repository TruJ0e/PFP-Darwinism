"""v0.28: a falsification verdict kills only on a readable verdict and commits against live state."""

from __future__ import annotations

import threading
from typing import Callable, List, Optional

from pfp import BranchState, BudgetLedger, FailJudge, PFPTracker

from test_hardening import _response


class _HookedClient:
    """Replays scripted replies; `hooks[i]` (if set) runs just before reply i is returned."""

    def __init__(self, replies: List[str], hooks: Optional[dict] = None) -> None:
        self.replies = list(replies)
        self.hooks = dict(hooks or {})
        self.calls = 0
        outer = self

        class _C:
            def create(self, messages, **kwargs):
                idx = outer.calls
                outer.calls += 1
                hook: Optional[Callable[[], None]] = outer.hooks.get(idx)
                if hook is not None:
                    hook()
                return _response(outer.replies.pop(0) if outer.replies else "{}")

        self.chat = type("Chat", (), {})()
        self.chat.completions = _C()


def _judge(replies: List[str], hooks: Optional[dict] = None):
    tracker = PFPTracker()
    tracker.register_branch("b1", None, 0.05, "Host firewall blocks inbound TCP/22")
    ledger = BudgetLedger()
    client = _HookedClient(replies, hooks)
    events: List[str] = []
    judge = FailJudge(client=client, tracker=tracker, budget_ledger=ledger)
    judge.event_sink = lambda e: events.append(e.event_type)
    return judge, tracker, ledger, client, events


def test_verdictless_appeal_does_not_uphold_the_kill() -> None:
    # Unfixed, an unparseable appeal reply fell through to "Kill upheld": an irreversible death
    # on no verdict at all, while a verdictless primary call kills nothing
    judge, tracker, ledger, client, events = _judge(['{"survived": false, "objection": "x"}', "garbled"])
    out = judge.falsify_branch("b1", schedule_trigger="background_sample")
    rec = tracker.get("b1")
    assert rec.state == BranchState.POSSIBLE
    assert out.inconclusive and not out.survived
    assert out.appeal_attempted and out.appeal_judge_id
    assert out.transitioned_to == BranchState.POSSIBLE.value
    assert not rec.appeal_used  # the unheard appeal is handed back
    assert client.calls == 2 and ledger.total_spent_usd > 0.0  # both paid calls are booked
    assert "fail" not in events and "falsification_inconclusive" in events


def test_handed_back_appeal_is_available_for_the_next_kill() -> None:
    judge, tracker, _, _, _ = _judge(
        ['{"survived": false}', "garbled", '{"survived": false}', '{"survived": true}']
    )
    judge.falsify_branch("b1", schedule_trigger="background_sample")
    out = judge.falsify_branch("b1", schedule_trigger="background_sample")
    assert out.appeal_attempted and out.appeal_overturned and out.survived
    assert tracker.get("b1").state == BranchState.POSSIBLE
    assert tracker.get("b1").appeal_used


def test_branch_absorbed_during_primary_call_is_not_appealed_or_journaled_as_failed() -> None:
    # Unfixed: a stale record paid for an appeal on the dead branch, then emitted "fail" and
    # reported transitioned_to=Fail for a branch that was Absorbed
    judge, tracker, _, client, events = _judge(['{"survived": false}', '{"survived": false}'])
    client.hooks[0] = lambda: tracker.transition("b1", BranchState.ABSORBED, reason="merged_into_m1")
    out = judge.falsify_branch("b1", schedule_trigger="background_sample")
    assert client.calls == 1  # no appeal bought for a dead branch
    assert not out.appeal_attempted
    assert not out.survived
    assert out.transitioned_to == BranchState.ABSORBED.value
    assert tracker.get("b1").state == BranchState.ABSORBED
    assert tracker.get("b1").open_critic_objections == 0
    assert "fail" not in events


def test_promotion_during_judge_call_is_revoked_not_raised() -> None:
    # A worker promoted b1 to Pass while the kill verdict was in flight; the kill must revoke
    # that Pass (rewriting the main line), never raise Pass -> Fail as an illegal transition
    judge, tracker, _, client, events = _judge(['{"survived": false}', '{"survived": false}'])
    client.hooks[1] = lambda: tracker.transition("b1", BranchState.PASS, reason="concurrent_gauntlet")
    out = judge.falsify_branch("b1", schedule_trigger="background_sample")
    assert tracker.get("b1").state == BranchState.FAIL
    assert out.transitioned_to == BranchState.FAIL.value
    assert "fail" in events and "revocation" in events


def test_survival_on_a_branch_killed_meanwhile_earns_no_credit_and_blocks_pass() -> None:
    judge, tracker, _, client, events = _judge(['{"survived": true}'] * 3)
    client.hooks[2] = lambda: tracker.transition("b1", BranchState.FAIL, reason="killed_concurrently")
    outcomes = judge.run_pass_readiness_gauntlet("b1", rounds=3)
    rec = tracker.get("b1")
    assert rec.state == BranchState.FAIL
    assert rec.survived_falsifications == 2
    assert len(rec.pass_provenance) == 2
    assert not outcomes[-1].survived
    assert outcomes[-1].transitioned_to == BranchState.FAIL.value
    assert judge.verified_main_line == []
    assert "pass" not in events
    assert events.count("falsification_survived") == 2


def test_contradiction_loser_already_dead_is_not_killed_twice() -> None:
    judge, tracker, _, client, events = _judge(['{"survived": true}', '{"survived": true}'])
    tracker.register_branch("b2", None, 0.05, "sshd is not listening")
    client.hooks[1] = lambda: tracker.transition("b1", BranchState.ABSORBED, reason="merged_into_m1")
    out_a, out_b = judge.resolve_contradiction("b1", "b2", "port 22 state")
    assert tracker.get("b1").state == BranchState.ABSORBED
    assert tracker.get("b2").state == BranchState.POSSIBLE
    assert out_b.survived
    assert "fail" not in events


def test_only_one_concurrent_kill_can_claim_the_appeal() -> None:
    gate = threading.Barrier(2, timeout=5)
    tracker = PFPTracker()
    tracker.register_branch("b1", None, 0.05, "Host firewall blocks inbound TCP/22")
    lock = threading.Lock()
    calls: List[str] = []

    class _C:
        def create(self, messages, **kwargs):
            text = "\n".join(m.get("content", "") for m in messages)
            appeal = "appeal_review=true" in text
            with lock:
                calls.append("appeal" if appeal else "primary")
            if not appeal:
                gate.wait()  # both primaries return together, so both race for the one appeal
            return _response('{"survived": false}')

    client = type("Client", (), {})()
    client.chat = type("Chat", (), {})()
    client.chat.completions = _C()
    judge = FailJudge(client=client, tracker=tracker, budget_ledger=BudgetLedger())
    threads = [
        threading.Thread(target=judge.falsify_branch, args=("b1",), kwargs={"schedule_trigger": "background_sample"})
        for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert calls.count("primary") == 2
    assert calls.count("appeal") <= 1
    assert tracker.get("b1").state == BranchState.FAIL
