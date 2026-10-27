"""v0.53 merge-quality batch: garbage synthesis retries once then aborts with
parents preserved (run 15: 4 merged children died post_merge on
fallback-concatenated marker/label forks)."""

from __future__ import annotations

import pytest

from pfp import BranchState, BudgetLedger, MergeSynthesizer, PFPTracker
from pfp.merge_synthesizer import _synthesis_fell_back

from test_v26_merge_custody import _CountingClient


class _Msg:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Msg(content)


class _Usage:
    cost_usd = 0.001


class _Resp:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]
        self.usage = _Usage()


class _ScriptedClient:
    """Returns scripted replies in order (last reply repeats). Counts calls."""

    def __init__(self, replies) -> None:
        self.replies = list(replies)
        self.calls = 0
        outer = self

        class _Completions:
            @staticmethod
            def create(*args, **kwargs):
                outer.calls += 1
                idx = min(outer.calls - 1, len(outer.replies) - 1)
                return _Resp(outer.replies[idx])

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def _tracker_with_parents(fork_a: str, fork_b: str) -> PFPTracker:
    tracker = PFPTracker()
    tracker.register_branch("p1", parent="root", budget_remaining=0.05, fork_under_exploration=fork_a)
    tracker.register_branch("p2", parent="root", budget_remaining=0.05, fork_under_exploration=fork_b)
    return tracker


def test_synthesis_fell_back_detected() -> None:
    assert _synthesis_fell_back("this is not json at all")
    assert _synthesis_fell_back('{"unified_reasoning": "merged them"}')
    assert _synthesis_fell_back('{"unified_fork": "short", "unified_reasoning": "x"}')
    assert not _synthesis_fell_back(
        '{"unified_fork": "Unified host-ingress remediation: enable laptop sshd and allow inbound TCP/22 on the firewall", '
        '"unified_reasoning": "Merged synthesis of both parents."}'
    )


def test_garbage_synthesis_retries_once_then_aborts_with_parents_alive() -> None:
    tracker = _tracker_with_parents(
        "Use a shared visual board with movable icons to signal when reading time starts and ends.",
        "Pair the schedule with a personalized self-regulation toolkit for transitions.",
    )
    client = _ScriptedClient(["garbage {{{", "also garbage {{{"])
    synth = MergeSynthesizer(client=client, tracker=tracker, budget_ledger=BudgetLedger())
    with pytest.raises(ValueError, match="refused"):
        synth.merge_branches("p1", "p2", "m1")
    assert client.calls == 2
    assert tracker.get("p1").state == BranchState.POSSIBLE
    assert tracker.get("p2").state == BranchState.POSSIBLE
    assert not tracker.has("m1")


def test_good_merge_still_commits() -> None:
    tracker = _tracker_with_parents(
        "Use a shared visual board with movable icons to signal when reading time starts and ends.",
        "Pair the schedule with a personalized self-regulation toolkit for transitions.",
    )
    synth = MergeSynthesizer(client=_CountingClient(), tracker=tracker, budget_ledger=BudgetLedger())
    res = synth.merge_branches("p1", "p2", "m1")
    assert res.child_state == BranchState.POSSIBLE
    assert tracker.get("p1").state == BranchState.ABSORBED
    assert tracker.get("p2").state == BranchState.ABSORBED
    assert len(tracker.get("m1").fork_under_exploration) >= 40
