"""v0.43: NS-born branches are capped at a share of the arena's spawns (default 0.25).

Run 5 grew 97 NS children out of 174 branches (56%): on live latency the wall-clock NS cadence
elapses during every branch's own probe round, so every exploring branch mutated, and the
population came out too mutually alien to recombine (0 merges). The mock never showed it (its
calls are instant, so only the starters' forced ticks fire); these tests force the live cadence.
"""

from __future__ import annotations

import collections

import pytest

import run_arena
from pfp import MainHarness, MockOpenAIClient, PFPTracker
from pfp.branch_spawner import NSShareGate


def _live_cadence_arena(cap: float, starters: int = 3):
    harness = MainHarness(
        client=MockOpenAIClient(),
        starter_agents=starters,
        hard_cap_usd=5.0,
        ns_share_cap=cap,
        depth_decay=0.85,
        finish_reserve_ratio=0.10,
        root_budget_override_usd=2.0,
    )
    # Live latency: the cadence has always elapsed by the time a branch reaches its NS check
    harness.budget_ledger.expected_agent_seconds = 1e-9
    summary = harness.run(run_arena.DEFAULT_PROBLEM)
    origins = collections.Counter(r.origin for r in harness.tracker.all_records())
    spawned = sum(n for origin, n in origins.items() if origin != "merge")
    return summary, origins, spawned


@pytest.mark.parametrize("starters", [3, 9])
def test_ns_share_is_capped_under_live_cadence(starters):
    summary, origins, spawned = _live_cadence_arena(0.25, starters)
    assert summary["halted_reason"] is None
    # (How many ticks were refused depends on how spawns and NS checks interleave; the share doesn't)
    assert origins["ns"] / spawned <= 0.25 + 1e-9
    assert summary["ns_share"]["ns_ticks_granted"] == origins["ns"]


def test_uncapped_live_cadence_overgrows_ns():
    _, capped, capped_spawned = _live_cadence_arena(0.25)
    _, free, free_spawned = _live_cadence_arena(1.0)
    assert free["ns"] / free_spawned > 0.25
    assert free["ns"] > capped["ns"]


def test_gate_counts_forced_ticks_but_never_refuses_them():
    t = PFPTracker()
    for i in range(3):
        t.register_branch(f"s{i}", None, 0.1, f"starter {i}", origin="starter")
    gate = NSShareGate(t, cap=0.25)
    # 4 spawns incl. the prospective child -> 1 NS fits; forced ticks always run
    assert gate.try_reserve(forced=True)
    assert not gate.try_reserve()
    assert gate.try_reserve(forced=True)
    assert gate.reserved == 2 and gate.refused == 1
    for i in range(10):
        t.register_branch(f"p{i}", "s0", 0.1, f"plaus {i}", origin="plaus")
    # 13 spawns + 1 -> 3.5 NS allowed: one more unforced slot
    assert gate.try_reserve()
    assert not gate.try_reserve()
    gate.release()
    assert gate.try_reserve()


def test_merges_are_not_spawns():
    t = PFPTracker()
    t.register_branch("s", None, 0.1, "s", origin="starter")
    for i in range(8):
        t.register_branch(f"m{i}", "s", 0.1, f"m{i}", origin="merge")
    # 1 spawn + 1 -> 0.8 NS allowed; counting the 8 merges would have allowed 4
    assert not NSShareGate(t, cap=0.4).try_reserve()


@pytest.mark.parametrize("cap", [0.0, -0.1, 1.5, float("nan")])
def test_cap_is_validated(cap):
    with pytest.raises(ValueError):
        MainHarness(client=MockOpenAIClient(), ns_share_cap=cap)


def test_cli_ns_share_cap(monkeypatch):
    captured = {}
    real = run_arena.MainHarness

    def spy(**kwargs):
        captured.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(run_arena, "MainHarness", spy)
    assert run_arena.main(["--mock", "--quiet", "--no-improve", "--ns-share-cap", "0.4"]) == 0
    assert captured["ns_share_cap"] == 0.4
