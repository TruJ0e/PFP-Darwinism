"""v0.43: blind variation is capped at a share of the population (default 25%)."""

from __future__ import annotations

import collections

import pytest

from pfp import MainHarness, MockOpenAIClient

PROBLEM = "SSH works laptop->desktop but not desktop->laptop. Diagnose root causes and verify."


class _CountingMock(MockOpenAIClient):
    def __init__(self) -> None:
        super().__init__()
        self.ns_calls = 0
        inner = self.chat.completions
        outer = self

        class _C:
            def create(self, messages, **kwargs):
                if "ns_fanout" in messages[-1].get("content", ""):
                    outer.ns_calls += 1
                return inner.create(messages, **kwargs)

        self.chat.completions = _C()


def _live_cadence_run(cap: float):
    # The mock answers instantly, so the wall-clock NS cadence almost never elapses and NS stays
    # rare. Live calls take seconds, so the cadence elapses on nearly every branch (run 5: 97 NS
    # of 174). Shrinking the expected agent time reproduces the live firing rate on the mock.
    client = _CountingMock()
    harness = MainHarness(
        client=client,
        starter_agents=9,
        hard_cap_usd=5.0,
        depth_decay=0.85,
        finish_reserve_ratio=0.10,
        root_budget_override_usd=2.0,
        ns_share_cap=cap,
    )
    harness.budget_ledger.expected_agent_seconds = 1e-9
    summary = harness.run(PROBLEM)
    origins = collections.Counter(r["origin"] for r in summary["scoreboard"])
    spawned = sum(n for origin, n in origins.items() if origin != "merge")
    return summary, origins, spawned, client


def test_uncapped_live_cadence_overgrows_ns() -> None:
    summary, origins, spawned, _ = _live_cadence_run(1.0)
    assert origins["ns"] / spawned > 0.25
    assert summary["ns"]["capped_ticks"] == 0


def test_cap_holds_ns_to_its_share_and_frees_room_for_merges() -> None:
    uncapped, _, _, _ = _live_cadence_run(1.0)
    summary, origins, spawned, _ = _live_cadence_run(0.25)
    assert 0 < origins["ns"] <= 0.25 * spawned
    assert summary["ns"]["spawned"] == origins["ns"]
    assert summary["ns"]["capped_ticks"] > 0
    assert len(summary["selection"]["merges_performed"]) > len(uncapped["selection"]["merges_performed"])


def test_refused_ticks_buy_no_probe() -> None:
    summary, origins, _, client = _live_cadence_run(0.0)
    assert origins["ns"] == 0 and summary["ns"]["capped_ticks"] > 0
    assert client.ns_calls == 0


@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan"), True])
def test_cap_is_validated(bad) -> None:
    with pytest.raises(ValueError):
        MainHarness(client=MockOpenAIClient(), ns_share_cap=bad)
