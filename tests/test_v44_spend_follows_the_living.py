"""v0.44: spend follows the living — probe token caps follow the same certainty that splits dollars.

Certainty-weighted allocation used to stop at money: every Plaus probe and NS continuation asked
for 768 completion tokens, whether the branch was its parent's strongest fork or a long shot.
Siblings are now tiered at spawn: the top quartile (at least one) keeps full completions, those
scoring below the certainty floor get half, the rest three quarters, never below 384 tokens.
Conclusions (1024) are untouched: run 1 showed what truncated conclusions do to selection.
"""

from __future__ import annotations

import math
import threading
import types
from typing import List

import pytest

import run_arena
from pfp import MainHarness, MockOpenAIClient
from pfp.branch_spawner import DEFAULT_PROBE_TOKEN_TIERS, sibling_token_scales
from pfp.issue_detector import MIN_PROBE_TOKENS, PROBE_MAX_TOKENS


def test_top_quartile_full_floor_cut_rest_middle():
    top, middle, cut = DEFAULT_PROBE_TOKEN_TIERS
    assert sibling_token_scales([2.0]) == [top]
    assert sibling_token_scales([2.0, 1.0, -0.5]) == [top, middle, cut]
    eight = sibling_token_scales([3, 2, 1.5, 1, 0.5, 0.1, -0.2, -1])
    assert eight == [top, top, middle, middle, middle, middle, cut, cut]
    # A non-finite score never earns more than the cut (unless it leads the siblings)
    assert sibling_token_scales([1.0, float("nan")]) == [top, cut]
    assert sibling_token_scales([1.0, 0.5], certainty_floor=0.75) == [top, cut]


class RecordingClient:
    """Mock backend that records max_tokens per probe / NS call."""

    def __init__(self):
        self.inner = MockOpenAIClient()
        self.model, self.base_url, self.is_mock = self.inner.model, self.inner.base_url, True
        self.cost_scale, self.max_billable_attempts = self.inner.cost_scale, 1
        self.probe_tokens: List[int] = []
        self._lock = threading.Lock()
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, *args, **kwargs):
        text = kwargs["messages"][-1]["content"]
        if text.startswith(("probe_continuation", "ns_fanout")):
            with self._lock:
                self.probe_tokens.append(kwargs["max_tokens"])
        return self.inner.chat.completions.create(*args, **kwargs)


def test_mock_arena_tiers_probe_budgets():
    client = RecordingClient()
    harness = MainHarness(client=client, starter_agents=3, hard_cap_usd=2.0)
    summary = harness.run(run_arena.DEFAULT_PROBLEM)
    assert summary["halted_reason"] is None and summary["winners_box"]
    tokens = set(client.probe_tokens)
    assert PROBE_MAX_TOKENS in tokens
    assert any(t < PROBE_MAX_TOKENS for t in tokens)
    assert min(tokens) >= MIN_PROBE_TOKENS
    spawns = [e for e in harness.events if e.event_type == "spawn"]
    assert all(e.details["probe_max_tokens"] == PROBE_MAX_TOKENS for e in spawns if e.snapshot["origin"] == "starter")
    # Every branch's tier is on the journal
    assert all("probe_max_tokens" in e.details for e in spawns)


def test_probe_floor_is_respected():
    client = RecordingClient()
    harness = MainHarness(client=client, starter_agents=3, hard_cap_usd=2.0, probe_token_tiers=(1.0, 0.2, 0.1))
    harness.run(run_arena.DEFAULT_PROBLEM)
    assert min(client.probe_tokens) == MIN_PROBE_TOKENS


@pytest.mark.parametrize("tiers", [(1.0, 0.5), (0.5, 0.75, 0.25), (1.0, 0.5, 0.0), (1.2, 1.0, 0.5), (1.0, math.nan, 0.5)])
def test_tiers_are_validated(tiers):
    with pytest.raises(ValueError):
        MainHarness(client=MockOpenAIClient(), probe_token_tiers=tiers)
