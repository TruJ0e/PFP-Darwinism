"""v0.16: selection decisions don't depend on sub-second wall-clock jitter."""

from __future__ import annotations

import math
import random
import time

import pytest

from pfp import MainHarness, MockOpenAIClient
from pfp.issue_detector import AGE_QUANTUM_S, IssueDetector, untested_age_units

SANDBOX_PROBLEM = "Sandbox verification: SSH works laptop->desktop but not desktop->laptop."


@pytest.mark.parametrize(
    "seconds, units",
    [(0.0, 0.0), (0.0004, 0.0), (0.9999, 0.0), (1.0, 1.0), (5.7, 5.0),
     (-3.0, 0.0), (math.nan, 0.0), (math.inf, 0.0), (-math.inf, 0.0), ("junk", 0.0), (None, 0.0)],
)
def test_untested_age_counts_whole_quanta_and_ignores_bad_clocks(seconds, units) -> None:
    assert AGE_QUANTUM_S == 1.0
    assert untested_age_units(seconds) == units


def test_siblings_born_microseconds_apart_rank_equal() -> None:
    detector = IssueDetector(MockOpenAIClient())
    kw = dict(survived_falsifications=0, merges_attracted=0, open_critic_objections=1)
    young = detector.compute_certainty(seconds_since_last_test=0.0002, **kw)
    older = detector.compute_certainty(seconds_since_last_test=0.0009, **kw)
    assert young == older == -1.25
    # Real age still decays certainty, a whole second at a time
    assert detector.compute_certainty(seconds_since_last_test=2.5, **kw) == round(-1.25 - 2 * 0.15, 4)


def _sandbox_outcome() -> tuple:
    harness = MainHarness(
        client=MockOpenAIClient(),
        starter_agents=2,
        hard_cap_usd=0.5,
        depth_dial=2,
        k_probes=2,
        gauntlet_rounds=3,
        enable_selection=True,
    )
    summary = harness.run(SANDBOX_PROBLEM)
    return tuple(
        sorted(
            (w["fork_under_exploration"], w["certainty_score"], w["survived_falsifications"])
            for w in summary["winners_box"]
        )
    )


def test_mock_arena_outcome_is_stable_under_clock_jitter(monkeypatch: pytest.MonkeyPatch) -> None:
    # Before v0.16 the background falsifier's target was decided by microsecond age
    # differences between equal siblings: the paired sandbox reached 22.5 or 31.5 depending
    # on thread scheduling, and the self-improvement test flaked on adoption. Exaggerate
    # that jitter (up to 5 ms per clock read, still far below the 1 s quantum) and demand one outcome.
    real_time = time.time
    rng = random.Random(16)
    monkeypatch.setattr(time, "time", lambda: real_time() + rng.random() * 0.005)

    outcomes = {_sandbox_outcome() for _ in range(8)}
    assert len(outcomes) == 1, outcomes
    (outcome,) = outcomes
    assert outcome, "sandbox produced no winner"
