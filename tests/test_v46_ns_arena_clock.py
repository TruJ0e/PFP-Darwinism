"""
v0.46 — NS firing runs on the arena clock, not the wall clock.

Before v0.46 the NS cadence was wall-clock seconds (expected_agent_seconds / budget_calls,
checked against time.time() once per step). Live calls take seconds, so NS fired on nearly
every eligible branch (run 5: ~97 NS-born of 174); the mock answers instantly, so it almost
never fired (3 ticks of 37 branches in the CI config, all of them forced starters). The mock
under-represented NS badly, and which branches won the v0.43 share cap's slots depended on
thread timing.

Now a tick is DUE on the arena's own clock — the branch's subtree must complete a
cadence-worth of arena steps — and the share cap's slots are allocated deterministically:
each branch draws a slot priority from the run seed and its content identity (lineage
digest), stratified by its rank within its spawn cohort. Same seed + same settings -> the
same branches tick in the same order, regardless of thread scheduling or machine speed.
The population count check remains as the cap's hard backstop, and a wall-clock mode is
kept behind --ns-clock for comparison.
"""

import pytest

from pfp import MainHarness, MockOpenAIClient


RICH_PROBLEM = "SSH works laptop->desktop but not desktop->laptop. Diagnose root causes and verify."
SMALL_PROBLEM = "Sandbox verification: SSH works laptop->desktop but not desktop->laptop."


def _rich_harness(**overrides):
    """The run-5-shaped config the v0.43 tests use, with default (unhacked) NS timing."""
    kwargs = dict(
        client=MockOpenAIClient(),
        starter_agents=9,
        hard_cap_usd=5.0,
        depth_decay=0.85,
        finish_reserve_ratio=0.10,
        root_budget_override_usd=2.0,
    )
    kwargs.update(overrides)
    return MainHarness(**kwargs)


def _small_harness(**overrides):
    """The CI-shaped config from the original finding (3 starters, where mock NS was ~8%)."""
    kwargs = dict(
        client=MockOpenAIClient(),
        starter_agents=3,
        hard_cap_usd=2.0,
        depth_dial=3,
        k_probes=3,
        gauntlet_rounds=3,
        enable_selection=True,
    )
    kwargs.update(overrides)
    return MainHarness(**kwargs)


def _run(harness, problem):
    summary = harness.run(problem)
    records = list(harness.tracker.all_records())
    origins = {}
    for record in records:
        origins[record.origin] = origins.get(record.origin, 0) + 1
    spawned = sum(n for origin, n in origins.items() if origin != "merge")
    ticks = [e for e in harness.events if e.event_type == "ns_tick"]
    tick_keys = sorted(harness.tracker.get(e.branch_id).lineage_digest for e in ticks)
    return summary, origins, spawned, ticks, tick_keys


def test_identical_mock_runs_allocate_identical_ns_ticks():
    """Same seed + settings -> the same branches (by content) win NS slots every run."""
    outcomes = []
    for _ in range(3):
        summary, origins, spawned, ticks, tick_keys = _run(_small_harness(), SMALL_PROBLEM)
        outcomes.append((
            tick_keys,
            summary["budget"]["total_spent_usd"],
            summary["total_branches_spawned"],
            summary["ns"]["spawned"],
            summary["ns"]["capped_ticks"],
        ))
    assert outcomes[0] == outcomes[1] == outcomes[2]
    assert outcomes[0][3] > 0  # the runs really did allocate NS ticks


def test_seed_changes_which_branches_get_ns_slots():
    """The allocation is seed-driven: a different seed hands the slots to other branches."""
    _, _, _, _, keys_seed0 = _run(_small_harness(ns_seed=0), SMALL_PROBLEM)
    _, _, _, _, keys_seed1 = _run(_small_harness(ns_seed=1), SMALL_PROBLEM)
    assert keys_seed0 != keys_seed1


def test_mock_ns_share_in_live_regime_without_wall_clock_hack():
    """
    With the cap lifted, the mock's NS share lands in the live regime under DEFAULT
    timing parameters. The v0.43 tests could only make mock NS fire by shrinking
    expected_agent_seconds to 1e-9; the arena clock needs no such hack, because due-ness
    no longer depends on how fast calls return.
    """
    summary, origins, spawned, ticks, _ = _run(
        _rich_harness(ns_share_cap=1.0), RICH_PROBLEM
    )
    assert origins.get("ns", 0) / spawned > 0.25


def test_share_cap_still_enforced_under_arena_clock():
    """The v0.43 cap still binds: share within the cap, over-cap ticks refused + counted."""
    summary, origins, spawned, ticks, _ = _run(
        _rich_harness(ns_share_cap=0.25), RICH_PROBLEM
    )
    share = origins.get("ns", 0) / spawned
    assert 0 < share <= 0.25
    assert summary["ns"]["capped_ticks"] > 0
    # NS still feeds selection: capped runs merge at least as eagerly as uncapped ones
    merges = summary["selection"]["merges_performed"]
    assert len(merges) >= 8


def test_zero_cap_still_means_no_ns_calls_under_arena_clock():
    summary, origins, spawned, ticks, _ = _run(
        _rich_harness(ns_share_cap=0.0), RICH_PROBLEM
    )
    assert origins.get("ns", 0) == 0
    assert summary["ns"]["spawned"] == 0
    assert summary["ns"]["capped_ticks"] > 0


def test_finding_config_no_longer_at_eight_percent():
    """
    The original finding's config (3 starters): NS-born share of spawned branches was
    ~8% under the wall clock (3 forced ticks, nothing else ever due). On the arena
    clock the cap binds and the share lands in the live regime under the 0.25 cap.
    """
    summary, origins, spawned, ticks, _ = _run(
        _small_harness(ns_share_cap=0.25), SMALL_PROBLEM
    )
    share = origins.get("ns", 0) / spawned
    assert share >= 0.15
    assert summary["ns"]["capped_ticks"] > 0
    # Non-forced ticks exist now — NS is exercised beyond the starter force.
    assert any(e.details.get("allocation") == "arena" for e in ticks)


def test_ns_journal_records_arena_clock_basis():
    """Every tick event carries why the branch got its mutation (the proof trail)."""
    _, _, _, ticks, _ = _run(_small_harness(ns_share_cap=0.25), SMALL_PROBLEM)
    assert ticks
    for event in ticks:
        details = event.details
        assert details["clock"] == "arena"
        assert details["allocation"] in ("arena", "forced")
        assert details["arena_steps"] >= 1
        assert details["period_steps"] >= 1
        assert 0.0 <= details["slot_priority"] < 1.0


def test_wall_clock_mode_kept_behind_dial():
    """--ns-clock wall restores the pre-v0.46 behavior shape (forced starters only here)."""
    summary, origins, spawned, ticks, _ = _run(
        _small_harness(ns_clock="wall"), SMALL_PROBLEM
    )
    assert summary["ns"]["clock"] == "wall"
    assert all(e.details.get("allocation") in ("forced", "wall") for e in ticks)


def test_invalid_ns_clock_rejected():
    with pytest.raises(ValueError):
        MainHarness(client=MockOpenAIClient(), ns_clock="sideways")
