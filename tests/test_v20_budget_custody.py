"""v0.20: non-finite or invalid values cannot corrupt the budget ledger."""

from __future__ import annotations

import math

import pytest

from pfp import BudgetLedger, PFPTracker


@pytest.mark.parametrize(
    "field,value",
    [
        ("hard_cap_usd", float("nan")),
        ("depth_decay", float("inf")),
        ("depth_decay", 0),
        ("depth_decay", 1.01),
        ("finish_reserve_ratio", True),
        ("min_branch_floor_usd", 0),
        ("conclusion_reserve_usd", -0.01),
        ("expected_agent_seconds", 0),
        ("estimated_cost_per_call_usd", float("nan")),
        ("estimated_cost_per_call_usd", 0),
        ("high_certainty_threshold", float("-inf")),
        ("verification_share", float("nan")),
    ],
)
def test_constructor_rejects_values_that_break_budget_invariants(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        BudgetLedger(**{field: value})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.01, True, "broken"])
def test_spend_paths_reject_poison_without_mutating_totals(bad: object) -> None:
    ledger = BudgetLedger()

    for operation in (
        lambda: ledger.record_spend(bad),
        lambda: ledger.cover_from_finish_reserve(bad),
        lambda: ledger.return_to_finish_reserve(bad),
        lambda: ledger.observe_call_cost("probe", bad),
    ):
        with pytest.raises(ValueError):
            operation()
        assert ledger.total_spent_usd == 0.0
        assert ledger.overdraft_usd == 0.0
        assert math.isfinite(ledger.finish_reserve_usd)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, True])
def test_branch_budget_math_rejects_poison(bad: object) -> None:
    ledger = BudgetLedger()

    for operation in (
        lambda: ledger.compute_ns_cadence(bad),
        lambda: ledger.max_fundable_siblings(bad),
        lambda: ledger.split_among_siblings(bad, [1.0]),
        lambda: ledger.protected_budget(bad),
    ):
        with pytest.raises(ValueError):
            operation()


def test_failed_charge_is_transactional() -> None:
    ledger = BudgetLedger()
    tracker = PFPTracker()
    tracker.register_branch("b", None, 0.1, "fork")

    with pytest.raises(ValueError):
        ledger.charge(tracker, "b", float("inf"), allow_finish_reserve=True)

    assert tracker.get("b").budget_remaining == 0.1
    assert ledger.total_spent_usd == 0.0
    assert ledger.overdraft_usd == 0.0
