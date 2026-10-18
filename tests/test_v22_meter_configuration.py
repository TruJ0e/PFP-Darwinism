"""v0.22: malformed client configuration cannot weaken spend admission."""

from __future__ import annotations

import math

import pytest

from pfp import MeteredClient, MockOpenAIClient, OpenAICompatibleClient


@pytest.mark.parametrize("bad", [True, math.nan, math.inf, -0.01, "broken"])
def test_meter_rejects_invalid_hard_caps(bad: object) -> None:
    with pytest.raises(ValueError):
        MeteredClient(MockOpenAIClient(), hard_cap_usd=bad)


@pytest.mark.parametrize("bad", [True, 0, -1, 1.5, "2"])
def test_meter_rejects_lossy_concurrency_configuration(bad: object) -> None:
    with pytest.raises(ValueError):
        MeteredClient(MockOpenAIClient(), max_concurrent_calls=bad)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("cost_scale", math.nan),
        ("cost_scale", 0.5),
        ("max_billable_attempts", True),
        ("max_billable_attempts", 1.5),
        ("max_billable_attempts", 0),
    ],
)
def test_meter_rejects_unsafe_inner_reservation_metadata(field: str, bad: object) -> None:
    inner = MockOpenAIClient()
    setattr(inner, field, bad)
    with pytest.raises(ValueError):
        MeteredClient(inner)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout": True},
        {"timeout": 0},
        {"timeout": math.inf},
        {"max_retries": True},
        {"max_retries": -1},
        {"max_retries": 1.5},
        {"backoff_base_s": True},
        {"backoff_base_s": -0.1},
        {"backoff_base_s": math.nan},
    ],
)
def test_live_client_rejects_invalid_transport_controls(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        OpenAICompatibleClient(base_url="http://unused.invalid", api_key="k", **kwargs)


def test_valid_configuration_preserves_exact_reservation_metadata() -> None:
    inner = MockOpenAIClient()
    inner.cost_scale = 2.5
    inner.max_billable_attempts = 4
    meter = MeteredClient(inner, hard_cap_usd=1, max_concurrent_calls=2)

    assert meter.hard_cap_usd == 1.0
    assert meter.cost_scale == 2.5
    assert meter.max_billable_attempts == 4
