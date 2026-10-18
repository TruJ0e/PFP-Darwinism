"""v0.9: retry-aware and positional-call hard-cap admission."""

from __future__ import annotations

import pytest
import types

from pfp import HardCapExceededError, MeteredClient, MockOpenAIClient
from pfp.backend import estimate_call_cost_usd, prompt_token_upper_bound
from test_v08_live_resilience import MESSAGES, MODEL, _RawServer, _live, _ok


def _bound(max_tokens: int, attempts: int = 1, messages=MESSAGES, n: int = 1) -> float:
    return attempts * estimate_call_cost_usd(
        prompt_token_upper_bound(messages), max_tokens * n, MODEL
    )


def test_live_meter_reserves_every_potentially_billable_retry_before_send():
    """One-call headroom must not admit a request that can bill three attempts."""
    one_attempt = _bound(50)
    with _RawServer([_ok({"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})]) as srv:
        meter = MeteredClient(_live(srv.url, retries=2), hard_cap_usd=one_attempt * 2.5)
        with pytest.raises(HardCapExceededError):
            meter.chat.completions.create(messages=MESSAGES, max_tokens=50)
        assert srv.hits == 0
    assert meter.stats()["in_flight_reserved_usd"] == 0.0


def test_retry_reservation_is_released_and_only_actual_spend_is_booked():
    good = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    with _RawServer([(200, b"not json"), _ok(good)]) as srv:
        meter = MeteredClient(_live(srv.url, retries=2), hard_cap_usd=_bound(50, attempts=3))
        meter.chat.completions.create(messages=MESSAGES, max_tokens=50)
        assert srv.hits == 2
    stats = meter.stats()
    expected = _bound(50) + estimate_call_cost_usd(10, 5, MODEL)
    assert stats["total_cost_usd"] == pytest.approx(round(expected, 6))
    assert stats["in_flight_reserved_usd"] == 0.0
    assert stats["reservation_overruns"] == 0
    assert stats["hard_cap_overrun_usd"] == 0.0


def test_positional_openai_arguments_receive_the_same_preflight_as_keywords():
    messages = [{"role": "user", "content": "x" * 20_000}]
    max_tokens = 4_000
    n = 3
    required = _bound(max_tokens, messages=messages, n=n) * 50.0
    meter = MeteredClient(MockOpenAIClient(), hard_cap_usd=required - 1e-8)

    # Signature: messages, model, temperature, max_tokens, logprobs,
    # top_logprobs, n. The inner mock must never receive this rejected call.
    with pytest.raises(HardCapExceededError):
        meter.chat.completions.create(messages, MODEL, 0.7, max_tokens, False, None, n)

    assert meter.inner.chat.completions._call_count == 0
    assert meter.stats()["in_flight_reserved_usd"] == 0.0


def test_exception_cost_overrun_is_visible_in_stats():
    class BilledFailure:
        model = MODEL

        class _Completions:
            @staticmethod
            def create(**kwargs):
                from pfp import BackendError

                raise BackendError("bad billed response", billed_usd=1.0)

        def __init__(self):
            self.chat = types.SimpleNamespace(completions=self._Completions())

    meter = MeteredClient(BilledFailure(), hard_cap_usd=2.0)
    with pytest.raises(Exception):
        meter.chat.completions.create(messages=MESSAGES, max_tokens=1)
    stats = meter.stats()
    assert stats["reservation_overruns"] == 1
    assert stats["hard_cap_overrun_usd"] == 0.0
