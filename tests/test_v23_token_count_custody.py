"""v0.23: token counts cannot weaken reservations or erase billed usage."""

from __future__ import annotations

import copy

import pytest

from pfp.backend import (
    HardCapExceededError,
    UNBOUNDED_COMPLETION_TOKENS,
    MeteredClient,
    OpenAICompatibleClient,
    completion_token_bound,
    estimate_call_cost_usd,
    prompt_token_upper_bound,
    request_token_upper_bound,
)
from tests.test_hardening import _FakeServer, _OK_BODY


@pytest.mark.parametrize("bad", [-1, -0.1, 1.5, "-1", "2.5"])
def test_invalid_completion_limits_reserve_full_context(bad: object) -> None:
    assert completion_token_bound(bad) == UNBOUNDED_COMPLETION_TOKENS


@pytest.mark.parametrize("bad", [True, 0, -2, -0.5, 1.5, "3"])
def test_invalid_sample_counts_are_rejected_before_dispatch(bad: object) -> None:
    with pytest.raises(ValueError, match="n must be an integer"):
        completion_token_bound(10, bad)


def test_live_client_rejects_invalid_sample_count_without_network_io() -> None:
    client = OpenAICompatibleClient(base_url="http://unused.invalid", api_key="k", max_retries=0)
    with pytest.raises(ValueError, match="n must be an integer"):
        client.chat.completions.create(messages=[], n=1.5)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("prompt_tokens", -1),
        ("prompt_tokens", 0.5),
        ("completion_tokens", -1),
        ("completion_tokens", 0.5),
    ],
)
def test_malformed_provider_usage_is_conservatively_estimated(field: str, bad: object) -> None:
    body = copy.deepcopy(_OK_BODY)
    body["usage"][field] = bad
    messages = [{"role": "user", "content": "account for me"}]

    with _FakeServer([(200, {}, body)]) as server:
        client = OpenAICompatibleClient(base_url=server.url, api_key="k", max_retries=0)
        response = client.chat.completions.create(messages=messages, max_tokens=17)

    expected_prompt = 1000 if field == "completion_tokens" else request_token_upper_bound(
        messages,
        {
            "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
            "temperature": 0.7,
            "max_tokens": 17,
        },
    )
    expected_completion = 500 if field == "prompt_tokens" else 17
    assert response.usage.estimated is True
    assert response.usage.prompt_tokens == expected_prompt
    assert response.usage.completion_tokens == expected_completion
    assert response.usage.cost_usd == pytest.approx(
        estimate_call_cost_usd(expected_prompt, expected_completion, response.model)
    )


def test_negative_limit_cannot_enter_under_a_tiny_hard_cap() -> None:
    messages = [{"role": "user", "content": "x"}]
    full_bound = estimate_call_cost_usd(
        prompt_token_upper_bound(messages), UNBOUNDED_COMPLETION_TOKENS
    )
    meter = MeteredClient(
        OpenAICompatibleClient(base_url="http://unused.invalid", api_key="k", max_retries=0),
        hard_cap_usd=full_bound / 2,
    )

    with pytest.raises(HardCapExceededError, match="Hard cap"):
        meter.chat.completions.create(messages=messages, max_tokens=-1)
