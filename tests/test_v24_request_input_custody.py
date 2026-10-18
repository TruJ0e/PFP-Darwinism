"""v0.24: every billable request field participates in cap custody."""

from __future__ import annotations

import copy

import pytest

from pfp.backend import (
    HardCapExceededError,
    MeteredClient,
    OpenAICompatibleClient,
    estimate_call_cost_usd,
    request_token_upper_bound,
)
from tests.test_hardening import _FakeServer, _OK_BODY


def _large_tools() -> list[dict[str, object]]:
    return [{
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "x" * 20_000,
            "parameters": {"type": "object", "properties": {}},
        },
    }]


def test_request_bound_includes_tool_definitions() -> None:
    messages = [{"role": "user", "content": "x"}]
    plain = request_token_upper_bound(messages)
    with_tools = request_token_upper_bound(messages, {"tools": _large_tools()})
    assert with_tools >= plain + 20_000


def test_large_tools_cannot_enter_under_message_only_cap() -> None:
    messages = [{"role": "user", "content": "x"}]
    tools = _large_tools()
    full_bound = estimate_call_cost_usd(
        request_token_upper_bound(
            messages,
            {
                "tools": tools,
                "model": None,
                "temperature": 0.7,
                "max_tokens": 1,
                "logprobs": False,
                "top_logprobs": None,
                "n": 1,
            },
        ),
        1,
    )
    meter = MeteredClient(
        OpenAICompatibleClient(base_url="http://unused.invalid", api_key="k", max_retries=0),
        hard_cap_usd=full_bound / 2,
    )

    with pytest.raises(HardCapExceededError, match="Hard cap"):
        meter.chat.completions.create(messages=messages, max_tokens=1, tools=tools)


def test_missing_usage_charges_full_request_input() -> None:
    body = copy.deepcopy(_OK_BODY)
    body.pop("usage")
    messages = [{"role": "user", "content": "x"}]
    tools = _large_tools()

    with _FakeServer([(200, {}, body)]) as server:
        response = OpenAICompatibleClient(
            base_url=server.url, api_key="k", max_retries=0
        ).chat.completions.create(messages=messages, max_tokens=7, tools=tools)

    sent = server.requests[0]["body"]
    expected_input = request_token_upper_bound(
        messages, {key: value for key, value in sent.items() if key != "messages"}
    )
    assert response.usage.estimated is True
    assert response.usage.prompt_tokens == expected_input
    assert response.usage.completion_tokens == 7
