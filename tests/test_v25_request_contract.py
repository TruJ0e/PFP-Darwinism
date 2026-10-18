"""v0.25: the request contract fails closed on the live, mock, and metered paths."""

from __future__ import annotations

import pytest

from pfp.backend import MeteredClient, MockOpenAIClient, OpenAICompatibleClient
from tests.test_hardening import _FakeServer, _OK_BODY

_MESSAGES = [{"role": "user", "content": "hello"}]


@pytest.mark.parametrize("stream", [True, 1, "yes", {"include_usage": True}])
def test_live_client_refuses_streaming_before_dispatch(stream: object) -> None:
    # Every reply would be an unreadable SSE body: billed, retried, billed again.
    with _FakeServer([(200, {}, _OK_BODY)] * 4) as server:
        client = OpenAICompatibleClient(base_url=server.url, api_key="k", max_retries=3)
        with pytest.raises(ValueError, match="stream"):
            client.chat.completions.create(messages=_MESSAGES, stream=stream)
    assert server.requests == []


@pytest.mark.parametrize("stream", [None, False])
def test_live_client_accepts_explicit_non_streaming(stream: object) -> None:
    with _FakeServer([(200, {}, _OK_BODY)]) as server:
        client = OpenAICompatibleClient(base_url=server.url, api_key="k", max_retries=0)
        client.chat.completions.create(messages=_MESSAGES, stream=stream)
    assert len(server.requests) == 1


def test_meter_refuses_streaming_without_reserving_or_counting() -> None:
    class _Inner:
        model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
        calls = 0

        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(*args, **kwargs):
                    _Inner.calls += 1
                    raise AssertionError("streamed request reached the inner client")

    meter = MeteredClient(_Inner(), hard_cap_usd=1.0)
    with pytest.raises(ValueError, match="stream"):
        meter.chat.completions.create(messages=_MESSAGES, stream=True)
    stats = meter.stats()
    assert _Inner.calls == 0
    assert stats["in_flight_reserved_usd"] == 0.0
    assert stats["failed_calls"] == 0 and stats["calls"] == 0


@pytest.mark.parametrize("bad", [0, -1, True, 2.5, "2"])
def test_mock_rejects_sample_counts_the_live_client_rejects(bad: object) -> None:
    mock = MockOpenAIClient()
    with pytest.raises(ValueError, match="n must be"):
        mock.chat.completions.create(messages=_MESSAGES, n=bad)
    with _FakeServer([]) as server:
        live = OpenAICompatibleClient(base_url=server.url, api_key="k", max_retries=0)
        with pytest.raises(ValueError, match="n must be"):
            live.chat.completions.create(messages=_MESSAGES, n=bad)
    assert server.requests == []


def test_mock_refuses_streaming_like_live() -> None:
    with pytest.raises(ValueError, match="stream"):
        MockOpenAIClient().chat.completions.create(messages=_MESSAGES, stream=True)


def test_mock_still_returns_n_choices() -> None:
    resp = MockOpenAIClient().chat.completions.create(messages=_MESSAGES, n=3)
    assert [c.index for c in resp.choices] == [0, 1, 2]
