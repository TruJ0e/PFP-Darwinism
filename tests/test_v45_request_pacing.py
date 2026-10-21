"""
v0.45 request pacing: live run 6 halted on HTTP 429 after 82 spawns in its first minute
(Token Factory: 100 requests/min). The live client now paces every HTTP attempt, retries
included, through a thread-safe limiter; a 429 holds the whole run back. The mock is never paced.
No network: urlopen is patched, and the limiter runs on a fake clock wherever timing matters.
"""

from __future__ import annotations

import concurrent.futures
import email.message
import io
import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any, List

import pytest

import run_arena
from pfp import HardCapExceededError, MeteredClient, MockOpenAIClient, OpenAICompatibleClient, RequestRateLimiter, create_client
from pfp.backend import (
    DEFAULT_REQUESTS_PER_MINUTE,
    REQUESTS_PER_MINUTE_ENV,
    parse_requests_per_minute,
    resolve_requests_per_minute,
)

_OK_BODY = {
    "id": "c1",
    "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
    "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
}


class FakeClock:
    """Monotonic clock that only moves when someone sleeps on it."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: List[float] = []
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self.now

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.sleeps.append(seconds)
            self.now += seconds


def _http_error(code: int, headers: dict) -> urllib.error.HTTPError:
    msg = email.message.Message()
    for k, v in headers.items():
        msg[k] = v
    return urllib.error.HTTPError("http://x/v1/chat/completions", code, "err", msg, io.BytesIO(b'{"error": "quota"}'))


class ScriptedUrlopen:
    """Stands in for urllib.request.urlopen: replays a plan of statuses, records send times."""

    def __init__(self, plan: List[Any], clock=time.monotonic) -> None:
        self.plan = list(plan)
        self.sent: List[float] = []
        self._clock = clock
        self._lock = threading.Lock()

    def __call__(self, req, timeout=None):
        with self._lock:
            self.sent.append(self._clock())
            step = self.plan.pop(0) if self.plan else 200
        if isinstance(step, tuple):
            raise _http_error(*step)
        return io.BytesIO(json.dumps(_OK_BODY).encode("utf-8"))


def _paced_client(rpm: float, clock: FakeClock, max_retries: int = 3) -> OpenAICompatibleClient:
    client = OpenAICompatibleClient(base_url="http://x/v1", api_key="k", max_retries=max_retries, requests_per_minute=rpm)
    client.rate_limiter._clock = clock
    client.chat.completions._sleep = clock.sleep
    return client


def _msg() -> list:
    return [{"role": "user", "content": "hi"}]


# ---------------------------------------------------------------------------
# The limiter itself
# ---------------------------------------------------------------------------
def test_limiter_spaces_back_to_back_requests_at_the_rate():
    clock = FakeClock()
    limiter = RequestRateLimiter(60, clock=clock)
    assert [limiter.reserve() for _ in range(5)] == [0.0, 1.0, 2.0, 3.0, 4.0]
    clock.sleep(30)  # idle time does not bank a burst beyond `burst`
    assert limiter.reserve() == 0.0
    assert limiter.reserve() == 1.0
    stats = limiter.stats()
    assert stats["requests"] == 7 and stats["paced_requests"] == 5


def test_limiter_burst_admits_burst_then_paces():
    clock = FakeClock()
    limiter = RequestRateLimiter(60, burst=3, clock=clock)
    assert [limiter.reserve() for _ in range(5)] == [0.0, 0.0, 0.0, 1.0, 2.0]


def test_any_minute_carries_at_most_rpm_plus_burst_requests():
    clock = FakeClock()
    rpm = 80
    limiter = RequestRateLimiter(rpm, clock=clock)
    start = clock()
    # 300 callers arrive in the first second, like run 6's growth burst
    sends = []
    for _ in range(300):
        clock.sleep(1 / 300.0)
        sends.append(clock() + limiter.reserve())
    for t0 in sends:
        in_window = sum(1 for t in sends if t0 <= t < t0 + 60.0)
        assert in_window <= rpm + 1
    # ...and the first minute is far below the provider's 100/min
    assert sum(1 for t in sends if t < start + 60.0) <= rpm + 1


def test_limiter_hands_out_unique_slots_under_concurrency():
    clock = FakeClock()  # frozen: every caller arrives at the same instant
    limiter = RequestRateLimiter(600, clock=clock)
    barrier = threading.Barrier(16)

    def worker() -> List[float]:
        barrier.wait()
        return [limiter.reserve() for _ in range(25)]

    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        waits = [w for f in [pool.submit(worker) for _ in range(16)] for w in f.result()]
    slots = sorted(round(w / limiter.interval_s) for w in waits)
    assert slots == list(range(400))  # no two requests share a slot, none skipped


def test_penalize_holds_every_unreserved_slot():
    clock = FakeClock()
    limiter = RequestRateLimiter(60, clock=clock)
    assert limiter.reserve() == 0.0
    limiter.penalize(30.0)
    assert limiter.reserve() == pytest.approx(30.0)
    assert limiter.reserve() == pytest.approx(31.0)
    limiter.penalize(5.0)  # never pulls slots earlier than already scheduled
    assert limiter.reserve() == pytest.approx(32.0)
    limiter.penalize(float("nan"))
    assert limiter.stats()["rate_limit_penalties"] == 2


@pytest.mark.parametrize("rpm", [0, -1, float("nan"), float("inf"), True, "fast"])
def test_limiter_rejects_bad_rates(rpm):
    with pytest.raises(ValueError):
        RequestRateLimiter(rpm)


# ---------------------------------------------------------------------------
# The dial
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    (None, DEFAULT_REQUESTS_PER_MINUTE), ("", DEFAULT_REQUESTS_PER_MINUTE), ("default", DEFAULT_REQUESTS_PER_MINUTE),
    ("0", None), (0, None), ("off", None), ("OFF", None), ("none", None),
    ("50", 50.0), (90, 90.0), ("12.5", 12.5),
])
def test_parse_requests_per_minute(raw, expected):
    assert parse_requests_per_minute(raw) == expected


@pytest.mark.parametrize("raw", ["-5", "nan", "inf", "fast", True])
def test_parse_requests_per_minute_rejects_garbage(raw):
    with pytest.raises(ValueError):
        parse_requests_per_minute(raw)


def test_env_var_sets_the_dial_and_explicit_value_wins(monkeypatch):
    monkeypatch.setenv(REQUESTS_PER_MINUTE_ENV, "40")
    assert resolve_requests_per_minute() == 40.0
    assert resolve_requests_per_minute("70") == 70.0
    monkeypatch.setenv(REQUESTS_PER_MINUTE_ENV, "off")
    assert resolve_requests_per_minute() is None
    monkeypatch.delenv(REQUESTS_PER_MINUTE_ENV)
    assert resolve_requests_per_minute() == DEFAULT_REQUESTS_PER_MINUTE


def test_create_client_paces_live_by_default_and_never_the_mock(monkeypatch):
    monkeypatch.delenv(REQUESTS_PER_MINUTE_ENV, raising=False)
    live = create_client(use_mock=False, api_key="k")
    assert live.rate_limiter is not None
    assert live.rate_limiter.requests_per_minute == DEFAULT_REQUESTS_PER_MINUTE
    assert live.chat.completions.rate_limiter is live.rate_limiter
    assert create_client(use_mock=False, api_key="k", requests_per_minute="off").rate_limiter is None
    assert create_client(use_mock=False, api_key="k", requests_per_minute=0).rate_limiter is None
    assert create_client(use_mock=False, api_key="k", requests_per_minute=30).rate_limiter.requests_per_minute == 30
    monkeypatch.setenv(REQUESTS_PER_MINUTE_ENV, "1")
    mock = create_client(use_mock=True)
    assert getattr(mock, "rate_limiter", None) is None
    assert getattr(mock.chat.completions, "rate_limiter", None) is None


def test_mock_run_is_instant_even_with_a_crawling_dial(monkeypatch, tmp_path):
    monkeypatch.setenv(REQUESTS_PER_MINUTE_ENV, "1")
    started = time.monotonic()
    rc = run_arena.main(["--mock", "--quiet", "--no-improve", "--requests-per-minute", "1",
                         "--summary-out", str(tmp_path / "s.json")])
    assert rc == 0
    assert time.monotonic() - started < 60  # 1/min pacing would need hours
    summary = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert "request_pacing" not in summary


@pytest.mark.parametrize("raw", ["-1", "nan", "abc"])
def test_cli_rejects_bad_pacing_dial(raw):
    with pytest.raises(SystemExit):
        run_arena.main(["--mock", "--quiet", "--requests-per-minute", raw])


# ---------------------------------------------------------------------------
# The live client
# ---------------------------------------------------------------------------
def test_live_client_paces_sequential_calls(monkeypatch):
    clock = FakeClock()
    fake = ScriptedUrlopen([], clock=clock)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    client = _paced_client(30, clock)  # one request per 2 s
    for _ in range(4):
        client.chat.completions.create(messages=_msg())
    gaps = [b - a for a, b in zip(fake.sent, fake.sent[1:])]
    assert gaps == [2.0, 2.0, 2.0]


def test_live_client_paces_concurrent_callers(monkeypatch):
    # Real clock, low rate, no network: 12 parallel calls at 1200/min must leave ~50 ms apart
    fake = ScriptedUrlopen([])
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    client = OpenAICompatibleClient(base_url="http://x/v1", api_key="k", requests_per_minute=1200)
    with concurrent.futures.ThreadPoolExecutor(12) as pool:
        list(pool.map(lambda _: client.chat.completions.create(messages=_msg()), range(12)))
    sent = sorted(fake.sent)
    assert len(sent) == 12
    assert sent[-1] - sent[0] >= 11 * 0.05 * 0.9
    assert min(b - a for a, b in zip(sent, sent[1:])) >= 0.05 * 0.5
    assert client.rate_limiter.stats()["requests"] == 12


def test_429_retry_waits_behind_retry_after_through_the_limiter(monkeypatch):
    clock = FakeClock()
    fake = ScriptedUrlopen([(429, {"Retry-After": "7"}), 200], clock=clock)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    client = _paced_client(60, clock)
    resp = client.chat.completions.create(messages=_msg())
    assert resp.choices[0].message.content == "ok"
    assert len(fake.sent) == 2
    assert fake.sent[1] - fake.sent[0] == pytest.approx(7.0)  # Retry-After honored, not 1 s pacing
    stats = client.rate_limiter.stats()
    assert stats["requests"] == 2  # the retry took a limiter slot too
    assert stats["rate_limit_penalties"] == 1


def test_429_pause_holds_other_threads_too(monkeypatch):
    clock = FakeClock()
    fake = ScriptedUrlopen([(429, {"Retry-After": "20"})], clock=clock)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    client = _paced_client(60, clock, max_retries=0)
    with pytest.raises(Exception):
        client.chat.completions.create(messages=_msg())  # out of retries, still penalizes
    client.chat.completions.create(messages=_msg())  # another caller
    assert fake.sent[1] - fake.sent[0] == pytest.approx(20.0)


def test_429_without_retry_after_backs_off_seconds_not_milliseconds(monkeypatch):
    clock = FakeClock()
    fake = ScriptedUrlopen([(429, {}), (429, {}), 200], clock=clock)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    client = _paced_client(600, clock)
    client.chat.completions.create(messages=_msg())
    first, second = (b - a for a, b in zip(fake.sent, fake.sent[1:]))
    assert 2.5 <= first <= 5.0
    assert 5.0 <= second <= 10.0


def test_unpaced_live_client_keeps_the_old_retry_path(monkeypatch):
    fake = ScriptedUrlopen([(503, {}), (429, {"Retry-After": "2"}), 200])
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    client = OpenAICompatibleClient(base_url="http://x/v1", api_key="k", max_retries=3)
    sleeps: List[float] = []
    client.chat.completions._sleep = sleeps.append
    assert client.rate_limiter is None
    client.chat.completions.create(messages=_msg())
    assert len(sleeps) == 2 and sleeps[1] == 2.0


# ---------------------------------------------------------------------------
# Custody: pacing changes WHEN, never WHETHER spend is accounted
# ---------------------------------------------------------------------------
def test_hard_cap_refusal_happens_before_any_paced_slot(monkeypatch):
    clock = FakeClock()
    fake = ScriptedUrlopen([], clock=clock)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    client = _paced_client(60, clock)
    meter = MeteredClient(client, hard_cap_usd=1e-9)
    with pytest.raises(HardCapExceededError):
        meter.chat.completions.create(messages=_msg(), max_tokens=100)
    assert client.rate_limiter.stats()["requests"] == 0
    assert fake.sent == []


def test_paced_and_unpaced_calls_book_identical_spend(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(urllib.request, "urlopen", ScriptedUrlopen([], clock=clock))
    paced = MeteredClient(_paced_client(60, clock), hard_cap_usd=1.0)
    unpaced = MeteredClient(OpenAICompatibleClient(base_url="http://x/v1", api_key="k"), hard_cap_usd=1.0)
    for meter in (paced, unpaced):
        for _ in range(3):
            meter.chat.completions.create(messages=_msg(), max_tokens=100)
    assert paced.stats() == unpaced.stats()
    assert paced.stats()["in_flight_reserved_usd"] == 0.0
    assert clock.sleeps == [1.0, 1.0]  # the paced meter waited; it did not pay differently


def test_mock_client_has_no_limiter_and_stays_deterministic():
    a = MockOpenAIClient()
    b = MockOpenAIClient()
    assert not hasattr(a, "rate_limiter")
    ra = a.chat.completions.create(messages=_msg(), n=2)
    rb = b.chat.completions.create(messages=_msg(), n=2)
    assert [c.message.content for c in ra.choices] == [c.message.content for c in rb.choices]
    assert ra.usage.to_dict() == rb.usage.to_dict()
