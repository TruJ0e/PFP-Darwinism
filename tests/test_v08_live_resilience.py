"""
v0.8: live-provider resilience and crash-safe accounting.

- The live client never raises on a billed-but-malformed reply (garbage logprobs,
  non-dict choices, content-part lists), so the meter can't book a paid call as free.
- HTTP 200 attempts with an unreadable body are charged, both when a retry succeeds
  and when every attempt fails (`BackendError.billed_usd` -> `MeteredClient`).
- Providers that ignore `n` are topped up with single-sample calls: K probes stay K
  and the arena still splits.
- A failing branch trips the run-wide stop for its siblings (fail fast, no wasted calls).
- Unexpected exceptions halt as `internal_error` with every artifact still written;
  Ctrl-C during Piece 8 keeps the arena run and records no failed experiment.
"""

from __future__ import annotations

import contextlib
import io
import json
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, List

import pytest

import run_arena
from pfp import (
    BackendError,
    IssueDetector,
    MainHarness,
    MeteredClient,
    MockOpenAIClient,
    OpenAICompatibleClient,
    SelfImprovementLoop,
    VersionRegistry,
)
from pfp.backend import estimate_call_cost_usd, prompt_token_upper_bound
from pfp.branch_spawner import MiniHarness
from pfp.issue_detector import compute_token_entropy_from_choice

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
PROBLEM = "A small business's SSH works laptop->desktop but not desktop->laptop."


class _RawServer:
    """Serves a fixed plan of (status, raw_body_bytes) replies to POSTs."""

    def __init__(self, plan: List[tuple[int, bytes]]):
        self.plan = list(plan)
        self.hits = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                outer.hits += 1
                status, payload = outer.plan.pop(0)
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_RawServer":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"


def _live(url: str, retries: int = 3) -> OpenAICompatibleClient:
    client = OpenAICompatibleClient(base_url=url, api_key="k", model=MODEL, max_retries=retries)
    client.chat.completions._sleep = lambda s: None
    return client


def _ok(body: dict) -> tuple[int, bytes]:
    return 200, json.dumps(body).encode("utf-8")


MESSAGES = [{"role": "user", "content": "probe"}]


def _attempt_bound(max_tokens: int = 50, n: int = 1) -> float:
    return estimate_call_cost_usd(prompt_token_upper_bound(MESSAGES), max_tokens * n, MODEL)


# ---------------------------------------------------------------------------
# 1. Malformed-but-billed replies parse instead of raising
# ---------------------------------------------------------------------------
def test_live_client_tolerates_malformed_choices_logprobs_and_content_parts():
    body = {
        "choices": [
            "not-a-choice",
            {
                "index": "zero",
                "message": {"role": None, "content": [{"type": "text", "text": "{\"a\": "}, "1}", {"type": "image"}]},
                "logprobs": {
                    "content": [
                        "garbage",
                        {"token": "x", "logprob": "nan", "top_logprobs": [{"token": "y", "logprob": None}, 7]},
                        {"token": "z", "logprob": 1e308, "top_logprobs": "bad"},
                    ]
                },
            },
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    with _RawServer([_ok(body)]) as srv:
        resp = _live(srv.url).chat.completions.create(messages=MESSAGES, max_tokens=50, logprobs=True)
    assert len(resp.choices) == 1
    choice = resp.choices[0]
    assert choice.message.content == '{"a": 1}'
    assert choice.message.role == "assistant"
    assert choice.index == 1  # unreadable index falls back to position
    toks = choice.logprobs.content
    assert len(toks) == 2
    assert toks[0].logprob == -0.5 and toks[0].top_logprobs[0].logprob == -10.0
    entropy = compute_token_entropy_from_choice(choice)
    assert entropy == entropy and entropy >= 0.0  # finite, no OverflowError from logprob=1e308
    assert resp.usage.estimated is False


def test_live_client_choices_not_a_list_yields_no_choices_but_keeps_usage():
    with _RawServer([_ok({"choices": {"oops": 1}, "usage": {"prompt_tokens": 3, "completion_tokens": 2}})]) as srv:
        resp = _live(srv.url).chat.completions.create(messages=MESSAGES, max_tokens=50)
    assert resp.choices == []
    assert resp.usage.prompt_tokens == 3


# ---------------------------------------------------------------------------
# 2. HTTP 200 with an unreadable body is billed, never free
# ---------------------------------------------------------------------------
def test_unreadable_200_then_success_charges_the_wasted_attempt():
    good = {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 100, "completion_tokens": 10}}
    with _RawServer([(200, b"<html>gateway hiccup</html>"), (200, b"[1, 2]"), _ok(good)]) as srv:
        resp = _live(srv.url).chat.completions.create(messages=MESSAGES, max_tokens=50)
        assert srv.hits == 3
    exact = estimate_call_cost_usd(100, 10, MODEL)
    assert resp.usage.cost_usd == pytest.approx(exact + 2 * _attempt_bound())
    assert resp.usage.estimated is True


def test_all_attempts_unreadable_raise_backend_error_that_meter_still_books():
    with _RawServer([(200, b"not json")] * 3) as srv:
        meter = MeteredClient(_live(srv.url, retries=2), hard_cap_usd=1.0)
        with pytest.raises(BackendError) as ctx:
            meter.chat.completions.create(messages=MESSAGES, max_tokens=50)
    assert ctx.value.billed_usd == pytest.approx(3 * _attempt_bound())
    stats = meter.stats()
    assert stats["failed_calls"] == 1
    assert stats["estimated_usage_calls"] == 1
    assert stats["total_cost_usd"] == pytest.approx(round(3 * _attempt_bound(), 6))
    assert stats["in_flight_reserved_usd"] == 0.0


def test_unreachable_endpoint_is_not_billed():
    meter = MeteredClient(_live("http://127.0.0.1:9/v1", retries=0), hard_cap_usd=1.0)
    with pytest.raises(BackendError) as ctx:
        meter.chat.completions.create(messages=MESSAGES, max_tokens=50)
    assert ctx.value.billed_usd == 0.0
    assert meter.stats()["total_cost_usd"] == 0.0


# ---------------------------------------------------------------------------
# 3. Providers that ignore `n` are topped up
# ---------------------------------------------------------------------------
class IgnoresNClient:
    """Mock that behaves like a provider dropping `n`: always one choice."""

    def __init__(self) -> None:
        self.inner = MockOpenAIClient()
        self.model = self.inner.model
        self.base_url = self.inner.base_url
        self.is_mock = True
        self.cost_scale = self.inner.cost_scale
        self.requested_n: List[int] = []
        outer = self

        class _C:
            def create(self, **kwargs: Any):
                outer.requested_n.append(int(kwargs.get("n", 1)))
                resp = outer.inner.chat.completions.create(**kwargs)
                resp.choices = resp.choices[:1]
                return resp

        self.chat = types.SimpleNamespace(completions=_C())


def test_detector_tops_up_short_batches_and_learns_provider_ignores_n():
    client = IgnoresNClient()
    detector = IssueDetector(client, k_probes=3, depth_dial=4)
    first = detector.detect_plaus(problem=PROBLEM, parent_reasoning="start")
    assert detector.provider_ignores_n is True
    assert detector.n_topup_calls == 2
    assert client.requested_n[:3] == [3, 1, 1]
    assert len(first.distinct_forks) >= 2  # distinct top-up prompts still diverge

    client.requested_n.clear()
    detector.detect_plaus(problem=PROBLEM, parent_reasoning="later")
    probe_calls = client.requested_n[:3]
    assert probe_calls == [1, 1, 1]  # no wasted n=3 request once learned
    assert detector.n_topup_calls == 4


def test_detector_does_not_top_up_providers_that_honor_n():
    detector = IssueDetector(MockOpenAIClient(), k_probes=3)
    detector.detect_plaus(problem=PROBLEM, parent_reasoning="start")
    assert detector.n_topup_calls == 0 and detector.provider_ignores_n is False


def test_arena_still_splits_on_a_provider_that_ignores_n():
    harness = MainHarness(client=IgnoresNClient(), starter_agents=2, hard_cap_usd=2.0, k_probes=3)
    summary = harness.run(PROBLEM)
    assert summary["halted_reason"] is None
    assert summary["total_branches_spawned"] > 2
    assert summary["budget"]["api_n_topup_calls"] > 0
    assert any(e.event_type == "branch" for e in harness.events)


# ---------------------------------------------------------------------------
# 4. A failing branch stops its siblings
# ---------------------------------------------------------------------------
def test_child_failure_trips_run_wide_abort_before_siblings_drain():
    aborted = threading.Event()
    sibling_saw_abort: List[bool] = []

    class Boom:
        def step(self) -> None:
            raise BackendError("HTTP 401")

    class Slow:
        def step(self) -> None:
            sibling_saw_abort.append(aborted.wait(timeout=5))

    owner = types.SimpleNamespace(abort_run=aborted.set)
    with pytest.raises(BackendError):
        MiniHarness._run_children_parallel(owner, [Boom(), Slow()])
    assert aborted.is_set()
    # The sibling either never started (cancelled) or saw the abort while still running
    assert sibling_saw_abort in ([], [True])


def test_backend_failure_mid_run_halts_and_labels_sibling_kills():
    calls = {"n": 0}

    class FailsLater:
        def __init__(self) -> None:
            self.inner = MockOpenAIClient()
            self.model, self.base_url, self.is_mock = self.inner.model, self.inner.base_url, True
            self.cost_scale = self.inner.cost_scale
            outer = self

            class _C:
                def create(self, **kwargs: Any):
                    calls["n"] += 1
                    if calls["n"] == 6:
                        raise BackendError("HTTP 401 key revoked", status=401)
                    return outer.inner.chat.completions.create(**kwargs)

            self.chat = types.SimpleNamespace(completions=_C())

    harness = MainHarness(client=FailsLater(), starter_agents=3, hard_cap_usd=2.0)
    summary = harness.run(PROBLEM)
    assert summary["halted_reason"] == "backend_error"
    kill_reasons = {e.details.get("reason") for e in harness.events if e.event_type == "kill"}
    assert "emergency_stop" not in kill_reasons


# ---------------------------------------------------------------------------
# 5. Crash safety: internal errors and Ctrl-C during Piece 8
# ---------------------------------------------------------------------------
class ExplodesClient:
    """Mock whose Nth call raises an unexpected (non-backend) exception."""

    def __init__(self, at: int) -> None:
        self.inner = MockOpenAIClient()
        self.model, self.base_url, self.is_mock = self.inner.model, self.inner.base_url, True
        self.cost_scale = self.inner.cost_scale
        self.count = 0
        outer = self

        class _C:
            def create(self, **kwargs: Any):
                outer.count += 1
                if outer.count == at:
                    raise RuntimeError("provider returned a shape nobody expected")
                return outer.inner.chat.completions.create(**kwargs)

        self.chat = types.SimpleNamespace(completions=_C())


def test_unexpected_exception_halts_as_internal_error_with_partial_results():
    harness = MainHarness(client=ExplodesClient(at=4), starter_agents=2, hard_cap_usd=2.0)
    summary = harness.run(PROBLEM)
    assert summary["halted_reason"] == "internal_error"
    halts = [e for e in harness.events if e.event_type == "halt"]
    assert halts and "RuntimeError" in halts[-1].details["detail"]
    assert summary["budget"]["api_spend_usd"] > 0


def _cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = run_arena.main(argv)
    return code, out.getvalue(), err.getvalue()


def test_cli_internal_error_still_writes_artifacts_and_exits_2(tmp_path, monkeypatch):
    monkeypatch.setattr(run_arena, "create_client", lambda **kw: ExplodesClient(at=5))
    stream, summary = tmp_path / "run.jsonl", tmp_path / "summary.json"
    code, _, err = _cli(
        ["--quiet", "--no-improve", "--stream-out", str(stream), "--summary-out", str(summary)]
    )
    assert code == 2
    assert "HALTED: internal_error" in err
    assert stream.exists() and json.loads(summary.read_text())["halted_reason"] == "internal_error"


def test_cli_ctrl_c_during_improvement_keeps_arena_run(tmp_path, monkeypatch):
    def interrupted(self, *args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(SelfImprovementLoop, "run_post_run_improvement", interrupted)
    state = tmp_path / "state"
    stream, summary = tmp_path / "run.jsonl", tmp_path / "summary.json"
    code, _, _ = _cli(
        ["--mock", "--quiet", "--state-dir", str(state), "--stream-out", str(stream), "--summary-out", str(summary)]
    )
    assert code == 2
    data = json.loads(summary.read_text())
    assert data["halted_reason"] is None
    assert data["self_improvement"]["interrupted"] is True
    assert stream.exists()
    registry = VersionRegistry.load(state / "registry.json")
    assert len(registry.all_versions()) == 1 and registry.failed_experiments == []


def test_sandbox_interrupt_is_not_recorded_as_a_failed_experiment():
    harness = MainHarness(client=MockOpenAIClient(), starter_agents=2, hard_cap_usd=2.0)
    harness.run(PROBLEM)

    class InterruptsSandbox:
        def __init__(self) -> None:
            self.inner = MockOpenAIClient()
            self.model, self.base_url, self.is_mock = self.inner.model, self.inner.base_url, True
            self.cost_scale = self.inner.cost_scale
            outer = self

            class _C:
                def create(self, **kwargs: Any):
                    text = "\n".join(m.get("content", "") for m in kwargs.get("messages", []))
                    if "Sandbox verification" in text:
                        raise KeyboardInterrupt
                    return outer.inner.chat.completions.create(**kwargs)

            self.chat = types.SimpleNamespace(completions=_C())

    registry = VersionRegistry()
    loop = SelfImprovementLoop(client=InterruptsSandbox(), registry=registry)
    with pytest.raises(KeyboardInterrupt):
        loop.run_post_run_improvement(harness.arena_feed.events, problem=PROBLEM, run_sandbox=True)
    assert len(registry.all_versions()) == 1
    assert registry.failed_experiments == []
    assert not registry.run_in_progress
