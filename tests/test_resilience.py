"""
Resilience tests (v0.3): spend stays honest when the provider omits usage, the hard
cap covers every paid call (including Piece 8), and backend failures / Ctrl-C halt a
run cleanly with its partial results instead of crashing it.
"""

from __future__ import annotations

import contextlib
import email.utils
import http.client
import io
import json
import math
import pytest
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, List

import run_arena
from pfp import (
    AppendOnlyViolationError,
    ArenaFeed,
    BackendError,
    BudgetLedger,
    MainHarness,
    MeteredClient,
    MockOpenAIClient,
    OpenAICompatibleClient,
    SelfImprovementLoop,
)
from pfp.backend import prompt_token_upper_bound


def test_arena_feed_payloads_are_deeply_detached_and_hash_chained(tmp_path):
    feed = ArenaFeed()
    source_snapshot = {"id": "b1", "state": "Possible", "nested": {"score": 1}}
    source_details = {"claims": ["original"]}
    subscriber_events = []
    feed.subscribe(subscriber_events.append)

    returned = feed.append("spawn", "b1", source_snapshot, source_details, timestamp=10.0)
    source_snapshot["nested"]["score"] = 99
    source_details["claims"].append("mutated")
    returned.snapshot["nested"]["score"] = 88
    subscriber_events[0].details["claims"].append("subscriber mutation")
    exposed = feed.events
    exposed[0].snapshot["nested"]["score"] = 77

    second = feed.append(
        "conclude", "b1", {"id": "b1", "state": "Possible"}, timestamp=11.0
    )
    persisted = feed.events
    assert persisted[0].snapshot["nested"]["score"] == 1
    assert persisted[0].details["claims"] == ["original"]
    assert len(persisted[0].digest) == 64
    assert second.previous_digest == persisted[0].digest

    path = feed.save_jsonl(tmp_path / "arena.jsonl")
    loaded = ArenaFeed.load_jsonl(path)
    assert [event.to_dict() for event in loaded.events] == [event.to_dict() for event in persisted]


def test_arena_feed_rejects_tampering_mixed_chain_and_invalid_identity():
    feed = ArenaFeed()
    feed.append("spawn", "b1", {"id": "b1", "state": "Possible"}, timestamp=10.0)
    feed.append("conclude", "b1", {"id": "b1", "state": "Possible"}, timestamp=11.0)
    records = [json.loads(line) for line in feed.to_jsonl().splitlines()]

    tampered = [dict(record) for record in records]
    tampered[0]["details"] = {"rewritten": True}
    with pytest.raises(AppendOnlyViolationError, match="hash chain"):
        ArenaFeed.from_jsonl("\n".join(json.dumps(record) for record in tampered))

    mixed = [dict(record) for record in records]
    mixed[1].pop("digest")
    mixed[1].pop("previous_digest")
    mixed[1].pop("journal_version")
    with pytest.raises(AppendOnlyViolationError, match="mixes chained and legacy"):
        ArenaFeed.from_jsonl("\n".join(json.dumps(record) for record in mixed))

    invalid = [dict(records[0])]
    invalid[0]["snapshot"] = {"id": "other", "state": "Possible"}
    with pytest.raises(ValueError, match="snapshot id"):
        ArenaFeed.from_jsonl(json.dumps(invalid[0]))

    invalid[0]["snapshot"] = {"id": "b1", "state": "Possible"}
    invalid[0]["timestamp"] = math.nan
    with pytest.raises(ValueError, match="non-finite timestamp"):
        ArenaFeed.from_jsonl(json.dumps(invalid[0]))

    with pytest.raises(ValueError, match="snapshot id"):
        feed.append("spawn", "b2", {"id": "other", "state": "Possible"})
    with pytest.raises(ValueError, match="finite"):
        feed.append("spawn", "b2", {"id": "b2", "state": "Possible"}, timestamp=math.inf)
from pfp.backend import ChatCompletionResponse, ChatMessage, Choice, CompletionUsage
from test_hardening import PROBLEM, InterceptingClient, _FakeServer, _OK_BODY


class TestHonestUsage(unittest.TestCase):
    def _client(self, url: str) -> OpenAICompatibleClient:
        client = OpenAICompatibleClient(base_url=url, api_key="k-test", max_retries=1)
        client.chat.completions._sleep = lambda s: None
        return client

    def test_missing_usage_is_charged_a_conservative_estimate_not_zero(self) -> None:
        body = {k: v for k, v in _OK_BODY.items() if k != "usage"}
        with _FakeServer([(200, {}, body)]) as srv:
            resp = self._client(srv.url).chat.completions.create(
                messages=[{"role": "user", "content": "hello"}], max_tokens=200
            )
        self.assertTrue(resp.usage.estimated)
        self.assertGreaterEqual(resp.usage.completion_tokens, 200)
        self.assertGreater(resp.usage.prompt_tokens, len("hello"))
        self.assertGreater(resp.usage.cost_usd, 0.0)

    def test_garbled_usage_fields_never_go_negative_or_crash(self) -> None:
        body = dict(_OK_BODY)
        body["usage"] = {"prompt_tokens": -50, "completion_tokens": "lots", "total_tokens": None}
        with _FakeServer([(200, {}, body)]) as srv:
            resp = self._client(srv.url).chat.completions.create(
                messages=[{"role": "user", "content": "x"}], max_tokens=64
            )
        self.assertTrue(resp.usage.estimated)
        # Negative counts are unreadable, not zero: treating them as zero would
        # erase billed input and weaken the hard-cap ledger.
        self.assertEqual(resp.usage.prompt_tokens, prompt_token_upper_bound([{"role": "user", "content": "x"}]))
        self.assertEqual(resp.usage.completion_tokens, 64)  # unreadable -> max_tokens bound
        self.assertEqual(resp.usage.total_tokens, resp.usage.prompt_tokens + 64)

    def test_reported_usage_is_not_marked_estimated(self) -> None:
        with _FakeServer([(200, {}, _OK_BODY)]) as srv:
            resp = self._client(srv.url).chat.completions.create(messages=[{"role": "user", "content": "x"}])
        self.assertFalse(resp.usage.estimated)
        self.assertEqual(resp.usage.total_tokens, 1500)

    def test_meter_charges_reservation_when_inner_client_reports_no_cost(self) -> None:
        class NoUsage:
            model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"

            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    def create(**kwargs: Any) -> Any:
                        return type("R", (), {"usage": None, "choices": []})()

        meter = MeteredClient(NoUsage(), hard_cap_usd=1.0)
        meter.chat.completions.create(messages=[{"role": "user", "content": "x"}], max_tokens=100)
        stats = meter.stats()
        self.assertGreater(stats["total_cost_usd"], 0.0)
        self.assertEqual(stats["estimated_usage_calls"], 1)
        self.assertEqual(stats["in_flight_reserved_usd"], 0.0)


class TestHttpRobustness(unittest.TestCase):
    def test_retry_after_http_date_is_honored(self) -> None:
        completions = OpenAICompatibleClient(base_url="http://x", api_key="k").chat.completions
        future = email.utils.formatdate(time.time() + 7, usegmt=True)
        delay = completions._backoff_delay(0, future)
        self.assertGreater(delay, 4.0)
        self.assertLessEqual(delay, 7.5)
        past = email.utils.formatdate(time.time() - 100, usegmt=True)
        self.assertEqual(completions._backoff_delay(0, past), 0.0)

    def test_dropped_connection_mid_body_is_retried(self) -> None:
        client = OpenAICompatibleClient(base_url="http://x", api_key="k", max_retries=2)
        sleeps: List[float] = []
        client.chat.completions._sleep = sleeps.append
        calls = {"n": 0}
        payload = json.dumps(_OK_BODY).encode("utf-8")

        class _Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise http.client.IncompleteRead(b"{\"id\":")
            return _Resp(payload)

        import urllib.request

        original = urllib.request.urlopen
        urllib.request.urlopen = fake_urlopen
        try:
            resp = client.chat.completions.create(messages=[{"role": "user", "content": "x"}])
        finally:
            urllib.request.urlopen = original
        self.assertEqual(calls["n"], 2)
        self.assertEqual(len(sleeps), 1)
        self.assertEqual(resp.usage.prompt_tokens, 1000)


class TestHardCapIsAbsolute(unittest.TestCase):
    def test_mock_demo_scale_never_pushes_spend_past_the_cap(self) -> None:
        for cap in (0.01, 0.05, 0.2, 0.5):
            with self.subTest(cap=cap):
                harness = MainHarness(client=MockOpenAIClient(), hard_cap_usd=cap)
                summary = harness.run(PROBLEM)
                self.assertLessEqual(summary["budget"]["api_spend_usd"], cap + 1e-9)
                self.assertEqual(harness.client.stats()["reservation_overruns"], 0)

    def test_cli_meters_self_improvement_under_the_same_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "summary.json"
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code = run_arena.main(["--mock", "--quiet", "--hard-cap", "1.2", "--summary-out", str(out)])
            summary = json.loads(out.read_text(encoding="utf-8"))
        self.assertIn(code, (0, 2))
        self.assertIn("self_improvement", summary)
        self.assertLessEqual(summary["total_api_spend_usd"], 1.2 + 1e-9)
        self.assertAlmostEqual(
            summary["total_api_spend_usd"],
            summary["budget"]["api_spend_usd"] + summary.get("self_improvement_spend_usd", 0.0),
            places=6,
        )

    def test_halted_sandbox_is_never_coherent(self) -> None:
        loop = SelfImprovementLoop(client=MockOpenAIClient(), sandbox_hard_cap_usd=0.005)
        result = loop.run_sandbox_test({})
        self.assertFalse(result.coherent)
        self.assertIn("halted", result.details)
        with self.assertRaises(ValueError):
            SelfImprovementLoop(client=MockOpenAIClient(), sandbox_hard_cap_usd=0)


class TestCleanHalts(unittest.TestCase):
    def _failing_after(self, n_ok: int, exc: BaseException) -> InterceptingClient:
        counter = {"n": 0}

        def hook(prompt: str) -> Any:
            counter["n"] += 1
            if counter["n"] > n_ok:
                raise exc
            return None

        return InterceptingClient(hook)

    def test_backend_error_mid_run_halts_with_partial_summary(self) -> None:
        client = self._failing_after(6, BackendError("HTTP 401: bad key", status=401))
        harness = MainHarness(client=client)
        summary = harness.run(PROBLEM)
        self.assertEqual(summary["halted_reason"], "backend_error")
        self.assertGreater(summary["budget"]["api_calls"], 0)
        self.assertGreater(summary["budget"]["api_failed_calls"], 0)
        halts = [e for e in harness.arena_feed.events if e.event_type == "halt"]
        self.assertEqual(len(halts), 1)
        self.assertEqual(halts[0].details["reason"], "backend_error")
        self.assertIn("401", halts[0].details["detail"])

    def test_keyboard_interrupt_halts_and_stops_new_calls(self) -> None:
        client = self._failing_after(4, KeyboardInterrupt())
        harness = MainHarness(client=client)
        summary = harness.run(PROBLEM)
        self.assertEqual(summary["halted_reason"], "interrupted")
        calls_at_halt = len(client.prompts)
        # No more paid calls may start once the run is halted
        self.assertTrue(harness._should_stop())
        self.assertEqual(len(client.prompts), calls_at_halt)
        self.assertEqual(harness.client.stats()["in_flight_reserved_usd"], 0.0)

    def test_cli_still_writes_outputs_after_backend_failure(self) -> None:
        def broken(**kwargs: Any) -> Any:
            return InterceptingClient(lambda p: (_ for _ in ()).throw(BackendError("HTTP 500 after retries", status=500)))

        original = run_arena.create_client
        run_arena.create_client = broken
        try:
            with tempfile.TemporaryDirectory() as tmp:
                stream = Path(tmp) / "run.jsonl"
                out = Path(tmp) / "s.json"
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    code = run_arena.main(["--quiet", "--stream-out", str(stream), "--summary-out", str(out)])
                self.assertEqual(code, 2)
                self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["halted_reason"], "backend_error")
                self.assertEqual(ArenaFeed.load_jsonl(stream).events[-1].event_type, "halt")
        finally:
            run_arena.create_client = original


class TestSizing(unittest.TestCase):
    def _size(self, reply: str) -> BudgetLedger:
        resp = ChatCompletionResponse(
            id="s",
            model="m",
            choices=[Choice(index=0, message=ChatMessage(role="assistant", content=reply))],
            usage=CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2, cost_usd=0.0),
        )
        client = type("C", (), {})()
        client.chat = type("Chat", (), {})()
        client.chat.completions = type("Comp", (), {"create": staticmethod(lambda **kw: resp)})()
        ledger = BudgetLedger(hard_cap_usd=2.0)
        ledger.size_problem_once(client, PROBLEM)
        return ledger

    def test_fenced_live_reply_is_parsed(self) -> None:
        ledger = self._size('<think>hmm</think>\n```json\n{"difficulty_estimate_usd": 0.4}\n```')
        self.assertAlmostEqual(ledger.difficulty_estimate_usd, 0.4)
        self.assertAlmostEqual(ledger.root_budget_usd, 0.8)

    def test_non_finite_or_negative_estimates_fall_back(self) -> None:
        for reply in ('{"difficulty_estimate_usd": NaN}', '{"difficulty_estimate_usd": -3}', '{"difficulty_estimate_usd": "x"}'):
            with self.subTest(reply=reply):
                ledger = self._size(reply)
                self.assertAlmostEqual(ledger.difficulty_estimate_usd, 0.05)
                self.assertAlmostEqual(ledger.root_budget_usd, 0.1)


class TestReplayCli(unittest.TestCase):
    def test_replay_rerenders_a_saved_stream_without_model_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stream = Path(tmp) / "run.jsonl"
            html = Path(tmp) / "replay.html"
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(run_arena.main(["--mock", "--quiet", "--no-improve", "--stream-out", str(stream)]), 0)

            def no_calls(**kwargs: Any) -> Any:
                raise AssertionError("replay must not build a model client")

            original = run_arena.create_client
            run_arena.create_client = no_calls
            try:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
                    code = run_arena.main(["--replay", str(stream), "--html-out", str(html)])
            finally:
                run_arena.create_client = original
            self.assertEqual(code, 0)
            self.assertTrue(html.exists())
            self.assertIn("WINNER", buf.getvalue().upper())
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["replay.html", "run.jsonl"])  # no .tmp left

    def test_replay_of_missing_or_corrupt_stream_fails_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.jsonl"
            bad.write_text('{"seq": 2, "event_type": "x"}\n', encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(run_arena.main(["--replay", str(bad)]), 1)
                self.assertEqual(run_arena.main(["--replay", str(Path(tmp) / "missing.jsonl")]), 1)


if __name__ == "__main__":
    unittest.main()
