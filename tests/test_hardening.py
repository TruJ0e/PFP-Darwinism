"""
Hardening tests (v0.2): live-model robustness, honest budget accounting, safety rails,
selection correctness, Piece 8 persistence/validation, and arena renderer safety.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List

import run_arena
from pfp import (
    ArenaFeed,
    BackendError,
    BranchState,
    BudgetLedger,
    EventLogMiner,
    FailJudge,
    HardCapExceededError,
    MainHarness,
    MeteredClient,
    MockOpenAIClient,
    OpenAICompatibleClient,
    PFPTracker,
    SelfImprovementLoop,
    TronArenaRenderer,
    VersionRegistry,
    coerce_bool,
    extract_json,
    parse_json_object,
    sanitize_dials,
)
from pfp.alignment_detector import AlignmentDetector
from pfp.backend import ChatCompletionResponse, ChatMessage, Choice, CompletionUsage

PROBLEM = "A small business's SSH works laptop->desktop but not desktop->laptop."


def _response(content: str, cost: float = 0.001) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="scripted",
        model="scripted-model",
        choices=[Choice(index=0, message=ChatMessage(role="assistant", content=content))],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20, cost_usd=cost),
    )


class _ScriptedCompletions:
    def __init__(self, replies: List[str]):
        self.replies = list(replies)
        self.prompts: List[str] = []

    def create(self, messages, **kwargs):
        self.prompts.append("\n".join(m.get("content", "") for m in messages))
        return _response(self.replies.pop(0) if self.replies else "{}")


class ScriptedClient:
    """Replays a fixed list of model replies (one per call)."""

    def __init__(self, replies: List[str]):
        self.model = "scripted-model"
        self.base_url = "scripted://"
        self.is_mock = True
        self.chat = type("Chat", (), {})()
        self.chat.completions = _ScriptedCompletions(replies)


class InterceptingClient:
    """Wraps the mock; `hook(prompt)` may return a replacement reply. Records every prompt."""

    def __init__(self, hook: Callable[[str], Any] = lambda p: None):
        self.inner = MockOpenAIClient()
        self.model = self.inner.model
        self.base_url = self.inner.base_url
        self.is_mock = True
        self.hook = hook
        self.prompts: List[str] = []
        outer = self

        class _C:
            def create(self, messages, **kwargs):
                prompt = "\n".join(m.get("content", "") for m in messages)
                outer.prompts.append(prompt)
                override = outer.hook(prompt)
                if override is not None:
                    return _response(override)
                return outer.inner.chat.completions.create(messages=messages, **kwargs)

        self.chat = type("Chat", (), {})()
        self.chat.completions = _C()


# ---------------------------------------------------------------------------
# Tolerant JSON parsing for live model output
# ---------------------------------------------------------------------------
class TestJsonUtil(unittest.TestCase):
    def test_extracts_json_from_real_world_model_wrappers(self) -> None:
        self.assertEqual(parse_json_object('{"a": 1}'), {"a": 1})
        self.assertEqual(parse_json_object('```json\n{"a": 2}\n```'), {"a": 2})
        self.assertEqual(parse_json_object('Sure! Here it is: {"a": 3} hope that helps'), {"a": 3})
        self.assertEqual(parse_json_object('<think>maybe {"a": 0}?</think>\n{"a": 4}'), {"a": 4})
        self.assertEqual(parse_json_object('{"outer": {"inner": [1, 2]}}'), {"outer": {"inner": [1, 2]}})
        self.assertEqual(extract_json("[1, 2, 3]"), [1, 2, 3])
        self.assertIsNone(parse_json_object("no json here"))
        self.assertIsNone(parse_json_object(""))
        self.assertIsNone(parse_json_object(None))
        self.assertIsNone(parse_json_object("[1, 2]"))  # array is not an object

    def test_coerce_bool_never_treats_false_strings_as_true(self) -> None:
        self.assertIs(coerce_bool("false"), False)
        self.assertIs(coerce_bool("No"), False)
        self.assertIs(coerce_bool("true"), True)
        self.assertIs(coerce_bool(True), True)
        self.assertIs(coerce_bool(0), False)
        self.assertIsNone(coerce_bool("maybe"))
        self.assertIsNone(coerce_bool(None))


# ---------------------------------------------------------------------------
# Fail judge: strict verdicts (fail closed), Pass loss rewrites the main line
# ---------------------------------------------------------------------------
class TestFailJudgeVerdicts(unittest.TestCase):
    def _judge(self, replies: List[str]) -> tuple[FailJudge, PFPTracker]:
        tracker = PFPTracker()
        tracker.register_branch("b1", None, 0.05, "Host firewall blocks inbound TCP/22")
        judge = FailJudge(client=ScriptedClient(replies), tracker=tracker, budget_ledger=BudgetLedger())
        return judge, tracker

    def test_string_false_verdict_kills_after_appeal(self) -> None:
        judge, tracker = self._judge(
            ['{"survived": "false", "objection": "counterexample"}', '```json\n{"survived": "no"}\n```']
        )
        out = judge.falsify_branch("b1", schedule_trigger="background_sample")
        self.assertFalse(out.survived)
        self.assertTrue(out.appeal_attempted)
        self.assertEqual(tracker.get("b1").state, BranchState.FAIL)

    def test_unparseable_verdict_is_inconclusive_not_survival(self) -> None:
        judge, tracker = self._judge(["I think it's probably fine?"])
        out = judge.falsify_branch("b1", schedule_trigger="background_sample")
        self.assertTrue(out.inconclusive)
        self.assertFalse(out.survived)
        rec = tracker.get("b1")
        self.assertEqual(rec.state, BranchState.POSSIBLE)  # no kill
        self.assertEqual(rec.survived_falsifications, 0)  # no credit
        self.assertFalse(rec.appeal_used)  # no appeal consumed

    def test_inconclusive_round_blocks_pass(self) -> None:
        judge, tracker = self._judge(['{"survived": true}', "garbled", '{"survived": true}'])
        outcomes = judge.run_pass_readiness_gauntlet("b1", rounds=3)
        self.assertEqual(len(outcomes), 2)
        self.assertEqual(tracker.get("b1").state, BranchState.POSSIBLE)
        self.assertEqual(judge.verified_main_line, [])

    def test_killing_a_pass_branch_rewrites_main_line_and_reverifies_dependents(self) -> None:
        events: List[str] = []
        judge, tracker = self._judge(
            ['{"survived": true}'] * 3 + ['{"survived": false}', '{"survived": false}', '{"survived": false}']
        )
        judge.event_sink = lambda e: events.append(e.event_type)
        tracker.register_branch("dep", "b1", 0.02, "Depends on firewall finding")
        judge.run_pass_readiness_gauntlet("b1", rounds=3)
        self.assertEqual(tracker.get("b1").state, BranchState.PASS)
        self.assertTrue(any(c.startswith("[b1]") for c in judge.verified_main_line))

        out = judge.falsify_branch("b1", schedule_trigger="pre_exit")
        self.assertFalse(out.survived)
        self.assertEqual(tracker.get("b1").state, BranchState.FAIL)
        self.assertFalse(any(c.startswith("[b1]") for c in judge.verified_main_line))
        self.assertIn("revocation", events)
        # Dependent was re-verified (and died on the scripted "false")
        self.assertEqual(tracker.get("dep").state, BranchState.FAIL)


# ---------------------------------------------------------------------------
# Live HTTP client: retries, non-retryable errors, reasoning-model replies
# ---------------------------------------------------------------------------
class _FakeServer:
    def __init__(self, plan: List[tuple[int, Dict[str, str], Dict[str, Any]]]):
        self.plan = list(plan)
        self.requests: List[Dict[str, Any]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                outer.requests.append(
                    {"path": self.path, "auth": self.headers.get("Authorization"), "body": json.loads(self.rfile.read(length))}
                )
                status, headers, body = outer.plan.pop(0)
                payload = json.dumps(body).encode("utf-8")
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> "_FakeServer":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"


_OK_BODY = {
    "id": "c1",
    "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": '```json\n{"survived": true}\n```'}}],
    "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
}


class TestLiveClient(unittest.TestCase):
    def _client(self, url: str, sleeps: List[float]) -> OpenAICompatibleClient:
        client = OpenAICompatibleClient(base_url=url, api_key="k-test", max_retries=3)
        client.chat.completions._sleep = sleeps.append
        return client

    def test_retries_transient_errors_and_honors_retry_after(self) -> None:
        plan = [(503, {}, {"error": "busy"}), (429, {"Retry-After": "2"}, {"error": "slow down"}), (200, {}, _OK_BODY)]
        sleeps: List[float] = []
        with _FakeServer(plan) as srv:
            resp = self._client(srv.url, sleeps).chat.completions.create(
                messages=[{"role": "user", "content": "hi"}], temperature=0.1
            )
        self.assertEqual(len(srv.requests), 3)
        self.assertEqual(srv.requests[0]["path"], "/v1/chat/completions")
        self.assertEqual(srv.requests[0]["auth"], "Bearer k-test")
        self.assertEqual(len(sleeps), 2)
        self.assertEqual(sleeps[1], 2.0)  # Retry-After honored
        self.assertEqual(parse_json_object(resp.choices[0].message.content), {"survived": True})
        # Cost computed from live catalog pricing: 1000 in @0.06/M + 500 out @0.24/M
        self.assertAlmostEqual(resp.usage.cost_usd, 0.00006 + 0.00012, places=9)

    def test_non_retryable_error_raises_immediately_with_status(self) -> None:
        sleeps: List[float] = []
        with _FakeServer([(401, {}, {"error": "bad key"})]) as srv:
            with self.assertRaises(BackendError) as ctx:
                self._client(srv.url, sleeps).chat.completions.create(messages=[{"role": "user", "content": "x"}])
        self.assertEqual(ctx.exception.status, 401)
        self.assertIn("bad key", ctx.exception.body)
        self.assertEqual(sleeps, [])

    def test_gives_up_after_max_retries(self) -> None:
        sleeps: List[float] = []
        with _FakeServer([(500, {}, {})] * 4) as srv:
            with self.assertRaises(BackendError) as ctx:
                self._client(srv.url, sleeps).chat.completions.create(messages=[{"role": "user", "content": "x"}])
        self.assertEqual(ctx.exception.status, 500)
        self.assertEqual(len(srv.requests), 4)

    def test_reasoning_model_null_content_falls_back_to_reasoning_content(self) -> None:
        body = dict(_OK_BODY)
        body["choices"] = [{"index": 0, "message": {"role": "assistant", "content": None, "reasoning_content": "{\"x\": 1}"}}]
        with _FakeServer([(200, {}, body)]) as srv:
            resp = self._client(srv.url, []).chat.completions.create(messages=[{"role": "user", "content": "x"}])
        self.assertEqual(resp.choices[0].message.content, '{"x": 1}')


# ---------------------------------------------------------------------------
# Safety rails: true spend metering, hard cap, emergency stop
# ---------------------------------------------------------------------------
class TestSafetyRails(unittest.TestCase):
    def test_metered_client_refuses_calls_past_hard_cap(self) -> None:
        metered = MeteredClient(ScriptedClient(["{}"] * 5), hard_cap_usd=0.0025)
        metered.chat.completions.create(messages=[{"role": "user", "content": "a"}])
        metered.chat.completions.create(messages=[{"role": "user", "content": "b"}])
        metered.chat.completions.create(messages=[{"role": "user", "content": "c"}])
        with self.assertRaises(HardCapExceededError):
            metered.chat.completions.create(messages=[{"role": "user", "content": "d"}])
        stats = metered.stats()
        self.assertEqual(stats["calls"], 3)
        self.assertTrue(stats["cap_reached"])

    def test_ledger_spend_matches_independent_meter(self) -> None:
        harness = MainHarness(client=MockOpenAIClient(), starter_agents=3)
        b = harness.run(PROBLEM)["budget"]
        self.assertGreater(b["api_calls"], 10)
        self.assertAlmostEqual(b["total_spent_usd"], b["api_spend_usd"], places=5)

    def test_hard_cap_halts_run_cleanly(self) -> None:
        harness = MainHarness(client=MockOpenAIClient(), starter_agents=3, hard_cap_usd=0.05)
        summary = harness.run(PROBLEM)
        self.assertEqual(summary["halted_reason"], "hard_cap")
        self.assertEqual(summary["winners_box"], [])
        # Overshoot is bounded by the calls already in flight when the cap was crossed
        self.assertLess(summary["budget"]["api_spend_usd"], 0.05 + 0.05)
        types = [e.event_type for e in harness.arena_feed.events]
        self.assertEqual(types.count("halt"), 1)
        ascii_view = harness.render_arena(problem=PROBLEM)
        self.assertIn("RUN HALTED (hard_cap)", ascii_view)
        # Arena-level events never appear as agents
        state = TronArenaRenderer.reconstruct_state_from_stream(harness.arena_feed.events)
        self.assertNotIn("arena", state["branches"])
        self.assertNotIn("arena", {b["branch_id"] for b in EventLogMiner().mine(harness.arena_feed.events).winning_branch_shapes})

    def test_registry_emergency_stop_kills_live_run(self) -> None:
        registry = VersionRegistry()
        harness = MainHarness(client=MockOpenAIClient(), starter_agents=3, version_registry=registry)
        fired = []

        def stop_on_first_split(evt) -> None:
            if evt.event_type == "branch" and not fired:
                fired.append(evt.seq)
                registry.emergency_stop()

        harness.arena_feed.subscribe(stop_on_first_split)
        summary = harness.run(PROBLEM)
        self.assertEqual(summary["halted_reason"], "emergency_stop")
        self.assertEqual(summary["winners_box"], [])
        kills = [e for e in harness.arena_feed.events if e.event_type == "kill"]
        self.assertTrue(any(k.details.get("reason") == "emergency_stop" for k in kills))
        self.assertFalse(registry.run_in_progress)

    def test_harness_rejects_invalid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            MainHarness(client=MockOpenAIClient(), starter_agents=0)
        with self.assertRaises(ValueError):
            MainHarness(client=MockOpenAIClient(), hard_cap_usd=0)
        with self.assertRaises(ValueError):
            MainHarness(client=MockOpenAIClient()).run("   ")


class TestLedgerCharge(unittest.TestCase):
    def test_charge_uses_branch_then_reserve_then_overdraft(self) -> None:
        tracker = PFPTracker()
        tracker.register_branch("b", None, 0.010, "fork")
        ledger = BudgetLedger()
        ledger.finish_reserve_usd = 0.005
        c1 = ledger.charge(tracker, "b", 0.008)
        self.assertEqual((c1["from_branch_usd"], c1["overdraft_usd"]), (0.008, 0.0))
        c2 = ledger.charge(tracker, "b", 0.004)  # exploration: may not touch the reserve
        self.assertEqual((c2["from_branch_usd"], c2["from_finish_reserve_usd"], c2["overdraft_usd"]), (0.002, 0.0, 0.002))
        c3 = ledger.charge(tracker, "b", 0.007, allow_finish_reserve=True)
        self.assertEqual((c3["from_finish_reserve_usd"], c3["overdraft_usd"]), (0.005, 0.002))
        self.assertAlmostEqual(ledger.total_spent_usd, 0.019, places=6)
        self.assertAlmostEqual(ledger.overdraft_usd, 0.004, places=6)

    def test_tracker_refuses_duplicate_ids(self) -> None:
        tracker = PFPTracker()
        tracker.register_branch("x", None, 0.01, "fork")
        tracker.transition("x", BranchState.FAIL)
        with self.assertRaises(ValueError):
            tracker.register_branch("x", None, 0.01, "fork")  # would resurrect a dead branch
        self.assertEqual(tracker.get("x").state, BranchState.FAIL)


# ---------------------------------------------------------------------------
# Selection: frontier-only exits, distinct winners, cooperation compounds, no re-judging
# ---------------------------------------------------------------------------
class TestSelectionPhase(unittest.TestCase):
    def test_winners_are_distinct_frontier_branches(self) -> None:
        harness = MainHarness(client=MockOpenAIClient(), starter_agents=3)
        summary = harness.run(PROBLEM)
        winners = summary["winners_box"]
        self.assertGreater(len(winners), 0)
        split_parents = {r.parent for r in harness.tracker.all_records() if r.parent and r.origin != "merge"}
        for w in winners:
            self.assertNotIn(w["id"], split_parents)
            self.assertEqual(w["state"], "Pass")
            self.assertTrue(w["solution"])
        keys = [" ".join(w["fork_under_exploration"].lower().split()) for w in winners]
        self.assertEqual(len(keys), len(set(keys)))
        # Every NS-born branch faced the falsifier
        for r in harness.tracker.all_records():
            if r.origin == "ns" and r.state != BranchState.ABSORBED:
                self.assertTrue(r.survived_falsifications > 0 or r.state == BranchState.FAIL, r.id)
        # Merged children merged again in a later round (cooperation compounds)
        merges = summary["selection"]["merges_performed"]
        merged_ids = {m["child_branch_id"] for m in merges}
        self.assertTrue(any(m["parent_a_id"] in merged_ids or m["parent_b_id"] in merged_ids for m in merges))
        self.assertEqual(summary["verified_main_line"], harness.fail_judge.verified_main_line)

    def test_unchanged_pairs_are_never_rejudged(self) -> None:
        tracker = PFPTracker()
        tracker.register_branch("a", "r1", 0.01, "Inspect laptop firewall inbound TCP/22")
        tracker.register_branch("b", "r2", 0.01, "Check sshd daemon listening on laptop port 22")
        detector = AlignmentDetector(ScriptedClient(['{"core_claims": ["x"]}'] * 2 + ['{"verdict": "unrelated"}'] * 4))
        first = detector.detect_alignments(tracker.active_branches())
        second = detector.detect_alignments(tracker.active_branches())
        self.assertEqual(len(first), 1)
        self.assertEqual(second, [])
        tracker.update_branch("a", reasoning_step="new evidence")  # state changed -> judge again
        self.assertEqual(len(detector.detect_alignments(tracker.active_branches())), 1)


# ---------------------------------------------------------------------------
# Piece 8: validated dials, wired prompts, persistence, recall
# ---------------------------------------------------------------------------
class TestPiece8Hardening(unittest.TestCase):
    def test_sanitize_dials_rejects_unsafe_changes(self) -> None:
        accepted, rejected = sanitize_dials(
            {
                "merges_attracted_weight": "1.75",
                "gauntlet_rounds": 0,
                "open_critic_objections_weight": 2.0,
                "survived_falsifications_weight": float("nan"),
                "prefilter_threshold": True,
                "made_up_dial": 1,
                "age_without_testing_weight": -0.2,
            }
        )
        self.assertEqual(accepted, {"merges_attracted_weight": 1.75, "age_without_testing_weight": -0.2})
        self.assertEqual(
            set(rejected),
            {"gauntlet_rounds", "open_critic_objections_weight", "survived_falsifications_weight",
             "prefilter_threshold", "made_up_dial"},
        )

    def test_unsafe_proposition_never_reaches_registry(self) -> None:
        def hook(prompt: str):
            if "synthesize_proposition" in prompt:
                return json.dumps({"summary": "disable selection", "dial_changes": {"gauntlet_rounds": 0, "prefilter_threshold": 0.3}})
            return None

        client = InterceptingClient(hook)
        loop = SelfImprovementLoop(client=client)
        harness = MainHarness(client=client, version_registry=loop.registry)
        harness.run(PROBLEM)
        res = loop.run_post_run_improvement(harness.arena_feed.events, problem=PROBLEM, run_sandbox=False)
        self.assertIn("gauntlet_rounds", res["proposition"]["rejected_dial_changes"])
        dials = res["registered_version"]["config_dials"]
        self.assertEqual(dials["gauntlet_rounds"], 3)
        self.assertEqual(dials["prefilter_threshold"], 0.3)

    def test_registry_prompt_reaches_probe_prompts(self) -> None:
        registry = VersionRegistry(initial_prompts={"plaus_probe_hint": "Check the boring host-level causes first."})
        client = InterceptingClient()
        MainHarness(client=client, version_registry=registry).run(PROBLEM)
        probe_prompts = [p for p in client.prompts if p.startswith("probe_continuation")]
        self.assertTrue(probe_prompts)
        self.assertTrue(all("Check the boring host-level causes first." in p for p in probe_prompts))

    def test_registry_persists_and_detects_tampering(self) -> None:
        reg = VersionRegistry()
        reg.register_rewrite(
            config_dials={"gauntlet_rounds": 4},
            prompts={"plaus_probe_hint": "x"},
            proposition={"summary": "p"},
            sandbox_result={"coherent": True, "score": 9.0},
        )
        reg.pin_version(reg.all_versions()[0].version_id)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            reg.save(path)
            loaded = VersionRegistry.load(path)
            self.assertEqual([v.version_id for v in loaded.all_versions()], [v.version_id for v in reg.all_versions()])
            self.assertEqual(loaded.pinned_version_id, reg.pinned_version_id)
            self.assertEqual(loaded.get_version().version_id, reg.get_version().version_id)

            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["versions"][1]["config_dials"]["gauntlet_rounds"] = 1  # tamper
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                VersionRegistry.load(path)

            reg.begin_run()
            with self.assertRaises(Exception):
                reg.save(path)  # never mid-arena
            reg.end_run()

    def test_case_bank_dedupes_recalls_by_similarity_and_persists(self) -> None:
        loop = SelfImprovementLoop(client=MockOpenAIClient())
        harness = MainHarness(client=MockOpenAIClient())
        harness.run(PROBLEM)
        metrics = loop.miner.mine(harness.arena_feed.events)
        added = loop.record_wins_for_recall(metrics, problem=PROBLEM)
        self.assertGreater(len(added), 0)
        self.assertEqual(loop.record_wins_for_recall(metrics, problem=PROBLEM), [])  # deduped
        self.assertTrue(all(c["wins"] == 2 for c in loop.case_bank))
        self.assertEqual(loop.recall("Bake a sourdough loaf with a crisp crust."), [])
        recalled = loop.recall("SSH works laptop->desktop but not the other way", k=2)
        self.assertGreater(len(recalled), 0)
        self.assertLessEqual(len(recalled), 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.json"
            loop.save_case_bank(path)
            other = SelfImprovementLoop(client=MockOpenAIClient())
            self.assertEqual(other.load_case_bank(path), len(loop.case_bank))

    def test_cli_carries_recall_and_versions_across_processes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = ["--mock", "--quiet", "--state-dir", tmp, "--json"]
            outs = []
            for _ in range(2):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(run_arena.main(args), 0)
                outs.append(json.loads(buf.getvalue().split("--- RUN SUMMARY JSON ---", 1)[1]))
            self.assertEqual(outs[0]["seeded_starters_count"], 0)
            self.assertGreater(outs[1]["seeded_starters_count"], 0)
            reg = VersionRegistry.load(Path(tmp) / "registry.json")
            self.assertGreaterEqual(len(reg.all_versions()), 3)


# ---------------------------------------------------------------------------
# Arena renderer safety + scale
# ---------------------------------------------------------------------------
class TestArenaRendering(unittest.TestCase):
    def test_html_player_escapes_hostile_stream_content(self) -> None:
        feed = ArenaFeed()
        evil = '</script><img src=x onerror=alert(1)> __EVENTS_JSON__'
        feed.append("spawn", "b1", {"id": "b1", "state": "Possible", "fork_under_exploration": evil, "parent": None})
        with tempfile.TemporaryDirectory() as tmp:
            out = TronArenaRenderer().export_html_arena(feed.events, Path(tmp) / "a.html", problem=evil)
            html_text = out.read_text(encoding="utf-8")
        self.assertEqual(html_text.count("</script>"), 1)
        self.assertNotIn("<img src=x", html_text)
        self.assertEqual(html_text.count("const ARENA_EVENTS = "), 1)
        self.assertIn("renderUpTo", html_text)

    def test_real_scale_nine_starters_render_within_width(self) -> None:
        harness = MainHarness(client=MockOpenAIClient(), starter_agents=9)
        summary = harness.run(PROBLEM)
        self.assertEqual(len(harness.starters), 9)
        renderer = TronArenaRenderer()
        view = renderer.render_ascii(harness.arena_feed.events, problem=PROBLEM)
        ignition = [ln for ln in view.splitlines() if "IGNITION" in ln or "▼]" in ln]
        self.assertTrue(ignition)
        self.assertTrue(all(len(ln) <= renderer.width for ln in ignition))
        self.assertLessEqual(summary["budget"]["api_spend_usd"], summary["budget"]["hard_cap_usd"])


if __name__ == "__main__":
    unittest.main()
