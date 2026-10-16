"""
Tests for Phase 3 (Piece 7: Arena Feed & Tron-style Renderer).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pfp import (
    AppendOnlyViolationError,
    ArenaFeed,
    BranchState,
    MainHarness,
    MockOpenAIClient,
    TronArenaRenderer,
)


class TestPiece7ArenaFeedAndRenderer(unittest.TestCase):
    def test_append_only_event_stream_and_tron_renderer(self) -> None:
        client = MockOpenAIClient()
        harness = MainHarness(
            client=client,
            starter_agents=3,
            hard_cap_usd=2.00,
            depth_dial=3,
            k_probes=3,
            gauntlet_rounds=3,
            enable_selection=True,
        )
        problem = "A small business's SSH works laptop->desktop but not desktop->laptop."
        harness.run(problem)

        # Also exercise a revocation event so the stream contains all canonical event types
        passed_branches = [r for r in harness.tracker.all_records() if r.state == BranchState.PASS]
        if len(passed_branches) >= 2:
            harness.fail_judge.revoke_and_reverify_dependents(
                passed_branches[-1].id,
                demote_to=BranchState.POSSIBLE,
                reason="late_counterexample_test",
            )

        feed = harness.arena_feed
        events = feed.events
        self.assertGreater(len(events), 10)

        # 1. Verify every event is timestamped and carries a full branch snapshot
        for idx, evt in enumerate(events):
            self.assertEqual(evt.seq, idx + 1)
            self.assertGreater(evt.timestamp, 0.0)
            self.assertIn("id", evt.snapshot)
            self.assertIn("state", evt.snapshot)
            self.assertIn("time_alive", evt.snapshot)
            self.assertIn("budget_remaining", evt.snapshot)

        # 2. Verify required event types appear in the stream
        seen_types = {e.event_type for e in events}
        for req_type in ("spawn", "branch", "merge", "fail", "pass", "budget_change"):
            self.assertIn(req_type, seen_types, f"Missing expected event type: {req_type}")

        # 3. Verify append-only immutability (history never rewrites)
        with self.assertRaises(AppendOnlyViolationError):
            events[0] = events[-1]  # type: ignore[index]
        with self.assertRaises(AppendOnlyViolationError):
            del events[0]  # type: ignore[arg-type]
        with self.assertRaises(AppendOnlyViolationError):
            events.pop()  # type: ignore[attr-defined]

        # 4. Verify Tron-style renderer reconstructs state purely from the stream (live & JSONL replay)
        renderer = TronArenaRenderer()
        live_ascii = renderer.render_ascii(feed.events, problem=problem)
        self.assertIn("TRON ARENA", live_ascii)
        self.assertIn("IGNITION (TOP DROP)", live_ascii)
        self.assertIn("2D TRON LIGHT-CYCLE GRID", live_ascii)
        self.assertIn("VERIFIED MAIN LINE", live_ascii)
        self.assertIn("SCOREBOARD HUD", live_ascii)
        self.assertIn("WINNER'S BOX", live_ascii)

        # Save to JSONL, reload, and verify replay produces identical output from the same stream
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = Path(tmpdir) / "arena_stream.jsonl"
            html_path = Path(tmpdir) / "arena_replay.html"
            feed.save_jsonl(jsonl_path)
            reloaded_feed = ArenaFeed.load_jsonl(jsonl_path)
            replay_ascii = renderer.render_ascii(reloaded_feed.events, problem=problem)
            self.assertEqual(live_ascii, replay_ascii)

            frames = renderer.render_replay_frames(reloaded_feed.events, problem=problem, step_size=10)
            self.assertGreaterEqual(len(frames), 2)
            self.assertEqual(frames[-1], replay_ascii)

            renderer.export_html_arena(reloaded_feed.events, html_path, problem=problem)
            self.assertTrue(html_path.exists())
            html_text = html_path.read_text(encoding="utf-8")
            self.assertIn("WINNER'S BOX", html_text)
            self.assertIn("SCOREBOARD HUD", html_text)
            self.assertIn("renderUpTo", html_text)
            self.assertIn("seqSlider", html_text)


if __name__ == "__main__":
    unittest.main()
