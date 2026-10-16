"""
Tests for Phase 4 (Piece 8: Self-Improvement Loop & Failsafe Version Registry).
"""

from __future__ import annotations

import unittest

from pfp import (
    EventLogMiner,
    MainHarness,
    MockOpenAIClient,
    RunBoundarySwitchError,
    SelfImprovementLoop,
    VersionRegistry,
)


class TestPiece8SelfImprovementAndRegistry(unittest.TestCase):
    def test_event_log_mining_and_self_improvement_loop(self) -> None:
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

        # 1. Event-log mining: winning branch shapes, merge patterns, falsifier angles, spend efficiency
        miner = EventLogMiner()
        metrics = miner.mine(harness.arena_feed.events)

        self.assertGreater(metrics.total_events, 0)
        self.assertGreater(len(metrics.winning_branch_shapes), 0)
        self.assertGreater(metrics.merge_patterns["total_merges"], 0)
        self.assertIn("merge_success_rate", metrics.merge_patterns)
        self.assertGreater(len(metrics.falsifier_angles), 0)
        self.assertGreater(metrics.spend_efficiency["total_spent_usd"], 0.0)
        self.assertGreater(metrics.spend_efficiency["winner_certainty_per_usd"], 0.0)

        # 2. Full post-run self-improvement cycle (mining -> recall -> self-assessment -> anti-model -> proposition -> sandbox -> registry)
        loop = SelfImprovementLoop(client=client)
        baseline_vid = loop.registry.active_version_id
        res = loop.run_post_run_improvement(harness.arena_feed.events, problem=problem, run_sandbox=True)

        self.assertGreater(res["recall_cases_added"], 0)
        self.assertGreater(len(loop.case_bank), 0)
        self.assertTrue(res["sandbox_result"]["coherent"])
        self.assertNotEqual(res["registered_version"]["version_id"], baseline_vid)
        self.assertEqual(len(res["registered_version"]["sha256_hash"]), 64)
        self.assertEqual(loop.registry.active_version_id, res["registered_version"]["version_id"])

    def test_version_registry_auto_revert_pin_restore_and_run_boundary(self) -> None:
        reg = VersionRegistry(degradation_threshold=0.15)
        v1_id = reg.active_version_id

        # Register v2 rewrite between runs
        v2 = reg.register_rewrite(
            config_dials={"survived_falsifications_weight": 3.5},
            prompts={"plaus_probe_hint": "v2 prompt"},
            proposition={"summary": "v2 test rewrite"},
            sandbox_result={"coherent": True, "score": 12.0, "baseline_score": 10.0},
            adopt=True,
        )
        self.assertEqual(reg.active_version_id, v2.version_id)

        # Run-boundary enforcement: while a live run is active, switching/pinning/restoring/registering is blocked
        reg.begin_run()
        self.assertTrue(reg.run_in_progress)
        with self.assertRaises(RunBoundarySwitchError):
            reg.restore_version(v1_id)
        with self.assertRaises(RunBoundarySwitchError):
            reg.pin_version(v1_id)
        with self.assertRaises(RunBoundarySwitchError):
            reg.register_rewrite(
                config_dials={"survived_falsifications_weight": 4.0},
                prompts={},
                proposition={"summary": "mid-run illegal rewrite"},
                sandbox_result={"coherent": True},
                adopt=True,
            )

        # Emergency stop ends the live run immediately and rolls back for the next run
        stop_info = reg.emergency_stop()
        self.assertTrue(stop_info["emergency_stopped"])
        self.assertFalse(reg.run_in_progress)
        self.assertEqual(reg.active_version_id, v1_id)

        # Manual restore to v2 between runs
        reg.restore_version(v2.version_id)
        self.assertEqual(reg.active_version_id, v2.version_id)

        # Auto-revert on measured degradation (score 6.0 vs baseline 10.0 -> 40% degradation > 15% threshold)
        rev = reg.evaluate_and_auto_revert(measured_score=6.0, baseline_score=10.0)
        self.assertTrue(rev["reverted"])
        self.assertEqual(rev["reverted_from"], v2.version_id)
        self.assertEqual(rev["reverted_to"], v1_id)
        self.assertEqual(reg.active_version_id, v1_id)
        self.assertEqual(len(reg.failed_experiments), 1)

        # Manual pin locks active version across subsequent rewrites
        reg.pin_version(v1_id)
        v3 = reg.register_rewrite(
            config_dials={"survived_falsifications_weight": 5.0},
            prompts={},
            proposition={"summary": "v3 while v1 is pinned"},
            sandbox_result={"coherent": True, "score": 15.0, "baseline_score": 10.0},
            adopt=True,
        )
        self.assertEqual(reg.get_version().version_id, v1_id)
        reg.unpin()
        reg.restore_version(v3.version_id)
        self.assertEqual(reg.get_version().version_id, v3.version_id)

    def test_multi_run_seeded_evolution_and_registry_lifecycle(self) -> None:
        client = MockOpenAIClient()
        reg = VersionRegistry()
        loop = SelfImprovementLoop(client=client, registry=reg)
        problem = "A small business's SSH works laptop->desktop but not desktop->laptop."

        # Run 1 (unseeded baseline)
        h1 = MainHarness(client=client, starter_agents=3, version_registry=reg, seed_cases=loop.case_bank)
        s1 = h1.run(problem)
        self.assertEqual(s1["seeded_starters_count"], 0)
        self.assertFalse(reg.run_in_progress)

        # Post-run improvement mines Run 1, populates recall case_bank, and adopts v2 in registry
        imp = loop.run_post_run_improvement(h1.arena_feed.events, problem=problem, run_sandbox=True)
        v2_id = imp["active_version_id"]
        self.assertGreater(len(loop.case_bank), 0)

        # Run 2 starts seeded with recalled wins from Run 1 and uses active v2 configuration
        h2 = MainHarness(client=client, starter_agents=3, version_registry=reg, seed_cases=loop.case_bank)
        s2 = h2.run(problem)
        self.assertEqual(s2["active_version_id"], v2_id)
        self.assertGreater(s2["seeded_starters_count"], 0)
        self.assertGreater(len(s2["winners_box"]), 0)


if __name__ == "__main__":
    unittest.main()
