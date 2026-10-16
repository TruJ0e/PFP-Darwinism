"""
End-to-end and unit tests for PFP Darwinism Phase 1 (Pieces 1–3 + Backend),
covering both Pass 1 literal requirements and Pass 2 intent & refinement guarantees.
"""

from __future__ import annotations

import os
import unittest

from pfp import (
    BranchState,
    BudgetLedger,
    CertaintyWeights,
    InvalidStateTransitionError,
    IssueDetector,
    MainHarness,
    MockOpenAIClient,
    OpenAICompatibleClient,
    PFPTracker,
    create_client,
)


class TestBackendInterface(unittest.TestCase):
    def test_env_configuration_and_mock_fallback(self) -> None:
        old_url = os.environ.get("NEBIUS_BASE_URL")
        old_model = os.environ.get("NEBIUS_MODEL")
        old_key = os.environ.get("NEBIUS_API_KEY")
        try:
            os.environ["NEBIUS_BASE_URL"] = "https://custom.nebius.example/v1"
            os.environ["NEBIUS_MODEL"] = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
            os.environ.pop("NEBIUS_API_KEY", None)

            client = create_client()
            self.assertIsInstance(client, MockOpenAIClient)
            self.assertEqual(client.base_url, "https://custom.nebius.example/v1")
            self.assertEqual(client.model, "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B")

            resp = client.chat.completions.create(
                messages=[{"role": "user", "content": "probe_continuation depth=0"}],
                temperature=0.9,
                logprobs=True,
                n=3,
            )
            self.assertEqual(len(resp.choices), 3)
            self.assertGreater(resp.usage.total_tokens, 0)
            self.assertGreater(resp.usage.cost_usd, 0.0)

            # When NEBIUS_API_KEY is present, create_client() returns OpenAICompatibleClient
            os.environ["NEBIUS_API_KEY"] = "test-live-key-placeholder"
            live_client = create_client()
            self.assertIsInstance(live_client, OpenAICompatibleClient)
            self.assertEqual(live_client.base_url, "https://custom.nebius.example/v1")
            self.assertEqual(live_client.model, "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B")
        finally:
            if old_url is None:
                os.environ.pop("NEBIUS_BASE_URL", None)
            else:
                os.environ["NEBIUS_BASE_URL"] = old_url
            if old_model is None:
                os.environ.pop("NEBIUS_MODEL", None)
            else:
                os.environ["NEBIUS_MODEL"] = old_model
            if old_key is None:
                os.environ.pop("NEBIUS_API_KEY", None)
            else:
                os.environ["NEBIUS_API_KEY"] = old_key


class TestPiece3PFPTracker(unittest.TestCase):
    def test_ledger_fields_and_one_way_transitions(self) -> None:
        tracker = PFPTracker()
        rec = tracker.register_branch(
            branch_id="b1",
            parent=None,
            budget_remaining=0.05,
            fork_under_exploration="firewall_check",
        )
        self.assertEqual(rec.state, BranchState.POSSIBLE)
        self.assertEqual(rec.fork_under_exploration, "firewall_check")
        self.assertEqual(len(rec.state_history), 1)

        # Possible -> Pass is valid
        tracker.transition("b1", BranchState.PASS, reason="survived_3_gauntlet_rounds")
        self.assertEqual(tracker.get("b1").state, BranchState.PASS)

        # Pass -> Absorbed is valid ("anything -> absorbed on merge")
        tracker.transition("b1", BranchState.ABSORBED, reason="merged_into_b3")
        self.assertEqual(tracker.get("b1").state, BranchState.ABSORBED)
        self.assertEqual(len(tracker.get("b1").state_history), 3)

        # Absorbed -> Possible is illegal (no resurrection)
        with self.assertRaises(InvalidStateTransitionError):
            tracker.transition("b1", BranchState.POSSIBLE)

        # Possible -> Fail is valid, and Fail -> Pass/Possible is illegal
        tracker.register_branch(
            branch_id="b2",
            parent="b1",
            budget_remaining=0.02,
            fork_under_exploration="sshd_check",
            depth=1,
        )
        self.assertIn("firewall_check", tracker.known_forks_in_lineage("b2"))
        tracker.transition("b2", BranchState.FAIL)
        self.assertEqual(tracker.get("b2").state, BranchState.FAIL)
        with self.assertRaises(InvalidStateTransitionError):
            tracker.transition("b2", BranchState.POSSIBLE)
        with self.assertRaises(InvalidStateTransitionError):
            tracker.transition("b2", BranchState.PASS)


class TestPiece1IssueDetector(unittest.TestCase):
    def test_plaus_divergence_vs_clustering_and_ns(self) -> None:
        client = MockOpenAIClient()
        detector = IssueDetector(client, k_probes=4, depth_dial=4)

        # At depth=0, high-temp K continuations genuinely split -> Plaus issue discovered
        res_split = detector.detect_plaus(
            problem="SSH works laptop->desktop but not desktop->laptop.",
            parent_reasoning="Initial observation.",
            depth=0,
        )
        self.assertTrue(res_split.should_split)
        self.assertEqual(res_split.trigger, "plaus")
        self.assertGreater(len(res_split.distinct_forks), 1)
        self.assertGreater(res_split.mean_token_entropy, 0.0)

        # Cross-lineage novelty exhaustion filters out already-explored forks
        already_seen = [f.fork_description for f in res_split.distinct_forks]
        res_filtered = detector.detect_plaus(
            problem="SSH works laptop->desktop but not desktop->laptop.",
            parent_reasoning="Initial observation.",
            depth=0,
            known_forks=already_seen,
        )
        self.assertTrue(res_filtered.novelty_exhausted)

        # At depth=2, continuations cluster -> no issue, merge and flow straight
        res_cluster = detector.detect_plaus(
            problem="SSH works laptop->desktop but not desktop->laptop.",
            parent_reasoning="Verified solution narrowed to firewall rule.",
            depth=2,
        )
        self.assertFalse(res_cluster.should_split)
        self.assertIsNotNone(res_cluster.merged_continuation)
        self.assertGreaterEqual(res_cluster.distinct_forks[0].cluster_size, 1)

        # NS blind variation trigger
        res_ns = detector.trigger_ns(
            problem="SSH works laptop->desktop but not desktop->laptop.",
            parent_reasoning="Checking standard ports.",
            depth=0,
            fanout_count=2,
        )
        self.assertTrue(res_ns.should_split)
        self.assertEqual(res_ns.trigger, "ns")
        self.assertGreaterEqual(len(res_ns.distinct_forks), 1)

    def test_certainty_ranking_dials(self) -> None:
        client = MockOpenAIClient()
        weights = CertaintyWeights(
            survived_falsifications=3.0,
            merges_attracted=1.5,
            open_critic_objections=-1.0,
            age_without_testing=-0.2,
        )
        detector = IssueDetector(client, weights=weights)
        high_c = detector.compute_certainty(
            survived_falsifications=2,
            merges_attracted=1,
            open_critic_objections=0,
            seconds_since_last_test=0.0,
        )
        low_c = detector.compute_certainty(
            survived_falsifications=0,
            merges_attracted=0,
            open_critic_objections=2,
            seconds_since_last_test=5.0,
        )
        self.assertGreater(high_c, low_c)


class TestPiece2BranchSpawnerAndBudget(unittest.TestCase):
    def test_strict_budget_conservation_and_floor_pruning(self) -> None:
        client = MockOpenAIClient()
        ledger = BudgetLedger(
            hard_cap_usd=2.00,
            finish_reserve_ratio=0.125,
            min_branch_floor_usd=0.002,
        )
        root = ledger.size_problem_once(client, "Diagnose asymmetric SSH.")
        self.assertGreater(root, 0.0)
        self.assertLessEqual(root, 2.00)
        self.assertAlmostEqual(ledger.finish_reserve_usd, root * 0.125, places=5)

        # Tight budget -> slower NS cadence
        fast_cadence = ledger.compute_ns_cadence(0.05)
        slow_cadence = ledger.compute_ns_cadence(0.003)
        self.assertGreater(slow_cadence, fast_cadence)

        # Certainty-weighted sibling split with depth decay (x0.5), floor, and exact conservation
        parent_budget = 0.04
        splits = ledger.split_among_siblings(parent_budget, [2.0, 0.0, -1.0], apply_depth_decay=True)
        self.assertEqual(len(splits), 3)
        self.assertGreater(splits[0], splits[1])
        self.assertGreater(splits[1], splits[2])
        self.assertTrue(all(s >= ledger.min_branch_floor_usd for s in splits))
        self.assertAlmostEqual(sum(splits), parent_budget * 0.5, places=5)

        # When decayed pool (0.5 * 0.009 = 0.0045) can only fund 2 children at floor 0.002,
        # asking for 4 siblings trims to 2 siblings and conserves pool!
        tight_splits = ledger.split_among_siblings(0.009, [3.0, 2.0, 1.0, 0.0], apply_depth_decay=True)
        self.assertEqual(len(tight_splits), 2)
        self.assertTrue(all(s >= ledger.min_branch_floor_usd for s in tight_splits))
        self.assertAlmostEqual(sum(tight_splits), 0.0045, places=5)

        # Finish reserve protection: branches with open critic objections cannot touch finish reserve
        tracker = PFPTracker()
        unready = tracker.register_branch("u1", None, 0.0005, "fork_a", certainty_score=1.0)
        tracker.update_branch("u1", open_critic_objections=1)
        self.assertEqual(ledger.draw_from_finish_reserve(tracker.get("u1"), 0.002), 0.0)

        # Near-conclusion branch (high certainty, 0 open critic objections) CAN draw from finish reserve
        ready = tracker.register_branch("r1", None, 0.0005, "fork_b", certainty_score=1.0)
        tracker.update_branch("r1", open_critic_objections=0)
        drawn = ledger.draw_from_finish_reserve(tracker.get("r1"), 0.002)
        self.assertEqual(drawn, 0.002)

    def test_end_to_end_parallel_harness_and_budget_cap(self) -> None:
        client = MockOpenAIClient()
        harness = MainHarness(
            client=client,
            starter_agents=3,
            hard_cap_usd=2.00,
            depth_dial=3,
            k_probes=3,
        )
        result = harness.run("A small business's SSH works laptop->desktop but not desktop->laptop.")
        self.assertGreater(result["total_branches_spawned"], 3)
        self.assertEqual(len(result["scoreboard"]), result["total_branches_spawned"])
        self.assertGreater(result["events_count"], 0)

        # Verify strict budget conservation. total_spent is the TRUE API spend (it must match
        # the independent metering client); only overdraft — spend no budget or reserve
        # covered — may exceed the root, and it is reported, never hidden.
        b = result["budget"]
        self.assertAlmostEqual(b["total_spent_usd"], b["api_spend_usd"], places=5)
        total_accounted = (
            b["total_spent_usd"]
            - b["overdraft_usd"]
            + b["unspent_branch_budgets_usd"]
            + b["finish_reserve_remaining_usd"]
        )
        self.assertLessEqual(total_accounted, b["root_budget_usd"] + 1e-4)
        self.assertLessEqual(b["root_budget_usd"], b["hard_cap_usd"])
        self.assertLessEqual(b["api_spend_usd"], b["hard_cap_usd"])
        # Demo scale (3 starters) runs (almost) entirely inside its root budget; the exact
        # overdraft varies with thread interleaving, so bound it rather than pin it
        self.assertLessEqual(b["overdraft_usd"], 0.10 * b["root_budget_usd"])

        # Verify no alive branch died mid-thought (all alive branches reached conclusion)
        for rec in harness.tracker.active_branches():
            has_conclusion = any("[Conclusion]" in step for step in rec.reasoning_trace)
            self.assertTrue(has_conclusion, f"Branch {rec.id} did not conclude cleanly")


if __name__ == "__main__":
    unittest.main()
