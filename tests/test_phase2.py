"""
Tests for Phase 2 (Pieces 4-6: Selection — Alignment Detector, Merge Synthesizer, Fail Judge).
"""

from __future__ import annotations

import unittest

from pfp import (
    AlignmentDetector,
    AlignmentVerdict,
    BranchState,
    BudgetLedger,
    FailJudge,
    FalsifierAngle,
    InvalidStateTransitionError,
    MainHarness,
    MergeSynthesizer,
    MockOpenAIClient,
    PFPTracker,
)


class TestPiece4AlignmentDetector(unittest.TestCase):
    def test_two_stage_embedding_prefilter_and_super_judge(self) -> None:
        client = MockOpenAIClient()
        tracker = PFPTracker()

        # Branch A: firewall hypothesis
        ba = tracker.register_branch(
            "b_firewall",
            parent="root_1",
            budget_remaining=0.01,
            fork_under_exploration="Inspect stateful firewall and inbound port 22 rules on the laptop",
        )
        tracker.update_branch(
            "b_firewall",
            reasoning_step="Hypothesis A: desktop->laptop inbound TCP/22 blocked by laptop firewall.",
        )

        # Branch B: sshd daemon hypothesis (complementary / overlapping on laptop TCP/22 ingress)
        bb = tracker.register_branch(
            "b_sshd",
            parent="root_2",
            budget_remaining=0.01,
            fork_under_exploration="Verify OpenSSH server daemon is installed and listening on the laptop",
        )
        tracker.update_branch(
            "b_sshd",
            reasoning_step="Hypothesis B: sshd service disabled or unlistening on TCP/22 on laptop.",
        )

        # Branch C: contradictory fail2ban hypothesis (claims port 22 is open & listening, fail2ban drops)
        bc = tracker.register_branch(
            "b_fail2ban",
            parent="root_3",
            budget_remaining=0.01,
            fork_under_exploration="Blind variation: Check fail2ban ban list where port 22 is open and listening",
        )
        tracker.update_branch(
            "b_fail2ban",
            reasoning_step="Contradict: port 22 is open and listening, fail2ban lockout drops desktop IP.",
        )

        # Branch D: unrelated client-side IPv6 DNS scope hypothesis
        bd = tracker.register_branch(
            "b_ipv6",
            parent="root_4",
            budget_remaining=0.01,
            fork_under_exploration="Blind variation: Test IPv6 link-local scope ID mismatch during DNS resolution",
        )
        tracker.update_branch(
            "b_ipv6",
            reasoning_step="NS Mutation 2: Dual-stack AAAA DNS scope mismatch on client.",
        )

        detector = AlignmentDetector(client, prefilter_threshold=0.20)

        # Stage 1: Embedding pre-filter recalls SSH port 22 host pairs (A-B, A-C, B-C) above unrelated D
        candidates = detector.embedding_prefilter([ba, bb, bc, bd])
        passed_pairs = {(c.branch_a_id, c.branch_b_id) for c in candidates if c.passed_prefilter}
        self.assertIn(("b_firewall", "b_sshd"), passed_pairs)
        self.assertIn(("b_firewall", "b_fail2ban"), passed_pairs)

        # Stage 2: Super judge distinguishes overlap vs contradictory vs unrelated
        res_ab = detector.judge_pair(ba, bb)
        self.assertEqual(res_ab.verdict, AlignmentVerdict.OVERLAP)
        self.assertTrue(res_ab.is_alignable)
        self.assertGreater(len(res_ab.overlapping_claims), 0)

        res_ac = detector.judge_pair(ba, bc)
        self.assertEqual(res_ac.verdict, AlignmentVerdict.CONTRADICTORY)
        self.assertTrue(res_ac.is_contradictory)
        self.assertGreater(len(res_ac.contradictory_claims), 0)

        res_ad = detector.judge_pair(ba, bd)
        self.assertEqual(res_ad.verdict, AlignmentVerdict.UNRELATED)
        self.assertFalse(res_ad.is_alignable)
        self.assertFalse(res_ad.is_contradictory)


class TestPiece5MergeSynthesizer(unittest.TestCase):
    def test_merge_unification_provenance_and_budget_transfer(self) -> None:
        client = MockOpenAIClient(force_cost_scale=1.0)
        tracker = PFPTracker()
        ledger = BudgetLedger()

        tracker.register_branch(
            "p1",
            parent="root",
            budget_remaining=0.025,
            fork_under_exploration="Inspect stateful firewall and inbound port 22 rules on the laptop",
            certainty_score=1.5,
            depth=1,
        )
        tracker.update_branch("p1", reasoning_step="Firewall rule check on laptop TCP/22")

        tracker.register_branch(
            "p2",
            parent="root",
            budget_remaining=0.030,
            fork_under_exploration="Verify OpenSSH server daemon is installed and listening on the laptop",
            certainty_score=2.0,
            depth=1,
        )
        tracker.update_branch("p2", reasoning_step="sshd listener check on laptop TCP/22")

        events = []
        synthesizer = MergeSynthesizer(
            client=client,
            tracker=tracker,
            budget_ledger=ledger,
            event_sink=events.append,
        )
        res = synthesizer.merge_branches("p1", "p2", "m_child_1")

        # 1. Child starts Possible, NEVER Passed
        child = tracker.get("m_child_1")
        self.assertEqual(child.state, BranchState.POSSIBLE)
        self.assertEqual(res.child_state, BranchState.POSSIBLE)

        # 2. Both parents are absorbed (terminal) and their budgets are zeroed
        p1_after = tracker.get("p1")
        p2_after = tracker.get("p2")
        self.assertEqual(p1_after.state, BranchState.ABSORBED)
        self.assertEqual(p2_after.state, BranchState.ABSORBED)
        self.assertEqual(p1_after.budget_remaining, 0.0)
        self.assertEqual(p2_after.budget_remaining, 0.0)
        with self.assertRaises(InvalidStateTransitionError):
            tracker.transition("p1", BranchState.POSSIBLE)

        # 3. Unused parent budgets transferred to child (born richer than either parent)
        self.assertAlmostEqual(
            child.budget_remaining + res.synthesis_cost_usd,
            0.025 + 0.030,
            places=5,
        )
        self.assertGreater(child.budget_remaining, 0.030)

        # 4. Provenance preserved (which parent contributed what)
        self.assertIn("p1", child.merge_provenance)
        self.assertIn("p2", child.merge_provenance)
        self.assertGreater(len(child.merge_provenance["p1"]), 0)
        self.assertGreater(len(child.merge_provenance["p2"]), 0)
        self.assertEqual(child.merge_parents, ["p1", "p2"])
        self.assertGreaterEqual(child.merges_attracted, 1)


class TestPiece6FailJudge(unittest.TestCase):
    def test_rotated_angles_gauntlet_appeal_and_revocation(self) -> None:
        client = MockOpenAIClient(force_cost_scale=1.0)
        tracker = PFPTracker()
        judge = FailJudge(client, tracker, gauntlet_rounds=3)

        # 1. Verify attack style rotation (logic -> counterexample -> assumption_audit)
        a1 = judge.next_angle()
        a2 = judge.next_angle()
        a3 = judge.next_angle()
        self.assertEqual([a1, a2, a3], [FalsifierAngle.LOGIC, FalsifierAngle.COUNTEREXAMPLE, FalsifierAngle.ASSUMPTION_AUDIT])

        # 2. 3-round Pass-readiness gauntlet on a solid branch -> transitions Possible -> Pass with provenance
        tracker.register_branch(
            "b_solid",
            parent=None,
            budget_remaining=0.05,
            fork_under_exploration="Enable sshd and allow inbound TCP/22 on laptop firewall",
            certainty_score=3.0,
        )
        outcomes = judge.run_pass_readiness_gauntlet("b_solid")
        self.assertEqual(len(outcomes), 3)
        self.assertTrue(all(o.survived for o in outcomes))
        solid_rec = tracker.get("b_solid")
        self.assertEqual(solid_rec.state, BranchState.PASS)
        self.assertEqual(len(solid_rec.pass_provenance), 3)
        self.assertEqual(
            {p.falsifier_angle for p in solid_rec.pass_provenance},
            {"logic", "counterexample", "assumption_audit"},
        )

        # 3. One appeal per kill with a different judge:
        #    Branch with `force_fail=true` + `overturn_appeal=true` fails primary judge,
        #    appeals to secondary judge (`super_judge_appeal_secondary`), and overturns the kill!
        tracker.register_branch(
            "b_appealable",
            parent=None,
            budget_remaining=0.05,
            fork_under_exploration="force_fail=true overturn_appeal=true salvageable edge hypothesis",
        )
        out_appeal = judge.falsify_branch("b_appealable", schedule_trigger="background_sample")
        self.assertTrue(out_appeal.survived)
        self.assertTrue(out_appeal.appeal_attempted)
        self.assertTrue(out_appeal.appeal_overturned)
        self.assertNotEqual(out_appeal.judge_id, out_appeal.appeal_judge_id)
        self.assertTrue(tracker.get("b_appealable").appeal_used)

        # A second kill on the same branch has NO appeal left -> dies immediately (Fail)
        out_second_kill = judge.falsify_branch(
            "b_appealable",
            schedule_trigger="background_sample",
            extra_context="force_fail=true",
        )
        self.assertFalse(out_second_kill.survived)
        self.assertFalse(out_second_kill.appeal_attempted)
        self.assertEqual(tracker.get("b_appealable").state, BranchState.FAIL)

        # 4. Revokable Pass + dependent re-verification:
        #    Create dependent child of `b_solid`, then revoke `b_solid` and force dependent re-verification
        tracker.register_branch(
            "b_dep_child",
            parent="b_solid",
            budget_remaining=0.04,
            fork_under_exploration="Dependent child relying on b_solid",
        )
        rev_res = judge.revoke_and_reverify_dependents(
            "b_solid",
            demote_to=BranchState.POSSIBLE,
            reason="late_counterexample_at_t40s",
            force_fail_dependents=True,
        )
        self.assertEqual(tracker.get("b_solid").state, BranchState.POSSIBLE)
        self.assertIn("b_dep_child", rev_res["dependents_checked"])
        self.assertEqual(tracker.get("b_dep_child").state, BranchState.FAIL)


class TestPhase2EndToEndSelection(unittest.TestCase):
    def test_end_to_end_align_merge_falsify_pass_fail(self) -> None:
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
        summary = harness.run("A small business's SSH works laptop->desktop but not desktop->laptop.")

        selection = summary["selection"]
        self.assertGreater(len(selection["alignments_evaluated"]), 0)
        self.assertGreater(len(selection["merges_performed"]), 0)
        self.assertGreater(selection["falsifications_count"], 0)
        self.assertGreater(len(summary["winners_box"]), 0)

        states = {r["state"] for r in summary["scoreboard"]}
        self.assertIn("Pass", states)
        self.assertIn("Fail", states)
        self.assertIn("Absorbed", states)

        # Verify DAG merge lineage novelty includes both merged parents' forks
        for m in selection["merges_performed"]:
            cid = m["child_branch_id"]
            pa = harness.tracker.get(m["parent_a_id"]).fork_under_exploration
            pb = harness.tracker.get(m["parent_b_id"]).fork_under_exploration
            known = harness.tracker.known_forks_in_lineage(cid)
            self.assertIn(pa, known)
            self.assertIn(pb, known)


if __name__ == "__main__":
    unittest.main()
