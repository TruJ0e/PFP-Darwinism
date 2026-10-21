"""v0.42: selection coverage scales with the population; a quiet round no longer ends selection."""

from __future__ import annotations

import threading
from typing import List

from pfp import MainHarness, MockOpenAIClient, PFPTracker
from pfp.alignment_detector import AlignmentDetector, AlignmentJudgeResult, AlignmentVerdict

from test_hardening import _response

PROBLEM = "Sandbox verification: SSH works laptop->desktop but not desktop->laptop."

FIREWALL = "laptop host firewall blocks inbound tcp port 22 from the desktop; check ufw status and sshd listening"
ROUTING = "desktop dns resolves the laptop name to a stale ipv6 link_local address across the wifi vlan routing"


def _population(n_firewall: int, n_routing: int = 0) -> PFPTracker:
    tracker = PFPTracker()
    for i in range(n_firewall):
        tracker.register_branch(f"fw{i:02d}", None, 0.05, f"{FIREWALL} variant {i}")
    for i in range(n_routing):
        tracker.register_branch(f"rt{i:02d}", None, 0.05, f"{ROUTING} variant {i}")
    return tracker


class _CountingClient:
    """Thread-safe scripted client: claims / verdict replies by prompt tag, every call counted."""

    def __init__(self, verdict: str = "unrelated") -> None:
        self.verdict = verdict
        self.tags: List[str] = []
        self._lock = threading.Lock()
        outer = self

        class _C:
            def create(self, messages, **kwargs):
                user = messages[-1]["content"]
                with outer._lock:
                    outer.tags.append(user.split(" ", 1)[0])
                if user.startswith("extract_claims"):
                    return _response('{"core_claims": ["c"]}', cost=0.002)
                return _response('{"verdict": "%s"}' % outer.verdict, cost=0.001)

        self.chat = type("Chat", (), {})()
        self.chat.completions = _C()


def test_plan_covers_the_population_not_a_constant_eight() -> None:
    # Before v0.42 every round judged at most 8 pairs whatever the population: run 5 put 174
    # branches through selection and judged 8 alignments in total
    tracker = _population(30)
    det = AlignmentDetector(_CountingClient())
    plan = det.plan_pairs(tracker.active_branches())
    assert len(plan) == 15  # a full matching: every branch in exactly one within-cluster pair
    seen = [bid for p in plan for bid in (p.branch_a_id, p.branch_b_id)]
    assert len(seen) == len(set(seen)) == 30
    assert det.last_unjudged_remaining > 0  # the rest waits for later rounds


def test_cluster_champions_meet_across_clusters() -> None:
    tracker = _population(4, 4)
    tracker.update_branch("rt02", certainty_score=2.0)
    tracker.update_branch("fw03", certainty_score=1.0)
    det = AlignmentDetector(_CountingClient(), prefilter_threshold=0.0, cluster_threshold=0.5)
    plan = det.plan_pairs(tracker.active_branches())
    within = [p for p in plan if p.branch_a_id[:2] == p.branch_b_id[:2]]
    cross = [p for p in plan if p.branch_a_id[:2] != p.branch_b_id[:2]]
    assert len(within) == 4
    assert [{p.branch_a_id, p.branch_b_id} for p in cross] == [{"rt02", "fw03"}]


def test_judged_pairs_are_never_planned_twice() -> None:
    tracker = _population(6)
    det = AlignmentDetector(_CountingClient())
    first = det.plan_pairs(tracker.active_branches())
    det.judge_planned_pairs(tracker.active_branches(), first)
    second = det.plan_pairs(tracker.active_branches())
    key = lambda p: frozenset((p.branch_a_id, p.branch_b_id))  # noqa: E731
    assert second and not ({key(p) for p in first} & {key(p) for p in second})


def test_concurrent_judging_extracts_each_branch_once_and_keeps_plan_order() -> None:
    tracker = _population(4, 4)
    client = _CountingClient()
    det = AlignmentDetector(client, prefilter_threshold=0.0)
    plan = det.plan_pairs(tracker.active_branches())
    results = det.judge_planned_pairs(tracker.active_branches(), plan, max_workers=8)
    assert [(r.branch_a_id, r.branch_b_id) for r in results] == [(p.branch_a_id, p.branch_b_id) for p in plan]
    involved = {bid for p in plan for bid in (p.branch_a_id, p.branch_b_id)}
    assert client.tags.count("extract_claims") == len(involved)
    assert client.tags.count("alignment_judge") == len(plan)
    # Every extraction is billed exactly once, on the first pair that uses the branch
    assert round(sum(r.cost_usd for r in results), 6) == round(0.002 * len(involved) + 0.001 * len(plan), 6)


class _QuietDetector:
    """Round 1 finds nothing but leaves pairs unjudged; round 2 finds nothing and none are left."""

    def __init__(self) -> None:
        self.rounds = 0
        self.last_unjudged_remaining = 0

    def plan_pairs(self, branches, **kwargs):
        self.rounds += 1
        self.last_unjudged_remaining = 5 if self.rounds == 1 else 0
        return []

    def judge_planned_pairs(self, branches, plan, **kwargs) -> List[AlignmentJudgeResult]:
        return []


def test_a_quiet_round_does_not_end_selection_while_pairs_remain_unjudged() -> None:
    harness = MainHarness(client=MockOpenAIClient(), starter_agents=3, hard_cap_usd=2.0, selection_rounds=5)
    quiet = _QuietDetector()
    harness.alignment_detector = quiet
    summary = harness.run(PROBLEM)
    assert quiet.rounds == 2
    assert summary["selection"]["rounds_run"] == 2


def test_nine_starter_arena_judges_more_than_the_old_constant() -> None:
    harness = MainHarness(client=MockOpenAIClient(), starter_agents=9, hard_cap_usd=2.0)
    summary = harness.run(PROBLEM)
    sel = summary["selection"]
    # The old selection could judge at most 8 pairs a round
    assert len(sel["alignments_evaluated"]) > 8 * sel["rounds_run"]
    assert sel["merges_performed"]
    assert summary["winners_box"]
    verdicts = {a["verdict"] for a in sel["alignments_evaluated"]}
    assert AlignmentVerdict.OVERLAP.value in verdicts
