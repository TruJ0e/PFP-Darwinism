"""v0.42: selection coverage scales with the population, and its judge calls run concurrently.

Run 5 (174 branches) judged 8 alignment pairs: `max_pairs_to_judge=8` was hard-coded, and a round
with no merge and no clash ended selection, so round 1 was the only round. 0 merges. Selection now
clusters branches, plans a matching per round (merge candidates within clusters, champions across
them), keeps going while rounds still buy coverage, and batches judge, merge and falsification
calls (concurrent calls, in-order commits, so outcomes and spend stay reproducible).
"""

from __future__ import annotations

import json
import threading
import types
import pytest

from pfp import BranchState, BudgetLedger, FailJudge, MainHarness, MergeSynthesizer, MockOpenAIClient, PFPTracker
from pfp.alignment_detector import AlignmentDetector
from pfp.backend import ChatCompletionResponse, ChatMessage, Choice, CompletionUsage, HardCapExceededError
from pfp.fail_judge import FalsificationRequest

SANDBOX_PROBLEM = (
    "A small business's SSH works laptop->desktop but not desktop->laptop. "
    "Diagnose the root causes and verify the solution."
)
FORKS = [
    "sshd not listening on the laptop tcp 22",
    "laptop firewall drops inbound tcp 22 from desktop",
    "authorized_keys permissions rejected by strictmodes",
    "wifi client isolation blocks desktop to laptop",
]


def _response(content: str, cost: float = 0.001) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="scripted",
        model="scripted",
        choices=[Choice(index=0, message=ChatMessage(role="assistant", content=content))],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20, cost_usd=cost),
    )


def _population(n: int) -> PFPTracker:
    t = PFPTracker()
    for i in range(n):
        fork = f"{FORKS[i % len(FORKS)]} variant {i // len(FORKS)}"
        t.register_branch(f"b{i:03d}", None, 0.1, fork, origin="starter")
        t.update_branch(f"b{i:03d}", reasoning_step=f"[Conclusion] {fork}")
    return t


class ScriptedJudge:
    def __init__(self, verdict: str = "unrelated"):
        self.verdict = verdict
        self.judge_calls = 0
        self._lock = threading.Lock()
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, messages, **kwargs):
        user = messages[-1]["content"]
        if user.startswith("extract_claims"):
            return _response(json.dumps({"core_claims": [user[-60:]]}))
        with self._lock:
            self.judge_calls += 1
        return _response(json.dumps({"verdict": self.verdict}))


@pytest.mark.parametrize("n", [6, 40, 174])
def test_round_coverage_scales_with_population(n):
    t = _population(n)
    det = AlignmentDetector(ScriptedJudge(), prefilter_threshold=0.0)
    plan = det.plan_pairs(t.active_branches(), cluster_threshold=0.5)
    used = [bid for c in plan.pairs for bid in (c.branch_a_id, c.branch_b_id)]
    assert len(used) == len(set(used)), "a branch was planned into two pairs"
    # A matching over everyone: no constant cap (v0.40 judged 8 pairs at any size). Only an odd
    # member per cluster can wait for a later round (champions also pair across clusters).
    assert sum(len(m) for m in plan.clusters.values()) == n
    assert n // 2 - len(plan.clusters) <= len(plan.pairs) <= n // 2
    if n > 20:
        assert len(plan.pairs) > 8
    for champ, members in plan.clusters.items():
        assert members[0] == champ


def test_within_cluster_pairs_come_first_and_champions_meet_across():
    t = PFPTracker()
    for bid, fork in (("a1", FORKS[0]), ("a2", FORKS[0] + " confirmed"),
                      ("b1", FORKS[3]), ("b2", FORKS[3] + " confirmed")):
        t.register_branch(bid, None, 0.1, fork, origin="starter")
    det = AlignmentDetector(ScriptedJudge(), prefilter_threshold=0.0)
    plan = det.plan_pairs(t.active_branches(), cluster_threshold=0.6)
    assert len(plan.clusters) == 2
    assert {frozenset((c.branch_a_id, c.branch_b_id)) for c in plan.pairs} == {
        frozenset(("a1", "a2")), frozenset(("b1", "b2"))
    }
    assert plan.within_cluster_pairs == 2 and plan.champion_pairs == 0
    # Once the within-cluster pairs are judged, the champions face each other
    det.judge_planned(t.active_branches(), plan)
    nxt = det.plan_pairs(t.active_branches(), cluster_threshold=0.6)
    assert nxt.champion_pairs == 1 and len(nxt.pairs) == 1
    assert set(plan.clusters) == {nxt.pairs[0].branch_a_id, nxt.pairs[0].branch_b_id}


def test_max_pairs_caps_a_round_only_when_asked():
    t = _population(20)
    det = AlignmentDetector(ScriptedJudge(), prefilter_threshold=0.0)
    assert len(det.plan_pairs(t.active_branches(), max_pairs=3).pairs) == 3


def test_planned_pairs_are_judged_concurrently_in_plan_order():
    t = _population(8)
    barrier = threading.Barrier(4, timeout=10)

    class BarrierJudge(ScriptedJudge):
        def _create(self, messages, **kwargs):
            if messages[-1]["content"].startswith("alignment_judge"):
                barrier.wait()  # deadlocks (BrokenBarrierError) unless 4 verdicts are in flight
            return super()._create(messages, **kwargs)

    det = AlignmentDetector(BarrierJudge(), prefilter_threshold=0.0)
    plan = det.plan_pairs(t.active_branches())
    results, failure = det.judge_planned(t.active_branches(), plan, max_parallel=4)
    assert failure is None
    assert [(r.branch_a_id, r.branch_b_id) for r in results] == [
        (c.branch_a_id, c.branch_b_id) for c in plan.pairs
    ]


def test_quiet_round_does_not_end_selection_while_coverage_remains():
    # Every verdict "unrelated": nothing merges or clashes. v0.40 stopped after round 1.
    client = ScriptedJudge()
    harness = MainHarness(client=MockOpenAIClient(), starter_agents=3, hard_cap_usd=2.0, selection_rounds=4)
    t = _population(10)
    harness.tracker = t
    harness.alignment_detector = AlignmentDetector(client, prefilter_threshold=0.0)
    harness.fail_judge.tracker = t
    harness.merge_synthesizer.tracker = t
    harness.budget_ledger.root_budget_usd = 10.0
    harness.budget_ledger.finish_reserve_usd = 10.0
    harness.enable_selection = True
    sel = harness._run_selection_phase()
    assert not sel["merges_performed"] and not sel["contradictions_resolved"]
    assert sel["rounds_run"] >= 2
    assert all(c["pairs_judged"] > 0 for c in sel["coverage"][:2])
    assert len(sel["alignments_evaluated"]) == client.judge_calls > 5


class FalsifyClient:
    """Kills branches whose text says 'weak' (appeal upholds); everything else survives."""

    def __init__(self, cap_on: str = ""):
        self.cap_on = cap_on
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, messages, **kwargs):
        user = messages[-1]["content"]
        if self.cap_on and self.cap_on in user:
            raise HardCapExceededError("cap")
        survived = "weak" not in user
        return _response(json.dumps({"survived": survived, "objection": "" if survived else "hole", "rationale": "r"}), 0.01)


def _judge_with(n: int, client) -> tuple:
    t = PFPTracker()
    for i in range(n):
        t.register_branch(f"b{i}", None, 0.5, f"{'weak' if i % 2 else 'solid'} claim {i}")
    ledger = BudgetLedger(hard_cap_usd=5.0)
    return FailJudge(client, t, budget_ledger=ledger), t, ledger


def test_batched_falsification_matches_one_by_one():
    seq_judge, seq_t, seq_ledger = _judge_with(6, FalsifyClient())
    seq = [seq_judge.falsify_branch(f"b{i}", "background_sample").to_dict() for i in range(6)]
    bat_judge, bat_t, bat_ledger = _judge_with(6, FalsifyClient())
    bat = [o.to_dict() for o in bat_judge.falsify_many(
        [FalsificationRequest(f"b{i}", "background_sample") for i in range(6)]
    )]
    assert bat == seq
    assert [r.state for r in bat_t.all_records()] == [r.state for r in seq_t.all_records()]
    assert bat_ledger.total_spent_usd == seq_ledger.total_spent_usd == pytest.approx(0.09)
    assert [r.state for r in bat_t.all_records()][:2] == [BranchState.POSSIBLE, BranchState.FAIL]


def test_a_branch_named_twice_is_judged_after_its_first_verdict():
    judge, t, _ = _judge_with(1, FalsifyClient())
    outs = judge.falsify_many([FalsificationRequest("b0", "background_sample")] * 2)
    assert [o.survived for o in outs] == [True, True]
    assert t.get("b0").survived_falsifications == 2


def test_a_cap_hit_in_a_batch_still_books_the_paid_verdicts():
    judge, t, ledger = _judge_with(4, FalsifyClient(cap_on="claim 2"))
    with pytest.raises(HardCapExceededError):
        judge.falsify_many([FalsificationRequest(f"b{i}", "background_sample") for i in range(4)])
    # b0, b1, b3 were answered and paid: booked, never silently dropped
    assert ledger.total_spent_usd == pytest.approx(0.03)


def test_merge_many_requires_disjoint_parents_and_spends_nothing_when_refused():
    t = _population(3)
    client = ScriptedJudge()
    synth = MergeSynthesizer(client, t, budget_ledger=BudgetLedger())
    with pytest.raises(ValueError):
        synth.merge_many([("b000", "b001", "m1", None), ("b001", "b002", "m2", None)])
    assert client.judge_calls == 0
    assert all(r.state == BranchState.POSSIBLE for r in t.all_records())


def test_nine_starter_mock_judges_past_the_old_cap():
    harness = MainHarness(client=MockOpenAIClient(), starter_agents=9, hard_cap_usd=2.0)
    summary = harness.run(SANDBOX_PROBLEM)
    assert summary["halted_reason"] is None
    sel = summary["selection"]
    assert sel["coverage"][0]["pairs_judged"] > 8
    assert sel["coverage"][0]["pairs_judged"] >= sel["coverage"][0]["alive"] // 2 - sel["coverage"][0]["clusters"]
    assert sel["merges_performed"]
    assert summary["winners_box"]


def test_selection_knobs_are_validated():
    with pytest.raises(ValueError):
        MainHarness(client=MockOpenAIClient(), cluster_threshold=0.0)
    with pytest.raises(ValueError):
        MainHarness(client=MockOpenAIClient(), max_pairs_per_round=0)


def test_batched_attacks_are_in_flight_together():
    barrier = threading.Barrier(3, timeout=10)

    class BarrierClient(FalsifyClient):
        def _create(self, messages, **kwargs):
            if "appeal_review=false" in messages[-1]["content"]:
                barrier.wait()  # raises BrokenBarrierError unless 3 primaries run at once
            return super()._create(messages, **kwargs)

    judge, t, _ = _judge_with(3, BarrierClient())
    outs = judge.falsify_many([FalsificationRequest(f"b{i}", "background_sample") for i in range(3)])
    assert [o.survived for o in outs] == [True, False, True]
