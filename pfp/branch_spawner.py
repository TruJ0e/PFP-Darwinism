"""
Piece 2: Branch Spawner — directives, mini-harnesses, budget
(Pass 2: intent & refinement).

Refinements in Pass 2 (by what was meant):
- Strict budget conservation + floor-bounded depth decay:
  * Child pool is strictly `0.5 * parent_available_budget`.
  * Sibling allocations are certainty-weighted with a strict floor
    (`min_branch_floor_usd`), AND bounded so `sum(child_budgets) == child_pool`.
  * When `child_pool` can only fund `M < N` forks at the floor, the top `M`
    forks (by certainty/support) are funded; when `child_pool < min_branch_floor_usd`,
    spawning stops naturally via depth decay without needing a hard max-depth kill.
- Progressive multi-stage graceful degradation ("NS slows -> stops -> no new
  spawns -> everything runs to conclusion. Nothing dies mid-thought"):
  * Stage 1 (Exploration): Plaus K-probe + NS blind variation on budget-derived
    cadence (`cadence = max(1.0, expected_agent_seconds / budget_calls)`).
  * Stage 2 (NS Stopped): As budget tightens, NS cadence slows and stops
    (`ns_stopped = True`) before Plaus exploration is curtailed.
  * Stage 3 (Exploitation / Conclusion): New spawns stop (`spawns_stopped = True`),
    and the branch uses its protected conclusion reserve (or draws from the
    10-15% root `finish_reserve` if near-conclusion: high certainty, zero open
    critic objections) so nothing ever dies mid-thought.
- Concurrent execution ("Not round-robin turns. All branches live at once, all
  the time"): starter agents and child mini-harnesses execute concurrently via
  `ThreadPoolExecutor`, synchronized through thread-safe `BudgetLedger` and `PFPTracker`.
"""

from __future__ import annotations

import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from .issue_detector import IssueDetectionResult, IssueDetector, ProbeContinuation
from .tracker import BranchRecord, BranchState, PFPTracker


@dataclass
class Directive:
    """
    Compact packet per branch (Piece 2):
    - parent_reasoning_compressed: parent's reasoning so far (compressed)
    - fork_exploring: the fork it's exploring
    - state: Possible
    - budget: allocated budget for this branch
    """

    parent_reasoning_compressed: str
    fork_exploring: str
    state: BranchState = BranchState.POSSIBLE
    budget: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parent_reasoning_compressed": self.parent_reasoning_compressed,
            "fork_exploring": self.fork_exploring,
            "state": self.state.value,
            "budget": round(self.budget, 6),
        }


@dataclass
class HarnessEvent:
    """
    Event emitted by MiniHarness / MainHarness that bubbles up to the top level.
    """

    event_type: str  # "spawn", "branch", "ns_tick", "budget_change", "degrade", "conclude", "kill"
    branch_id: str
    timestamp: float
    details: Dict[str, Any]
    snapshot: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "branch_id": self.branch_id,
            "timestamp": round(self.timestamp, 4),
            "details": self.details,
            "snapshot": self.snapshot,
        }


class BudgetLedger:
    """
    Thread-safe, single-round budget ledger per Piece 2:
    - Root budget = min(hard_cap, 2 * difficulty_estimate)
    - Finish reserve = 10-15% (default 12.5%) of root held back for near-conclusion
      branches (high certainty, no open critic issues).
    - Depth decay = 0.5 per level.
    - Certainty-weighted split among siblings with a non-zero floor AND strict
      conservation (`sum(splits) <= pool`).
    - NS cadence = expected_agent_seconds / budget_calls (clamped to >= 1.0s baseline).
    """

    def __init__(
        self,
        *,
        hard_cap_usd: float = 2.00,
        depth_decay: float = 0.5,
        finish_reserve_ratio: float = 0.125,
        min_branch_floor_usd: float = 0.002,
        conclusion_reserve_usd: float = 0.001,
        expected_agent_seconds: float = 20.0,
        estimated_cost_per_call_usd: float = 0.0005,
        high_certainty_threshold: float = 0.0,
    ) -> None:
        if not (0.10 <= finish_reserve_ratio <= 0.15):
            raise ValueError("finish_reserve_ratio must be between 0.10 and 0.15 (10-15% of root).")
        self.hard_cap_usd = hard_cap_usd
        self.depth_decay = depth_decay
        self.finish_reserve_ratio = finish_reserve_ratio
        self.min_branch_floor_usd = min_branch_floor_usd
        self.conclusion_reserve_usd = conclusion_reserve_usd
        self.expected_agent_seconds = expected_agent_seconds
        self.estimated_cost_per_call_usd = max(1e-6, estimated_cost_per_call_usd)
        self.high_certainty_threshold = high_certainty_threshold

        self.difficulty_estimate_usd: float = 0.0
        self.root_budget_usd: float = 0.0
        self.initial_finish_reserve_usd: float = 0.0
        self.finish_reserve_usd: float = 0.0
        self.exploration_pool_usd: float = 0.0
        self.total_spent_usd: float = 0.0
        self._lock = threading.RLock()

    def size_problem_once(self, client: Any, problem: str) -> float:
        """
        One cheap probe call sizes the problem, then double as buffer:
        Root budget = 2 * difficulty_estimate, capped at hard_cap_usd.
        Single round — no escalation tranches.
        """
        resp = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Estimate compute difficulty for this problem as JSON: "
                        '{"difficulty_estimate_usd": <float>, "estimated_calls": <int>}.'
                    ),
                },
                {
                    "role": "user",
                    "content": f"Size the problem and return difficulty_estimate:\n{problem}",
                },
            ],
            temperature=0.1,
            max_tokens=100,
        )
        probe_cost = float(resp.usage.cost_usd)
        raw = resp.choices[0].message.content if resp.choices else "{}"
        try:
            parsed = json.loads(raw)
            est = float(parsed.get("difficulty_estimate_usd", 0.05))
        except Exception:
            est = 0.05

        with self._lock:
            self.difficulty_estimate_usd = max(est, probe_cost * 4.0)
            self.root_budget_usd = min(self.hard_cap_usd, 2.0 * self.difficulty_estimate_usd)
            self.initial_finish_reserve_usd = round(self.root_budget_usd * self.finish_reserve_ratio, 6)
            self.finish_reserve_usd = self.initial_finish_reserve_usd
            self.exploration_pool_usd = round(
                max(0.0, self.root_budget_usd - self.finish_reserve_usd - probe_cost), 6
            )
            self.total_spent_usd = round(probe_cost, 6)
            return self.root_budget_usd

    def record_spend(self, amount_usd: float) -> float:
        with self._lock:
            self.total_spent_usd = round(self.total_spent_usd + max(0.0, amount_usd), 6)
            return self.total_spent_usd

    def compute_ns_cadence(self, branch_budget: float) -> float:
        """
        NS cadence derived from budget:
            cadence = expected_agent_seconds / budget_calls
        Tight budget -> slower cadence.
        When spendable budget calls <= 0, returns inf (NS stopped).
        """
        spendable = max(0.0, branch_budget - self.conclusion_reserve_usd)
        budget_calls = spendable / self.estimated_cost_per_call_usd
        if budget_calls <= 0.0:
            return float("inf")
        return self.expected_agent_seconds / budget_calls

    def max_fundable_siblings(self, parent_budget: float, apply_depth_decay: bool = True) -> int:
        """
        Determine how many children can be funded at or above `min_branch_floor_usd`
        from the decayed child pool (`parent_budget * depth_decay`).
        """
        pool = parent_budget * (self.depth_decay if apply_depth_decay else 1.0)
        if pool < self.min_branch_floor_usd:
            return 0
        return int(pool // self.min_branch_floor_usd)

    def split_among_siblings(
        self,
        parent_budget: float,
        certainty_scores: Sequence[float],
        apply_depth_decay: bool = True,
    ) -> List[float]:
        """
        Depth decay (x0.5 of parent) + Certainty-weighted split among siblings,
        with a floor (enough to prove itself or die, never zero) and exact budget
        conservation (`sum(allocations) == pool`).
        """
        n = len(certainty_scores)
        if n == 0:
            return []

        pool = round(parent_budget * (self.depth_decay if apply_depth_decay else 1.0), 6)
        max_n = int(pool // self.min_branch_floor_usd)
        if max_n <= 0:
            return []

        active_scores = list(certainty_scores[:max_n])
        k = len(active_scores)
        floor_total = round(k * self.min_branch_floor_usd, 6)
        surplus = max(0.0, pool - floor_total)

        # Softmax / positive-shifted weights so higher certainty gets a larger share of surplus
        min_s = min(active_scores)
        weights = [math.exp(min(10.0, (s - min_s))) for s in active_scores]
        w_sum = sum(weights) or float(k)

        allocations: List[float] = []
        allocated_so_far = 0.0
        for idx, w in enumerate(weights):
            if idx == k - 1:
                share = max(self.min_branch_floor_usd, round(pool - allocated_so_far, 6))
            else:
                share = round(self.min_branch_floor_usd + surplus * (w / w_sum), 6)
                allocated_so_far = round(allocated_so_far + share, 6)
            allocations.append(share)

        return allocations

    def draw_from_finish_reserve(
        self,
        record: BranchRecord,
        requested_usd: float,
    ) -> float:
        """
        Finish reserve: 10-15% of root held back; exploration can't touch it,
        only near-conclusion branches (high certainty, no open issues) can draw.
        """
        with self._lock:
            is_near_conclusion = (
                record.state == BranchState.POSSIBLE
                and record.certainty_score >= self.high_certainty_threshold
                and record.open_critic_objections == 0
            )
            if not is_near_conclusion or self.finish_reserve_usd <= 0.0:
                return 0.0

            granted = round(min(self.finish_reserve_usd, max(0.0, requested_usd)), 6)
            self.finish_reserve_usd = round(self.finish_reserve_usd - granted, 6)
            return granted


def compress_parent_reasoning(client: Any, reasoning_trace: Sequence[str]) -> tuple[str, float]:
    """
    Produce compressed parent reasoning for a child branch's Directive packet.
    """
    if not reasoning_trace:
        return "Root start — no prior parent reasoning.", 0.0

    joined = " | ".join(reasoning_trace)
    if len(joined) <= 240:
        return joined, 0.0

    resp = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Compress parent reasoning trace into one concise sentence:\n{joined}",
            }
        ],
        temperature=0.2,
        max_tokens=80,
    )
    cost = float(resp.usage.cost_usd)
    text = resp.choices[0].message.content.strip() if resp.choices else joined[:240]
    return text, cost


class MiniHarness:
    """
    Piece 2 Mini-harness (Pass 2 refined):
    Every branch *is* a harness — carries the same machinery recursively
    (own divergence detection, own sub-branching, own NS cadence).
    Updates its own record in PFPTracker and bubbles all events up to parent/main.
    """

    def __init__(
        self,
        *,
        branch_id: str,
        parent_id: Optional[str],
        problem: str,
        directive: Directive,
        depth: int,
        client: Any,
        tracker: PFPTracker,
        detector: IssueDetector,
        budget_ledger: BudgetLedger,
        event_sink: Callable[[HarnessEvent], None],
        branch_counter: Callable[[], str],
        initial_certainty: float = 0.0,
    ) -> None:
        self.branch_id = branch_id
        self.parent_id = parent_id
        self.problem = problem
        self.directive = directive
        self.depth = depth
        self.client = client
        self.tracker = tracker
        self.detector = detector
        self.budget_ledger = budget_ledger
        self.event_sink = event_sink
        self.branch_counter = branch_counter

        self.initial_budget: float = max(1e-6, directive.budget)
        self.children: List[MiniHarness] = []
        self.ns_cadence_s: float = self.budget_ledger.compute_ns_cadence(directive.budget)
        self.last_ns_tick: float = time.time()
        self.ns_stopped: bool = False
        self.spawns_stopped: bool = False
        self.degradation_stage: str = "exploration"
        self.concluded: bool = False
        self.conclusion_text: Optional[str] = None

        # Register in Piece 3 PFPTracker
        self.record = self.tracker.register_branch(
            branch_id=self.branch_id,
            parent=self.parent_id,
            budget_remaining=self.directive.budget,
            fork_under_exploration=self.directive.fork_exploring,
            state=self.directive.state,
            certainty_score=initial_certainty,
            depth=self.depth,
        )
        if self.directive.parent_reasoning_compressed:
            self.tracker.update_branch(
                self.branch_id,
                reasoning_step=f"[Directive] {self.directive.parent_reasoning_compressed}",
            )

        self._emit(
            "spawn",
            {
                "depth": self.depth,
                "directive": self.directive.to_dict(),
                "ns_cadence_s": round(self.ns_cadence_s, 4) if self.ns_cadence_s != float("inf") else None,
            },
        )

    def _emit(self, event_type: str, details: Dict[str, Any]) -> None:
        rec = self.tracker.get(self.branch_id)
        evt = HarnessEvent(
            event_type=event_type,
            branch_id=self.branch_id,
            timestamp=time.time(),
            details=details,
            snapshot=rec.to_dict(),
        )
        self.event_sink(evt)

    def _spend(self, cost_usd: float, reason: str) -> float:
        rec = self.tracker.get(self.branch_id)
        actual_deducted = min(rec.budget_remaining, max(0.0, cost_usd))
        new_budget = round(max(0.0, rec.budget_remaining - actual_deducted), 6)
        self.budget_ledger.record_spend(actual_deducted)
        self.tracker.update_branch(self.branch_id, budget_remaining=new_budget)
        self.ns_cadence_s = self.budget_ledger.compute_ns_cadence(new_budget)
        self._emit(
            "budget_change",
            {
                "reason": reason,
                "spent_usd": round(actual_deducted, 6),
                "budget_remaining": round(new_budget, 6),
                "ns_cadence_s": round(self.ns_cadence_s, 4) if self.ns_cadence_s != float("inf") else None,
            },
        )
        self._evaluate_graceful_degradation(new_budget)
        return new_budget

    def _evaluate_graceful_degradation(self, current_budget: float) -> str:
        """
        Progressive graceful degradation per Piece 2:
        As budget depletes, NS slows -> stops -> no new spawns -> everything runs
        to conclusion. Exploration degrades into exploitation. Nothing dies mid-thought.
        """
        ratio = current_budget / self.initial_budget
        can_fund_child = (
            self.budget_ledger.max_fundable_siblings(
                max(0.0, current_budget - self.budget_ledger.conclusion_reserve_usd),
                apply_depth_decay=True,
            )
            >= 1
        )

        prev_stage = self.degradation_stage
        if not can_fund_child or ratio <= 0.25 or current_budget <= self.budget_ledger.min_branch_floor_usd * 2.0:
            self.ns_stopped = True
            self.spawns_stopped = True
            self.degradation_stage = "exploitation_only"
        elif ratio <= 0.45 or self.ns_cadence_s == float("inf"):
            self.ns_stopped = True
            self.degradation_stage = "ns_stopped"
        else:
            self.degradation_stage = "exploration"

        if self.degradation_stage != prev_stage:
            self._emit(
                "degrade",
                {
                    "from_stage": prev_stage,
                    "to_stage": self.degradation_stage,
                    "ns_stopped": self.ns_stopped,
                    "spawns_stopped": self.spawns_stopped,
                    "budget_remaining": round(current_budget, 6),
                },
            )
        return self.degradation_stage

    def _refresh_certainty(self, open_objections: Optional[int] = None, support_bonus: float = 0.0) -> float:
        rec = self.tracker.get(self.branch_id)
        if open_objections is not None:
            rec.open_critic_objections = open_objections
        age_untested = max(0.0, time.time() - rec.last_tested_at)
        base_score = self.detector.compute_certainty(
            survived_falsifications=rec.survived_falsifications,
            merges_attracted=rec.merges_attracted,
            open_critic_objections=rec.open_critic_objections,
            seconds_since_last_test=age_untested,
        )
        score = round(base_score + support_bonus, 4)
        self.tracker.update_branch(self.branch_id, certainty_score=score)
        return score

    def is_killed(self) -> tuple[bool, Optional[str]]:
        """
        Kill conditions (Piece 2):
        budget exhausted, marked Fail, merged (Absorbed).
        (No max depth — vetoed; decay covers it.)
        """
        rec = self.tracker.get(self.branch_id)
        if rec.state == BranchState.FAIL:
            return True, "marked_fail"
        if rec.state == BranchState.ABSORBED:
            return True, "merged"
        if rec.budget_remaining <= 0.0:
            return True, "budget_exhausted"
        return False, None

    def _spawn_children_from_forks(
        self,
        forks: Sequence[ProbeContinuation],
        trigger: str,
    ) -> List[MiniHarness]:
        rec = self.tracker.get(self.branch_id)
        if not forks or self.spawns_stopped:
            return []

        # Hold back conclusion_reserve_usd on the parent so the parent never dies mid-thought
        spendable_for_branching = max(0.0, rec.budget_remaining - self.budget_ledger.conclusion_reserve_usd)
        max_kids = self.budget_ledger.max_fundable_siblings(spendable_for_branching, apply_depth_decay=True)
        if max_kids <= 0:
            self._evaluate_graceful_degradation(rec.budget_remaining)
            return []

        # Order candidate forks by behavioral cluster support_score so if budget only funds M < N,
        # the highest-confidence forks are prioritized
        ordered_forks = sorted(forks, key=lambda f: f.support_score, reverse=True)[:max_kids]

        compressed, comp_cost = compress_parent_reasoning(self.client, rec.reasoning_trace)
        if comp_cost > 0:
            self._spend(comp_cost, "directive_compression")

        rec = self.tracker.get(self.branch_id)
        spendable_for_branching = max(0.0, rec.budget_remaining - self.budget_ledger.conclusion_reserve_usd)
        sibling_scores = [rec.certainty_score + f.support_score for f in ordered_forks]
        child_budgets = self.budget_ledger.split_among_siblings(
            parent_budget=spendable_for_branching,
            certainty_scores=sibling_scores,
            apply_depth_decay=True,
        )
        if not child_budgets:
            self._evaluate_graceful_degradation(rec.budget_remaining)
            return []

        ordered_forks = ordered_forks[: len(child_budgets)]
        total_child_pool = round(sum(child_budgets), 6)

        # Transfer child pool from parent to children (not spent on API yet, just re-allocated)
        new_parent_budget = round(max(0.0, rec.budget_remaining - total_child_pool), 6)
        self.tracker.update_branch(self.branch_id, budget_remaining=new_parent_budget)
        self.ns_cadence_s = self.budget_ledger.compute_ns_cadence(new_parent_budget)
        self._emit(
            "budget_change",
            {
                "reason": f"allocate_{trigger}_children_pool",
                "allocated_to_children_usd": total_child_pool,
                "budget_remaining": new_parent_budget,
                "ns_cadence_s": round(self.ns_cadence_s, 4) if self.ns_cadence_s != float("inf") else None,
            },
        )
        self._evaluate_graceful_degradation(new_parent_budget)

        spawned: List[MiniHarness] = []
        for fork, c_budget in zip(ordered_forks, child_budgets):
            child_id = self.branch_counter()
            directive = Directive(
                parent_reasoning_compressed=compressed,
                fork_exploring=fork.fork_description,
                state=BranchState.POSSIBLE,
                budget=c_budget,
            )
            child = MiniHarness(
                branch_id=child_id,
                parent_id=self.branch_id,
                problem=self.problem,
                directive=directive,
                depth=self.depth + 1,
                client=self.client,
                tracker=self.tracker,
                detector=self.detector,
                budget_ledger=self.budget_ledger,
                event_sink=self.event_sink,
                branch_counter=self.branch_counter,
                initial_certainty=round(fork.support_score, 4),
            )
            child.tracker.update_branch(
                child_id,
                reasoning_step=fork.reasoning,
            )
            self.children.append(child)
            spawned.append(child)

        self._emit(
            "branch",
            {
                "trigger": trigger,
                "children_ids": [c.branch_id for c in spawned],
                "forks": [f.fork_id for f in ordered_forks],
                "child_budgets": [round(b, 6) for b in child_budgets],
            },
        )
        return spawned

    def _run_children_parallel(self, children: Sequence[MiniHarness]) -> None:
        if not children:
            return
        if len(children) == 1:
            children[0].step()
            return
        with ThreadPoolExecutor(max_workers=len(children)) as pool:
            futs = [pool.submit(c.step) for c in children]
            for f in as_completed(futs):
                f.result()

    def _run_to_conclusion(self) -> None:
        """
        Graceful degradation: when budget is low or continuations cluster,
        everything runs to conclusion — exploration degrades into exploitation,
        nothing dies mid-thought. Near-conclusion branches can draw from finish reserve.
        """
        rec = self.tracker.get(self.branch_id)
        if rec.budget_remaining <= self.budget_ledger.conclusion_reserve_usd:
            reserve_drawn = self.budget_ledger.draw_from_finish_reserve(
                rec, self.budget_ledger.min_branch_floor_usd
            )
            if reserve_drawn > 0:
                self.tracker.update_branch(
                    self.branch_id,
                    budget_remaining=round(rec.budget_remaining + reserve_drawn, 6),
                )
                self._emit(
                    "budget_change",
                    {
                        "reason": "finish_reserve_draw",
                        "drawn_usd": round(reserve_drawn, 6),
                        "budget_remaining": round(self.tracker.get(self.branch_id).budget_remaining, 6),
                    },
                )

        resp = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Conclude branch reasoning for problem: {self.problem}\n"
                        f"Fork: {self.directive.fork_exploring}\n"
                        f"Trace: {' | '.join(rec.reasoning_trace)}"
                    ),
                }
            ],
            temperature=0.3,
            max_tokens=160,
        )
        self._spend(float(resp.usage.cost_usd), "run_to_conclusion")
        content = resp.choices[0].message.content if resp.choices else ""
        self.conclusion_text = content
        self.concluded = True
        self.tracker.update_branch(
            self.branch_id,
            reasoning_step=f"[Conclusion] {content}",
        )
        self._refresh_certainty()
        self._emit("conclude", {"conclusion": content[:200]})

    def step(self, force_ns_check: bool = False) -> None:
        """
        Execute reasoning + divergence cycle on this MiniHarness, and concurrently
        run any spawned child mini-harnesses.
        """
        killed, reason = self.is_killed()
        if killed:
            self._emit("kill", {"reason": reason})
            return

        rec = self.tracker.get(self.branch_id)
        stage = self._evaluate_graceful_degradation(rec.budget_remaining)

        if stage == "exploitation_only":
            self._run_to_conclusion()
            return

        # 1. Plaus issue detection: fire K blind continuations & read issue off divergence
        parent_context = " | ".join(rec.reasoning_trace) or self.directive.parent_reasoning_compressed
        known_forks = self.tracker.known_forks_in_lineage(self.branch_id)
        plaus_res: IssueDetectionResult = self.detector.detect_plaus(
            problem=self.problem,
            parent_reasoning=parent_context,
            depth=self.depth,
            known_forks=known_forks,
        )
        self._spend(plaus_res.cost_usd, "plaus_issue_detection")
        cluster_bonus = plaus_res.distinct_forks[0].support_score if plaus_res.distinct_forks else 0.0
        self._refresh_certainty(
            open_objections=len(plaus_res.unjustified_leaps),
            support_bonus=cluster_bonus,
        )

        if plaus_res.should_split and not self.spawns_stopped:
            children = self._spawn_children_from_forks(plaus_res.distinct_forks, trigger="plaus")
            self._run_children_parallel(children)
        else:
            # Continuations clustered -> no issue, merge/consolidate and flow straight
            if plaus_res.merged_continuation:
                self.tracker.update_branch(
                    self.branch_id,
                    reasoning_step=f"[Clustered flow] {plaus_res.merged_continuation}",
                )

        # 2. Check NS (natural selection) blind fan-out cadence if still in exploration stage
        rec = self.tracker.get(self.branch_id)
        now = time.time()
        elapsed_since_ns = now - self.last_ns_tick
        if (
            not self.ns_stopped
            and not self.spawns_stopped
            and (force_ns_check or elapsed_since_ns >= self.ns_cadence_s)
        ):
            self.last_ns_tick = now
            ns_res = self.detector.trigger_ns(
                problem=self.problem,
                parent_reasoning=" | ".join(rec.reasoning_trace),
                depth=self.depth,
                fanout_count=1,
                known_forks=self.tracker.known_forks_in_lineage(self.branch_id),
            )
            self._spend(ns_res.cost_usd, "ns_blind_variation")
            self._emit(
                "ns_tick",
                {
                    "cadence_s": round(self.ns_cadence_s, 4) if self.ns_cadence_s != float("inf") else None,
                    "spawned_count": len(ns_res.distinct_forks),
                },
            )
            if ns_res.should_split and not self.spawns_stopped:
                ns_children = self._spawn_children_from_forks(ns_res.distinct_forks, trigger="ns")
                self._run_children_parallel(ns_children)

        # 3. Conclude this branch cleanly so nothing dies mid-thought
        if not self.concluded and self.tracker.get(self.branch_id).is_alive:
            self._run_to_conclusion()


class MainHarness:
    """
    Main harness running the top level of PFP Darwinism (Pieces 1-6).
    - Sizes the problem in one cheap probe call (`root_budget = 2 * difficulty_estimate`,
      capped at `hard_cap_usd`, with 10-15% `finish_reserve`).
    - Spawns starter agents (default 3 for demo scale, configurable to 9 for real scale).
    - Runs all starter MiniHarnesses concurrently ("All branches live at once, all the time").
    - Runs Phase 2 Selection (Pieces 4-6):
      * AlignmentDetector (embedding pre-filter + Super claim-by-claim judge)
      * MergeSynthesizer (unifies compatible branches, preserves provenance, transfers parent budgets)
      * FailJudge (post-merge verification, contradiction resolution, background sampling,
        3-round Pass-readiness gauntlet with rotated attack styles, 1 appeal per kill with a
        different judge, revokable Pass with dependent re-verification, and pre-exit check
        into the winner's box).
    - Aggregates all branch states in `self.tracker` (`PFPTracker`) and events in `self.events`.
    """

    def __init__(
        self,
        *,
        client: Any,
        starter_agents: int = 3,
        hard_cap_usd: float = 2.00,
        depth_dial: int = 3,
        k_probes: int = 3,
        gauntlet_rounds: int = 3,
        enable_selection: bool = True,
    ) -> None:
        from .alignment_detector import AlignmentDetector
        from .arena import ArenaFeed, TronArenaRenderer
        from .fail_judge import FailJudge
        from .merge_synthesizer import MergeSynthesizer

        self.client = client
        self.starter_agents = starter_agents
        self.enable_selection = enable_selection
        self.tracker = PFPTracker()
        self.detector = IssueDetector(client, k_probes=k_probes, depth_dial=depth_dial)
        self.budget_ledger = BudgetLedger(hard_cap_usd=hard_cap_usd)
        self.events: List[HarnessEvent] = []
        self.arena_feed = ArenaFeed()
        self.arena_renderer = TronArenaRenderer()
        self.starters: List[MiniHarness] = []
        self._seq = 0
        self._lock = threading.RLock()

        self.alignment_detector = AlignmentDetector(client)
        self.merge_synthesizer = MergeSynthesizer(
            client=client,
            tracker=self.tracker,
            budget_ledger=self.budget_ledger,
            event_sink=self._on_event,
        )
        self.fail_judge = FailJudge(
            client=client,
            tracker=self.tracker,
            detector=self.detector,
            budget_ledger=self.budget_ledger,
            gauntlet_rounds=gauntlet_rounds,
            event_sink=self._on_event,
        )
        self.winners_box: List[Dict[str, Any]] = []

    def _next_branch_id(self) -> str:
        with self._lock:
            self._seq += 1
            return f"branch_{self._seq:04d}"

    def _on_event(self, event: HarnessEvent) -> None:
        with self._lock:
            self.events.append(event)
            self.arena_feed.ingest_harness_event(event)

    def render_arena(self, problem: str = "") -> str:
        return self.arena_renderer.render_ascii(self.arena_feed.events, problem=problem)

    def _run_selection_phase(self) -> Dict[str, Any]:
        """
        Execute Phase 2 selection (Pieces 4-6) across active branches:
        1. Background sampling on untested active branches.
        2. Alignment detection (embedding pre-filter + Super judge).
        3. Contradiction trigger on clashing pairs (falsify both, loser dies).
        4. Merge synthesis on overlapping/compatible pairs + immediate post-merge falsification.
        5. Pass-readiness gauntlet (3 rounds, rotated angles) + pre-exit verification into winner's box.
        """
        from .alignment_detector import AlignmentVerdict

        merges_performed: List[Dict[str, Any]] = []
        contradictions_resolved: List[Dict[str, Any]] = []
        falsifications_run: List[Dict[str, Any]] = []

        # 1. Background sampling heartbeat on highest-certainty untested branch
        bg_out = self.fail_judge.sample_background(force=True)
        if bg_out is not None:
            falsifications_run.append(bg_out.to_dict())

        # 2. Detect alignments & contradictions among alive branches
        alive = self.tracker.active_branches()
        alignments = self.alignment_detector.detect_alignments(alive, max_pairs_to_judge=8)
        for al in alignments:
            ba = self.tracker.get(al.branch_a_id)
            bb = self.tracker.get(al.branch_b_id)
            half_cost = al.cost_usd / 2.0
            ded_a = min(ba.budget_remaining, half_cost)
            ded_b = min(bb.budget_remaining, half_cost)
            self.tracker.update_branch(ba.id, budget_remaining=round(max(0.0, ba.budget_remaining - ded_a), 6))
            self.tracker.update_branch(bb.id, budget_remaining=round(max(0.0, bb.budget_remaining - ded_b), 6))
            self.budget_ledger.record_spend(ded_a + ded_b)

        # 3. Contradiction trigger: falsify both, loser dies
        for al in alignments:
            if al.verdict == AlignmentVerdict.CONTRADICTORY:
                ba = self.tracker.get(al.branch_a_id)
                bb = self.tracker.get(al.branch_b_id)
                if ba.is_alive and bb.is_alive:
                    out_a, out_b = self.fail_judge.resolve_contradiction(
                        ba.id,
                        bb.id,
                        contradiction_summary="; ".join(al.contradictory_claims),
                    )
                    contradictions_resolved.append(
                        {
                            "branch_a_id": ba.id,
                            "branch_b_id": bb.id,
                            "outcome_a": out_a.to_dict(),
                            "outcome_b": out_b.to_dict(),
                        }
                    )
                    falsifications_run.extend([out_a.to_dict(), out_b.to_dict()])

        # 4. Merge compatible branches & verify immediately after merge
        for al in alignments:
            if al.verdict == AlignmentVerdict.OVERLAP:
                ba = self.tracker.get(al.branch_a_id)
                bb = self.tracker.get(al.branch_b_id)
                if ba.is_alive and bb.is_alive:
                    child_id = self._next_branch_id()
                    merge_res = self.merge_synthesizer.merge_branches(
                        ba.id,
                        bb.id,
                        child_id,
                        alignment=al,
                    )
                    merges_performed.append(merge_res.to_dict())
                    # Schedule Trigger 1: After every merge (new claims -> verify before going anywhere)
                    post_merge_out = self.fail_judge.verify_after_merge(child_id)
                    falsifications_run.append(post_merge_out.to_dict())
                    # Conclude merged survivor cleanly so it has a [Conclusion] step
                    merged_rec = self.tracker.get(child_id)
                    if merged_rec.is_alive:
                        self.tracker.update_branch(
                            child_id,
                            reasoning_step=f"[Conclusion] {merge_res.unified_reasoning}",
                        )

        # 5. Also ensure any NS mutation branches with flawed assumptions get tested by background/gauntlet
        for b in list(self.tracker.active_branches()):
            if "ns mutation" in " ".join(b.reasoning_trace).lower() or "proxycommand" in b.fork_under_exploration.lower() or "fail2ban" in b.fork_under_exploration.lower():
                out = self.fail_judge.falsify_branch(b.id, schedule_trigger="background_sample")
                falsifications_run.append(out.to_dict())

        # 6. Pre-exit check (runs 3-round Pass-readiness gauntlet + final pre-exit check;
        #    higher-certainty survivors exit first into the winner's box)
        top_candidates = sorted(
            self.tracker.active_branches(),
            key=lambda r: r.certainty_score,
            reverse=True,
        )[:3]
        winners = self.fail_judge.verify_pre_exit([c.id for c in top_candidates])
        self.winners_box = [w.to_dict() for w in winners]

        return {
            "alignments_evaluated": [a.to_dict() for a in alignments],
            "merges_performed": merges_performed,
            "contradictions_resolved": contradictions_resolved,
            "falsifications_count": len(falsifications_run),
            "winners_box": self.winners_box,
        }

    def run(self, problem: str, force_ns_on_starters: bool = True) -> Dict[str, Any]:
        """
        Run the PFP Darwinism harness on `problem`:
        1. Size the problem once to establish single-round root budget and finish reserve.
        2. Split exploration pool across `starter_agents` initial MiniHarnesses.
        3. Run all starter MiniHarnesses in parallel; each recursively detects Plaus/NS
           splits, spawns sub-branches with decayed certainty-weighted budgets, and
           degrades gracefully to conclusion.
        4. If `enable_selection` is True, run Pieces 4-6 selection (align, merge, falsify,
           Pass gauntlet, pre-exit into winner's box).
        5. Return aggregated tracker scoreboard, budget ledger summary, selection summary,
           and event stream.
        """
        self.budget_ledger.size_problem_once(self.client, problem)

        starter_scores = [0.0 for _ in range(self.starter_agents)]
        starter_budgets = self.budget_ledger.split_among_siblings(
            parent_budget=self.budget_ledger.exploration_pool_usd,
            certainty_scores=starter_scores,
            apply_depth_decay=False,
        )

        for idx, s_budget in enumerate(starter_budgets):
            b_id = self._next_branch_id()
            directive = Directive(
                parent_reasoning_compressed="Root arena entry — starter agent.",
                fork_exploring=f"starter_agent_{idx + 1}",
                state=BranchState.POSSIBLE,
                budget=s_budget,
            )
            mini = MiniHarness(
                branch_id=b_id,
                parent_id=None,
                problem=problem,
                directive=directive,
                depth=0,
                client=self.client,
                tracker=self.tracker,
                detector=self.detector,
                budget_ledger=self.budget_ledger,
                event_sink=self._on_event,
                branch_counter=self._next_branch_id,
            )
            self.starters.append(mini)

        # All starter branches live at once in parallel
        with ThreadPoolExecutor(max_workers=max(1, len(self.starters))) as pool:
            futs = [
                pool.submit(mini.step, force_ns_on_starters)
                for mini in self.starters
            ]
            for f in as_completed(futs):
                f.result()

        selection_summary: Dict[str, Any] = {}
        if self.enable_selection:
            selection_summary = self._run_selection_phase()

        remaining_across_branches = round(
            sum(r.budget_remaining for r in self.tracker.all_records()), 6
        )

        return {
            "problem": problem,
            "model": getattr(self.client, "model", ""),
            "base_url": getattr(self.client, "base_url", ""),
            "is_mock": getattr(self.client, "is_mock", False),
            "budget": {
                "difficulty_estimate_usd": round(self.budget_ledger.difficulty_estimate_usd, 6),
                "root_budget_usd": round(self.budget_ledger.root_budget_usd, 6),
                "initial_finish_reserve_usd": round(self.budget_ledger.initial_finish_reserve_usd, 6),
                "finish_reserve_remaining_usd": round(self.budget_ledger.finish_reserve_usd, 6),
                "exploration_pool_usd": round(self.budget_ledger.exploration_pool_usd, 6),
                "total_spent_usd": round(self.budget_ledger.total_spent_usd, 6),
                "unspent_branch_budgets_usd": remaining_across_branches,
                "hard_cap_usd": round(self.budget_ledger.hard_cap_usd, 6),
            },
            "total_branches_spawned": len(self.tracker.all_records()),
            "selection": selection_summary,
            "winners_box": self.winners_box,
            "scoreboard": self.tracker.scoreboard(),
            "events_count": len(self.events),
        }
