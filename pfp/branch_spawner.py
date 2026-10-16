"""
Piece 2: Branch Spawner — directives, mini-harnesses, budget
(Pass 1: literal implementation per DESIGN.md).

Components:
- Directive: compact packet per branch containing parent's reasoning so far
  (compressed), the fork it's exploring, its state (Possible), its budget.
- BudgetLedger:
  * Root budget = 2 * difficulty_estimate (one cheap probe call sizes the
    problem, then double as buffer). Single round, hard cap as safety rail
    (user-set, default $2.00).
  * Depth decay: each level gets a fraction of its parent (* 0.5).
  * Certainty-weighted split among siblings, with a floor (never zero).
  * NS cadence derived from budget: `cadence = expected_agent_seconds / budget_calls`.
  * Kill conditions: budget exhausted, marked Fail, merged (Absorbed). No max depth.
  * Finish reserve: 10-15% of root held back; exploration can't touch it,
    only near-conclusion branches (high certainty, no open issues) can draw.
  * Graceful degradation: as budget depletes, NS slows -> stops -> no new
    spawns -> everything runs to conclusion.
- MiniHarness: every branch *is* a harness carrying the same machinery
  recursively (own divergence detection, own sub-branching, own NS cadence).
- MainHarness: runs the top level (default 3 starter agents, or 9 at real scale);
  mini-harnesses run subtrees; all events bubble up.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
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
    Single-round budget mechanics per Piece 2:
    - Root budget = min(hard_cap, 2 * difficulty_estimate)
    - Finish reserve = 10-15% (default 12.5%) of root held back for near-conclusion
      branches (high certainty, no open critic issues).
    - Depth decay = 0.5 per level.
    - Certainty-weighted split among siblings with a non-zero floor.
    - NS cadence = expected_agent_seconds / budget_calls.
    """

    def __init__(
        self,
        *,
        hard_cap_usd: float = 2.00,
        depth_decay: float = 0.5,
        finish_reserve_ratio: float = 0.125,
        min_branch_floor_usd: float = 0.002,
        expected_agent_seconds: float = 4.0,
        estimated_cost_per_call_usd: float = 0.0005,
        high_certainty_threshold: float = 0.0,
    ) -> None:
        if not (0.10 <= finish_reserve_ratio <= 0.15):
            raise ValueError("finish_reserve_ratio must be between 0.10 and 0.15 (10-15% of root).")
        self.hard_cap_usd = hard_cap_usd
        self.depth_decay = depth_decay
        self.finish_reserve_ratio = finish_reserve_ratio
        self.min_branch_floor_usd = min_branch_floor_usd
        self.expected_agent_seconds = expected_agent_seconds
        self.estimated_cost_per_call_usd = max(1e-6, estimated_cost_per_call_usd)
        self.high_certainty_threshold = high_certainty_threshold

        self.difficulty_estimate_usd: float = 0.0
        self.root_budget_usd: float = 0.0
        self.finish_reserve_usd: float = 0.0
        self.exploration_pool_usd: float = 0.0
        self.total_spent_usd: float = 0.0

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

        self.difficulty_estimate_usd = max(est, probe_cost * 4.0)
        self.root_budget_usd = min(self.hard_cap_usd, 2.0 * self.difficulty_estimate_usd)
        self.finish_reserve_usd = round(self.root_budget_usd * self.finish_reserve_ratio, 6)
        self.exploration_pool_usd = round(
            max(0.0, self.root_budget_usd - self.finish_reserve_usd - probe_cost), 6
        )
        self.total_spent_usd = probe_cost
        return self.root_budget_usd

    def compute_ns_cadence(self, branch_budget: float) -> float:
        """
        NS cadence derived from budget:
            cadence = expected_agent_seconds / budget_calls
        Tight budget -> slower cadence.
        """
        budget_calls = branch_budget / self.estimated_cost_per_call_usd
        if budget_calls <= 0:
            return float("inf")
        return self.expected_agent_seconds / budget_calls

    def split_among_siblings(
        self,
        parent_budget: float,
        certainty_scores: Sequence[float],
        apply_depth_decay: bool = True,
    ) -> List[float]:
        """
        Depth decay (x0.5 of parent) + Certainty-weighted split among siblings,
        with a floor (enough to prove itself or die, never zero).
        """
        n = len(certainty_scores)
        if n == 0:
            return []

        pool = parent_budget * (self.depth_decay if apply_depth_decay else 1.0)
        if pool <= 0:
            return [self.min_branch_floor_usd for _ in range(n)]

        # Shift certainty scores so weights are strictly positive
        min_score = min(certainty_scores)
        shifted = [(s - min_score) + 1.0 for s in certainty_scores]
        total_weight = sum(shifted)

        allocations: List[float] = []
        for w in shifted:
            raw_share = pool * (w / total_weight)
            allocations.append(max(self.min_branch_floor_usd, round(raw_share, 6)))
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
        is_near_conclusion = (
            record.state == BranchState.POSSIBLE
            and record.certainty_score >= self.high_certainty_threshold
            and record.open_critic_objections == 0
        )
        if not is_near_conclusion or self.finish_reserve_usd <= 0.0:
            return 0.0

        granted = min(self.finish_reserve_usd, requested_usd)
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
    Piece 2 Mini-harness:
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

        self.children: List[MiniHarness] = []
        self.ns_cadence_s: float = self.budget_ledger.compute_ns_cadence(directive.budget)
        self.last_ns_tick: float = time.time()
        self.ns_stopped: bool = False
        self.spawns_stopped: bool = False
        self.concluded: bool = False
        self.conclusion_text: Optional[str] = None

        # Register in Piece 3 PFPTracker
        self.record = self.tracker.register_branch(
            branch_id=self.branch_id,
            parent=self.parent_id,
            budget_remaining=self.directive.budget,
            fork_under_exploration=self.directive.fork_exploring,
            state=self.directive.state,
            certainty_score=0.0,
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

    def _spend(self, cost_usd: float, reason: str) -> None:
        rec = self.tracker.get(self.branch_id)
        new_budget = max(0.0, rec.budget_remaining - cost_usd)
        self.budget_ledger.total_spent_usd += cost_usd
        self.tracker.update_branch(self.branch_id, budget_remaining=new_budget)
        self.ns_cadence_s = self.budget_ledger.compute_ns_cadence(new_budget)
        self._emit(
            "budget_change",
            {
                "reason": reason,
                "spent_usd": round(cost_usd, 6),
                "budget_remaining": round(new_budget, 6),
                "ns_cadence_s": round(self.ns_cadence_s, 4) if self.ns_cadence_s != float("inf") else None,
            },
        )

    def _refresh_certainty(self, open_objections: Optional[int] = None) -> float:
        rec = self.tracker.get(self.branch_id)
        if open_objections is not None:
            rec.open_critic_objections = open_objections
        age_untested = max(0.0, time.time() - rec.last_tested_at)
        score = self.detector.compute_certainty(
            survived_falsifications=rec.survived_falsifications,
            merges_attracted=rec.merges_attracted,
            open_critic_objections=rec.open_critic_objections,
            seconds_since_last_test=age_untested,
        )
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
        if not forks or self.spawns_stopped or rec.budget_remaining <= self.budget_ledger.min_branch_floor_usd:
            return []

        compressed, comp_cost = compress_parent_reasoning(self.client, rec.reasoning_trace)
        if comp_cost > 0:
            self._spend(comp_cost, "directive_compression")

        rec = self.tracker.get(self.branch_id)
        # Sibling certainty weights (initial scores before selection reveals differences)
        sibling_scores = [rec.certainty_score for _ in forks]
        child_budgets = self.budget_ledger.split_among_siblings(
            parent_budget=rec.budget_remaining,
            certainty_scores=sibling_scores,
            apply_depth_decay=True,
        )

        # Deduct the decayed child pool (0.5 * parent_budget) from parent
        allocated_pool = min(rec.budget_remaining, rec.budget_remaining * self.budget_ledger.depth_decay)
        self._spend(allocated_pool, f"spawn_{trigger}_children")

        spawned: List[MiniHarness] = []
        for fork, c_budget in zip(forks, child_budgets):
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
                "forks": [f.fork_id for f in forks],
            },
        )
        return spawned

    def _run_to_conclusion(self) -> None:
        """
        Graceful degradation: when budget is low or continuations cluster,
        everything runs to conclusion — exploration degrades into exploitation,
        nothing dies mid-thought. Near-conclusion branches can draw from finish reserve.
        """
        rec = self.tracker.get(self.branch_id)
        if rec.budget_remaining < self.budget_ledger.min_branch_floor_usd:
            reserve_drawn = self.budget_ledger.draw_from_finish_reserve(
                rec, self.budget_ledger.min_branch_floor_usd * 2.0
            )
            if reserve_drawn > 0:
                self.tracker.update_branch(
                    self.branch_id,
                    budget_remaining=rec.budget_remaining + reserve_drawn,
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
        Execute one reasoning + divergence cycle on this MiniHarness, and recursively
        step any spawned child mini-harnesses.
        """
        killed, reason = self.is_killed()
        if killed:
            self._emit("kill", {"reason": reason})
            return

        rec = self.tracker.get(self.branch_id)

        # Graceful degradation check: as budget depletes, NS slows -> stops -> no new spawns -> conclude
        low_budget_threshold = self.budget_ledger.min_branch_floor_usd * 3.0
        if rec.budget_remaining <= low_budget_threshold:
            self.ns_stopped = True
            self.spawns_stopped = True
            self._emit(
                "degrade",
                {
                    "stage": "exploitation_only",
                    "ns_stopped": True,
                    "spawns_stopped": True,
                    "budget_remaining": round(rec.budget_remaining, 6),
                },
            )
            self._run_to_conclusion()
            return

        # 1. Plaus issue detection: fire K blind continuations & read issue off divergence
        parent_context = " | ".join(rec.reasoning_trace) or self.directive.parent_reasoning_compressed
        plaus_res: IssueDetectionResult = self.detector.detect_plaus(
            problem=self.problem,
            parent_reasoning=parent_context,
            depth=self.depth,
        )
        self._spend(plaus_res.cost_usd, "plaus_issue_detection")
        self._refresh_certainty(open_objections=len(plaus_res.unjustified_leaps))

        if plaus_res.should_split and not self.spawns_stopped:
            children = self._spawn_children_from_forks(plaus_res.distinct_forks, trigger="plaus")
            for child in children:
                child.step()
        else:
            # Continuations clustered -> no issue, merge/consolidate and flow straight
            if plaus_res.merged_continuation:
                self.tracker.update_branch(
                    self.branch_id,
                    reasoning_step=f"[Clustered flow] {plaus_res.merged_continuation}",
                )

        # 2. Check NS (natural selection) blind fan-out cadence if budget remains
        rec = self.tracker.get(self.branch_id)
        now = time.time()
        elapsed_since_ns = now - self.last_ns_tick
        if (
            not self.ns_stopped
            and not self.spawns_stopped
            and rec.budget_remaining > low_budget_threshold
            and (force_ns_check or elapsed_since_ns >= self.ns_cadence_s)
        ):
            self.last_ns_tick = now
            ns_res = self.detector.trigger_ns(
                problem=self.problem,
                parent_reasoning=" | ".join(rec.reasoning_trace),
                depth=self.depth,
                fanout_count=1,
            )
            self._spend(ns_res.cost_usd, "ns_blind_variation")
            self._emit(
                "ns_tick",
                {
                    "cadence_s": round(self.ns_cadence_s, 4),
                    "spawned_count": len(ns_res.distinct_forks),
                },
            )
            if ns_res.should_split:
                ns_children = self._spawn_children_from_forks(ns_res.distinct_forks, trigger="ns")
                for ns_child in ns_children:
                    ns_child.step()

        # 3. Conclude this branch if it hasn't concluded yet and still has alive state
        if not self.concluded and self.tracker.get(self.branch_id).is_alive:
            self._run_to_conclusion()


class MainHarness:
    """
    Main harness running the top level of PFP Darwinism (Pieces 1-3).
    - Sizes the problem in one cheap probe call (`root_budget = 2 * difficulty_estimate`,
      capped at `hard_cap_usd`, with 10-15% `finish_reserve`).
    - Spawns starter agents (default 3 for demo scale, configurable to 9 for real scale).
    - Each starter agent is a recursive `MiniHarness`; all events bubble up to `self.events`.
    - Aggregates all branch states in `self.tracker` (`PFPTracker`).
    """

    def __init__(
        self,
        *,
        client: Any,
        starter_agents: int = 3,
        hard_cap_usd: float = 2.00,
        depth_dial: int = 3,
        k_probes: int = 3,
    ) -> None:
        self.client = client
        self.starter_agents = starter_agents
        self.tracker = PFPTracker()
        self.detector = IssueDetector(client, k_probes=k_probes, depth_dial=depth_dial)
        self.budget_ledger = BudgetLedger(hard_cap_usd=hard_cap_usd)
        self.events: List[HarnessEvent] = []
        self.starters: List[MiniHarness] = []
        self._seq = 0

    def _next_branch_id(self) -> str:
        self._seq += 1
        return f"branch_{self._seq:04d}"

    def _on_event(self, event: HarnessEvent) -> None:
        self.events.append(event)

    def run(self, problem: str, force_ns_on_starters: bool = True) -> Dict[str, Any]:
        """
        Run the Phase 1 harness (Pieces 1-3) on `problem`:
        1. Size the problem once to establish single-round root budget and finish reserve.
        2. Fire starter probe to seed `starter_agents` initial MiniHarnesses.
        3. Run each starter MiniHarness (which recursively detects Plaus/NS splits,
           spawns sub-branches with decayed certainty-weighted budgets, and degrades
           gracefully to conclusion).
        4. Return aggregated tracker scoreboard, budget ledger summary, and event stream.
        """
        self.budget_ledger.size_problem_once(self.client, problem)

        # Split exploration pool across starter agents
        starter_scores = [0.0 for _ in range(self.starter_agents)]
        starter_budgets = self.budget_ledger.split_among_siblings(
            parent_budget=self.budget_ledger.exploration_pool_usd,
            certainty_scores=starter_scores,
            apply_depth_decay=False,
        )

        for idx in range(self.starter_agents):
            b_id = self._next_branch_id()
            directive = Directive(
                parent_reasoning_compressed="Root arena entry — starter agent.",
                fork_exploring=f"starter_agent_{idx + 1}",
                state=BranchState.POSSIBLE,
                budget=starter_budgets[idx],
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

        for mini in self.starters:
            mini.step(force_ns_check=force_ns_on_starters)

        return {
            "problem": problem,
            "model": getattr(self.client, "model", ""),
            "base_url": getattr(self.client, "base_url", ""),
            "is_mock": getattr(self.client, "is_mock", False),
            "budget": {
                "difficulty_estimate_usd": round(self.budget_ledger.difficulty_estimate_usd, 6),
                "root_budget_usd": round(self.budget_ledger.root_budget_usd, 6),
                "finish_reserve_usd": round(self.budget_ledger.finish_reserve_usd, 6),
                "exploration_pool_usd": round(self.budget_ledger.exploration_pool_usd, 6),
                "total_spent_usd": round(self.budget_ledger.total_spent_usd, 6),
                "hard_cap_usd": round(self.budget_ledger.hard_cap_usd, 6),
            },
            "total_branches_spawned": len(self.tracker.all_records()),
            "scoreboard": self.tracker.scoreboard(),
            "events_count": len(self.events),
        }
