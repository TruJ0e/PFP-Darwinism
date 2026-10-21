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

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from .backend import FAST_JUDGE_MODEL
from .issue_detector import IssueDetectionResult, IssueDetector, ProbeContinuation
from .jsonutil import parse_json_object
from .tracker import BranchRecord, BranchState, PFPTracker, id_free

# Smallest arena that can still do selection: 3 starters, their probe cycles,
# alignment/merge rounds and a 3-round falsification gauntlet. The difficulty
# estimate never sizes below this many calls (see BudgetLedger.size_problem_once).
MIN_ARENA_CALLS = 60


def _finite_number(value: Any, name: str) -> float:
    """Coerce a public budget input without admitting bool, NaN, or infinity."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, not bool.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def _non_negative_amount(value: Any, name: str) -> float:
    number = _finite_number(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be >= 0.")
    return number


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
        verification_share: float = 0.3,
        root_budget_override_usd: Optional[float] = None,
    ) -> None:
        hard_cap_usd = _non_negative_amount(hard_cap_usd, "hard_cap_usd")
        depth_decay = _finite_number(depth_decay, "depth_decay")
        finish_reserve_ratio = _finite_number(finish_reserve_ratio, "finish_reserve_ratio")
        min_branch_floor_usd = _finite_number(min_branch_floor_usd, "min_branch_floor_usd")
        conclusion_reserve_usd = _non_negative_amount(conclusion_reserve_usd, "conclusion_reserve_usd")
        expected_agent_seconds = _finite_number(expected_agent_seconds, "expected_agent_seconds")
        estimated_cost_per_call_usd = _finite_number(
            estimated_cost_per_call_usd, "estimated_cost_per_call_usd"
        )
        high_certainty_threshold = _finite_number(high_certainty_threshold, "high_certainty_threshold")
        verification_share = _finite_number(verification_share, "verification_share")
        if not (0.0 < depth_decay <= 1.0):
            raise ValueError("depth_decay must be in (0, 1].")
        if not (0.10 <= finish_reserve_ratio <= 0.15):
            raise ValueError("finish_reserve_ratio must be between 0.10 and 0.15 (10-15% of root).")
        if min_branch_floor_usd <= 0.0:
            raise ValueError("min_branch_floor_usd must be > 0.")
        if expected_agent_seconds <= 0.0:
            raise ValueError("expected_agent_seconds must be > 0.")
        if estimated_cost_per_call_usd <= 0.0:
            raise ValueError("estimated_cost_per_call_usd must be > 0.")
        if not (0.0 <= verification_share < 1.0):
            raise ValueError("verification_share must be in [0, 1).")
        # Fraction of every branch's budget that exploration and child spawning can't touch:
        # it pays for the branch's own falsification / alignment / merge (selection is not free)
        self.verification_share = verification_share
        self._cost_ema: Dict[str, float] = {}
        self.hard_cap_usd = hard_cap_usd
        if root_budget_override_usd is not None:
            root_budget_override_usd = _finite_number(
                root_budget_override_usd, "root_budget_override_usd"
            )
            if root_budget_override_usd <= 0.0:
                raise ValueError("root_budget_override_usd must be > 0.")
        self.root_budget_override_usd = root_budget_override_usd
        self.depth_decay = depth_decay
        self.finish_reserve_ratio = finish_reserve_ratio
        self.min_branch_floor_usd = min_branch_floor_usd
        self.conclusion_reserve_usd = conclusion_reserve_usd
        self.expected_agent_seconds = expected_agent_seconds
        self.estimated_cost_per_call_usd = estimated_cost_per_call_usd
        self.high_certainty_threshold = high_certainty_threshold

        self.difficulty_estimate_usd: float = 0.0
        self.root_budget_usd: float = 0.0
        self.initial_finish_reserve_usd: float = 0.0
        self.finish_reserve_usd: float = 0.0
        self.exploration_pool_usd: float = 0.0
        # True API spend (every call, whether or not a branch budget could cover it)
        self.total_spent_usd: float = 0.0
        # Spend no branch budget or reserve could cover — surfaced, never hidden
        self.overdraft_usd: float = 0.0
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
                        "Estimate compute difficulty for this problem as JSON. Price anchor: one model call costs about $0.0003 "
                        "(fan-out model $0.06/M input, $0.24/M output tokens; judge calls cost ~3x that). "
                        "difficulty_estimate_usd MUST be roughly estimated_calls x $0.0003 — do not guess a round dollar figure. "
                        "Context for estimated_calls: the arena runs several agents in parallel, each making probe, "
                        "falsification and merge calls — even a small problem takes 60-150 calls in total. "
                        '{"difficulty_estimate_usd": <float>, "estimated_calls": <int>}.'
                    ),
                },
                {
                    "role": "user",
                    "content": f"Size the problem and return difficulty_estimate:\n{problem}",
                },
            ],
            temperature=0.1,
            max_tokens=320,
        )
        probe_cost = _non_negative_amount(resp.usage.cost_usd, "probe cost_usd")
        raw = resp.choices[0].message.content if resp.choices else "{}"
        # Live models wrap JSON in fences / <think> blocks; a NaN, inf or negative
        # estimate must never become the root budget.
        parsed = parse_json_object(raw) or {}
        try:
            est = float(parsed.get("difficulty_estimate_usd", 0.05))
        except (TypeError, ValueError):
            est = 0.05
        if not math.isfinite(est) or est <= 0:
            est = 0.05

        # Structural floor: the sizing model prices a call well once anchored, but it
        # cannot know the arena's shape and chronically under-counts calls (the first
        # calibrated live run estimated ~6 calls and funded a single starter; the arena
        # actually used 83). The smallest useful arena — 3 starters, their probes,
        # selection and a 3-round gauntlet — takes ~60 calls, so never size below that.
        from .backend import estimate_call_cost_usd, get_model_name

        structural_floor = MIN_ARENA_CALLS * estimate_call_cost_usd(1200, 800, get_model_name())
        est = max(est, structural_floor)

        with self._lock:
            self.difficulty_estimate_usd = max(est, probe_cost * 4.0)
            if self.root_budget_override_usd is not None:
                # Operator override (e.g. an unfiltered-profile run): size the arena
                # directly instead of from the model's difficulty guess. The hard
                # cap still binds — the override can only ever lower the root.
                self.root_budget_usd = min(self.hard_cap_usd, self.root_budget_override_usd)
            else:
                self.root_budget_usd = min(self.hard_cap_usd, 2.0 * self.difficulty_estimate_usd)
            self.initial_finish_reserve_usd = round(self.root_budget_usd * self.finish_reserve_ratio, 6)
            self.finish_reserve_usd = self.initial_finish_reserve_usd
            self.exploration_pool_usd = round(
                max(0.0, self.root_budget_usd - self.finish_reserve_usd - probe_cost), 6
            )
            self.total_spent_usd = round(probe_cost, 6)
            return self.root_budget_usd

    def record_spend(self, amount_usd: float, covered_usd: Optional[float] = None) -> float:
        """
        Record a real API cost. `covered_usd` is the part paid from a branch budget
        or reserve; anything above it is booked as overdraft so the run report
        never understates spend.
        """
        with self._lock:
            amount = _non_negative_amount(amount_usd, "amount_usd")
            covered = amount if covered_usd is None else min(
                amount, _non_negative_amount(covered_usd, "covered_usd")
            )
            self.total_spent_usd = round(self.total_spent_usd + amount, 6)
            self.overdraft_usd = round(self.overdraft_usd + (amount - covered), 6)
            return self.total_spent_usd

    def charge(
        self,
        tracker: PFPTracker,
        branch_id: str,
        cost_usd: float,
        *,
        allow_finish_reserve: bool = False,
    ) -> Dict[str, float]:
        """
        Pay a real API cost on behalf of `branch_id`:
        1. Deduct from the branch's own budget.
        2. If `allow_finish_reserve` (verification / conclusion work), cover the
           shortfall from the finish reserve — exploration never touches it.
        3. Book whatever is still uncovered as overdraft.
        Always records the full cost as spent.
        """
        cost = _non_negative_amount(cost_usd, "cost_usd")
        with self._lock:
            rec = tracker.get(branch_id)
            from_branch = min(rec.budget_remaining, cost)
            new_budget = round(max(0.0, rec.budget_remaining - from_branch), 6)
            tracker.update_branch(branch_id, budget_remaining=new_budget)
            shortfall = round(cost - from_branch, 8)
            from_reserve = 0.0
            if shortfall > 0.0 and allow_finish_reserve and self.finish_reserve_usd > 0.0:
                from_reserve = round(min(self.finish_reserve_usd, shortfall), 6)
                self.finish_reserve_usd = round(self.finish_reserve_usd - from_reserve, 6)
            self.record_spend(cost, covered_usd=from_branch + from_reserve)
            return {
                "cost_usd": round(cost, 6),
                "from_branch_usd": round(from_branch, 6),
                "from_finish_reserve_usd": round(from_reserve, 6),
                "overdraft_usd": round(max(0.0, cost - from_branch - from_reserve), 6),
                "budget_remaining": new_budget,
            }

    def cover_from_finish_reserve(self, amount_usd: float) -> float:
        """Pay up to `amount_usd` of selection work from the finish reserve; returns what was granted."""
        amount = _non_negative_amount(amount_usd, "amount_usd")
        with self._lock:
            granted = round(min(self.finish_reserve_usd, amount), 6)
            self.finish_reserve_usd = round(self.finish_reserve_usd - granted, 6)
            return granted

    def return_to_finish_reserve(self, amount_usd: float) -> float:
        """Idle branch budget flowing back to the finish reserve (never creates money)."""
        amount = _non_negative_amount(amount_usd, "amount_usd")
        with self._lock:
            self.finish_reserve_usd = round(self.finish_reserve_usd + amount, 6)
            return self.finish_reserve_usd

    def observe_call_cost(self, kind: str, cost_usd: float) -> None:
        """Learn what each kind of call really costs (EMA) so affordability checks track reality."""
        cost = _non_negative_amount(cost_usd, "cost_usd")
        if cost == 0.0:
            return
        with self._lock:
            prev = self._cost_ema.get(kind)
            self._cost_ema[kind] = cost if prev is None else round(0.7 * prev + 0.3 * cost, 8)

    def observed_cost(self, kind: str) -> Optional[float]:
        """The learned (EMA) cost of `kind`, or None if no such call has been paid for yet."""
        with self._lock:
            return self._cost_ema.get(kind)

    def estimate_call_cost(self, kind: str) -> float:
        with self._lock:
            return self._cost_ema.get(kind, self.estimated_cost_per_call_usd)

    def protected_budget(self, initial_budget: float) -> float:
        """The verification share of a branch's initial budget (never spent on exploration)."""
        budget = _non_negative_amount(initial_budget, "initial_budget")
        return max(self.conclusion_reserve_usd, round(budget * self.verification_share, 6))

    def compute_ns_cadence(self, branch_budget: float) -> float:
        """
        NS cadence derived from budget:
            cadence = expected_agent_seconds / budget_calls
        Tight budget -> slower cadence.
        When spendable budget calls <= 0, returns inf (NS stopped).
        """
        budget = _non_negative_amount(branch_budget, "branch_budget")
        spendable = max(0.0, budget - self.conclusion_reserve_usd)
        budget_calls = spendable / self.estimated_cost_per_call_usd
        if budget_calls <= 0.0:
            return float("inf")
        return self.expected_agent_seconds / budget_calls

    def max_fundable_siblings(self, parent_budget: float, apply_depth_decay: bool = True) -> int:
        """
        Determine how many children can be funded at or above `min_branch_floor_usd`
        from the decayed child pool (`parent_budget * depth_decay`).
        """
        budget = _non_negative_amount(parent_budget, "parent_budget")
        pool = budget * (self.depth_decay if apply_depth_decay else 1.0)
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

        budget = _non_negative_amount(parent_budget, "parent_budget")
        pool = round(budget * (self.depth_decay if apply_depth_decay else 1.0), 6)
        max_n = int(pool // self.min_branch_floor_usd)
        if max_n <= 0:
            return []

        # Non-finite scores (a poisoned recall case, a NaN critic signal) get no extra share
        # instead of turning the whole split into NaN.
        active_scores = [
            float(s) if isinstance(s, (int, float)) and math.isfinite(s) else 0.0
            for s in certainty_scores[:max_n]
        ]
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
        requested = _non_negative_amount(requested_usd, "requested_usd")
        with self._lock:
            is_near_conclusion = (
                record.state == BranchState.POSSIBLE
                and record.certainty_score >= self.high_certainty_threshold
                and record.open_critic_objections == 0
            )
            if not is_near_conclusion or self.finish_reserve_usd <= 0.0:
                return 0.0

            granted = round(min(self.finish_reserve_usd, requested), 6)
            self.finish_reserve_usd = round(self.finish_reserve_usd - granted, 6)
            return granted


def _seed_certainty(case: Dict[str, Any]) -> float:
    """Recalled cases come from disk: a missing, non-numeric or non-finite score seeds nothing."""
    try:
        val = float(case.get("certainty_score", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return val if math.isfinite(val) and val > 0.0 else 0.0


def _seed_weight(case: Dict[str, Any]) -> float:
    """
    Budget weight of a recalled starter: its past certainty scaled by how similar its
    problem was to this one. A barely-related recall (similarity 0.15) must not draw the
    same head start as a near-exact match. Cases without a `similarity` (seeded directly,
    not via `recall`) count as exact matches; a bad value seeds nothing.
    """
    raw = case.get("similarity", 1.0)
    try:
        sim = float(raw) if not isinstance(raw, bool) else 0.0
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(sim):
        return 0.0
    return _seed_certainty(case) * min(1.0, max(0.0, sim))


def _conclusion_text(raw: str) -> str:
    """Models often answer the conclusion prompt in JSON; surface the conclusion itself."""
    data = parse_json_object(raw)
    if data is not None:
        for key in ("conclusion", "solution", "answer", "unified_reasoning", "reasoning"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return raw.strip()


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
        max_tokens=320,
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
        origin: str = "starter",
        should_stop: Optional[Callable[[], bool]] = None,
        abort_run: Optional[Callable[[], None]] = None,
        stop_reason: Optional[Callable[[], str]] = None,
        ns_admit: Optional[Callable[[], bool]] = None,
        ns_settle: Optional[Callable[[int], None]] = None,
    ) -> None:
        self.branch_id = branch_id
        # NS share cap (v0.43): the arena admits an NS tick only while blind variation stays
        # within its share of the population; a refused tick buys no probe
        self.ns_admit: Callable[[], bool] = ns_admit or (lambda: True)
        self.ns_settle: Callable[[int], None] = ns_settle or (lambda spawned: None)
        self.origin = origin
        self.should_stop: Callable[[], bool] = should_stop or (lambda: False)
        # Trips the run-wide stop when any branch fails, so its siblings and cousins start
        # no further paid calls while the failure propagates (nested pools wait on them).
        self.abort_run: Callable[[], None] = abort_run or (lambda: None)
        self.stop_reason: Callable[[], str] = stop_reason or (lambda: "emergency_stop")
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
        # Verification share: exploration and child spawning never touch it (see BudgetLedger)
        self.protected_usd: float = budget_ledger.protected_budget(directive.budget)
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
            origin=origin,
        )
        if self.directive.parent_reasoning_compressed:
            self.tracker.update_branch(
                self.branch_id,
                reasoning_step=f"[Directive] {self.directive.parent_reasoning_compressed}",
            )

        spawn_details: Dict[str, Any] = {
            "depth": self.depth,
            "directive": self.directive.to_dict(),
            "ns_cadence_s": round(self.ns_cadence_s, 4) if self.ns_cadence_s != float("inf") else None,
        }
        if self.parent_id is None and origin == "starter":
            from .arena import MAX_STREAM_PROBLEM_CHARS

            # The journal names its own problem: replays and proof reports need no --problem.
            spawn_details["problem"] = str(problem)[:MAX_STREAM_PROBLEM_CHARS]
        self._emit("spawn", spawn_details)

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

    def _can_afford(self, kind: str) -> bool:
        """Exploration is affordable only from budget above the protected verification share."""
        rec = self.tracker.get(self.branch_id)
        return rec.budget_remaining - self.protected_usd >= self.budget_ledger.estimate_call_cost(kind)

    def _spend(self, cost_usd: float, reason: str, kind: Optional[str] = None) -> float:
        self.budget_ledger.observe_call_cost(kind or reason, cost_usd)
        charged = self.budget_ledger.charge(self.tracker, self.branch_id, cost_usd)
        new_budget = charged["budget_remaining"]
        self.ns_cadence_s = self.budget_ledger.compute_ns_cadence(new_budget)
        self._emit(
            "budget_change",
            {
                "reason": reason,
                "spent_usd": charged["cost_usd"],
                "overdraft_usd": charged["overdraft_usd"],
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
                max(0.0, current_budget - self.protected_usd),
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
        base_score = self.detector.compute_certainty(
            survived_falsifications=rec.survived_falsifications,
            merges_attracted=rec.merges_attracted,
            open_critic_objections=rec.open_critic_objections,
            untested_rounds=self.tracker.untested_rounds(rec),
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

        # Hold back the protected verification share on the parent so it can still conclude
        # and face falsification — the parent never dies mid-thought
        spendable_for_branching = max(0.0, rec.budget_remaining - self.protected_usd)
        max_kids = self.budget_ledger.max_fundable_siblings(spendable_for_branching, apply_depth_decay=True)
        if max_kids <= 0:
            self._evaluate_graceful_degradation(rec.budget_remaining)
            return []

        # Order candidate forks by behavioral cluster support_score so if budget only funds M < N,
        # the highest-confidence forks are prioritized
        ordered_forks = sorted(forks, key=lambda f: f.support_score, reverse=True)[:max_kids]

        compressed, comp_cost = compress_parent_reasoning(self.client, rec.reasoning_trace)
        if comp_cost > 0:
            self._spend(comp_cost, "directive_compression", kind="compress")

        rec = self.tracker.get(self.branch_id)
        spendable_for_branching = max(0.0, rec.budget_remaining - self.protected_usd)
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
                origin=trigger,
                should_stop=self.should_stop,
                abort_run=self.abort_run,
                stop_reason=self.stop_reason,
                ns_admit=self.ns_admit,
                ns_settle=self.ns_settle,
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
            try:
                for f in as_completed(futs):
                    f.result()
            except BaseException:
                # Leaving the `with` waits for every sibling: stop them starting paid calls first
                self.abort_run()
                for f in futs:
                    f.cancel()
                raise

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
                        f"Trace: {' | '.join(rec.reasoning_trace)}\n"
                        "State this branch's concrete conclusion (the solution it proposes) in 1-3 sentences."
                    ),
                }
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        self._spend(float(resp.usage.cost_usd), "run_to_conclusion", kind="conclude")
        content = _conclusion_text((resp.choices[0].message.content if resp.choices else "") or "")
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
        if self.should_stop():
            # A run-wide stop kills the branch; the ledger keeps it as-is for post-mortem
            self._emit("kill", {"reason": self.stop_reason()})
            return

        rec = self.tracker.get(self.branch_id)
        stage = self._evaluate_graceful_degradation(rec.budget_remaining)

        if stage == "exploitation_only" or not self._can_afford("plaus"):
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
        self._spend(plaus_res.cost_usd, "plaus_issue_detection", kind="plaus")
        cluster_bonus = plaus_res.distinct_forks[0].support_score if plaus_res.distinct_forks else 0.0
        self._refresh_certainty(
            open_objections=len(plaus_res.unjustified_leaps),
            support_bonus=cluster_bonus,
        )

        if self.should_stop():
            self._emit("kill", {"reason": self.stop_reason()})
            return

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
            # An unseeded starter has no fork of its own yet: the cluster it flows along is its fork
            if self.directive.fork_exploring.startswith("starter_agent_") and plaus_res.distinct_forks:
                self.tracker.update_branch(
                    self.branch_id,
                    fork_under_exploration=plaus_res.distinct_forks[0].fork_description,
                )

        # 2. Check NS (natural selection) blind fan-out cadence if still in exploration stage
        rec = self.tracker.get(self.branch_id)
        now = time.time()
        elapsed_since_ns = now - self.last_ns_tick
        if (
            not self.ns_stopped
            and not self.spawns_stopped
            and not self.should_stop()
            and self._can_afford("ns")
            and (force_ns_check or elapsed_since_ns >= self.ns_cadence_s)
            and self.ns_admit()
        ):
            self.last_ns_tick = now
            ns_children: List[MiniHarness] = []
            try:
                ns_res = self.detector.trigger_ns(
                    problem=self.problem,
                    parent_reasoning=" | ".join(rec.reasoning_trace),
                    depth=self.depth,
                    fanout_count=1,
                    known_forks=self.tracker.known_forks_in_lineage(self.branch_id),
                )
                self._spend(ns_res.cost_usd, "ns_blind_variation", kind="ns")
                self._emit(
                    "ns_tick",
                    {
                        "cadence_s": round(self.ns_cadence_s, 4) if self.ns_cadence_s != float("inf") else None,
                        "spawned_count": len(ns_res.distinct_forks),
                    },
                )
                if ns_res.should_split and not self.spawns_stopped:
                    ns_children = self._spawn_children_from_forks(ns_res.distinct_forks, trigger="ns")
            finally:
                # Settle the admission before the children run: they compete for NS slots too
                self.ns_settle(len(ns_children))
            self._run_children_parallel(ns_children)

        # 3. Conclude this branch cleanly so nothing dies mid-thought
        if not self.concluded and not self.should_stop() and self.tracker.get(self.branch_id).is_alive:
            self._run_to_conclusion()


class MainHarness:
    """
    Main harness running the top level of PFP Darwinism (Pieces 1-6).
    - Sizes the problem in one cheap probe call (`root_budget = 2 * difficulty_estimate`,
      capped at `hard_cap_usd`, with 10-15% `finish_reserve`).
    - Spawns starter agents (default 3 for demo scale, configurable to 9 for real scale).
    - Runs all starter MiniHarnesses concurrently ("All branches live at once, all the time").
    - Runs Phase 2 Selection (Pieces 4-6) in repeated rounds until the arena settles:
      * AlignmentDetector (embedding pre-filter + Super claim-by-claim judge)
      * MergeSynthesizer (unifies compatible branches, preserves provenance, transfers parent budgets)
      * FailJudge (post-merge verification, contradiction resolution, background sampling,
        3-round Pass-readiness gauntlet with rotated attack styles, 1 appeal per kill with a
        different judge, revokable Pass with dependent re-verification, and pre-exit check
        into the winner's box).
    - Safety rails: every model call goes through a `MeteredClient` (true spend, hard cap,
      bounded concurrency); `emergency_stop()` (or the registry's emergency stop) kills the run.
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
        version_registry: Optional[Any] = None,
        seed_cases: Optional[Sequence[Dict[str, Any]]] = None,
        selection_rounds: int = 3,
        max_concurrent_calls: int = 32,
        max_winners: int = 3,
        depth_decay: float = 0.5,
        finish_reserve_ratio: float = 0.125,
        root_budget_override_usd: Optional[float] = None,
        fast_judge_model: Optional[str] = FAST_JUDGE_MODEL,
        ns_share_cap: float = 0.25,
    ) -> None:
        from .alignment_detector import AlignmentDetector
        from .arena import ArenaFeed, TronArenaRenderer
        from .backend import MeteredClient
        from .fail_judge import FailJudge
        from .merge_synthesizer import MergeSynthesizer

        if starter_agents < 1:
            raise ValueError("starter_agents must be >= 1")
        ns_share_cap = _finite_number(ns_share_cap, "ns_share_cap")
        if not (0.0 <= ns_share_cap <= 1.0):
            raise ValueError("ns_share_cap must be in [0, 1].")
        if not isinstance(hard_cap_usd, (int, float)) or not math.isfinite(hard_cap_usd) or hard_cap_usd <= 0:
            # NaN compares False against every spend, inf never binds: either would silently
            # turn the hard cap off, so both are refused before a client is wrapped.
            raise ValueError("hard_cap_usd must be a finite number > 0")

        self.raw_client = client
        self.client = MeteredClient(
            client,
            hard_cap_usd=hard_cap_usd,
            max_concurrent_calls=max_concurrent_calls,
        )
        self.starter_agents = starter_agents
        self.enable_selection = enable_selection
        self.version_registry = version_registry
        self.seed_cases: List[Dict[str, Any]] = list(seed_cases or [])
        self.selection_rounds = max(1, int(selection_rounds))
        self.max_winners = max(1, int(max_winners))
        self.tracker = PFPTracker()
        self.detector = IssueDetector(self.client, k_probes=k_probes, depth_dial=depth_dial)
        self.budget_ledger = BudgetLedger(
            hard_cap_usd=hard_cap_usd,
            depth_decay=depth_decay,
            finish_reserve_ratio=finish_reserve_ratio,
            root_budget_override_usd=root_budget_override_usd,
        )
        self.events: List[HarnessEvent] = []
        self.arena_feed = ArenaFeed()
        self.arena_renderer = TronArenaRenderer()
        self.starters: List[MiniHarness] = []
        self.halted_reason: Optional[str] = None
        self._seq = 0
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._abort_reason: Optional[str] = None
        # NS share cap (v0.43): blind variation may be at most this share of all spawned branches.
        # Run 5 grew 97 NS of 174 (56%): mutations too mutually alien to recombine, while Plaus
        # children — the mergeable stock — were crowded out. 1.0 = uncapped (pre-v0.43).
        self.ns_share_cap = ns_share_cap
        self._ns_lock = threading.Lock()
        self._ns_spawned = 0
        self._ns_reserved = 0
        self._ns_capped_ticks = 0

        # Judge ladder (v0.41): first-pass judging on a fast instruct model, Super for appeals,
        # escalations, the final gauntlet round, the pre-exit check and merge synthesis.
        # None (or the Super model id) puts every judge back on Super.
        self.fast_judge_model: Optional[str] = fast_judge_model or None
        self.alignment_detector = AlignmentDetector(self.client, judge_model=self.fast_judge_model)
        self.merge_synthesizer = MergeSynthesizer(
            client=self.client,
            tracker=self.tracker,
            budget_ledger=self.budget_ledger,
            event_sink=self._on_event,
        )
        self.fail_judge = FailJudge(
            client=self.client,
            tracker=self.tracker,
            detector=self.detector,
            budget_ledger=self.budget_ledger,
            gauntlet_rounds=gauntlet_rounds,
            fast_model=self.fast_judge_model,
            event_sink=self._on_event,
        )
        self.winners_box: List[Dict[str, Any]] = []
        # Concurrent judge calls during selection (alignment verdicts, merges, the NS sweep).
        # Judge latency, not money, set run 5's 28 minutes: ~100 verdicts one after another.
        self.judge_parallelism: int = 8

    # ------------------------------------------------------------------
    # Run-boundary configuration + safety rails
    # ------------------------------------------------------------------
    def _apply_active_version_dials(self) -> Optional[str]:
        """Lock the active registry version at the run boundary and apply its (validated) dials + prompts."""
        from .dials import resolve_dials, weights_from_dials

        if self.version_registry is None:
            return None
        snap = self.version_registry.begin_run()
        dials = resolve_dials(snap.config_dials)
        self.detector.weights = weights_from_dials(dials)
        self.alignment_detector.prefilter_threshold = float(dials["prefilter_threshold"])
        self.fail_judge.gauntlet_rounds = int(dials["gauntlet_rounds"])
        self.detector.probe_hint = str(snap.prompts.get("plaus_probe_hint", ""))
        return snap.version_id

    def emergency_stop(self) -> None:
        """Kill the live run: no new model calls start in any branch; selection is skipped."""
        self._stop_event.set()

    def _abort_run(self) -> None:
        """A branch failed: stop every other branch before the failure reaches `run`."""
        with self._lock:
            if self._abort_reason is None and not self._stop_event.is_set():
                self._abort_reason = "run_halted"
        self._stop_event.set()

    def _stop_reason(self) -> str:
        return self._abort_reason or "emergency_stop"

    def _should_stop(self) -> bool:
        if self._stop_event.is_set():
            return True
        reg = self.version_registry
        return bool(reg is not None and getattr(reg, "emergency_stopped", False))

    def _ns_admit(self) -> bool:
        """
        Reserve one NS slot if the population can take another blind mutation within the cap:
        (NS spawned + in flight + 1) <= cap x (branches registered + in flight + 1). The tick
        fans out one fork, so a slot is one child. Refused ticks are counted, never paid for.
        """
        total = len(self.tracker.all_records())
        with self._ns_lock:
            in_flight = self._ns_reserved
            if self._ns_spawned + in_flight + 1 > self.ns_share_cap * (total + in_flight + 1):
                self._ns_capped_ticks += 1
                return False
            self._ns_reserved += 1
            return True

    def _ns_settle(self, spawned: int) -> None:
        with self._ns_lock:
            self._ns_reserved = max(0, self._ns_reserved - 1)
            self._ns_spawned += max(0, int(spawned))

    def _next_branch_id(self) -> str:
        with self._lock:
            self._seq += 1
            return f"branch_{self._seq:04d}"

    def _on_event(self, event: HarnessEvent) -> None:
        with self._lock:
            self.events.append(event)
            self.arena_feed.ingest_harness_event(event)

    def _emit_halt(self, reason: str, detail: str) -> None:
        """Arena-level event (not tied to a branch) so renderers can show why the run stopped."""
        self._on_event(
            HarnessEvent(
                event_type="halt",
                branch_id="arena",
                timestamp=time.time(),
                details={"reason": reason, "detail": detail},
                snapshot={"id": "arena", "state": "Halted"},
            )
        )

    def render_arena(self, problem: str = "") -> str:
        return self.arena_renderer.render_ascii(self.arena_feed.events, problem=problem)

    # ------------------------------------------------------------------
    # Selection (Pieces 4-6)
    # ------------------------------------------------------------------
    def _can_afford_merge(self) -> bool:
        """
        True while the root budget covers one more merge round trip (synthesis + post-merge check)
        *and* still leaves the finish for the child it creates: a merged child is only worth
        anything if it can run the Pass gauntlet plus the pre-exit check. Costs are the ones
        learned from merges already paid for this run; before the first merge nothing is known,
        so the first merge is gated by headroom alone.
        """
        ledger = self.budget_ledger
        headroom = ledger.root_budget_usd - ledger.total_spent_usd
        if headroom <= 0.0:
            return False
        if ledger.observed_cost("merge_round_trip") is None:
            return True
        finish_calls = self.fail_judge.gauntlet_rounds + 1
        need = ledger.estimate_call_cost("merge_round_trip") + finish_calls * ledger.estimate_call_cost(
            "merge_falsification"
        )
        return headroom >= need

    def _affordable_merges(self) -> int:
        """How many merge round trips (plus each child's finish) the root headroom still covers."""
        ledger = self.budget_ledger
        headroom = ledger.root_budget_usd - ledger.total_spent_usd
        if headroom <= 0.0:
            return 0
        finish_calls = self.fail_judge.gauntlet_rounds + 1
        need = ledger.estimate_call_cost("merge_round_trip") + finish_calls * ledger.estimate_call_cost(
            "merge_falsification"
        )
        if need <= 0.0:
            return 0
        return int(headroom // need)

    def _affordable_alignment_pairs(self) -> int:
        """
        Alignment verdicts the root headroom covers once the finish is set aside (a full gauntlet
        plus pre-exit for every winner's-box slot). Coverage scales with the population only as
        far as the money does: on the 9-starter unfiltered mock an ungated plan overdrew the root
        by 6%. Per-pair cost is learned from verdicts already paid for this run.
        """
        ledger = self.budget_ledger
        headroom = ledger.root_budget_usd - ledger.total_spent_usd
        finish = self.max_winners * (self.fail_judge.gauntlet_rounds + 1) * ledger.estimate_call_cost(
            "merge_falsification"
        )
        per_pair = ledger.estimate_call_cost("alignment_pair")
        spare = headroom - finish
        if spare <= 0.0 or per_pair <= 0.0:
            return 0
        return int(spare // per_pair)

    def _parallel_map(self, fn: Callable[[Any], Any], items: Sequence[Any]) -> List[Any]:
        """`fn` over `items` on up to `judge_parallelism` threads; results in `items` order."""
        items = list(items)
        workers = max(1, min(int(self.judge_parallelism), len(items)))
        if workers == 1:
            return [fn(item) for item in items]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(fn, item) for item in items]
            try:
                return [f.result() for f in futs]
            except BaseException:
                # A failed judge call (hard cap, backend error) stops the rest starting paid calls
                self._abort_run()
                for f in futs:
                    f.cancel()
                raise

    def _merge_and_verify(self, job: tuple) -> tuple:
        """One merge round trip: Super synthesis, then the child's post-merge falsification."""
        al, child_id = job
        merge_res = self.merge_synthesizer.merge_branches(al.branch_a_id, al.branch_b_id, child_id, alignment=al)
        # Schedule Trigger 1: after every merge (new claims -> verify before going anywhere)
        post_merge_out = self.fail_judge.verify_after_merge(child_id)
        return merge_res, post_merge_out

    def _charge_alignment(self, al: Any) -> None:
        """Split a judge call's real cost across the two branches it compared (verification work)."""
        half_cost = al.cost_usd / 2.0
        for bid in (al.branch_a_id, al.branch_b_id):
            charged = self.budget_ledger.charge(self.tracker, bid, half_cost, allow_finish_reserve=True)
            self._emit_budget(bid, "alignment_judge", charged)

    def _emit_budget(self, branch_id: str, reason: str, charged: Dict[str, float]) -> None:
        self._on_event(
            HarnessEvent(
                event_type="budget_change",
                branch_id=branch_id,
                timestamp=time.time(),
                details={
                    "reason": reason,
                    "spent_usd": charged["cost_usd"],
                    "from_finish_reserve_usd": charged["from_finish_reserve_usd"],
                    "overdraft_usd": charged["overdraft_usd"],
                    "budget_remaining": charged["budget_remaining"],
                },
                snapshot=self.tracker.get(branch_id).to_dict(),
            )
        )

    def _split_parent_ids(self) -> set:
        return {r.parent for r in self.tracker.all_records() if r.parent and r.origin != "merge"}

    def _reclaim_split_parent_budgets(self) -> float:
        """
        A branch that split handed its flow to its children; its idle budget would
        otherwise sit unused while verification overdraws. Return it to the finish
        reserve, which pays for selection (falsification, alignment, pre-exit).
        """
        reclaimed = 0.0
        split_parents = self._split_parent_ids()
        for rec in self.tracker.all_records():
            if rec.id not in split_parents or rec.budget_remaining <= 0.0:
                continue
            amount = rec.budget_remaining
            self.tracker.update_branch(rec.id, budget_remaining=0.0)
            self.budget_ledger.return_to_finish_reserve(amount)
            reclaimed += amount
            self._on_event(
                HarnessEvent(
                    event_type="budget_change",
                    branch_id=rec.id,
                    timestamp=time.time(),
                    details={
                        "reason": "returned_to_finish_reserve",
                        "returned_usd": round(amount, 6),
                        "budget_remaining": 0.0,
                    },
                    snapshot=self.tracker.get(rec.id).to_dict(),
                )
            )
        return round(reclaimed, 6)

    @staticmethod
    def _solution_key(rec: BranchRecord) -> str:
        # ID-free: a merged fork that names its parents is the same solution whichever IDs it drew
        return " ".join(id_free(rec.fork_under_exploration).lower().split())

    def _age_untested_branches(self) -> int:
        """
        Close one arena round: every live branch the round left untested decays by one unit of
        `age_without_testing`, then the logical clock ticks. Untested age used to be wall-clock
        seconds, so call latency and thread scheduling decided which of two equal branches decayed
        (and so the background target, sibling budgets and the pre-exit cut); counted in rounds it
        is a pure function of the run's decisions. Returns how many branches decayed.
        """
        from .tracker import content_order_key

        weight = float(self.detector.weights.age_without_testing)
        decayed = 0
        with self.tracker.atomic():
            closing = self.tracker.arena_round
            for rec in sorted(self.tracker.active_branches(), key=content_order_key):
                if rec.last_tested_round < closing and weight:
                    self.tracker.update_branch(rec.id, certainty_score=round(rec.certainty_score + weight, 4))
                    decayed += 1
            self.tracker.advance_round()
        return decayed

    def _run_selection_phase(self) -> Dict[str, Any]:
        """
        Execute Phase 2 selection (Pieces 4-6) across active branches:
        1. Background sampling on the highest-certainty untested branch.
        2. Repeated rounds (until nothing merges or clashes, max `selection_rounds`):
           alignment detection -> contradiction trigger (falsify both, loser dies) ->
           merge synthesis + immediate post-merge falsification. Merged children can
           merge again next round, so cooperation compounds.
        3. Natural selection on blind variation: every untested NS-born branch faces
           the falsifier before anything exits.
        4. Pass-readiness gauntlet + pre-exit verification into the winner's box
           (higher certainty exits first; duplicate solutions collapse to one entry).
        """
        from .alignment_detector import AlignmentVerdict
        from .tracker import content_order_key

        merges_performed: List[Dict[str, Any]] = []
        contradictions_resolved: List[Dict[str, Any]] = []
        falsifications_run: List[Dict[str, Any]] = []
        alignments_evaluated: List[Dict[str, Any]] = []
        rounds_run = 0
        reclaimed_usd = self._reclaim_split_parent_budgets()
        # The spawn phase is arena round 0; selection rounds count from 1, so a branch the first
        # round leaves untested has aged one round (no one is tested while spawning)
        self.tracker.advance_round()

        bg_out = self.fail_judge.sample_background(force=True)
        if bg_out is not None:
            falsifications_run.append(bg_out.to_dict())

        for round_idx in range(self.selection_rounds):
            if self._should_stop():
                break
            # Graceful degradation at arena level: once the root budget is spent, no further
            # cooperation rounds — only the finish (pre-exit verification) remains
            if round_idx > 0 and self.budget_ledger.total_spent_usd >= self.budget_ledger.root_budget_usd:
                break
            rounds_run += 1
            # Coverage scales with the population (v0.42): cluster, match within clusters,
            # pit cluster champions against each other, judge the whole plan concurrently
            # The first round keeps the pre-v0.42 guarantee of up to 8 verdicts; beyond that (and
            # in every later round) the plan is as large as the population and the money allow
            max_pairs = self._affordable_alignment_pairs()
            if round_idx == 0:
                max_pairs = max(8, max_pairs)
            if max_pairs <= 0:
                break
            active = self.tracker.active_branches()
            plan = self.alignment_detector.plan_pairs(
                active, max_pairs=max_pairs, roots_of=self.tracker.starter_roots
            )
            alignments = self.alignment_detector.judge_planned_pairs(
                active, plan, max_workers=self.judge_parallelism
            )
            for al in alignments:
                self._charge_alignment(al)
                self.budget_ledger.observe_call_cost("alignment_pair", al.cost_usd)
                alignments_evaluated.append(al.to_dict())

            changed = False
            for al in alignments:
                if al.verdict != AlignmentVerdict.CONTRADICTORY:
                    continue
                ba = self.tracker.get(al.branch_a_id)
                bb = self.tracker.get(al.branch_b_id)
                if not (ba.is_alive and bb.is_alive):
                    continue
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
                changed = True

            # Overlap verdicts on disjoint live pairs, in plan order (a branch in two overlap
            # verdicts merges once this round; its child is compared afresh next round)
            pending: List[Any] = []
            merging: set = set()
            for al in alignments:
                if al.verdict != AlignmentVerdict.OVERLAP:
                    continue
                if al.branch_a_id in merging or al.branch_b_id in merging:
                    continue
                if not (self.tracker.get(al.branch_a_id).is_alive and self.tracker.get(al.branch_b_id).is_alive):
                    continue
                merging.update((al.branch_a_id, al.branch_b_id))
                pending.append(al)

            while pending and not self._should_stop():
                # A merge buys a synthesis call plus its post-merge falsification. Once the root
                # budget can't cover another one (learned from the merges already paid for), stop
                # cooperating and keep what's left for the pre-exit finish. Before the first merge
                # nothing is known: it runs alone, gated by headroom; later merges run as a batch
                # the learned costs can fund.
                if not self._can_afford_merge():
                    break
                if self.budget_ledger.observed_cost("merge_round_trip") is None:
                    batch = pending[:1]
                else:
                    batch = pending[: max(1, self._affordable_merges())]
                pending = pending[len(batch):]
                # Child IDs are drawn here, in plan order, never by whichever worker finishes first
                jobs = [(al, self._next_branch_id()) for al in batch]
                for (al, child_id), (merge_res, post_merge_out) in zip(jobs, self._parallel_map(self._merge_and_verify, jobs)):
                    merges_performed.append(merge_res.to_dict())
                    changed = True
                    falsifications_run.append(post_merge_out.to_dict())
                    self.budget_ledger.observe_call_cost(
                        "merge_round_trip", merge_res.synthesis_cost_usd + post_merge_out.cost_usd
                    )
                    self.budget_ledger.observe_call_cost("merge_falsification", post_merge_out.cost_usd)
                    if self.tracker.get(child_id).is_alive:
                        self.tracker.update_branch(
                            child_id,
                            reasoning_step=f"[Conclusion] {merge_res.unified_reasoning}",
                        )
            self._age_untested_branches()
            # A quiet round ends selection only once coverage is exhausted: unjudged candidate
            # pairs left over (the plan matches each branch once per round) get the next round
            if not changed and self.alignment_detector.last_unjudged_remaining <= 0:
                break

        # Blind variation must face selection: NS-born branches never exit untested
        if not self._should_stop():
            ns_untested = [
                b.id
                for b in sorted(self.tracker.active_branches(), key=content_order_key)
                if b.origin == "ns" and b.state == BranchState.POSSIBLE and b.survived_falsifications == 0
            ]
            # Independent attacks on distinct branches: run them side by side (run 5 queued 97)
            for out in self._parallel_map(
                lambda bid: self.fail_judge.falsify_branch(bid, schedule_trigger="background_sample"),
                ns_untested,
            ):
                falsifications_run.append(out.to_dict())

        # Pre-exit: frontier branches only (a branch that split handed its flow to its
        # children — the river splits around the rock), distinct solutions, highest certainty first
        winners: List[BranchRecord] = []
        if not self._should_stop():
            split_parents = self._split_parent_ids()
            seen: set = set()
            top_candidates: List[BranchRecord] = []
            frontier = [r for r in self.tracker.active_branches() if r.id not in split_parents]
            # Certainty ties (common: equal evidence -> equal certainty) resolve on content, so which
            # duplicate stands for a solution, and who makes the max_winners cut, never follows the
            # registration order that parallel subtrees leave to thread scheduling
            for rec in sorted(frontier, key=lambda r: (-r.certainty_score, content_order_key(r))):
                key = self._solution_key(rec)
                if key in seen:
                    continue
                seen.add(key)
                top_candidates.append(rec)
                if len(top_candidates) >= self.max_winners:
                    break
            winners = self.fail_judge.verify_pre_exit([c.id for c in top_candidates])
        self.winners_box = [w.to_dict() for w in winners]

        return {
            "rounds_run": rounds_run,
            "reclaimed_to_finish_reserve_usd": reclaimed_usd,
            "alignments_evaluated": alignments_evaluated,
            "unjudged_pairs_remaining": self.alignment_detector.last_unjudged_remaining,
            "merges_performed": merges_performed,
            "contradictions_resolved": contradictions_resolved,
            "falsifications_count": len(falsifications_run),
            "winners_box": self.winners_box,
        }

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def _spawn_starters(self, problem: str) -> None:
        starter_scores = [
            _seed_weight(self.seed_cases[idx]) * 0.25 if idx < len(self.seed_cases) else 0.0
            for idx in range(self.starter_agents)
        ]
        starter_budgets = self.budget_ledger.split_among_siblings(
            parent_budget=self.budget_ledger.exploration_pool_usd,
            certainty_scores=starter_scores,
            apply_depth_decay=False,
        )
        for idx, s_budget in enumerate(starter_budgets):
            b_id = self._next_branch_id()
            if idx < len(self.seed_cases):
                seed_fork = str(self.seed_cases[idx].get("fork_shape", f"starter_agent_{idx + 1}"))
                seed_prefix = f"Root arena entry — seeded from recalled win: {seed_fork}"
            else:
                seed_fork = f"starter_agent_{idx + 1}"
                seed_prefix = "Root arena entry — starter agent."
            directive = Directive(
                parent_reasoning_compressed=seed_prefix,
                fork_exploring=seed_fork,
                state=BranchState.POSSIBLE,
                budget=s_budget,
            )
            self.starters.append(
                MiniHarness(
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
                    origin="starter",
                    should_stop=self._should_stop,
                    abort_run=self._abort_run,
                    stop_reason=self._stop_reason,
                    ns_admit=self._ns_admit,
                    ns_settle=self._ns_settle,
                )
            )

    def run(self, problem: str, force_ns_on_starters: bool = True) -> Dict[str, Any]:
        """
        Run the PFP Darwinism harness on `problem`:
        1. Lock active version in `version_registry` (if attached) at the run boundary.
        2. Size the problem once to establish single-round root budget and finish reserve.
        3. Split exploration pool across `starter_agents` initial MiniHarnesses (seeded with
           any recalled winning branch shapes from `seed_cases`).
        4. Run all starter MiniHarnesses in parallel; each recursively detects Plaus/NS
           splits, spawns sub-branches with decayed certainty-weighted budgets, and
           degrades gracefully to conclusion.
        5. If `enable_selection` is True, run Pieces 4-6 selection (align, merge, falsify,
           Pass gauntlet, pre-exit into winner's box).
        6. Unlock run boundary on `version_registry` and return full summary.
        A hard-cap hit or emergency stop halts the run cleanly (`halted_reason`), never mid-write.
        """
        from .backend import BackendError, HardCapExceededError

        if not problem or not problem.strip():
            raise ValueError("problem must be a non-empty string")

        active_version_id = self._apply_active_version_dials()
        selection_summary: Dict[str, Any] = {}
        try:
            try:
                self.budget_ledger.size_problem_once(self.client, problem)
                probe_spent = self.budget_ledger.total_spent_usd
                self._spawn_starters(problem)
                if self.starters and probe_spent > 0.0:
                    first_id = self.starters[0].branch_id
                    self._on_event(
                        HarnessEvent(
                            event_type="budget_change",
                            branch_id=first_id,
                            timestamp=time.time(),
                            details={
                                "reason": "problem_sizing_probe",
                                "spent_usd": round(probe_spent, 6),
                                "budget_remaining": self.tracker.get(first_id).budget_remaining,
                            },
                            snapshot=self.tracker.get(first_id).to_dict(),
                        )
                    )

                # All starter branches live at once in parallel
                with ThreadPoolExecutor(max_workers=max(1, len(self.starters))) as pool:
                    futs = [pool.submit(mini.step, force_ns_on_starters) for mini in self.starters]
                    try:
                        for f in as_completed(futs):
                            f.result()
                    except BaseException:
                        # One starter failing (or Ctrl-C) stops every other branch from
                        # starting new paid calls before the pool drains.
                        self._abort_run()
                        raise

                if self.enable_selection and not self._should_stop():
                    selection_summary = self._run_selection_phase()
            except HardCapExceededError as exc:
                self._stop_event.set()
                self.halted_reason = "hard_cap"
                self._emit_halt("hard_cap", str(exc))
            except BackendError as exc:
                # The endpoint failed after retries (or rejected the request outright):
                # keep everything spent so far reportable instead of crashing the run.
                self._stop_event.set()
                self.halted_reason = "backend_error"
                self._emit_halt("backend_error", str(exc)[:500])
            except KeyboardInterrupt:
                self._stop_event.set()
                self.halted_reason = "interrupted"
                self._emit_halt("interrupted", "Operator interrupted the run (Ctrl-C); partial results kept.")
            except Exception as exc:
                # A defect (or a reply shape nothing anticipated) must not lose the money already
                # spent: halt like any other failure so the stream, HTML and summary survive.
                self._stop_event.set()
                self.halted_reason = "internal_error"
                self._emit_halt("internal_error", f"{type(exc).__name__}: {str(exc)[:480]}")
            if self.halted_reason is None and self._should_stop():
                self.halted_reason = "emergency_stop"
                self._emit_halt("emergency_stop", "Emergency stop killed the run; rollback applies next run.")

            remaining_across_branches = round(
                sum(r.budget_remaining for r in self.tracker.all_records()), 6
            )
            meter = self.client.stats()

            return {
                "problem": problem,
                "model": getattr(self.client, "model", ""),
                "fast_judge_model": self.fast_judge_model,
                "base_url": getattr(self.client, "base_url", ""),
                "is_mock": getattr(self.client, "is_mock", False),
                "active_version_id": active_version_id,
                "seeded_starters_count": min(self.starter_agents, len(self.seed_cases)),
                "halted_reason": self.halted_reason,
                "budget": {
                    "difficulty_estimate_usd": round(self.budget_ledger.difficulty_estimate_usd, 6),
                    "root_budget_usd": round(self.budget_ledger.root_budget_usd, 6),
                    "initial_finish_reserve_usd": round(self.budget_ledger.initial_finish_reserve_usd, 6),
                    "finish_reserve_remaining_usd": round(self.budget_ledger.finish_reserve_usd, 6),
                    "exploration_pool_usd": round(self.budget_ledger.exploration_pool_usd, 6),
                    "total_spent_usd": round(self.budget_ledger.total_spent_usd, 6),
                    "overdraft_usd": round(self.budget_ledger.overdraft_usd, 6),
                    "unspent_branch_budgets_usd": remaining_across_branches,
                    "hard_cap_usd": round(self.budget_ledger.hard_cap_usd, 6),
                    "api_calls": meter["calls"],
                    "api_spend_usd": meter["total_cost_usd"],
                    "api_spend_by_model": meter["cost_by_model"],
                    "api_failed_calls": meter.get("failed_calls", 0),
                    "api_estimated_usage_calls": meter.get("estimated_usage_calls", 0),
                    "api_reservation_overruns": meter.get("reservation_overruns", 0),
                    "api_n_topup_calls": self.detector.n_topup_calls,
                },
                "total_branches_spawned": len(self.tracker.all_records()),
                "ns": {
                    "share_cap": self.ns_share_cap,
                    "spawned": self._ns_spawned,
                    "capped_ticks": self._ns_capped_ticks,
                },
                "selection": selection_summary,
                "verified_main_line": list(self.fail_judge.verified_main_line),
                "winners_box": self.winners_box,
                "scoreboard": self.tracker.scoreboard(),
                "events_count": len(self.events),
            }
        finally:
            if self.version_registry is not None and self.version_registry.run_in_progress:
                self.version_registry.end_run()
