"""
Piece 3: PFP Tracker — the state ledger (Pass 2: intent & refinement).

Every branch carries:
- id
- parent
- state (Possible / Pass / Fail / Absorbed)
- certainty_score
- time_alive
- budget_remaining
- fork_under_exploration

Refinements in Pass 2 (by what was meant):
- Thread-safe ledger (`threading.RLock`) so parallel MiniHarnesses ("All branches
  live at once, all the time") can update their records concurrently without races.
- Strict one-way transitions:
  * Possible -> Pass (survived falsification)
  * Possible -> Fail (contradiction proven / hit the verified line)
  * Possible | Pass -> Absorbed (absorbed on merge)
  * No resurrection — dead (Fail / Absorbed) stays dead.
- Transition provenance history (`state_history`) and lineage queries
  (`children_of`, `lineage`, `known_forks_in_lineage`) so the arena feed,
  scoreboard, and novelty detector read straight off the ledger.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, Iterator, List, Optional, Sequence, Tuple

_BRANCH_ID = re.compile(r"branch_\d+")


def id_free(text: str) -> str:
    """Text with branch IDs blanked: IDs follow thread scheduling, so they can't order content."""
    return _BRANCH_ID.sub("branch_#", text)


def _finite_number(value: Any, field_name: str) -> float:
    """Validate numeric ledger inputs before they can poison ordering or JSON custody."""
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number, not bool.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite.")
    return number


class BranchState(str, Enum):
    POSSIBLE = "Possible"
    PASS = "Pass"
    FAIL = "Fail"
    ABSORBED = "Absorbed"


class InvalidStateTransitionError(ValueError):
    """Raised when attempting an illegal or backward state transition (e.g. resurrection)."""


@dataclass
class StateTransitionRecord:
    from_state: Optional[str]
    to_state: str
    timestamp: float
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_state": self.from_state,
            "to_state": self.to_state,
            "timestamp": round(self.timestamp, 4),
            "reason": self.reason,
        }


@dataclass
class PassProvenanceRecord:
    survived_what: str
    timestamp: float
    falsifier_angle: str
    judge_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "survived_what": self.survived_what,
            "timestamp": round(self.timestamp, 4),
            "falsifier_angle": self.falsifier_angle,
            "judge_id": self.judge_id,
        }


@dataclass
class BranchRecord:
    id: str
    parent: Optional[str]
    state: BranchState = BranchState.POSSIBLE
    certainty_score: float = 0.0
    time_alive: float = 0.0
    budget_remaining: float = 0.0
    fork_under_exploration: str = ""
    depth: int = 0
    created_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    survived_falsifications: int = 0
    attacks_faced: int = 0
    merges_attracted: int = 0
    open_critic_objections: int = 0
    last_tested_at: float = field(default_factory=time.time)
    # Arena round (`PFPTracker.arena_round`) of the last test, or of birth if never tested.
    # Untested age is counted on this logical clock, never on wall time (see `untested_rounds`).
    last_tested_round: int = 0
    reasoning_trace: List[str] = field(default_factory=list)
    state_history: List[StateTransitionRecord] = field(default_factory=list)
    merge_parents: List[str] = field(default_factory=list)
    merge_provenance: Dict[str, List[str]] = field(default_factory=dict)
    pass_provenance: List[PassProvenanceRecord] = field(default_factory=list)
    appeal_used: bool = False
    # How the branch was born: "starter", "plaus", "ns", "merge" (drives arena colors + NS selection)
    origin: str = "starter"
    brief: str = ""
    brief_id: str = ""
    parent_brief_id: Optional[str] = None
    last_objection: str = ""
    # Scheduling-free identity of where the branch came from: a digest of its ID-free fork and
    # origin plus its parents' digests (set by `PFPTracker.register_branch`). Two branches with
    # byte-identical text but different ancestry (common for merged children) order apart on
    # it, where their IDs would order them by which thread registered first.
    lineage_digest: str = ""

    @property
    def is_merge_eligible(self) -> bool:
        """A branch is eligible for absorption/merging only after facing >= 1 falsification attack."""
        return self.attacks_faced >= 1 or self.survived_falsifications > 0

    @property
    def is_alive(self) -> bool:
        return self.state in (BranchState.POSSIBLE, BranchState.PASS)

    @property
    def last_step(self) -> str:
        return self.reasoning_trace[-1] if self.reasoning_trace else ""

    @property
    def solution(self) -> str:
        """The branch's latest conclusion (what the winner's box presents), else its fork."""
        for step in reversed(self.reasoning_trace):
            if step.startswith("[Conclusion]"):
                return step[len("[Conclusion]"):].strip()
        return self.fork_under_exploration

    def refresh_time_alive(self, now: Optional[float] = None) -> float:
        current = now if now is not None else time.time()
        end_ref = self.ended_at if self.ended_at is not None else current
        self.time_alive = max(0.0, round(end_ref - self.created_at, 4))
        return self.time_alive

    def to_dict(self) -> Dict[str, Any]:
        self.refresh_time_alive()
        d = {
            "id": self.id,
            "parent": self.parent,
            "merge_parents": list(self.merge_parents),
            "depth": self.depth,
            "state": self.state.value,
            "certainty_score": round(self.certainty_score, 4),
            "time_alive": self.time_alive,
            "budget_remaining": round(self.budget_remaining, 6),
            "fork_under_exploration": self.fork_under_exploration,
            "survived_falsifications": self.survived_falsifications,
            "attacks_faced": self.attacks_faced,
            "merges_attracted": self.merges_attracted,
            "open_critic_objections": self.open_critic_objections,
            "merge_provenance": self.merge_provenance,
            "pass_provenance": [p.to_dict() for p in self.pass_provenance],
            "appeal_used": self.appeal_used,
            "origin": self.origin,
            "last_step": self.last_step[:240],
            "solution": self.solution[:600],
        }
        if self.brief:
            d["brief"] = self.brief
            d["brief_id"] = self.brief_id
            if self.parent_brief_id is not None:
                d["parent_brief_id"] = self.parent_brief_id
        return d


def branch_position_text(record: BranchRecord) -> str:
    trace_str = " | ".join(record.reasoning_trace)
    return f"{record.fork_under_exploration} | {trace_str}".strip()


def content_order_key(record: BranchRecord) -> Tuple[str, str, str]:
    """
    Canonical tie-break for selection: the branch's ID-free position, its lineage digest,
    then its ID.

    Parallel subtrees register branches (and draw IDs) in thread-scheduling order, so the
    tracker's record order is not reproducible. Every selection decision that could tie
    (pair order, merge orientation, contradiction loser, pre-exit order and cut) falls back
    to this key instead, so equal evidence on equal content always resolves the same way.
    The ID only decides between branches with the same text *and* the same ancestry.
    """
    return (id_free(branch_position_text(record)), record.lineage_digest, record.id)


class PFPTracker:
    """
    Thread-safe state ledger for PFP Darwinism.
    Each branch's mini-harness updates its own record; the main harness aggregates
    and reads scoreboard / arena feed snapshots straight off these records.
    """

    def __init__(self) -> None:
        self._records: Dict[str, BranchRecord] = {}
        self._lock = threading.RLock()
        # Logical arena clock: advanced by the harness once per completed selection round, so
        # untested age is a function of the run's decisions, not of call latency or scheduling
        self._arena_round = 0

    def register_branch(
        self,
        branch_id: str,
        parent: Optional[str],
        budget_remaining: float,
        fork_under_exploration: str,
        state: BranchState = BranchState.POSSIBLE,
        certainty_score: float = 0.0,
        depth: int = 0,
        origin: str = "starter",
        merge_parents: Sequence[str] = (),
        brief: str = "",
        brief_id: str = "",
        parent_brief_id: Optional[str] = None,
    ) -> BranchRecord:
        with self._lock:
            if branch_id in self._records:
                # Re-registering would silently resurrect / overwrite history — never allowed
                raise ValueError(f"Branch id '{branch_id}' is already registered in the ledger.")
            clean_budget = _finite_number(budget_remaining, "budget_remaining")
            clean_certainty = _finite_number(certainty_score, "certainty_score")
            now = time.time()
            rec = BranchRecord(
                id=branch_id,
                parent=parent,
                state=state,
                certainty_score=clean_certainty,
                time_alive=0.0,
                budget_remaining=round(max(0.0, clean_budget), 6),
                fork_under_exploration=fork_under_exploration,
                depth=depth,
                origin=origin,
                merge_parents=list(merge_parents),
                brief=brief,
                brief_id=brief_id,
                parent_brief_id=parent_brief_id,
                lineage_digest=self._lineage_digest(branch_id, parent, merge_parents, fork_under_exploration, origin),
                created_at=now,
                last_tested_at=now,
                last_tested_round=self._arena_round,
                state_history=[
                    StateTransitionRecord(
                        from_state=None,
                        to_state=state.value,
                        timestamp=now,
                        reason="spawn",
                    )
                ],
            )
            self._records[branch_id] = rec
            return rec

    def _lineage_digest(
        self,
        branch_id: str,
        parent: Optional[str],
        merge_parents: Sequence[str],
        fork: str,
        origin: str,
    ) -> str:
        parent_digests = sorted(
            {self._records[p].lineage_digest for p in [parent, *merge_parents] if p in self._records}
        )
        # Starter IDs ("starter_agent_N") are assigned in order and kept; branch_N IDs are blanked
        material = json.dumps([id_free(branch_id), origin, id_free(fork), parent_digests])
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    @contextmanager
    def atomic(self) -> Iterator["PFPTracker"]:
        """
        Hold the ledger lock across a multi-record check-then-act (e.g. a merge commit), so no
        other worker can kill, absorb or register a branch between the check and the writes.
        Never call into the BudgetLedger while holding it: `BudgetLedger.charge` takes the
        ledger lock first and the tracker lock second.
        """
        with self._lock:
            yield self

    def has(self, branch_id: str) -> bool:
        with self._lock:
            return branch_id in self._records

    def get(self, branch_id: str) -> BranchRecord:
        with self._lock:
            if branch_id not in self._records:
                raise KeyError(f"Unknown branch_id: {branch_id}")
            rec = self._records[branch_id]
            rec.refresh_time_alive()
            return rec

    def transition(
        self,
        branch_id: str,
        new_state: BranchState,
        reason: str = "",
    ) -> BranchRecord:
        """
        Enforce strict one-way transitions per Piece 3:
        - Possible -> Pass (survived falsification)
        - Possible -> Fail (contradiction proven / hit the verified line)
        - anything alive -> Absorbed (absorbed on merge)
        No resurrection — dead stays dead.
        """
        with self._lock:
            rec = self.get(branch_id)
            old_state = rec.state
            now = time.time()

            if old_state == new_state:
                return rec

            # Dead stays dead: Fail and Absorbed cannot transition to anything
            if old_state in (BranchState.FAIL, BranchState.ABSORBED):
                raise InvalidStateTransitionError(
                    f"Cannot transition branch '{branch_id}' from terminal state "
                    f"{old_state.value} to {new_state.value} (no resurrection)."
                )

            # Anything alive -> Absorbed on merge
            if new_state == BranchState.ABSORBED:
                rec.state = BranchState.ABSORBED
                rec.ended_at = now
                rec.state_history.append(
                    StateTransitionRecord(
                        from_state=old_state.value,
                        to_state=new_state.value,
                        timestamp=now,
                        reason=reason or "absorbed_on_merge",
                    )
                )
                rec.refresh_time_alive(now)
                return rec

            # Possible -> Pass or Possible -> Fail
            if old_state == BranchState.POSSIBLE and new_state in (BranchState.PASS, BranchState.FAIL):
                rec.state = new_state
                if new_state == BranchState.FAIL:
                    rec.ended_at = now
                rec.state_history.append(
                    StateTransitionRecord(
                        from_state=old_state.value,
                        to_state=new_state.value,
                        timestamp=now,
                        reason=reason or ("survived_falsification" if new_state == BranchState.PASS else "falsified"),
                    )
                )
                rec.refresh_time_alive(now)
                return rec

            raise InvalidStateTransitionError(
                f"Illegal one-way state transition for branch '{branch_id}': "
                f"{old_state.value} -> {new_state.value}"
            )

    def revoke_pass(
        self,
        branch_id: str,
        demote_to: BranchState = BranchState.POSSIBLE,
        reason: str = "pass_revoked",
    ) -> BranchRecord:
        """
        Piece 6: Pass is revokable. A branch verified at t=20s can be demoted at t=40s
        (to Possible or Fail); the main line rewrites and dependents re-verify or die.
        Dead branches (Fail / Absorbed) can never be resurrected.
        """
        with self._lock:
            rec = self.get(branch_id)
            if rec.state != BranchState.PASS:
                raise InvalidStateTransitionError(
                    f"Cannot revoke Pass on branch '{branch_id}' with state {rec.state.value} (must be Pass)."
                )
            if demote_to not in (BranchState.POSSIBLE, BranchState.FAIL):
                raise InvalidStateTransitionError(
                    f"Pass revocation target must be Possible or Fail, got {demote_to.value}"
                )
            now = time.time()
            old_state = rec.state
            rec.state = demote_to
            if demote_to == BranchState.FAIL:
                rec.ended_at = now
            rec.state_history.append(
                StateTransitionRecord(
                    from_state=old_state.value,
                    to_state=demote_to.value,
                    timestamp=now,
                    reason=reason,
                )
            )
            rec.refresh_time_alive(now)
            return rec

    def dependents_of(self, branch_id: str) -> List[BranchRecord]:
        """
        Return all alive descendant or merge-child branches that depend on `branch_id`.
        """
        with self._lock:
            dependents: List[BranchRecord] = []
            queue = [branch_id]
            visited = {branch_id}
            all_recs = self.all_records()
            while queue:
                curr = queue.pop(0)
                for r in all_recs:
                    if r.id in visited:
                        continue
                    if r.parent == curr or curr in r.merge_parents or curr in r.merge_provenance:
                        visited.add(r.id)
                        queue.append(r.id)
                        if r.is_alive:
                            dependents.append(r)
            return dependents

    def update_branch(
        self,
        branch_id: str,
        *,
        certainty_score: Optional[float] = None,
        budget_remaining: Optional[float] = None,
        fork_under_exploration: Optional[str] = None,
        survived_falsifications: Optional[int] = None,
        attacks_faced: Optional[int] = None,
        merges_attracted: Optional[int] = None,
        open_critic_objections: Optional[int] = None,
        tested: bool = False,
        reasoning_step: Optional[str] = None,
        brief: Optional[str] = None,
        brief_id: Optional[str] = None,
        parent_brief_id: Optional[str] = None,
        last_objection: Optional[str] = None,
    ) -> BranchRecord:
        with self._lock:
            rec = self.get(branch_id)
            now = time.time()
            if certainty_score is not None:
                rec.certainty_score = _finite_number(certainty_score, "certainty_score")
            if budget_remaining is not None:
                clean_budget = _finite_number(budget_remaining, "budget_remaining")
                rec.budget_remaining = round(max(0.0, clean_budget), 6)
            if fork_under_exploration is not None:
                rec.fork_under_exploration = fork_under_exploration
            if survived_falsifications is not None:
                rec.survived_falsifications = survived_falsifications
            if attacks_faced is not None:
                rec.attacks_faced = attacks_faced
            if merges_attracted is not None:
                rec.merges_attracted = merges_attracted
            if open_critic_objections is not None:
                rec.open_critic_objections = open_critic_objections
            if tested:
                rec.last_tested_at = now
                rec.last_tested_round = self._arena_round
            if reasoning_step is not None:
                rec.reasoning_trace.append(reasoning_step)
            if brief is not None:
                rec.brief = brief
            if brief_id is not None:
                rec.brief_id = brief_id
            if parent_brief_id is not None:
                rec.parent_brief_id = parent_brief_id
            if last_objection is not None:
                rec.last_objection = last_objection
            rec.refresh_time_alive(now)
            return rec

    @property
    def arena_round(self) -> int:
        with self._lock:
            return self._arena_round

    def advance_round(self) -> int:
        """Tick the logical arena clock (one completed selection round); returns the new round."""
        with self._lock:
            self._arena_round += 1
            return self._arena_round

    def untested_rounds(self, record: BranchRecord) -> int:
        """Arena rounds since `record` was last tested (or born): its untested age."""
        with self._lock:
            return max(0, self._arena_round - int(record.last_tested_round))

    def all_records(self) -> List[BranchRecord]:
        with self._lock:
            now = time.time()
            for r in self._records.values():
                r.refresh_time_alive(now)
            return list(self._records.values())

    def active_branches(self) -> List[BranchRecord]:
        with self._lock:
            return [r for r in self.all_records() if r.is_alive]

    def children_of(self, parent_id: Optional[str]) -> List[BranchRecord]:
        with self._lock:
            return [r for r in self.all_records() if r.parent == parent_id]

    def lineage(self, branch_id: str) -> List[BranchRecord]:
        """Return ancestor chain from root down to `branch_id`."""
        with self._lock:
            chain: List[BranchRecord] = []
            curr_id: Optional[str] = branch_id
            visited = set()
            while curr_id is not None and curr_id in self._records and curr_id not in visited:
                visited.add(curr_id)
                rec = self._records[curr_id]
                chain.append(rec)
                curr_id = rec.parent
            chain.reverse()
            return chain

    def starter_roots(self, branch_id: str) -> FrozenSet[str]:
        """Starter IDs reachable through `parent` and every `merge_parents` edge (the lineage DAG)."""
        with self._lock:
            roots: set = set()
            queue: List[str] = [branch_id]
            visited: set = set()
            while queue:
                curr_id = queue.pop()
                if curr_id in visited or curr_id not in self._records:
                    continue
                visited.add(curr_id)
                rec = self._records[curr_id]
                parents = ([rec.parent] if rec.parent else []) + list(rec.merge_parents)
                if not parents:
                    roots.add(curr_id)
                queue.extend(parents)
            return frozenset(roots)

    def known_forks_in_lineage(self, branch_id: str) -> List[str]:
        """
        Return all fork descriptions already explored in the ancestor chain (including
        all `merge_parents` DAG lineages) and immediate siblings of `branch_id` so
        novelty exhaustion prevents redundant remixes after merges too.
        """
        with self._lock:
            forks: List[str] = []
            queue: List[str] = [branch_id]
            visited = set()
            while queue:
                curr_id = queue.pop(0)
                if curr_id in visited or curr_id not in self._records:
                    continue
                visited.add(curr_id)
                rec = self._records[curr_id]
                if rec.fork_under_exploration and rec.fork_under_exploration not in forks:
                    forks.append(rec.fork_under_exploration)
                for sib in self.children_of(rec.parent):
                    if sib.fork_under_exploration and sib.fork_under_exploration not in forks:
                        forks.append(sib.fork_under_exploration)
                if rec.parent is not None:
                    queue.append(rec.parent)
                for mp in rec.merge_parents:
                    if mp not in visited:
                        queue.append(mp)
            return forks

    def scoreboard(self) -> List[Dict[str, Any]]:
        """
        Scoreboard HUD ordered by state priority (Pass > Possible > Absorbed > Fail)
        and certainty ranking (highest certainty first), reading straight off the ledger.
        """
        state_rank = {
            BranchState.PASS: 3,
            BranchState.POSSIBLE: 2,
            BranchState.ABSORBED: 1,
            BranchState.FAIL: 0,
        }
        with self._lock:
            records = self.all_records()
            ordered = sorted(
                records,
                key=lambda r: (state_rank.get(r.state, 0), r.certainty_score, r.time_alive),
                reverse=True,
            )
            return [r.to_dict() for r in ordered]
