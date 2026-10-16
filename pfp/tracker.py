"""
Piece 3: PFP Tracker — the state ledger (Pass 1: literal implementation per DESIGN.md).

Every branch carries:
- id
- parent
- state (Possible / Pass / Fail / Absorbed)
- certainty_score
- time_alive
- budget_remaining
- fork_under_exploration

The branch's own mini-harness updates it; the main harness aggregates.
Transitions are one-way:
- Possible -> Pass (survived falsification)
- Possible -> Fail (contradiction proven / hit the verified line)
- anything -> Absorbed (absorbed on merge)
No resurrection — dead stays dead.
The arena feed and scoreboard read straight off these records.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class BranchState(str, Enum):
    POSSIBLE = "Possible"
    PASS = "Pass"
    FAIL = "Fail"
    ABSORBED = "Absorbed"


class InvalidStateTransitionError(ValueError):
    """Raised when attempting an illegal or backward state transition (e.g. resurrection)."""


@dataclass
class BranchRecord:
    id: str
    parent: Optional[str]
    state: BranchState = BranchState.POSSIBLE
    certainty_score: float = 0.0
    time_alive: float = 0.0
    budget_remaining: float = 0.0
    fork_under_exploration: str = ""
    created_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    survived_falsifications: int = 0
    merges_attracted: int = 0
    open_critic_objections: int = 0
    last_tested_at: float = field(default_factory=time.time)
    reasoning_trace: List[str] = field(default_factory=list)

    @property
    def is_alive(self) -> bool:
        return self.state in (BranchState.POSSIBLE, BranchState.PASS)

    def refresh_time_alive(self, now: Optional[float] = None) -> float:
        current = now if now is not None else time.time()
        end_ref = self.ended_at if self.ended_at is not None else current
        self.time_alive = max(0.0, round(end_ref - self.created_at, 4))
        return self.time_alive

    def to_dict(self) -> Dict[str, Any]:
        self.refresh_time_alive()
        return {
            "id": self.id,
            "parent": self.parent,
            "state": self.state.value,
            "certainty_score": round(self.certainty_score, 4),
            "time_alive": self.time_alive,
            "budget_remaining": round(self.budget_remaining, 6),
            "fork_under_exploration": self.fork_under_exploration,
            "survived_falsifications": self.survived_falsifications,
            "merges_attracted": self.merges_attracted,
            "open_critic_objections": self.open_critic_objections,
        }


class PFPTracker:
    """
    The state ledger for PFP Darwinism.
    Each branch's mini-harness updates its own record; the main harness aggregates
    and reads scoreboard / arena feed snapshots straight off these records.
    """

    def __init__(self) -> None:
        self._records: Dict[str, BranchRecord] = {}

    def register_branch(
        self,
        branch_id: str,
        parent: Optional[str],
        budget_remaining: float,
        fork_under_exploration: str,
        state: BranchState = BranchState.POSSIBLE,
        certainty_score: float = 0.0,
    ) -> BranchRecord:
        now = time.time()
        rec = BranchRecord(
            id=branch_id,
            parent=parent,
            state=state,
            certainty_score=certainty_score,
            time_alive=0.0,
            budget_remaining=budget_remaining,
            fork_under_exploration=fork_under_exploration,
            created_at=now,
            last_tested_at=now,
        )
        self._records[branch_id] = rec
        return rec

    def get(self, branch_id: str) -> BranchRecord:
        if branch_id not in self._records:
            raise KeyError(f"Unknown branch_id: {branch_id}")
        rec = self._records[branch_id]
        rec.refresh_time_alive()
        return rec

    def transition(self, branch_id: str, new_state: BranchState) -> BranchRecord:
        """
        Enforce strict one-way transitions per Piece 3:
        - Possible -> Pass (survived falsification)
        - Possible -> Fail (contradiction proven / hit the verified line)
        - anything -> Absorbed (absorbed on merge)
        No resurrection — dead stays dead.
        """
        rec = self.get(branch_id)
        old_state = rec.state

        if old_state == new_state:
            return rec

        # Dead stays dead: Fail and Absorbed cannot transition to anything
        if old_state in (BranchState.FAIL, BranchState.ABSORBED):
            raise InvalidStateTransitionError(
                f"Cannot transition branch '{branch_id}' from terminal state {old_state.value} to {new_state.value} (no resurrection)."
            )

        # Anything alive -> Absorbed on merge
        if new_state == BranchState.ABSORBED:
            rec.state = BranchState.ABSORBED
            rec.ended_at = time.time()
            rec.refresh_time_alive()
            return rec

        # Possible -> Pass or Possible -> Fail
        if old_state == BranchState.POSSIBLE and new_state in (BranchState.PASS, BranchState.FAIL):
            rec.state = new_state
            if new_state == BranchState.FAIL:
                rec.ended_at = time.time()
            rec.refresh_time_alive()
            return rec

        raise InvalidStateTransitionError(
            f"Illegal one-way state transition for branch '{branch_id}': {old_state.value} -> {new_state.value}"
        )

    def update_branch(
        self,
        branch_id: str,
        *,
        certainty_score: Optional[float] = None,
        budget_remaining: Optional[float] = None,
        fork_under_exploration: Optional[str] = None,
        survived_falsifications: Optional[int] = None,
        merges_attracted: Optional[int] = None,
        open_critic_objections: Optional[int] = None,
        tested: bool = False,
        reasoning_step: Optional[str] = None,
    ) -> BranchRecord:
        rec = self.get(branch_id)
        now = time.time()
        if certainty_score is not None:
            rec.certainty_score = certainty_score
        if budget_remaining is not None:
            rec.budget_remaining = max(0.0, budget_remaining)
        if fork_under_exploration is not None:
            rec.fork_under_exploration = fork_under_exploration
        if survived_falsifications is not None:
            rec.survived_falsifications = survived_falsifications
        if merges_attracted is not None:
            rec.merges_attracted = merges_attracted
        if open_critic_objections is not None:
            rec.open_critic_objections = open_critic_objections
        if tested:
            rec.last_tested_at = now
        if reasoning_step is not None:
            rec.reasoning_trace.append(reasoning_step)
        rec.refresh_time_alive(now)
        return rec

    def all_records(self) -> List[BranchRecord]:
        now = time.time()
        for r in self._records.values():
            r.refresh_time_alive(now)
        return list(self._records.values())

    def scoreboard(self) -> List[Dict[str, Any]]:
        """
        Scoreboard HUD ordered by certainty ranking (highest certainty first),
        reading straight off the BranchRecord ledger.
        """
        records = self.all_records()
        ordered = sorted(records, key=lambda r: (r.certainty_score, r.time_alive), reverse=True)
        return [r.to_dict() for r in ordered]
