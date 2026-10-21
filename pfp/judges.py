"""
Judge ladder (v0.41): which model rules on which verdict.

Every judge used to be Super. Live runs showed what that costs: Super reasons before it answers,
so each verdict took ~10-50 s, judging was 55-68% of spend, and at 512 tokens the JSON verdict was
often buried under the reasoning (fail-closed as inconclusive). The ladder splits the work:

- First pass (a fast instruct model): alignment claims and verdicts, background samples, the NS
  sweep, post-merge checks, contradiction attacks, dependent re-verification, and every gauntlet
  round before the last.
- Final tier (Super): every appeal, the final gauntlet round and the pre-exit check. A kill on the
  first pass still gets its one appeal heard by Super, and nothing is crowned Pass or enters the
  winner's box without surviving Super.

The gauntlet is therefore a gradient (fast, fast + Super appeal, Super), not three coin-flips
against the strongest predator. Three survivals stay the bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .backend import FAST_JUDGE_MODEL, SUPER_MODEL

# Schedule triggers ruled on by the final tier no matter which round they belong to.
FINAL_TIER_TRIGGERS = frozenset({"pre_exit"})
_GAUNTLET_PREFIX = "pass_gauntlet_round_"


@dataclass(frozen=True)
class JudgeLadder:
    first_pass_model: str = FAST_JUDGE_MODEL
    final_model: str = SUPER_MODEL

    @classmethod
    def super_only(cls, model: str = SUPER_MODEL) -> "JudgeLadder":
        """The pre-v0.41 routing: one model judges everything."""
        return cls(first_pass_model=model, final_model=model)

    def falsifier_model(self, schedule_trigger: str, gauntlet_rounds: int, *, is_appeal: bool = False) -> str:
        if is_appeal or schedule_trigger in FINAL_TIER_TRIGGERS:
            return self.final_model
        if schedule_trigger.startswith(_GAUNTLET_PREFIX):
            try:
                round_number = int(schedule_trigger[len(_GAUNTLET_PREFIX):])
            except ValueError:
                return self.final_model
            # The last round (and anything past it) is Super's: crowning requires Super
            return self.final_model if round_number >= max(1, int(gauntlet_rounds)) else self.first_pass_model
        return self.first_pass_model

    def judge_id(self, model: str, *, is_appeal: bool = False) -> str:
        if is_appeal:
            return "super_judge_appeal_secondary"
        if model == self.final_model:
            return "super_judge_primary"
        return "fast_judge_primary"

    def to_dict(self) -> Dict[str, Any]:
        return {"first_pass_model": self.first_pass_model, "final_model": self.final_model}
