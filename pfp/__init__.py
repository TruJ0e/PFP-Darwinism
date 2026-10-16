"""
PFP Darwinism — Phase 1 Core Harness (Pieces 1–3 + OpenAI-compatible Backend).

Pass, Fail, Possible — survival of the fittest reasoning.
Tagline: "Nebius — survival of the fittest."
"""

from .backend import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    MockOpenAIClient,
    OpenAICompatibleClient,
    create_client,
)
from .tracker import (
    BranchRecord,
    BranchState,
    InvalidStateTransitionError,
    PFPTracker,
)
from .issue_detector import (
    CertaintyWeights,
    IssueDetectionResult,
    IssueDetector,
    ProbeContinuation,
)
from .branch_spawner import (
    BudgetLedger,
    Directive,
    HarnessEvent,
    MainHarness,
    MiniHarness,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "MockOpenAIClient",
    "OpenAICompatibleClient",
    "create_client",
    "BranchRecord",
    "BranchState",
    "InvalidStateTransitionError",
    "PFPTracker",
    "CertaintyWeights",
    "IssueDetectionResult",
    "IssueDetector",
    "ProbeContinuation",
    "BudgetLedger",
    "Directive",
    "HarnessEvent",
    "MainHarness",
    "MiniHarness",
]
