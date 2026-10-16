"""
PFP Darwinism — Phase 1 Core Harness (Pieces 1–3 + OpenAI-compatible Backend).

Pass, Fail, Possible — survival of the fittest reasoning.
Tagline: "Nebius — survival of the fittest."
"""

from .backend import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    SUPER_MODEL,
    MockOpenAIClient,
    OpenAICompatibleClient,
    create_client,
)
from .tracker import (
    BranchRecord,
    BranchState,
    InvalidStateTransitionError,
    PFPTracker,
    PassProvenanceRecord,
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
from .alignment_detector import (
    AlignmentCandidatePair,
    AlignmentDetector,
    AlignmentJudgeResult,
    AlignmentVerdict,
    compute_branch_embedding,
    cosine_similarity,
)
from .merge_synthesizer import (
    MergeSynthesisResult,
    MergeSynthesizer,
)
from .fail_judge import (
    FailJudge,
    FalsificationOutcome,
    FalsifierAngle,
    ROTATED_ANGLES,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "SUPER_MODEL",
    "MockOpenAIClient",
    "OpenAICompatibleClient",
    "create_client",
    "BranchRecord",
    "BranchState",
    "InvalidStateTransitionError",
    "PFPTracker",
    "PassProvenanceRecord",
    "CertaintyWeights",
    "IssueDetectionResult",
    "IssueDetector",
    "ProbeContinuation",
    "BudgetLedger",
    "Directive",
    "HarnessEvent",
    "MainHarness",
    "MiniHarness",
    "AlignmentCandidatePair",
    "AlignmentDetector",
    "AlignmentJudgeResult",
    "AlignmentVerdict",
    "compute_branch_embedding",
    "cosine_similarity",
    "MergeSynthesisResult",
    "MergeSynthesizer",
    "FailJudge",
    "FalsificationOutcome",
    "FalsifierAngle",
    "ROTATED_ANGLES",
]

