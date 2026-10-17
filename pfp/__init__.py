"""
PFP Darwinism — all eight pieces: split, spawn, track, align, merge, falsify,
arena feed + Tron renderer, self-improvement loop + failsafe registry.

Pass, Fail, Possible — survival of the fittest reasoning.
Tagline: "Nebius — survival of the fittest."
"""

from .backend import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    SUPER_MODEL,
    BackendError,
    HardCapExceededError,
    MeteredClient,
    MockOpenAIClient,
    OpenAICompatibleClient,
    create_client,
)
from .jsonutil import coerce_bool, extract_json, parse_json_object
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
from .dials import DEFAULT_DIALS, DIAL_BOUNDS, sanitize_dials
from .arena import (
    ARENA_LEVEL_EVENT_TYPES,
    CANONICAL_EVENT_TYPES,
    AppendOnlyViolationError,
    ArenaEvent,
    ArenaFeed,
    TronArenaRenderer,
    stream_problem,
)
from .self_improvement import (
    EventLogMiner,
    MinedArenaMetrics,
    RewriteProposition,
    RunBoundarySwitchError,
    SandboxResult,
    SelfImprovementLoop,
    VersionRegistry,
    VersionSnapshot,
)
from .fsutil import StateLockError, atomic_write_text, exclusive_lock
from .proof_report import build_proof_report, render_proof_markdown, write_proof_report

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "SUPER_MODEL",
    "BackendError",
    "HardCapExceededError",
    "MeteredClient",
    "MockOpenAIClient",
    "OpenAICompatibleClient",
    "create_client",
    "coerce_bool",
    "extract_json",
    "parse_json_object",
    "DEFAULT_DIALS",
    "DIAL_BOUNDS",
    "sanitize_dials",
    "ARENA_LEVEL_EVENT_TYPES",
    "CANONICAL_EVENT_TYPES",
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
    "AppendOnlyViolationError",
    "ArenaEvent",
    "ArenaFeed",
    "TronArenaRenderer",
    "EventLogMiner",
    "MinedArenaMetrics",
    "RewriteProposition",
    "RunBoundarySwitchError",
    "SandboxResult",
    "SelfImprovementLoop",
    "VersionRegistry",
    "VersionSnapshot",
    "atomic_write_text",
    "build_proof_report",
    "render_proof_markdown",
    "write_proof_report",
    "stream_problem",
    "StateLockError",
    "exclusive_lock",
]

