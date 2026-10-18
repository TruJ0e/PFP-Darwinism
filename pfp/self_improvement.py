"""
Piece 8: Self-Improvement Loop & Failsafe Version Registry (Pass 1: literal spec).

From DESIGN.md:
1. Log: mine the arena stream for what worked — winning branch shapes,
   effective falsifier angles, merge success rates, spend-vs-certainty.
2. Recall: new runs start seeded with past wins (case-based). Fewer blind
   branches needed over time — this is how it gets faster.
3. Self-assessment: post-run summary — what worked, what to improve.
4. Anti-model: a purely negative critic that attacks every bit of the
   run. Grounded in the event log (evidence), not vibes — unbounded
   negativity without evidence is noise.
5. Proposition: synthesis of self-assessment + anti-model critique into
   concrete changes (dials, prompts, thresholds).
6. Self-rewrite + sandbox: changes applied to its own code, then run
   against a test problem in isolation. Adopted only if coherent.

Failsafe (drift protection):
- Version registry: every rewrite snapshots code/config/prompts, hashed,
  with proposition + sandbox result attached.
- Auto-revert on degradation past threshold (no human needed).
- Manual pin: any version, anytime; all future runs use it.
- Run-boundary switching: versions change between runs, never mid-arena.
  Emergency stop kills the run; rollback applies next.
- Failed versions feed back into the anti-model as failed experiments.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .arena import ARENA_LEVEL_EVENT_TYPES, ArenaEvent
from .backend import SUPER_MODEL
from .fsutil import atomic_write_text
from .dials import DEFAULT_DIALS, resolve_dials, sanitize_dials, weights_from_dials
from .issue_detector import _jaccard_similarity
from .jsonutil import as_str_list, parse_json_object


@dataclass
class MinedArenaMetrics:
    """
    Piece 8.1: Event-log mining output.
    - winning_branch_shapes: depth, merge count, survived falsifications, and fork patterns of winners
    - merge_patterns: total merges, merged branches that reached Pass, merge success rate
    - falsifier_angles: per-angle attack count, kill count, survival count, and kill effectiveness rate
    - spend_efficiency: total spend USD, winner certainty, certainty-per-dollar, spend by event category
    """

    total_events: int
    total_branches: int
    winning_branch_shapes: List[Dict[str, Any]]
    merge_patterns: Dict[str, Any]
    falsifier_angles: Dict[str, Dict[str, Any]]
    spend_efficiency: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_events": self.total_events,
            "total_branches": self.total_branches,
            "winning_branch_shapes": copy.deepcopy(self.winning_branch_shapes),
            "merge_patterns": copy.deepcopy(self.merge_patterns),
            "falsifier_angles": copy.deepcopy(self.falsifier_angles),
            "spend_efficiency": copy.deepcopy(self.spend_efficiency),
        }


class EventLogMiner:
    """
    Piece 8.1: Mines the append-only ArenaFeed stream for:
    - Winning branch shapes (depth, merges_attracted, survived_falsifications, lineage structure)
    - Merge patterns (merge count, merged survivor Pass rate, average budget transferred)
    - Falsifier angles (`logic`, `counterexample`, `assumption_audit` attack counts & kill rates)
    - Spend efficiency (certainty per USD spent, spend breakdown by reason)
    """

    def mine(self, events: Iterable[ArenaEvent | Dict[str, Any]]) -> MinedArenaMetrics:
        ev_list = [e if isinstance(e, ArenaEvent) else ArenaEvent.from_dict(e) for e in events]
        branches: Dict[str, Dict[str, Any]] = {}
        merges: List[Dict[str, Any]] = []
        angle_stats: Dict[str, Dict[str, Any]] = {}
        spend_by_reason: Dict[str, float] = {}
        total_spent_usd = 0.0

        for evt in ev_list:
            if evt.event_type in ARENA_LEVEL_EVENT_TYPES:
                continue
            branches[evt.branch_id] = copy.deepcopy(evt.snapshot)
            et = evt.event_type
            det = evt.details

            if et == "merge":
                merges.append(
                    {
                        "child_branch_id": evt.branch_id,
                        "parent_a_id": det.get("parent_a_id", ""),
                        "parent_b_id": det.get("parent_b_id", ""),
                        "transferred_budget_usd": float(det.get("transferred_budget_usd", 0.0)),
                    }
                )
                synth_spent = float(det.get("synthesis_cost_usd", 0.0) or 0.0)
                if math.isfinite(synth_spent) and synth_spent > 0.0:
                    spend_by_reason["merge_synthesis"] = round(
                        spend_by_reason.get("merge_synthesis", 0.0) + synth_spent, 6
                    )
                    total_spent_usd = round(total_spent_usd + synth_spent, 6)
            elif et in ("fail", "falsification_survived"):
                angle = str(det.get("angle") or "unknown")
                entry = angle_stats.setdefault(
                    angle,
                    {"attacks": 0, "kills": 0, "survivals": 0, "kill_rate": 0.0},
                )
                entry["attacks"] += 1
                if et == "fail" or not det.get("survived", True):
                    entry["kills"] += 1
                else:
                    entry["survivals"] += 1
            elif et == "budget_change":
                spent = float(det.get("spent_usd", 0.0) or 0.0)
                if math.isfinite(spent) and spent > 0.0:
                    reason = str(det.get("reason", "other"))
                    spend_by_reason[reason] = round(spend_by_reason.get(reason, 0.0) + spent, 6)
                    total_spent_usd = round(total_spent_usd + spent, 6)

        for ang, st in angle_stats.items():
            att = max(1, st["attacks"])
            st["kill_rate"] = round(st["kills"] / float(att), 4)

        # Winning branch shapes (all branches that ended in Pass)
        winners = [b for b in branches.values() if b.get("state") == "Pass"]
        winners.sort(key=lambda b: float(b.get("certainty_score", 0.0)), reverse=True)
        winning_shapes: List[Dict[str, Any]] = []
        for w in winners:
            winning_shapes.append(
                {
                    "branch_id": w.get("id"),
                    "depth": int(w.get("depth", 0)),
                    "is_merged_branch": bool(w.get("merge_parents")),
                    "merge_parents": list(w.get("merge_parents", [])),
                    "merges_attracted": int(w.get("merges_attracted", 0)),
                    "survived_falsifications": int(w.get("survived_falsifications", 0)),
                    "certainty_score": float(w.get("certainty_score", 0.0)),
                    "fork_shape": str(w.get("fork_under_exploration", "")),
                    "solution": str(w.get("solution", "")),
                    "origin": str(w.get("origin", "")),
                }
            )

        # Merge patterns
        merged_ids = [m["child_branch_id"] for m in merges]
        merged_passed = [
            cid for cid in merged_ids if branches.get(cid, {}).get("state") == "Pass"
        ]
        merge_success_rate = (
            round(len(merged_passed) / float(len(merged_ids)), 4) if merged_ids else 0.0
        )
        avg_transfer = (
            round(sum(m["transferred_budget_usd"] for m in merges) / float(len(merges)), 6)
            if merges
            else 0.0
        )
        merge_patterns = {
            "total_merges": len(merges),
            "merged_branches_passed": len(merged_passed),
            "merge_success_rate": merge_success_rate,
            "avg_transferred_budget_usd": avg_transfer,
            "merged_child_ids": merged_ids,
        }

        # Spend vs certainty efficiency
        max_winner_certainty = (
            max((float(w.get("certainty_score", 0.0)) for w in winners), default=0.0)
        )
        mean_certainty = (
            sum(float(b.get("certainty_score", 0.0)) for b in branches.values())
            / float(max(1, len(branches)))
        )
        denom = max(1e-6, total_spent_usd)
        spend_efficiency = {
            "total_spent_usd": round(total_spent_usd, 6),
            "max_winner_certainty": round(max_winner_certainty, 4),
            "mean_branch_certainty": round(mean_certainty, 4),
            "winner_certainty_per_usd": round(max_winner_certainty / denom, 4),
            "pass_rate": round(len(winners) / float(max(1, len(branches))), 4),
            "spend_by_reason": spend_by_reason,
        }

        return MinedArenaMetrics(
            total_events=len(ev_list),
            total_branches=len(branches),
            winning_branch_shapes=winning_shapes,
            merge_patterns=merge_patterns,
            falsifier_angles=angle_stats,
            spend_efficiency=spend_efficiency,
        )


@dataclass
class RewriteProposition:
    """
    Concrete rewrite proposition synthesized from self-assessment + evidence-grounded anti-model critique.
    """

    summary: str
    dial_changes: Dict[str, Any] = field(default_factory=dict)
    prompt_changes: Dict[str, str] = field(default_factory=dict)
    self_assessment: Dict[str, Any] = field(default_factory=dict)
    anti_model_critique: List[Dict[str, Any]] = field(default_factory=list)
    # Proposed dial changes that failed validation (unknown key, non-numeric, out of safe range)
    rejected_dial_changes: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "dial_changes": copy.deepcopy(self.dial_changes),
            "rejected_dial_changes": copy.deepcopy(self.rejected_dial_changes),
            "prompt_changes": copy.deepcopy(self.prompt_changes),
            "self_assessment": copy.deepcopy(self.self_assessment),
            "anti_model_critique": copy.deepcopy(self.anti_model_critique),
        }


@dataclass
class SandboxResult:
    coherent: bool
    test_problem: str
    score: float
    baseline_score: float
    passed_branches: int
    total_spent_usd: float
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coherent": self.coherent,
            "test_problem": self.test_problem,
            "score": round(self.score, 4),
            "baseline_score": round(self.baseline_score, 4),
            "passed_branches": self.passed_branches,
            "total_spent_usd": round(self.total_spent_usd, 6),
            "details": self.details,
        }


@dataclass
class VersionSnapshot:
    """
    Failsafe Version Registry entry:
    Every rewrite is snapshotted and SHA-256 hashed with proposition + sandbox result attached.
    """

    version_id: str
    sha256_hash: str
    created_at: float
    parent_version_id: Optional[str]
    config_dials: Dict[str, Any]
    prompts: Dict[str, str]
    code_snapshot_hashes: Dict[str, str]
    proposition: Dict[str, Any]
    sandbox_result: Dict[str, Any]
    status: str = "adopted"  # "baseline", "adopted", "reverted", "rejected_sandbox"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "sha256_hash": self.sha256_hash,
            "created_at": round(self.created_at, 4),
            "parent_version_id": self.parent_version_id,
            "config_dials": copy.deepcopy(self.config_dials),
            "prompts": copy.deepcopy(self.prompts),
            "code_snapshot_hashes": copy.deepcopy(self.code_snapshot_hashes),
            "proposition": copy.deepcopy(self.proposition),
            "sandbox_result": copy.deepcopy(self.sandbox_result),
            "status": self.status,
        }


class RunBoundarySwitchError(RuntimeError):
    """
    Raised when attempting to switch active versions mid-run while a live run is active.
    Per DESIGN.md: "Run-boundary switching: versions change between runs, never mid-arena.
    Emergency stop kills the run; rollback applies next."
    """


class VersionRegistry:
    """
    Piece 8 Failsafe (drift protection):
    - Every rewrite snapshots code/config/prompts, SHA-256 hashed, with proposition + sandbox result.
    - Auto-revert on measured degradation past threshold (no human needed).
    - Manual pin/restore (`pin_version`, `unpin`, `restore_version`).
    - Run-boundary switching (`begin_run`, `end_run`, `emergency_stop`):
      versions switch ONLY between runs, never mid-arena. If a switch or rollback is requested
      during a live run without stopping, raises `RunBoundarySwitchError` (or queues for next run
      after `emergency_stop` kills the live run).
    - Failed/reverted versions are recorded in `failed_experiments` so they feed back into the anti-model.
    """

    def __init__(
        self,
        initial_dials: Optional[Dict[str, Any]] = None,
        initial_prompts: Optional[Dict[str, str]] = None,
        degradation_threshold: float = 0.15,
    ) -> None:
        self.degradation_threshold = degradation_threshold
        self._versions: Dict[str, VersionSnapshot] = {}
        self._version_order: List[str] = []
        self.active_version_id: str = ""
        self.pinned_version_id: Optional[str] = None
        self.pending_next_run_version_id: Optional[str] = None
        self.run_in_progress: bool = False
        self.emergency_stopped: bool = False
        self.failed_experiments: List[Dict[str, Any]] = []

        default_dials = initial_dials or dict(DEFAULT_DIALS)
        default_prompts = initial_prompts or {
            "plaus_probe_hint": "Explore diverse root-cause hypotheses across host and network layers.",
        }
        base_snap = self.register_rewrite(
            config_dials=default_dials,
            prompts=default_prompts,
            proposition={"summary": "Initial baseline configuration"},
            sandbox_result={"coherent": True, "score": 10.0, "baseline_score": 10.0},
            adopt=True,
            status="baseline",
        )
        self.active_version_id = base_snap.version_id

    @staticmethod
    def _compute_code_hashes() -> Dict[str, str]:
        pkg_dir = Path(__file__).resolve().parent
        hashes: Dict[str, str] = {}
        for py_file in sorted(pkg_dir.glob("*.py")):
            hashes[py_file.name] = hashlib.sha256(py_file.read_bytes()).hexdigest()[:16]
        return hashes

    @staticmethod
    def _hash_payload(
        config_dials: Dict[str, Any],
        prompts: Dict[str, str],
        code_hashes: Dict[str, str],
        proposition: Dict[str, Any],
        sandbox_result: Dict[str, Any],
    ) -> str:
        canonical = json.dumps(
            {
                "config_dials": config_dials,
                "prompts": prompts,
                "code_hashes": code_hashes,
                "proposition": proposition,
                "sandbox_result": sandbox_result,
            },
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def register_rewrite(
        self,
        *,
        config_dials: Dict[str, Any],
        prompts: Dict[str, str],
        proposition: Dict[str, Any],
        sandbox_result: Dict[str, Any],
        adopt: bool = True,
        status: str = "adopted",
    ) -> VersionSnapshot:
        if self.run_in_progress and adopt:
            raise RunBoundarySwitchError(
                "Cannot adopt a new version while a live run is in progress; versions switch only between runs."
            )

        clean_dials, rejected_dials = sanitize_dials(config_dials or {})
        if rejected_dials:
            raise ValueError(f"Cannot register version with invalid config_dials: {rejected_dials}")
        if not isinstance(prompts, dict):
            raise ValueError("prompts must be a dict of string keys and values")
        clean_prompts: Dict[str, str] = {}
        for k, v in prompts.items():
            ks, vs = str(k), str(v)
            if len(vs) > 4000:
                raise ValueError(f"Prompt {ks!r} exceeds maximum length of 4000 characters")
            clean_prompts[ks] = vs

        code_hashes = self._compute_code_hashes()
        full_hash = self._hash_payload(clean_dials, clean_prompts, code_hashes, proposition, sandbox_result)
        vid = f"v{len(self._version_order) + 1:03d}_{full_hash[:8]}"
        parent_id = self.active_version_id or None

        coherent = bool(sandbox_result.get("coherent", True))
        if not coherent:
            status = "rejected_sandbox"
            adopt = False

        snap = VersionSnapshot(
            version_id=vid,
            sha256_hash=full_hash,
            created_at=time.time(),
            parent_version_id=parent_id,
            config_dials=copy.deepcopy(clean_dials),
            prompts=copy.deepcopy(clean_prompts),
            code_snapshot_hashes=code_hashes,
            proposition=copy.deepcopy(proposition),
            sandbox_result=copy.deepcopy(sandbox_result),
            status=status,
        )
        self._versions[vid] = snap
        self._version_order.append(vid)

        if not coherent:
            self.failed_experiments.append(
                {
                    "version_id": vid,
                    "sha256_hash": full_hash,
                    "reason": "rejected_by_sandbox",
                    "proposition": copy.deepcopy(proposition),
                    "sandbox_result": copy.deepcopy(sandbox_result),
                }
            )
            return snap

        # If manual pin is active, do not override pinned version
        if adopt and self.pinned_version_id is None:
            self.active_version_id = vid
        return snap

    # ------------------------------------------------------------------
    # Persistence: the failsafe only protects across runs if it survives the process
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "format": "pfp-version-registry/2",
            "degradation_threshold": self.degradation_threshold,
            "active_version_id": self.active_version_id,
            "pinned_version_id": self.pinned_version_id,
            "pending_next_run_version_id": self.pending_next_run_version_id,
            "failed_experiments": copy.deepcopy(self.failed_experiments),
            "versions": [self._versions[vid].to_dict() for vid in self._version_order],
        }
        payload["integrity_sha256"] = self._registry_digest(payload)
        return payload

    @staticmethod
    def _registry_digest(payload: Dict[str, Any]) -> str:
        """Hash every persisted control-plane field, excluding the digest itself."""
        unsigned = {k: v for k, v in payload.items() if k != "integrity_sha256"}
        canonical = json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> Path:
        if self.run_in_progress:
            raise RunBoundarySwitchError("Save the registry between runs, not mid-arena.")
        return atomic_write_text(Path(path), json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "VersionRegistry":
        """
        Restore a saved registry. Every snapshot's SHA-256 is re-verified against its
        recorded payload; a tampered or corrupted version refuses to load.
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not isinstance(raw.get("versions", []), list):
            raise ValueError("Registry file must contain a JSON object with a 'versions' list.")
        try:
            fmt = raw.get("format", "pfp-version-registry/1")
            if fmt not in ("pfp-version-registry/1", "pfp-version-registry/2"):
                raise ValueError(f"Unsupported registry format: {fmt!r}")
            if fmt == "pfp-version-registry/2":
                recorded = raw.get("integrity_sha256")
                expected = cls._registry_digest(raw)
                if not isinstance(recorded, str) or recorded != expected:
                    raise ValueError("Registry control-plane integrity check failed; refusing to load.")
            return cls._from_raw(raw)
        except (KeyError, TypeError, AttributeError, OverflowError) as exc:
            # A hand-edited or truncated registry must refuse to load, never half-load.
            raise ValueError(f"Malformed registry: {type(exc).__name__}: {exc}") from exc

    @classmethod
    def _from_raw(cls, raw: Dict[str, Any]) -> "VersionRegistry":
        reg = cls.__new__(cls)
        reg.degradation_threshold = float(raw.get("degradation_threshold", 0.15))
        if not math.isfinite(reg.degradation_threshold) or reg.degradation_threshold < 0.0:
            raise ValueError("Registry degradation_threshold must be finite and non-negative.")
        reg._versions = {}
        reg._version_order = []
        reg.run_in_progress = False
        reg.emergency_stopped = False
        failed = raw.get("failed_experiments", [])
        if not isinstance(failed, list) or any(not isinstance(item, dict) for item in failed):
            raise ValueError("Registry failed_experiments must be a list of objects.")
        reg.failed_experiments = copy.deepcopy(failed)
        for v in raw.get("versions", []):
            if not isinstance(v, dict):
                raise ValueError("Every registry version must be an object.")
            snap = VersionSnapshot(
                version_id=str(v["version_id"]),
                sha256_hash=str(v["sha256_hash"]),
                created_at=float(v.get("created_at", 0.0)),
                parent_version_id=v.get("parent_version_id"),
                config_dials=dict(v.get("config_dials", {})),
                prompts={str(k): str(val) for k, val in dict(v.get("prompts", {})).items()},
                code_snapshot_hashes=dict(v.get("code_snapshot_hashes", {})),
                proposition=dict(v.get("proposition", {})),
                sandbox_result=dict(v.get("sandbox_result", {})),
                status=str(v.get("status", "adopted")),
            )
            expected = cls._hash_payload(
                snap.config_dials,
                snap.prompts,
                snap.code_snapshot_hashes,
                snap.proposition,
                snap.sandbox_result,
            )
            if expected != snap.sha256_hash:
                raise ValueError(f"Version {snap.version_id} failed SHA-256 integrity check; refusing to load.")
            _, rejected_dials = sanitize_dials(snap.config_dials)
            if rejected_dials:
                raise ValueError(
                    f"Version {snap.version_id} contains invalid config_dials {rejected_dials}; refusing to load."
                )
            if any(len(val) > 4000 for val in snap.prompts.values()):
                raise ValueError(
                    f"Version {snap.version_id} contains an oversized prompt (>4000 chars); refusing to load."
                )
            if not snap.version_id or snap.version_id in reg._versions:
                raise ValueError(f"Duplicate or empty registry version id: {snap.version_id!r}")
            if not math.isfinite(snap.created_at):
                raise ValueError(f"Version {snap.version_id} has a non-finite creation timestamp.")
            if snap.parent_version_id is not None and snap.parent_version_id not in reg._versions:
                raise ValueError(
                    f"Version {snap.version_id} references a missing or out-of-order parent "
                    f"{snap.parent_version_id!r}."
                )
            reg._versions[snap.version_id] = snap
            reg._version_order.append(snap.version_id)
        if not reg._version_order:
            raise ValueError("Registry file contains no versions.")
        reg.active_version_id = str(raw.get("active_version_id") or reg._version_order[0])
        reg.pinned_version_id = raw.get("pinned_version_id")
        reg.pending_next_run_version_id = raw.get("pending_next_run_version_id")
        for ref in (reg.active_version_id, reg.pinned_version_id, reg.pending_next_run_version_id):
            if ref is not None and ref not in reg._versions:
                raise ValueError(f"Registry references unknown version {ref}")
        return reg

    def get_version(self, version_id: Optional[str] = None) -> VersionSnapshot:
        target = version_id or self.pinned_version_id or self.active_version_id
        if target not in self._versions:
            raise KeyError(f"Unknown version_id: {target}")
        return self._versions[target]

    def all_versions(self) -> List[VersionSnapshot]:
        return [self._versions[vid] for vid in self._version_order]

    def begin_run(self) -> VersionSnapshot:
        """
        Mark a run as active. Applies any queued `pending_next_run_version_id`
        at the run boundary before ignition starts.
        """
        if self.pending_next_run_version_id is not None and self.pinned_version_id is None:
            self.active_version_id = self.pending_next_run_version_id
            self.pending_next_run_version_id = None
        self.run_in_progress = True
        self.emergency_stopped = False
        return self.get_version()

    def end_run(self) -> None:
        """Mark the current run as finished, unlocking run-boundary version switching."""
        self.run_in_progress = False
        if self.pending_next_run_version_id is not None and self.pinned_version_id is None:
            self.active_version_id = self.pending_next_run_version_id
            self.pending_next_run_version_id = None

    def emergency_stop(self, rollback_to_version_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Emergency stop kills the live run immediately; rollback applies for the next run.
        """
        was_running = self.run_in_progress
        self.emergency_stopped = True
        self.run_in_progress = False

        target_rollback = rollback_to_version_id
        if target_rollback is None:
            cur = self.get_version()
            target_rollback = cur.parent_version_id or self._version_order[0]

        if target_rollback not in self._versions:
            raise KeyError(f"Unknown rollback version: {target_rollback}")

        if self.pinned_version_id is None:
            self.active_version_id = target_rollback
        self.pending_next_run_version_id = None

        return {
            "emergency_stopped": True,
            "was_running": was_running,
            "active_version_for_next_run": self.get_version().version_id,
        }

    def pin_version(self, version_id: str) -> VersionSnapshot:
        """
        Manual pin: pin any version; all future runs use it.
        Enforces run-boundary switching if a run is currently live.
        """
        if version_id not in self._versions:
            raise KeyError(f"Unknown version_id: {version_id}")
        if self.run_in_progress:
            raise RunBoundarySwitchError(
                "Cannot switch/pin version mid-run; call emergency_stop() or wait for end_run()."
            )
        self.pinned_version_id = version_id
        self.active_version_id = version_id
        return self._versions[version_id]

    def unpin(self) -> None:
        self.pinned_version_id = None

    def restore_version(self, version_id: str) -> VersionSnapshot:
        """
        Manual restore to a specific snapshot version between runs.
        """
        if version_id not in self._versions:
            raise KeyError(f"Unknown version_id: {version_id}")
        if self.run_in_progress:
            raise RunBoundarySwitchError(
                "Versions switch only between runs; cannot restore mid-run."
            )
        self.pinned_version_id = None
        self.active_version_id = version_id
        return self._versions[version_id]

    def evaluate_and_auto_revert(
        self,
        measured_score: float,
        baseline_score: float,
    ) -> Dict[str, Any]:
        """
        Auto-revert on measured degradation past threshold (no human needed).
        Failed versions feed back into the anti-model as failed experiments.
        """
        if self.run_in_progress:
            raise RunBoundarySwitchError("Auto-revert evaluation must run between runs, not mid-run.")

        cur = self.get_version()
        if baseline_score <= 0:
            degradation_ratio = 0.0
        else:
            degradation_ratio = (baseline_score - measured_score) / float(baseline_score)

        if degradation_ratio > self.degradation_threshold and cur.parent_version_id is not None:
            reverted_from = cur.version_id
            cur.status = "reverted"
            fallback_id = cur.parent_version_id
            if self.pinned_version_id == reverted_from:
                self.pinned_version_id = None
            self.active_version_id = fallback_id
            self.failed_experiments.append(
                {
                    "version_id": reverted_from,
                    "sha256_hash": cur.sha256_hash,
                    "reason": f"auto_reverted_degradation_{degradation_ratio:.4f}",
                    "measured_score": round(measured_score, 4),
                    "baseline_score": round(baseline_score, 4),
                    "proposition": copy.deepcopy(cur.proposition),
                }
            )
            return {
                "reverted": True,
                "reverted_from": reverted_from,
                "reverted_to": fallback_id,
                "degradation_ratio": round(degradation_ratio, 4),
            }

        return {
            "reverted": False,
            "active_version_id": cur.version_id,
            "degradation_ratio": round(degradation_ratio, 4),
        }


# Case-bank limits (recall memory is persisted and fed back into prompts and budgets).
CASE_TEXT_LIMIT = 1200
CASE_PROBLEM_LIMIT = 4000
CASE_CERTAINTY_CEILING = 1000.0


class SelfImprovementLoop:
    """
    Piece 8: Full 6-step Self-Improvement Loop + Failsafe integration.
    1. Log (`miner.mine`): mine arena stream for winning branch shapes, merge patterns,
       falsifier angles, and spend efficiency.
    2. Recall (`case_bank`): store winning branch shapes so subsequent runs start seeded
       with past wins.
    3. Self-assessment (`run_self_assessment`): post-run summary of what worked and what to improve.
    4. Anti-model (`run_anti_model`): purely negative critic grounded in the event log + failed experiments.
    5. Proposition (`synthesize_proposition`): concrete dial/prompt/threshold changes.
    6. Self-rewrite + sandbox (`propose_and_sandbox_rewrite`): apply changes in isolation against
       a sandbox test problem, snapshot & hash in `VersionRegistry`, and adopt only if coherent
       and non-degraded.
    """

    def __init__(
        self,
        client: Any,
        registry: Optional[VersionRegistry] = None,
        *,
        super_model: str = SUPER_MODEL,
        sandbox_hard_cap_usd: float = 1.00,
        max_cases: int = 500,
    ) -> None:
        if max_cases < 1:
            raise ValueError("max_cases must be >= 1")
        if not math.isfinite(float(sandbox_hard_cap_usd)) or sandbox_hard_cap_usd <= 0:
            raise ValueError("sandbox_hard_cap_usd must be a finite number > 0")
        self.client = client
        self.sandbox_hard_cap_usd = float(sandbox_hard_cap_usd)
        self.registry = registry or VersionRegistry()
        self.miner = EventLogMiner()
        self.super_model = super_model
        self.case_bank: List[Dict[str, Any]] = []
        self.max_cases = int(max_cases)

    def record_wins_for_recall(self, metrics: MinedArenaMetrics, problem: str) -> List[Dict[str, Any]]:
        """
        Step 2 (Recall): store winning branch shapes in the case bank to seed future runs.
        A repeated win for the same problem + shape updates the case (keeps the best
        certainty, counts wins) instead of piling up duplicates.
        """
        added: List[Dict[str, Any]] = []
        for shape in metrics.winning_branch_shapes:
            existing = next(
                (
                    c
                    for c in self.case_bank
                    if c.get("problem") == problem and c.get("fork_shape") == shape["fork_shape"]
                ),
                None,
            )
            if existing is not None:
                existing["certainty_score"] = max(float(existing.get("certainty_score", 0.0)), shape["certainty_score"])
                existing["wins"] = int(existing.get("wins", 1)) + 1
                continue
            entry = {
                "problem": problem,
                "fork_shape": shape["fork_shape"],
                "solution": shape.get("solution", ""),
                "is_merged_branch": shape["is_merged_branch"],
                "certainty_score": shape["certainty_score"],
                "wins": 1,
            }
            self.case_bank.append(entry)
            added.append(entry)
        self._trim_case_bank()
        return added

    def recall(self, problem: str, k: int = 3, min_similarity: float = 0.15) -> List[Dict[str, Any]]:
        """
        Case-based recall: past wins on similar problems, best match first
        (similarity to `problem`, then certainty). Distinct fork shapes only.
        """
        scored = []
        for case in self.case_bank:
            sim = _jaccard_similarity(problem, str(case.get("problem", "")))
            if sim < min_similarity:
                continue
            scored.append((sim, float(case.get("certainty_score", 0.0)), case))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        seen: set = set()
        out: List[Dict[str, Any]] = []
        for sim, _, case in scored:
            if case.get("fork_shape") in seen:
                continue
            seen.add(case.get("fork_shape"))
            out.append(dict(case, similarity=round(sim, 4)))
            if len(out) >= max(0, k):
                break
        return out

    def _trim_case_bank(self) -> None:
        """Bound recall memory: keep the most-proven cases (wins, then certainty), drop the rest."""
        if len(self.case_bank) <= self.max_cases:
            return
        ranked = sorted(
            range(len(self.case_bank)),
            key=lambda i: (int(self.case_bank[i].get("wins", 1)), float(self.case_bank[i].get("certainty_score", 0.0))),
            reverse=True,
        )
        keep = set(ranked[: self.max_cases])
        self.case_bank = [c for i, c in enumerate(self.case_bank) if i in keep]

    @staticmethod
    def sanitize_case(raw: Any) -> Optional[Dict[str, Any]]:
        """
        A case bank on disk is untrusted: it seeds starter budgets and prompts next run.
        Non-finite or negative certainty would poison the certainty-weighted budget split
        (inf/NaN shares), and unbounded strings would bloat every seeded prompt.
        Returns the cleaned case, or None if it is unusable.
        """
        if not isinstance(raw, dict):
            return None
        fork = " ".join(str(raw.get("fork_shape") or "").split())[:CASE_TEXT_LIMIT]
        if not fork:
            return None
        try:
            certainty = float(raw.get("certainty_score", 0.0))
        except (TypeError, ValueError):
            certainty = 0.0
        if not math.isfinite(certainty) or certainty < 0.0:
            certainty = 0.0
        try:
            wins = int(raw.get("wins", 1))
        except (TypeError, ValueError, OverflowError):
            wins = 1
        return {
            "problem": str(raw.get("problem") or "")[:CASE_PROBLEM_LIMIT],
            "fork_shape": fork,
            "solution": str(raw.get("solution") or "")[:CASE_TEXT_LIMIT],
            "is_merged_branch": raw.get("is_merged_branch") is True,
            "certainty_score": min(certainty, CASE_CERTAINTY_CEILING),
            "wins": max(1, wins),
        }

    def save_case_bank(self, path: str | Path) -> Path:
        cleaned = [case for case in (self.sanitize_case(c) for c in self.case_bank) if case is not None]
        self.case_bank = cleaned
        self._trim_case_bank()
        return atomic_write_text(Path(path), json.dumps(self.case_bank, indent=2, allow_nan=False))

    def load_case_bank(self, path: str | Path) -> int:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("Case bank file must contain a JSON list.")
        cleaned = (self.sanitize_case(c) for c in raw)
        self.case_bank = [c for c in cleaned if c is not None]
        self._trim_case_bank()
        return len(self.case_bank)

    def run_self_assessment(self, metrics: MinedArenaMetrics) -> Dict[str, Any]:
        """Step 3: Post-run self-assessment summary — what worked, what to improve."""
        resp = self.client.chat.completions.create(
            model=self.super_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are the PFP post-run self-assessment module. Return JSON with what_worked and what_to_improve.",
                },
                {
                    "role": "user",
                    "content": f"self_assessment — post-run summary of mined arena metrics:\n{json.dumps(metrics.to_dict())}",
                },
            ],
            temperature=0.2,
            max_tokens=200,
        )
        raw = (resp.choices[0].message.content if resp.choices else "") or ""
        data = parse_json_object(raw)
        if data is None:
            return {"what_worked": [raw.strip()[:400]] if raw.strip() else [], "what_to_improve": []}
        return {
            "what_worked": as_str_list(data.get("what_worked")),
            "what_to_improve": as_str_list(data.get("what_to_improve")),
        }

    def run_anti_model(self, metrics: MinedArenaMetrics) -> List[Dict[str, Any]]:
        """
        Step 4: Anti-model — purely negative critic that attacks every bit of the run,
        grounded in the event log (evidence) and past failed experiments in the registry.
        """
        resp = self.client.chat.completions.create(
            model=self.super_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the PFP anti-model: a purely negative critic grounded strictly in event-log evidence."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"anti_model — purely negative critic on mined log:\n{json.dumps(metrics.to_dict())}\n"
                        f"Past failed experiments: {json.dumps(self.registry.failed_experiments)}"
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=220,
        )
        raw = (resp.choices[0].message.content if resp.choices else "") or ""
        data = parse_json_object(raw)
        if data is None:
            return [{"target": "general", "evidence_seq": "log", "critique": raw.strip()[:400]}] if raw.strip() else []
        attacks = data.get("evidence_grounded_attacks")
        if not isinstance(attacks, list):
            return []
        # "Grounded in the event log (evidence), not vibes": drop attacks that cite no evidence
        return [
            a for a in attacks
            if isinstance(a, dict) and str(a.get("evidence_seq") or "").strip() and str(a.get("critique") or "").strip()
        ]

    def synthesize_proposition(
        self,
        metrics: MinedArenaMetrics,
        self_assessment: Dict[str, Any],
        anti_model_critique: List[Dict[str, Any]],
    ) -> RewriteProposition:
        """
        Step 5: Synthesize self-assessment + anti-model critique into concrete changes
        (dials, prompts, thresholds).
        """
        resp = self.client.chat.completions.create(
            model=self.super_model,
            messages=[
                {
                    "role": "system",
                    "content": "Synthesize self-assessment and anti-model critique into concrete dial_changes and prompt_changes JSON.",
                },
                {
                    "role": "user",
                    "content": (
                        "synthesize_proposition — concrete changes (dials, prompts, thresholds):\n"
                        f"Self-assessment: {json.dumps(self_assessment)}\n"
                        f"Anti-model: {json.dumps(anti_model_critique)}"
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=220,
        )
        raw = (resp.choices[0].message.content if resp.choices else "") or ""
        data = parse_json_object(raw)
        rejected: Dict[str, str] = {}
        if data is not None:
            summary = str(data.get("summary") or "Proposed dial and prompt refinement")
            proposed = data.get("dial_changes")
            dial_changes, rejected = sanitize_dials(proposed if isinstance(proposed, dict) else {})
            raw_prompts = data.get("prompt_changes")
            prompt_changes = (
                {str(k): str(v)[:1000] for k, v in raw_prompts.items() if str(v).strip()}
                if isinstance(raw_prompts, dict)
                else {}
            )
        else:
            summary = "Fallback proposition (unparseable synthesis reply)"
            dial_changes = {}
            prompt_changes = {}

        return RewriteProposition(
            summary=summary,
            dial_changes=dial_changes,
            prompt_changes=prompt_changes,
            self_assessment=self_assessment,
            anti_model_critique=anti_model_critique,
            rejected_dial_changes=rejected,
        )

    def run_sandbox_test(
        self,
        candidate_dials: Dict[str, Any],
        test_problem: str = "Sandbox verification: SSH works laptop->desktop but not desktop->laptop.",
        baseline_score: float = 5.0,
        candidate_prompts: Optional[Dict[str, str]] = None,
    ) -> SandboxResult:
        """
        Step 6: Run candidate dials/config against a test problem in an isolated sandbox
        harness before adopting.
        """
        from .branch_spawner import MainHarness

        dials = resolve_dials(candidate_dials)
        harness = MainHarness(
            client=self.client,
            starter_agents=2,
            hard_cap_usd=self.sandbox_hard_cap_usd,
            depth_dial=2,
            k_probes=2,
            gauntlet_rounds=int(dials["gauntlet_rounds"]),
            enable_selection=True,
        )
        harness.detector.weights = weights_from_dials(dials)
        harness.alignment_detector.prefilter_threshold = float(dials["prefilter_threshold"])
        harness.detector.probe_hint = str((candidate_prompts or {}).get("plaus_probe_hint", ""))

        summary = harness.run(test_problem)
        if summary.get("halted_reason") == "interrupted":
            # The operator's Ctrl-C is not evidence against the candidate: stop the whole
            # cycle instead of recording a failed experiment the anti-model would learn from.
            raise KeyboardInterrupt("sandbox run interrupted by the operator")
        winners = summary.get("winners_box", [])
        passed_count = len(winners)
        top_cert = max((float(w.get("certainty_score", 0.0)) for w in winners), default=0.0)
        halted = summary.get("halted_reason")
        # A sandbox that was cut short (cap, backend failure) proves nothing about the candidate
        coherent = halted is None and passed_count > 0 and top_cert > 0.0

        return SandboxResult(
            coherent=coherent,
            test_problem=test_problem,
            score=top_cert,
            baseline_score=baseline_score,
            passed_branches=passed_count,
            total_spent_usd=float(summary["budget"]["api_spend_usd"]),
            details=(
                f"Sandbox halted ({halted}) before it could validate the candidate"
                if halted
                else f"Sandbox produced {passed_count} winner(s) with max certainty {top_cert:.2f}"
            ),
        )

    def run_post_run_improvement(
        self,
        events: Iterable[ArenaEvent | Dict[str, Any]],
        problem: str = "",
        run_sandbox: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute the full Piece 8 self-improvement cycle between runs:
        1. Mine the event stream.
        2. Seed recall case bank with winning branch shapes.
        3. Generate post-run self-assessment.
        4. Run evidence-grounded anti-model critique.
        5. Synthesize concrete proposition (dials + prompts).
        6. Run sandbox validation & register version snapshot in VersionRegistry.
        """
        metrics = self.miner.mine(events)
        recalled = self.record_wins_for_recall(metrics, problem=problem)
        assessment = self.run_self_assessment(metrics)
        critique = self.run_anti_model(metrics)
        proposition = self.synthesize_proposition(metrics, assessment, critique)

        cur_ver = self.registry.get_version()
        new_dials = copy.deepcopy(cur_ver.config_dials)
        new_dials.update(proposition.dial_changes)  # already validated by sanitize_dials
        new_prompts = copy.deepcopy(cur_ver.prompts)
        new_prompts.update(proposition.prompt_changes)

        baseline_score = metrics.spend_efficiency.get("max_winner_certainty", 5.0)
        if run_sandbox:
            sb_res = self.run_sandbox_test(
                new_dials, baseline_score=baseline_score, candidate_prompts=new_prompts
            )
        else:
            sb_res = SandboxResult(
                coherent=True,
                test_problem=problem,
                score=baseline_score,
                baseline_score=baseline_score,
                passed_branches=len(metrics.winning_branch_shapes),
                total_spent_usd=metrics.spend_efficiency.get("total_spent_usd", 0.0),
                details="Sandbox skipped",
            )

        snap = self.registry.register_rewrite(
            config_dials=new_dials,
            prompts=new_prompts,
            proposition=proposition.to_dict(),
            sandbox_result=sb_res.to_dict(),
            adopt=sb_res.coherent,
        )

        revert_check = {"reverted": False, "active_version_id": self.registry.active_version_id}
        if sb_res.coherent:
            revert_check = self.registry.evaluate_and_auto_revert(
                measured_score=sb_res.score,
                baseline_score=baseline_score,
            )

        return {
            "mined_metrics": metrics.to_dict(),
            "recall_cases_added": len(recalled),
            "proposition": proposition.to_dict(),
            "sandbox_result": sb_res.to_dict(),
            "registered_version": snap.to_dict(),
            "auto_revert_check": revert_check,
            "active_version_id": self.registry.get_version().version_id,
        }
