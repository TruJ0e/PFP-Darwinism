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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .arena import ArenaEvent
from .backend import SUPER_MODEL


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
                spent = float(det.get("spent_usd", 0.0))
                if spent > 0.0:
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "dial_changes": copy.deepcopy(self.dial_changes),
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

        default_dials = initial_dials or {
            "survived_falsifications_weight": 3.0,
            "merges_attracted_weight": 1.5,
            "open_critic_objections_weight": -1.25,
            "age_without_testing_weight": -0.15,
            "prefilter_threshold": 0.20,
            "gauntlet_rounds": 3,
        }
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

        code_hashes = self._compute_code_hashes()
        full_hash = self._hash_payload(config_dials, prompts, code_hashes, proposition, sandbox_result)
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
            config_dials=copy.deepcopy(config_dials),
            prompts=copy.deepcopy(prompts),
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
    ) -> None:
        self.client = client
        self.registry = registry or VersionRegistry()
        self.miner = EventLogMiner()
        self.super_model = super_model
        self.case_bank: List[Dict[str, Any]] = []

    def record_wins_for_recall(self, metrics: MinedArenaMetrics, problem: str) -> List[Dict[str, Any]]:
        """Step 2 (Recall): store winning branch shapes in the case bank to seed future runs."""
        added: List[Dict[str, Any]] = []
        for shape in metrics.winning_branch_shapes:
            entry = {
                "problem": problem,
                "fork_shape": shape["fork_shape"],
                "is_merged_branch": shape["is_merged_branch"],
                "certainty_score": shape["certainty_score"],
            }
            self.case_bank.append(entry)
            added.append(entry)
        return added

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
        raw = resp.choices[0].message.content if resp.choices else "{}"
        try:
            return json.loads(raw)
        except Exception:
            return {"what_worked": [raw.strip()], "what_to_improve": []}

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
        raw = resp.choices[0].message.content if resp.choices else "{}"
        try:
            data = json.loads(raw)
            return list(data.get("evidence_grounded_attacks", []))
        except Exception:
            return [{"target": "general", "evidence_seq": "log", "critique": raw.strip()}]

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
        raw = resp.choices[0].message.content if resp.choices else "{}"
        try:
            data = json.loads(raw)
            summary = str(data.get("summary", "Proposed dial and prompt refinement"))
            dial_changes = dict(data.get("dial_changes", {}))
            prompt_changes = {str(k): str(v) for k, v in data.get("prompt_changes", {}).items()}
        except Exception:
            summary = "Fallback proposition"
            dial_changes = {}
            prompt_changes = {}

        return RewriteProposition(
            summary=summary,
            dial_changes=dial_changes,
            prompt_changes=prompt_changes,
            self_assessment=self_assessment,
            anti_model_critique=anti_model_critique,
        )

    def run_sandbox_test(
        self,
        candidate_dials: Dict[str, Any],
        test_problem: str = "Sandbox verification: SSH works laptop->desktop but not desktop->laptop.",
        baseline_score: float = 5.0,
    ) -> SandboxResult:
        """
        Step 6: Run candidate dials/config against a test problem in an isolated sandbox
        harness before adopting.
        """
        from .branch_spawner import MainHarness
        from .issue_detector import CertaintyWeights

        harness = MainHarness(
            client=self.client,
            starter_agents=2,
            hard_cap_usd=1.00,
            depth_dial=2,
            k_probes=2,
            gauntlet_rounds=int(candidate_dials.get("gauntlet_rounds", 3)),
            enable_selection=True,
        )
        harness.detector.weights = CertaintyWeights(
            survived_falsifications=float(candidate_dials.get("survived_falsifications_weight", 3.0)),
            merges_attracted=float(candidate_dials.get("merges_attracted_weight", 1.5)),
            open_critic_objections=float(candidate_dials.get("open_critic_objections_weight", -1.25)),
            age_without_testing=float(candidate_dials.get("age_without_testing_weight", -0.15)),
        )
        harness.alignment_detector.prefilter_threshold = float(
            candidate_dials.get("prefilter_threshold", 0.20)
        )

        summary = harness.run(test_problem)
        winners = summary.get("winners_box", [])
        passed_count = len(winners)
        top_cert = max((float(w.get("certainty_score", 0.0)) for w in winners), default=0.0)
        coherent = passed_count > 0 and top_cert > 0.0

        return SandboxResult(
            coherent=coherent,
            test_problem=test_problem,
            score=top_cert,
            baseline_score=baseline_score,
            passed_branches=passed_count,
            total_spent_usd=float(summary["budget"]["total_spent_usd"]),
            details=f"Sandbox produced {passed_count} winner(s) with max certainty {top_cert:.2f}",
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
        new_dials.update(proposition.dial_changes)
        new_prompts = copy.deepcopy(cur_ver.prompts)
        new_prompts.update(proposition.prompt_changes)

        baseline_score = metrics.spend_efficiency.get("max_winner_certainty", 5.0)
        if run_sandbox:
            sb_res = self.run_sandbox_test(new_dials, baseline_score=baseline_score)
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
