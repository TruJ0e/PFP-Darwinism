"""
Piece 1: Issue Detector — when to split (Pass 1: literal implementation per DESIGN.md).

Core mechanism:
- Branch first, read the issue off the divergence.
- At each step, fire K diverse continuations blind (high temperature, cheap).
- If they cluster — no issue, merge and flow straight.
- If they genuinely split — that split *is* the issue, discovered not predicted.

Two split triggers:
- Plaus: measurable uncertainty —
  (a) divergence across probe continuations (primary, behavioral),
  (b) token entropy via logprobs (cheap, noisy),
  (c) critic pass listing unjustified leaps.
- NS: blind fan-out on cadence (every 1–2s per agent, budget-derived).
  The mutation rate catching unknown unknowns.

Branch count: bounded by novelty exhaustion (stop when new continuations are
remixes) and the depth dial.

Certainty ranking:
survived falsifications (big plus), merges attracted (plus), open critic
objections (minus), age-without-testing (decays). Certainty is revealed by
selection, not computed upfront. Score weights are tunable dials.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set


@dataclass
class CertaintyWeights:
    """
    Tunable dials for certainty ranking (Piece 1):
    - survived_falsifications: big plus
    - merges_attracted: plus
    - open_critic_objections: minus
    - age_without_testing: decays (per second of untested age)
    """

    survived_falsifications: float = 3.0
    merges_attracted: float = 1.5
    open_critic_objections: float = -1.25
    age_without_testing: float = -0.15


@dataclass
class ProbeContinuation:
    fork_id: str
    fork_description: str
    reasoning: str
    token_entropy: float = 0.0
    raw_content: str = ""


@dataclass
class IssueDetectionResult:
    should_split: bool
    trigger: Optional[str]  # "plaus", "ns", or None
    issue_summary: str
    distinct_forks: List[ProbeContinuation] = field(default_factory=list)
    merged_continuation: Optional[str] = None
    mean_token_entropy: float = 0.0
    unjustified_leaps: List[str] = field(default_factory=list)
    novelty_exhausted: bool = False
    cost_usd: float = 0.0


def _tokenize_words(text: str) -> Set[str]:
    return {
        w
        for w in re.findall(r"[a-z0-9_]+", text.lower())
        if len(w) > 2 and w not in {"the", "and", "for", "with", "that", "this", "from", "are"}
    }


def _jaccard_similarity(a: str, b: str) -> float:
    sa, sb = _tokenize_words(a), _tokenize_words(b)
    if not sa or not sb:
        return 1.0 if a.strip() == b.strip() else 0.0
    return len(sa & sb) / float(len(sa | sb))


def compute_token_entropy_from_choice(choice: Any) -> float:
    """
    Compute mean token Shannon entropy (in nats) from OpenAI-compatible choice logprobs.
    """
    lp = getattr(choice, "logprobs", None)
    if lp is None and isinstance(choice, dict):
        lp = choice.get("logprobs")
    if not lp:
        return 0.0

    content = getattr(lp, "content", None)
    if content is None and isinstance(lp, dict):
        content = lp.get("content", [])
    if not content:
        return 0.0

    entropies: List[float] = []
    for tok_entry in content:
        tops = getattr(tok_entry, "top_logprobs", None)
        if tops is None and isinstance(tok_entry, dict):
            tops = tok_entry.get("top_logprobs", [])
        if tops:
            probs = []
            for tp in tops:
                val = getattr(tp, "logprob", None)
                if val is None and isinstance(tp, dict):
                    val = tp.get("logprob", -10.0)
                probs.append(math.exp(float(val)))
            total_p = sum(probs) or 1.0
            norm_probs = [p / total_p for p in probs if p > 0]
            h = -sum(p * math.log(p) for p in norm_probs)
            entropies.append(h)
        else:
            val = getattr(tok_entry, "logprob", None)
            if val is None and isinstance(tok_entry, dict):
                val = tok_entry.get("logprob", 0.0)
            p = min(max(math.exp(float(val)), 1e-6), 1.0 - 1e-6)
            h = -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))
            entropies.append(h)

    return sum(entropies) / len(entropies) if entropies else 0.0


def _parse_probe_json(raw_text: str, idx: int) -> ProbeContinuation:
    try:
        data = json.loads(raw_text)
        fork_id = str(data.get("fork_id") or data.get("approach") or f"fork_{idx}")
        fork_desc = str(data.get("fork") or data.get("step") or data.get("approach") or raw_text[:80])
        reasoning = str(data.get("reasoning") or data.get("first_step") or raw_text)
    except Exception:
        fork_id = f"fork_{idx}"
        fork_desc = raw_text.strip()[:80]
        reasoning = raw_text.strip()
    return ProbeContinuation(
        fork_id=fork_id,
        fork_description=fork_desc,
        reasoning=reasoning,
        raw_content=raw_text,
    )


class IssueDetector:
    """
    Piece 1: Issue detector.
    - Fires K blind continuations at high temperature to read issues off behavioral divergence.
    - Combines behavioral divergence (primary), token logprob entropy (cheap/noisy),
      and critic pass listing unjustified leaps (Plaus trigger).
    - Supports NS blind fan-out on budget-derived cadence (NS trigger).
    - Bounds branch count via novelty exhaustion and the depth dial.
    - Computes certainty ranking from selection history via tunable dial weights.
    """

    def __init__(
        self,
        client: Any,
        *,
        k_probes: int = 4,
        probe_temperature: float = 0.9,
        similarity_threshold: float = 0.55,
        entropy_threshold: float = 0.85,
        depth_dial: int = 4,
        weights: Optional[CertaintyWeights] = None,
    ) -> None:
        self.client = client
        self.k_probes = k_probes
        self.probe_temperature = probe_temperature
        self.similarity_threshold = similarity_threshold
        self.entropy_threshold = entropy_threshold
        self.depth_dial = depth_dial
        self.weights = weights or CertaintyWeights()

    def compute_certainty(
        self,
        *,
        survived_falsifications: int,
        merges_attracted: int,
        open_critic_objections: int,
        seconds_since_last_test: float,
    ) -> float:
        """
        Certainty ranking (Piece 1):
        survived falsifications (big plus), merges attracted (plus),
        open critic objections (minus), age-without-testing (decays).
        """
        score = (
            survived_falsifications * self.weights.survived_falsifications
            + merges_attracted * self.weights.merges_attracted
            + open_critic_objections * self.weights.open_critic_objections
            + max(0.0, seconds_since_last_test) * self.weights.age_without_testing
        )
        return round(score, 4)

    def run_critic_pass(self, reasoning_context: str) -> tuple[List[str], float]:
        """
        Run critic pass listing unjustified leaps in the current reasoning.
        Returns (list_of_unjustified_leaps, call_cost_usd).
        """
        resp = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a PFP critic pass. List any unjustified leaps in the reasoning as JSON: "
                        '{"unjustified_leaps": ["..."], "count": N}.'
                    ),
                },
                {
                    "role": "user",
                    "content": f"Critic pass — identify unjustified leaps in:\n{reasoning_context}",
                },
            ],
            temperature=0.2,
            max_tokens=150,
        )
        cost = float(resp.usage.cost_usd)
        raw = resp.choices[0].message.content if resp.choices else "{}"
        try:
            parsed = json.loads(raw)
            leaps = [str(x) for x in parsed.get("unjustified_leaps", [])]
        except Exception:
            leaps = [raw.strip()] if raw.strip() else []
        return leaps, cost

    def _deduplicate_by_novelty(
        self,
        continuations: Sequence[ProbeContinuation],
        max_branches: int,
    ) -> tuple[List[ProbeContinuation], bool]:
        """
        Filter probe continuations until novelty is exhausted (new continuations are remixes
        of already-accepted forks) or bounded by depth_dial / max_branches.
        """
        distinct: List[ProbeContinuation] = []
        novelty_exhausted = False

        for cand in continuations:
            is_remix = False
            for existing in distinct:
                if cand.fork_id == existing.fork_id:
                    is_remix = True
                    break
                sim = _jaccard_similarity(
                    f"{cand.fork_description} {cand.reasoning}",
                    f"{existing.fork_description} {existing.reasoning}",
                )
                if sim >= self.similarity_threshold:
                    is_remix = True
                    break
            if is_remix:
                novelty_exhausted = True
                continue
            distinct.append(cand)
            if len(distinct) >= max_branches:
                break

        return distinct, novelty_exhausted

    def detect_plaus(
        self,
        problem: str,
        parent_reasoning: str,
        depth: int = 0,
    ) -> IssueDetectionResult:
        """
        Fire K diverse continuations blind (high temperature, cheap).
        - If they cluster -> no issue, merge and flow straight.
        - If they genuinely split -> that split *is* the issue, discovered not predicted.
        Also computes token entropy via logprobs and runs the critic pass listing unjustified leaps.
        """
        prompt = (
            f"probe_continuation depth={depth}\n"
            f"Fire K diverse continuations for problem: {problem}\n"
            f"Parent reasoning so far: {parent_reasoning}\n"
            "Return JSON with fork_id, fork, and reasoning."
        )
        resp = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            temperature=self.probe_temperature,
            max_tokens=200,
            logprobs=True,
            top_logprobs=4,
            n=self.k_probes,
        )
        total_cost = float(resp.usage.cost_usd)

        probes: List[ProbeContinuation] = []
        for idx, ch in enumerate(resp.choices):
            p = _parse_probe_json(ch.message.content, idx)
            p.token_entropy = compute_token_entropy_from_choice(ch)
            probes.append(p)

        mean_entropy = (
            sum(p.token_entropy for p in probes) / len(probes) if probes else 0.0
        )

        # Critic pass listing unjustified leaps
        leaps, critic_cost = self.run_critic_pass(parent_reasoning or problem)
        total_cost += critic_cost

        # Bound branch count by novelty exhaustion and the depth dial
        max_allowed = max(1, self.depth_dial - depth)
        distinct_forks, novelty_exhausted = self._deduplicate_by_novelty(
            probes, max_branches=max_allowed
        )

        # Primary behavioral trigger: >1 genuinely distinct clusters
        behavioral_split = len(distinct_forks) > 1
        entropy_high = mean_entropy >= self.entropy_threshold
        has_critic_leaps = len(leaps) > 0

        if behavioral_split:
            fork_names = ", ".join(f.fork_id for f in distinct_forks)
            issue_summary = (
                f"Plaus divergence discovered across {len(distinct_forks)} forks [{fork_names}] "
                f"(entropy={mean_entropy:.2f}, critic_leaps={len(leaps)})"
            )
            return IssueDetectionResult(
                should_split=True,
                trigger="plaus",
                issue_summary=issue_summary,
                distinct_forks=distinct_forks,
                merged_continuation=None,
                mean_token_entropy=round(mean_entropy, 4),
                unjustified_leaps=leaps,
                novelty_exhausted=novelty_exhausted,
                cost_usd=total_cost,
            )

        # Clustered! No behavioral split -> merge/consolidate and flow straight
        consolidated = (
            distinct_forks[0].reasoning
            if distinct_forks
            else (probes[0].reasoning if probes else parent_reasoning)
        )
        return IssueDetectionResult(
            should_split=False,
            trigger=None,
            issue_summary=(
                f"Continuations clustered into 1 path "
                f"(entropy_high={entropy_high}, critic_leaps={ has_critic_leaps }) — flowing straight."
            ),
            distinct_forks=distinct_forks,
            merged_continuation=consolidated,
            mean_token_entropy=round(mean_entropy, 4),
            unjustified_leaps=leaps,
            novelty_exhausted=novelty_exhausted,
            cost_usd=total_cost,
        )

    def trigger_ns(
        self,
        problem: str,
        parent_reasoning: str,
        depth: int = 0,
        fanout_count: int = 2,
    ) -> IssueDetectionResult:
        """
        NS (natural selection): blind fan-out on cadence with no trigger.
        Catches unknown unknowns via blind variation.
        """
        max_allowed = max(1, min(fanout_count, self.depth_dial - depth))
        prompt = (
            f"ns_fanout natural selection blind variation depth={depth}\n"
            f"Problem: {problem}\n"
            f"Current reasoning: {parent_reasoning}\n"
            "Generate an orthogonal, unconsidered angle as JSON."
        )
        resp = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            max_tokens=180,
            logprobs=True,
            top_logprobs=4,
            n=max_allowed,
        )
        total_cost = float(resp.usage.cost_usd)

        probes: List[ProbeContinuation] = []
        for idx, ch in enumerate(resp.choices):
            p = _parse_probe_json(ch.message.content, idx)
            p.token_entropy = compute_token_entropy_from_choice(ch)
            probes.append(p)

        distinct_forks, novelty_exhausted = self._deduplicate_by_novelty(
            probes, max_branches=max_allowed
        )
        mean_entropy = (
            sum(p.token_entropy for p in probes) / len(probes) if probes else 0.0
        )

        return IssueDetectionResult(
            should_split=len(distinct_forks) > 0,
            trigger="ns",
            issue_summary=f"NS blind variation spawned {len(distinct_forks)} mutation fork(s)",
            distinct_forks=distinct_forks,
            merged_continuation=None,
            mean_token_entropy=round(mean_entropy, 4),
            unjustified_leaps=[],
            novelty_exhausted=novelty_exhausted,
            cost_usd=total_cost,
        )
