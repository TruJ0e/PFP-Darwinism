"""
Piece 1: Issue Detector — when to split (Pass 2: intent & refinement).

Refinements in Pass 2 (by what was meant):
- Cluster consolidation & support scoring (`cluster_size`, `support_score`):
  when multiple K blind continuations converge on the same fork, their reasoning
  is consolidated and `cluster_size` is tracked so sibling resource allocation
  knows which fork had stronger behavioral convergence vs. singletons.
- Cross-lineage novelty exhaustion (`known_forks`): filters out continuations
  that are remixes not only of earlier probes in the current K-batch, but also
  of forks already explored by ancestors or siblings in the arena.
- Tri-signal Plaus synthesis: when surface probes cluster onto a single path,
  yet logprob token entropy is high AND the critic pass flags multiple
  unjustified leaps, the detector converts the critic's unjustified leaps into
  explicit challenge forks so hidden uncertainty still splits properly.
"""

from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Set

from .jsonutil import as_str_list, parse_json_object


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
    cluster_size: int = 1
    support_score: float = 0.0
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


def _logprob(value: Any, default: float) -> float:
    """A log-probability is finite and <= 0; anything else (garbage, NaN, overflow bait) is `default`."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return min(0.0, number) if math.isfinite(number) else default


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
                probs.append(math.exp(_logprob(val, -10.0)))
            total_p = sum(probs) or 1.0
            norm_probs = [p / total_p for p in probs if p > 0]
            h = -sum(p * math.log(p) for p in norm_probs)
            entropies.append(h)
        else:
            val = getattr(tok_entry, "logprob", None)
            if val is None and isinstance(tok_entry, dict):
                val = tok_entry.get("logprob", 0.0)
            p = min(max(math.exp(_logprob(val, 0.0)), 1e-6), 1.0 - 1e-6)
            h = -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))
            entropies.append(h)

    return sum(entropies) / len(entropies) if entropies else 0.0


def _parse_probe_json(raw_text: str, idx: int) -> ProbeContinuation:
    raw_text = raw_text or ""
    data = parse_json_object(raw_text)
    if data is not None:
        fork_id = str(data.get("fork_id") or data.get("approach") or f"fork_{idx}")
        fork_desc = str(data.get("fork") or data.get("step") or data.get("approach") or raw_text[:80])
        reasoning = str(data.get("reasoning") or data.get("first_step") or raw_text)
    else:
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
    Piece 1: Issue detector (Pass 2 refined).
    - Fires K blind continuations at high temperature to read issues off behavioral divergence.
    - Clusters similar probes (tracking `cluster_size` and consolidating reasoning) while
      enforcing novelty exhaustion against both in-batch probes and `known_forks` in the lineage.
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
        # Versioned prompt text from the Piece 8 registry (`plaus_probe_hint`); empty = no hint
        self.probe_hint: str = ""
        # Learned on the first short batch: the provider returns one choice whatever `n` asks
        self.provider_ignores_n: bool = False
        self.n_topup_calls: int = 0
        self._stats_lock = threading.Lock()

    def _sample_choices(
        self, prompt: str, *, n: int, temperature: float, max_tokens: int
    ) -> tuple[List[Any], float]:
        """
        K blind samples of one prompt. Some OpenAI-compatible providers ignore `n` and
        return a single choice; without a top-up every probe batch would have one member,
        nothing could ever diverge, and the arena would never split. Missing samples are
        drawn with single-sample calls (each metered like any other), and once a provider
        has shown it drops `n`, later batches skip the wasted multi-sample request.
        """
        n = max(1, int(n))
        messages = [{"role": "user", "content": prompt}]

        def _call(count: int, sample_no: int = 0) -> Any:
            kwargs: dict = {"logprobs": True, "top_logprobs": 4}
            if count > 1:
                kwargs["n"] = count
            msgs = messages
            if sample_no:
                # A distinct prompt per top-up sample keeps caching or near-deterministic
                # providers from handing back the same continuation K times
                msgs = [{"role": "user", "content": f"{prompt}\nIndependent sample {sample_no + 1} of {n}."}]
            return self.client.chat.completions.create(
                messages=msgs, temperature=temperature, max_tokens=max_tokens, **kwargs
            )

        first = _call(1 if self.provider_ignores_n else n)
        choices: List[Any] = list(first.choices or [])[:n]
        cost = float(first.usage.cost_usd)
        if n > 1 and not self.provider_ignores_n and len(choices) == 1:
            self.provider_ignores_n = True
        # Top up a short batch; an empty reply can't be retried forever (bounded by n)
        for sample_no in range(1, n - len(choices) + 1):
            extra = _call(1, sample_no=sample_no)
            cost += float(extra.usage.cost_usd)
            with self._stats_lock:
                self.n_topup_calls += 1
            choices.extend(list(extra.choices or [])[:1])
            if len(choices) >= n:
                break
        return choices, cost

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
        raw = (resp.choices[0].message.content if resp.choices else "") or ""
        parsed = parse_json_object(raw)
        if parsed is not None:
            leaps = as_str_list(parsed.get("unjustified_leaps"))
        else:
            leaps = [raw.strip()[:240]] if raw.strip() else []
        return leaps, cost

    def _cluster_and_deduplicate(
        self,
        continuations: Sequence[ProbeContinuation],
        max_branches: int,
        known_forks: Sequence[str] = (),
    ) -> tuple[List[ProbeContinuation], bool]:
        """
        Cluster probe continuations and filter against `known_forks` until novelty is
        exhausted (new continuations are remixes of already-accepted forks) or bounded
        by `max_branches` (from `depth_dial`).
        """
        distinct: List[ProbeContinuation] = []
        novelty_exhausted = False
        total_probes = max(1, len(continuations))

        for cand in continuations:
            cand_text = f"{cand.fork_description} {cand.reasoning}"

            # Check if candidate is a remix of an already-explored ancestor/sibling fork
            already_known = False
            for kf in known_forks:
                if kf == cand.fork_description or _jaccard_similarity(cand.fork_description, kf) >= self.similarity_threshold:
                    already_known = True
                    break
            if already_known:
                novelty_exhausted = True
                continue

            # Check if candidate clusters with an existing fork in this probe batch
            matched_cluster: Optional[ProbeContinuation] = None
            for existing in distinct:
                if cand.fork_id == existing.fork_id:
                    matched_cluster = existing
                    break
                sim = _jaccard_similarity(
                    cand_text,
                    f"{existing.fork_description} {existing.reasoning}",
                )
                if sim >= self.similarity_threshold:
                    matched_cluster = existing
                    break

            if matched_cluster is not None:
                novelty_exhausted = True
                matched_cluster.cluster_size += 1
                if cand.reasoning not in matched_cluster.reasoning:
                    matched_cluster.reasoning = f"{matched_cluster.reasoning} | {cand.reasoning}"
                continue

            if len(distinct) < max_branches:
                distinct.append(cand)
            else:
                novelty_exhausted = True

        # Compute initial support_score for each cluster based on convergence & inverse entropy
        for cluster in distinct:
            convergence_bonus = (cluster.cluster_size / float(total_probes)) * 1.5
            entropy_penalty = cluster.token_entropy * 0.25
            cluster.support_score = round(convergence_bonus - entropy_penalty, 4)

        return distinct, novelty_exhausted

    def detect_plaus(
        self,
        problem: str,
        parent_reasoning: str,
        depth: int = 0,
        known_forks: Sequence[str] = (),
        k_override: Optional[int] = None,
    ) -> IssueDetectionResult:
        """
        Fire K diverse continuations blind (high temperature, cheap).
        - If they cluster (and critic/entropy don't flag hidden leaps) -> no issue,
          merge and flow straight.
        - If they genuinely split (or high entropy + critic leaps expose hidden forks) ->
          that split *is* the issue, discovered not predicted.
        """
        num_probes = max(1, k_override if k_override is not None else self.k_probes)
        prompt = (
            f"probe_continuation depth={depth}\n"
            f"Fire K diverse continuations for problem: {problem}\n"
            f"Parent reasoning so far: {parent_reasoning}\n"
            + (f"Guidance: {self.probe_hint}\n" if self.probe_hint else "")
            + "Return JSON with fork_id, fork, and reasoning."
        )
        choices, total_cost = self._sample_choices(
            prompt, n=num_probes, temperature=self.probe_temperature, max_tokens=200
        )

        probes: List[ProbeContinuation] = []
        for idx, ch in enumerate(choices):
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
        distinct_forks, novelty_exhausted = self._cluster_and_deduplicate(
            probes,
            max_branches=max_allowed,
            known_forks=known_forks,
        )

        entropy_high = mean_entropy >= self.entropy_threshold

        # Secondary Plaus synthesis: if surface continuations clustered into <=1 fork,
        # but high token entropy + multiple critic leaps indicate an unaddressed hidden fork,
        # turn the critic leaps into explicit alternative forks (up to max_allowed).
        if len(distinct_forks) <= 1 and entropy_high and len(leaps) >= 2 and max_allowed > 1:
            for idx, leap in enumerate(leaps):
                if len(distinct_forks) >= max_allowed:
                    break
                distinct_forks.append(
                    ProbeContinuation(
                        fork_id=f"critic_leap_fork_{idx + 1}",
                        fork_description=f"Resolve critic leap: {leap}",
                        reasoning=f"Testing assumption flagged by critic: {leap}",
                        token_entropy=mean_entropy,
                        cluster_size=1,
                        support_score=0.1,
                    )
                )

        behavioral_split = len(distinct_forks) > 1

        if behavioral_split:
            fork_names = ", ".join(f"{f.fork_id}(n={f.cluster_size})" for f in distinct_forks)
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
                f"(entropy_high={entropy_high}, critic_leaps={len(leaps)}) — flowing straight."
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
        known_forks: Sequence[str] = (),
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
        choices, total_cost = self._sample_choices(prompt, n=max_allowed, temperature=1.0, max_tokens=180)

        probes: List[ProbeContinuation] = []
        for idx, ch in enumerate(choices):
            p = _parse_probe_json(ch.message.content, idx)
            p.token_entropy = compute_token_entropy_from_choice(ch)
            probes.append(p)

        distinct_forks, novelty_exhausted = self._cluster_and_deduplicate(
            probes,
            max_branches=max_allowed,
            known_forks=known_forks,
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
