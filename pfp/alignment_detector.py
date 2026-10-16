"""
Piece 4: Alignment Detector — spotting mergeable branches (Pass 1: literal spec).

From DESIGN.md:
Two stages:
1. Embedding pre-filter (cheap): cosine similarity on branch positions.
   Recall only — similar-sounding != compatible.
2. Judge call (Super): extract each branch's core claims, compare
   claim-by-claim. Overlap -> alignable. Contradiction -> leave them to fight.
   Disjoint -> unrelated, walk away.

Safety net: a bad merge isn't fatal. Merged branches start Possible and must
survive falsification like everything else. The detector needs recall;
selection handles precision.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .backend import SUPER_MODEL
from .tracker import BranchRecord


class AlignmentVerdict(str, Enum):
    OVERLAP = "overlap"              # Overlap / compatible -> alignable (mergeable)
    CONTRADICTORY = "contradictory"  # Contradiction -> leave them to fight (triggers falsification)
    UNRELATED = "unrelated"          # Disjoint -> unrelated, walk away


@dataclass
class AlignmentCandidatePair:
    branch_a_id: str
    branch_b_id: str
    cosine_similarity: float
    passed_prefilter: bool


@dataclass
class AlignmentJudgeResult:
    branch_a_id: str
    branch_b_id: str
    cosine_similarity: float
    verdict: AlignmentVerdict
    claims_a: List[str] = field(default_factory=list)
    claims_b: List[str] = field(default_factory=list)
    overlapping_claims: List[str] = field(default_factory=list)
    contradictory_claims: List[str] = field(default_factory=list)
    disjoint_claims: List[str] = field(default_factory=list)
    rationale: str = ""
    cost_usd: float = 0.0

    @property
    def is_alignable(self) -> bool:
        return self.verdict == AlignmentVerdict.OVERLAP

    @property
    def is_contradictory(self) -> bool:
        return self.verdict == AlignmentVerdict.CONTRADICTORY

    def to_dict(self) -> Dict[str, Any]:
        return {
            "branch_a_id": self.branch_a_id,
            "branch_b_id": self.branch_b_id,
            "cosine_similarity": round(self.cosine_similarity, 4),
            "verdict": self.verdict.value,
            "is_alignable": self.is_alignable,
            "is_contradictory": self.is_contradictory,
            "claims_a": self.claims_a,
            "claims_b": self.claims_b,
            "overlapping_claims": self.overlapping_claims,
            "contradictory_claims": self.contradictory_claims,
            "disjoint_claims": self.disjoint_claims,
            "rationale": self.rationale,
            "cost_usd": round(self.cost_usd, 6),
        }


_STOP_WORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "into", "onto", "upon", "directive", "root", "start", "prior", "parent",
    "reasoning", "branch", "fork", "compressed", "trace", "hypothesis", "step",
}


def compute_branch_embedding(text: str, dim: int = 128) -> List[float]:
    """
    Cheap deterministic vector embedding of a branch's position (fork + reasoning)
    for high-recall cosine similarity pre-filtering.
    Uses non-negative bag-of-features hashing over word unigrams, character trigrams,
    and domain concept anchors so shared concepts strictly increase cosine similarity.
    """
    vec = [0.0] * dim
    tokens = [
        w
        for w in re.findall(r"[a-z0-9_]+", text.lower())
        if len(w) > 2 and w not in _STOP_WORDS
    ]
    if not tokens:
        return vec

    features: List[Tuple[str, float]] = [(t, 1.0) for t in tokens]
    for i in range(len(tokens) - 1):
        features.append((f"{tokens[i]}_{tokens[i + 1]}", 0.5))

    # Domain concept buckets so surface-related positions recall well in the pre-filter
    lower = text.lower()
    if any(k in lower for k in ("sshd", "port", "22", "tcp", "firewall", "ufw", "inbound", "daemon", "fail2ban", "listening")):
        features.append(("__concept_ssh_host_ingress__", 2.0))
    if any(k in lower for k in ("laptop", "desktop", "host", "server")):
        features.append(("__concept_endpoints__", 1.0))
    if any(k in lower for k in ("key", "authorized_keys", "chmod", "auth", "strictmodes")):
        features.append(("__concept_ssh_auth__", 2.0))
    if any(k in lower for k in ("vlan", "wifi", "wlan", "ipv6", "routing", "arp", "dns", "link_local")):
        features.append(("__concept_network_routing__", 2.0))

    for feat, weight in features:
        h = hashlib.md5(feat.encode("utf-8")).digest()
        idx = int.from_bytes(h[:2], "little") % dim
        vec[idx] += weight

    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0.0:
        vec = [v / norm for v in vec]
    return vec


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    na = math.sqrt(sum(a * a for a in vec_a))
    nb = math.sqrt(sum(b * b for b in vec_b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def branch_position_text(record: BranchRecord) -> str:
    trace_str = " | ".join(record.reasoning_trace)
    return f"{record.fork_under_exploration} | {trace_str}".strip()


class AlignmentDetector:
    """
    Piece 4: Alignment detector — spotting mergeable branches.
    Stage 1: Embedding pre-filter (cheap cosine similarity on branch positions, recall-oriented).
    Stage 2: Super judge call (`nvidia/nemotron-3-super-120b-a12b`) extracting core claims
             and comparing claim-by-claim into overlap/compatible, contradictory, or unrelated.
    """

    def __init__(
        self,
        client: Any,
        *,
        prefilter_threshold: float = 0.20,
        embedding_dim: int = 64,
        super_model: str = SUPER_MODEL,
    ) -> None:
        self.client = client
        self.prefilter_threshold = prefilter_threshold
        self.embedding_dim = embedding_dim
        self.super_model = super_model

    def embedding_prefilter(
        self,
        branches: Sequence[BranchRecord],
    ) -> List[AlignmentCandidatePair]:
        """
        Stage 1: Compute cosine similarity across candidate branch pairs.
        Recall-only filter: pairs with similarity >= prefilter_threshold advance to Super judge.
        """
        embeddings: Dict[str, List[float]] = {
            b.id: compute_branch_embedding(branch_position_text(b), dim=self.embedding_dim)
            for b in branches
        }
        pairs: List[AlignmentCandidatePair] = []
        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                ba, bb = branches[i], branches[j]
                # Skip direct parent-child pairs; we compare concurrent/peer branches
                if ba.parent == bb.id or bb.parent == ba.id:
                    continue
                sim = cosine_similarity(embeddings[ba.id], embeddings[bb.id])
                pairs.append(
                    AlignmentCandidatePair(
                        branch_a_id=ba.id,
                        branch_b_id=bb.id,
                        cosine_similarity=round(sim, 4),
                        passed_prefilter=(sim >= self.prefilter_threshold),
                    )
                )
        pairs.sort(key=lambda p: p.cosine_similarity, reverse=True)
        return pairs

    def extract_claims(self, branch: BranchRecord) -> Tuple[List[str], float]:
        """
        Extract core claims from a branch's position via the Super model.
        """
        pos_text = branch_position_text(branch)
        resp = self.client.chat.completions.create(
            model=self.super_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the PFP Super judge. Extract the branch's core claims as JSON: "
                        '{"core_claims": ["claim 1", "claim 2"]}.'
                    ),
                },
                {
                    "role": "user",
                    "content": f"extract_claims — extract core claims from branch {branch.id}:\n{pos_text}",
                },
            ],
            temperature=0.1,
            max_tokens=160,
        )
        cost = float(resp.usage.cost_usd)
        raw = resp.choices[0].message.content if resp.choices else "{}"
        try:
            parsed = json.loads(raw)
            claims = [str(c) for c in parsed.get("core_claims", []) if str(c).strip()]
        except Exception:
            claims = [pos_text[:160]] if pos_text else []
        if not claims:
            claims = [branch.fork_under_exploration or pos_text[:120]]
        return claims, cost

    def judge_pair(
        self,
        branch_a: BranchRecord,
        branch_b: BranchRecord,
        cosine_sim: Optional[float] = None,
    ) -> AlignmentJudgeResult:
        """
        Stage 2: Super judge call (`nvidia/nemotron-3-super-120b-a12b`).
        Extracts each branch's core claims and compares them claim-by-claim:
        - overlap (compatible -> alignable)
        - contradictory (clash -> leave them to fight / trigger falsification)
        - unrelated (disjoint -> walk away)
        """
        if cosine_sim is None:
            va = compute_branch_embedding(branch_position_text(branch_a), dim=self.embedding_dim)
            vb = compute_branch_embedding(branch_position_text(branch_b), dim=self.embedding_dim)
            cosine_sim = cosine_similarity(va, vb)

        claims_a, cost_a = self.extract_claims(branch_a)
        claims_b, cost_b = self.extract_claims(branch_b)

        prompt = (
            f"alignment_judge — compare claim-by-claim between two PFP branches.\n"
            f"Branch A ({branch_a.id}, fork={branch_a.fork_under_exploration}):\n"
            f"Claims A: {json.dumps(claims_a)}\n"
            f"Position A: {branch_position_text(branch_a)}\n\n"
            f"Branch B ({branch_b.id}, fork={branch_b.fork_under_exploration}):\n"
            f"Claims B: {json.dumps(claims_b)}\n"
            f"Position B: {branch_position_text(branch_b)}\n\n"
            "Return JSON with verdict ('overlap', 'contradictory', or 'unrelated'), "
            "overlapping_claims, contradictory_claims, disjoint_claims, and rationale."
        )
        resp = self.client.chat.completions.create(
            model=self.super_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the PFP Super alignment judge. Compare claims from Branch A and Branch B "
                        "claim-by-claim and classify as 'overlap', 'contradictory', or 'unrelated'."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=220,
        )
        total_cost = cost_a + cost_b + float(resp.usage.cost_usd)
        raw = resp.choices[0].message.content if resp.choices else "{}"

        try:
            data = json.loads(raw)
            v_str = str(data.get("verdict", "unrelated")).lower().strip()
            if v_str in ("overlap", "compatible", "alignable"):
                verdict = AlignmentVerdict.OVERLAP
            elif v_str in ("contradictory", "contradiction", "clash"):
                verdict = AlignmentVerdict.CONTRADICTORY
            else:
                verdict = AlignmentVerdict.UNRELATED
            overlapping = [str(x) for x in data.get("overlapping_claims", [])]
            contradictory = [str(x) for x in data.get("contradictory_claims", [])]
            disjoint = [str(x) for x in data.get("disjoint_claims", [])]
            rationale = str(data.get("rationale", ""))
        except Exception:
            verdict = AlignmentVerdict.UNRELATED
            overlapping, contradictory, disjoint = [], [], []
            rationale = raw.strip()

        return AlignmentJudgeResult(
            branch_a_id=branch_a.id,
            branch_b_id=branch_b.id,
            cosine_similarity=round(float(cosine_sim), 4),
            verdict=verdict,
            claims_a=claims_a,
            claims_b=claims_b,
            overlapping_claims=overlapping,
            contradictory_claims=contradictory,
            disjoint_claims=disjoint,
            rationale=rationale,
            cost_usd=round(total_cost, 6),
        )

    def detect_alignments(
        self,
        branches: Sequence[BranchRecord],
        max_pairs_to_judge: int = 6,
    ) -> List[AlignmentJudgeResult]:
        """
        Run two-stage alignment detection across active branches:
        1. Cheap embedding cosine similarity pre-filter (recall).
        2. Super judge claim-by-claim comparison on pairs that pass pre-filter.
        """
        by_id = {b.id: b for b in branches if b.is_alive}
        candidates = self.embedding_prefilter(list(by_id.values()))
        shortlisted = [c for c in candidates if c.passed_prefilter][:max_pairs_to_judge]

        results: List[AlignmentJudgeResult] = []
        for cand in shortlisted:
            ba = by_id.get(cand.branch_a_id)
            bb = by_id.get(cand.branch_b_id)
            if ba is None or bb is None or not ba.is_alive or not bb.is_alive:
                continue
            res = self.judge_pair(ba, bb, cosine_sim=cand.cosine_similarity)
            results.append(res)
        return results
