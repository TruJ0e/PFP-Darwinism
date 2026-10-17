"""
OpenAI-compatible Model Backend for PFP Darwinism.

Reads NEBIUS_BASE_URL, NEBIUS_MODEL, and NEBIUS_API_KEY from the environment.
Provides:
- OpenAICompatibleClient: live HTTP client against Nebius Token Factory
  (or any OpenAI-compatible /chat/completions endpoint).
- MockOpenAIClient: drop-in mock implementing the identical
  `client.chat.completions.create(...)` interface, returning plausible diverse
  reasoning continuations, token logprobs, critic evaluations, and difficulty
  estimates so the harness (branch, track, budget) runs end-to-end without a key.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1"
DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
SUPER_MODEL = "nvidia/nemotron-3-super-120b-a12b"

# Live catalog pricing from DESIGN.md (USD per 1M tokens in/out)
MODEL_PRICING_PER_1M: Dict[str, tuple[float, float]] = {
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B": (0.06, 0.24),
    "nvidia/nemotron-3-super-120b-a12b": (0.30, 0.90),
}
# Unknown models must not silently get the cheapest known price: that would make
# the hard-cap preflight optimistic.  Use the most expensive catalog rates as a
# conservative fallback (callers can add a catalog entry for exact accounting).
DEFAULT_PRICING_PER_1M = (
    max(rate[0] for rate in MODEL_PRICING_PER_1M.values()),
    max(rate[1] for rate in MODEL_PRICING_PER_1M.values()),
)


def get_base_url() -> str:
    return os.environ.get("NEBIUS_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def get_model_name() -> str:
    return os.environ.get("NEBIUS_MODEL", DEFAULT_MODEL)


def estimate_call_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    model: Optional[str] = None,
) -> float:
    model_id = model or get_model_name()
    in_rate, out_rate = MODEL_PRICING_PER_1M.get(model_id, DEFAULT_PRICING_PER_1M)
    return (prompt_tokens / 1_000_000.0) * in_rate + (completion_tokens / 1_000_000.0) * out_rate


@dataclass
class TopLogprob:
    token: str
    logprob: float

    def to_dict(self) -> Dict[str, Any]:
        return {"token": self.token, "logprob": self.logprob}


@dataclass
class TokenLogprob:
    token: str
    logprob: float
    top_logprobs: List[TopLogprob] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token": self.token,
            "logprob": self.logprob,
            "top_logprobs": [t.to_dict() for t in self.top_logprobs],
        }


@dataclass
class ChoiceLogprobs:
    content: List[TokenLogprob] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"content": [c.to_dict() for c in self.content]}


@dataclass
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass
class Choice:
    index: int
    message: ChatMessage
    finish_reason: str = "stop"
    logprobs: Optional[ChoiceLogprobs] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "message": self.message.to_dict(),
            "finish_reason": self.finish_reason,
            "logprobs": self.logprobs.to_dict() if self.logprobs else None,
        }


@dataclass
class CompletionUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
        }


@dataclass
class ChatCompletionResponse:
    id: str
    model: str
    choices: List[Choice]
    usage: CompletionUsage
    created: int = field(default_factory=lambda: int(time.time()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [c.to_dict() for c in self.choices],
            "usage": self.usage.to_dict(),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


class BackendError(RuntimeError):
    """Raised when the live endpoint fails after all retries (carries HTTP status + body snippet)."""

    def __init__(self, message: str, status: Optional[int] = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


class HardCapExceededError(RuntimeError):
    """Raised by `MeteredClient` when a call would start after true spend reached the hard cap."""


# HTTP statuses worth retrying: rate limits and transient upstream failures.
RETRYABLE_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class _LiveChatCompletions:
    """OpenAI-compatible `client.chat.completions` implementation over HTTP."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        default_model: str,
        timeout: float = 120.0,
        max_retries: int = 3,
        backoff_base_s: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_model = default_model
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.backoff_base_s = max(0.0, backoff_base_s)
        self._sleep = sleep

    def _backoff_delay(self, attempt: int, retry_after: Optional[str]) -> float:
        if retry_after:
            try:
                return min(60.0, max(0.0, float(retry_after)))
            except ValueError:
                pass
        # Exponential backoff with full jitter so hundreds of parallel branches don't stampede
        return random.uniform(0.0, self.backoff_base_s * (2 ** attempt))

    def _post_with_retries(self, body: bytes) -> Dict[str, Any]:
        last_error: Optional[BackendError] = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            retry_after: Optional[str] = None
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.load(resp)
            except urllib.error.HTTPError as exc:
                snippet = ""
                try:
                    snippet = exc.read().decode("utf-8", "replace")[:500]
                except Exception:
                    pass
                last_error = BackendError(
                    f"HTTP {exc.code} from {self.base_url}/chat/completions: {snippet}",
                    status=exc.code,
                    body=snippet,
                )
                if exc.code not in RETRYABLE_STATUSES:
                    raise last_error from exc
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last_error = BackendError(f"Network error calling {self.base_url}: {exc}")
            except json.JSONDecodeError as exc:
                last_error = BackendError(f"Non-JSON response from {self.base_url}: {exc}")
            if attempt < self.max_retries:
                self._sleep(self._backoff_delay(attempt, retry_after))
        assert last_error is not None
        raise last_error

    def create(
        self,
        messages: Sequence[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 256,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None,
        n: int = 1,
        **kwargs: Any,
    ) -> ChatCompletionResponse:
        target_model = model or self.default_model
        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if logprobs:
            payload["logprobs"] = True
            if top_logprobs is not None:
                payload["top_logprobs"] = top_logprobs
        if n > 1:
            payload["n"] = n
        payload.update(kwargs)

        body = json.dumps(payload).encode("utf-8")
        raw = self._post_with_retries(body)

        choices: List[Choice] = []
        for idx, ch in enumerate(raw.get("choices", [])):
            msg = ch.get("message", {})
            lp_raw = ch.get("logprobs")
            lp_obj: Optional[ChoiceLogprobs] = None
            if isinstance(lp_raw, dict) and isinstance(lp_raw.get("content"), list):
                tokens: List[TokenLogprob] = []
                for t in lp_raw["content"]:
                    tops = [
                        TopLogprob(token=str(tp.get("token", "")), logprob=float(tp.get("logprob", -1.0)))
                        for tp in t.get("top_logprobs", [])
                    ]
                    tokens.append(
                        TokenLogprob(
                            token=str(t.get("token", "")),
                            logprob=float(t.get("logprob", -0.5)),
                            top_logprobs=tops,
                        )
                    )
                lp_obj = ChoiceLogprobs(content=tokens)
            choices.append(
                Choice(
                    index=ch.get("index", idx),
                    message=ChatMessage(
                        role=msg.get("role") or "assistant",
                        # Reasoning models may return null content with the answer in reasoning_content
                        content=msg.get("content") or msg.get("reasoning_content") or "",
                    ),
                    finish_reason=ch.get("finish_reason") or "stop",
                    logprobs=lp_obj,
                )
            )

        usage_raw = raw.get("usage") or {}
        p_tok = int(usage_raw.get("prompt_tokens", 0))
        c_tok = int(usage_raw.get("completion_tokens", 0))
        t_tok = int(usage_raw.get("total_tokens", p_tok + c_tok))
        cost = estimate_call_cost_usd(p_tok, c_tok, target_model)

        return ChatCompletionResponse(
            id=str(raw.get("id", f"chatcmpl-{int(time.time() * 1000)}")),
            model=str(raw.get("model", target_model)),
            choices=choices,
            usage=CompletionUsage(
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                total_tokens=t_tok,
                cost_usd=cost,
            ),
        )


class _LiveChatNamespace:
    def __init__(self, completions: _LiveChatCompletions):
        self.completions = completions


class OpenAICompatibleClient:
    """
    OpenAI-compatible client configured from NEBIUS_BASE_URL, NEBIUS_MODEL,
    and NEBIUS_API_KEY.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        backoff_base_s: float = 0.5,
    ):
        self.base_url = (base_url or get_base_url()).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("NEBIUS_API_KEY", "")
        self.model = model or get_model_name()
        self.is_mock = False
        self.chat = _LiveChatNamespace(
            _LiveChatCompletions(
                base_url=self.base_url,
                api_key=self.api_key,
                default_model=self.model,
                timeout=timeout,
                max_retries=max_retries,
                backoff_base_s=backoff_base_s,
            )
        )


class _MeteredCompletions:
    def __init__(self, owner: "MeteredClient"):
        self._owner = owner

    def create(self, *args: Any, **kwargs: Any) -> Any:
        return self._owner._metered_create(*args, **kwargs)


class _MeteredChatNamespace:
    def __init__(self, completions: _MeteredCompletions):
        self.completions = completions


class MeteredClient:
    """
    Ground-truth spend meter + safety rail around any OpenAI-compatible client.

    - Every call's real `usage.cost_usd` is recorded, whether or not a branch
      budget could cover it, so reported spend is the truth (not what the
      ledger could afford).
    - Hard cap: once true spend reaches `hard_cap_usd`, further calls raise
      `HardCapExceededError` before any request is sent.
    - `max_concurrent_calls` bounds in-flight requests across all parallel
      mini-harnesses (hundreds of branches, one connection pool).
    """

    def __init__(
        self,
        inner: Any,
        *,
        hard_cap_usd: Optional[float] = None,
        max_concurrent_calls: int = 32,
    ) -> None:
        self.inner = inner
        self.hard_cap_usd = hard_cap_usd
        self.model = getattr(inner, "model", "")
        self.base_url = getattr(inner, "base_url", "")
        self.is_mock = getattr(inner, "is_mock", False)
        self.total_cost_usd: float = 0.0
        self.calls: int = 0
        self.failed_calls: int = 0
        self.cost_by_model: Dict[str, float] = {}
        self._reserved_cost_usd: float = 0.0
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(max(1, int(max_concurrent_calls)))
        self.chat = _MeteredChatNamespace(_MeteredCompletions(self))

    @property
    def cap_reached(self) -> bool:
        with self._lock:
            return self.hard_cap_usd is not None and (
                self.total_cost_usd + self._reserved_cost_usd >= self.hard_cap_usd
            )

    @staticmethod
    def _prompt_token_upper_bound(messages: Any) -> int:
        """Conservative preflight bound: a token cannot exceed its UTF-8 byte input."""
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            return 1024
        total = 1024  # protocol/template overhead across providers
        for message in messages:
            if isinstance(message, dict):
                total += len(str(message.get("role", "")).encode("utf-8"))
                total += len(str(message.get("content", "")).encode("utf-8"))
                total += 64
            else:
                total += len(str(message).encode("utf-8")) + 64
        return total

    def _reserve_call(self, kwargs: Dict[str, Any]) -> float:
        model = str(kwargs.get("model") or self.model)
        prompt_bound = self._prompt_token_upper_bound(kwargs.get("messages", []))
        completion_bound = max(0, int(kwargs.get("max_tokens", 256)))
        n = max(1, int(kwargs.get("n", 1)))
        reservation = estimate_call_cost_usd(prompt_bound, completion_bound * n, model)
        with self._lock:
            projected = self.total_cost_usd + self._reserved_cost_usd + reservation
            if self.hard_cap_usd is not None and projected > self.hard_cap_usd:
                raise HardCapExceededError(
                    f"Hard cap ${self.hard_cap_usd:.4f} would be exceeded by a new call "
                    f"(spent ${self.total_cost_usd:.4f}, in-flight reserve "
                    f"${self._reserved_cost_usd:.4f}, call reserve ${reservation:.4f}); refusing it."
                )
            self._reserved_cost_usd += reservation
        return reservation

    def _metered_create(self, *args: Any, **kwargs: Any) -> Any:
        reservation = self._reserve_call(kwargs)
        try:
            with self._slots:
                resp = self.inner.chat.completions.create(*args, **kwargs)
        except Exception:
            with self._lock:
                self._reserved_cost_usd = max(0.0, self._reserved_cost_usd - reservation)
                self.failed_calls += 1
            raise
        cost = float(getattr(getattr(resp, "usage", None), "cost_usd", 0.0) or 0.0)
        model = str(kwargs.get("model") or self.model)
        with self._lock:
            self._reserved_cost_usd = max(0.0, self._reserved_cost_usd - reservation)
            self.calls += 1
            self.total_cost_usd = round(self.total_cost_usd + cost, 8)
            self.cost_by_model[model] = round(self.cost_by_model.get(model, 0.0) + cost, 8)
        return resp

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "calls": self.calls,
                "failed_calls": self.failed_calls,
                "total_cost_usd": round(self.total_cost_usd, 6),
                "in_flight_reserved_usd": round(self._reserved_cost_usd, 6),
                "cost_by_model": {k: round(v, 6) for k, v in self.cost_by_model.items()},
                "hard_cap_usd": self.hard_cap_usd,
                "cap_reached": self.hard_cap_usd is not None and self.total_cost_usd >= self.hard_cap_usd,
            }


# ---------------------------------------------------------------------------
# Mock Backend (plausible diverse responses, zero code changes needed for live)
# ---------------------------------------------------------------------------

_PLAUS_FORK_POOL = [
    {
        "fork_id": "network_firewall_asymmetry",
        "approach": "Inspect stateful firewall and inbound port 22 rules on the laptop",
        "reasoning": (
            "Hypothesis A (Firewall/NAT asymmetry): Laptop->desktop succeeds because outbound "
            "SYN from laptop establishes stateful return path, whereas desktop->laptop inbound "
            "TCP/22 is blocked by laptop host firewall (ufw/Windows Defender) or network profile."
        ),
        "conclusion": (
            "Verified solution: Enable sshd and allow inbound TCP/22 on the laptop's local "
            "firewall profile; confirm with `ss -tlnp | grep :22`."
        ),
    },
    {
        "fork_id": "sshd_daemon_inactive",
        "approach": "Verify OpenSSH server daemon is installed and listening on the laptop",
        "reasoning": (
            "Hypothesis B (Missing/stopped sshd service): Client binaries (`ssh`) ship by default "
            "on laptops while the server daemon (`sshd`) is disabled or unconfigured on mobile hosts."
        ),
        "conclusion": (
            "Verified solution: Start and enable `sshd` service on laptop (`systemctl enable --now ssh`) "
            "and verify ListenAddress binds to the LAN interface."
        ),
    },
    {
        "fork_id": "auth_authorized_keys_mismatch",
        "approach": "Audit public key permissions and sshd_config Auth modes on laptop",
        "reasoning": (
            "Hypothesis C (Authentication / StrictModes rejection): Desktop key is missing from "
            "laptop `~/.ssh/authorized_keys` or home directory permissions violate sshd StrictModes."
        ),
        "conclusion": (
            "Verified solution: Fix `chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys` on laptop "
            "and install desktop's ed25519 public key."
        ),
    },
    {
        "fork_id": "wifi_client_isolation_routing",
        "approach": "Check AP client isolation and subnet boundary between wired desktop and Wi-Fi laptop",
        "reasoning": (
            "Hypothesis D (VLAN / Wi-Fi isolation): Desktop on wired LAN and laptop on guest/isolated "
            "WLAN have asymmetric ACLs or mDNS hostname resolution drift."
        ),
        "conclusion": (
            "Verified solution: Move laptop off isolated guest SSID onto primary LAN VLAN and verify "
            "bidirectional ARP/ICMP reachability."
        ),
    },
]

_NS_MUTATION_POOL = [
    {
        "fork_id": "ns_ssh_proxycommand_override",
        "approach": "Blind variation: Check ~/.ssh/config on desktop for stale ProxyCommand or Host alias",
        "reasoning": (
            "NS Mutation 1 (Client-side ssh_config override): Desktop's `~/.ssh/config` maps the laptop "
            "alias to a stale IP, jump host, or non-standard port, causing connection failure before "
            "packets ever reach the laptop's current LAN address."
        ),
    },
    {
        "fork_id": "ns_ipv6_link_local_scope",
        "approach": "Blind variation: Test IPv6 link-local scope ID mismatch during DNS resolution",
        "reasoning": (
            "NS Mutation 2 (Dual-stack AAAA blackhole): Desktop resolves laptop hostname to an IPv6 "
            "link-local address without interface scope (`%eth0`), stalling the handshake while IPv4 works."
        ),
    },
    {
        "fork_id": "ns_fail2ban_lockout",
        "approach": "Blind variation: Check fail2ban / sshguard ban list on the laptop for desktop IP",
        "reasoning": (
            "NS Mutation 3 (Intrusion prevention ban): Automated retry scripts on desktop triggered "
            "a temporary nftables ban in fail2ban on the laptop."
        ),
    },
]


class _MockChatCompletions:
    """Mock implementation of `client.chat.completions` with deterministic + diverse behavior."""

    def __init__(self, default_model: str, seed: int = 42, force_cost_scale: float = 1.0):
        self.default_model = default_model
        self._rng = random.Random(seed)
        self._call_count = 0
        self._count_lock = threading.Lock()
        self.force_cost_scale = force_cost_scale

    def _build_logprobs(self, high_entropy: bool) -> ChoiceLogprobs:
        tokens: List[TokenLogprob] = []
        for idx in range(6):
            if high_entropy:
                # Nearly uniform top-4 distribution -> high Shannon entropy (~1.35 nats)
                probs = [0.28, 0.26, 0.24, 0.22]
            else:
                # Peaked distribution -> low Shannon entropy (~0.25 nats)
                probs = [0.92, 0.04, 0.03, 0.01]
            tops = [
                TopLogprob(token=f"tok_{idx}_{j}", logprob=math.log(p))
                for j, p in enumerate(probs)
            ]
            tokens.append(
                TokenLogprob(
                    token=tops[0].token,
                    logprob=tops[0].logprob,
                    top_logprobs=tops,
                )
            )
        return ChoiceLogprobs(content=tokens)

    def _generate_one_content(
        self,
        prompt_text: str,
        temperature: float,
        choice_idx: int,
    ) -> tuple[str, bool]:
        lower = prompt_text.lower()
        # Content-addressed choices: the same prompt always yields the same reply, no matter
        # how parallel branches interleave — runs and replays are reproducible
        digest = int(hashlib.sha1(prompt_text.encode("utf-8")).hexdigest()[:8], 16)

        # 1. Difficulty estimation probe call
        if "difficulty_estimate" in lower or "size the problem" in lower:
            return (
                json.dumps(
                    {
                        "difficulty_tier": "medium",
                        "estimated_calls": 40,
                        "difficulty_estimate_usd": 0.5,
                        "rationale": "Multi-hypothesis systems diagnostic requiring parallel fork exploration.",
                    }
                ),
                False,
            )

        # 2. Critic pass listing unjustified leaps
        if "critic pass" in lower or "unjustified leaps" in lower:
            if "verified solution" in lower or "converged" in lower or "cluster" in lower:
                leaps: List[str] = []
            elif "ns mutation" in lower:
                leaps = ["Assumes client-side override without packet capture verification"]
            elif digest % 3 == 0:
                leaps = [
                    "Assumes TCP port 22 is reachable without checking host firewall state",
                    "Skips daemon status check before testing authentication keys",
                ]
            else:
                leaps = ["Needs direct verification command output on target host"]
            return json.dumps({"unjustified_leaps": leaps, "count": len(leaps)}), False

        # 3. NS (Natural Selection) blind variation prompt
        if (
            "ns_fanout" in lower
            or (
                ("natural selection" in lower or "blind variation" in lower)
                and "extract_claims" not in lower
                and "alignment_judge" not in lower
                and "merge_synthesizer" not in lower
                and "fail_judge" not in lower
            )
        ):
            pick = _NS_MUTATION_POOL[(digest + choice_idx) % len(_NS_MUTATION_POOL)]
            return (
                json.dumps(
                    {
                        "trigger": "NS",
                        "fork_id": pick["fork_id"],
                        "fork": pick["approach"],
                        "reasoning": pick["reasoning"],
                    }
                ),
                True,
            )

        # 4. Probe continuations (Plaus divergence check)
        if "probe_continuation" in lower or "fire k diverse continuations" in lower:
            # If the prompt indicates deep depth (>=2) or already narrowed to a specific verified check,
            # make the continuations cluster together so some branches flow straight without endless splitting!
            if "depth=2" in lower or "depth=3" in lower or "cluster_mode=converge" in lower:
                base = _PLAUS_FORK_POOL[0]
                return (
                    json.dumps(
                        {
                            "fork_id": base["fork_id"],
                            "fork": base["approach"],
                            "reasoning": base["reasoning"] + f" (confirmation probe #{choice_idx + 1})",
                        }
                    ),
                    False,
                )
            # Otherwise at high temperature, return genuinely diverse forks across the K probes
            if temperature >= 0.6:
                # Cycle through distinct forks, with an occasional remix to exercise novelty exhaustion
                pool_idx = (choice_idx if choice_idx < len(_PLAUS_FORK_POOL) else choice_idx % 2)
                fork = _PLAUS_FORK_POOL[pool_idx]
                return (
                    json.dumps(
                        {
                            "fork_id": fork["fork_id"],
                            "fork": fork["approach"],
                            "reasoning": fork["reasoning"],
                        }
                    ),
                    True,
                )
            else:
                fork = _PLAUS_FORK_POOL[0]
                return (
                    json.dumps(
                        {
                            "fork_id": fork["fork_id"],
                            "fork": fork["approach"],
                            "reasoning": fork["reasoning"],
                        }
                    ),
                    False,
                )

        # 5. Compression / Directive summary
        if "compress parent reasoning" in lower:
            digest = hashlib.sha1(prompt_text.encode("utf-8")).hexdigest()[:8]
            return f"Compressed parent trace [{digest}]: evaluated SSH asymmetry hypotheses; narrowing active fork.", False

        # 6. Piece 4: Claim extraction for a branch
        if "extract_claims" in lower or "extract core claims" in lower:
            claims: List[str] = []
            if "firewall" in lower or "port 22" in lower or "hypothesis a" in lower:
                claims.extend(
                    [
                        "Laptop outbound TCP/22 SYN succeeds due to stateful return path",
                        "Inbound TCP/22 from desktop to laptop is blocked or unlistening on laptop",
                    ]
                )
            if "sshd" in lower or "daemon" in lower or "hypothesis b" in lower:
                claims.extend(
                    [
                        "Inbound TCP/22 from desktop to laptop is blocked or unlistening on laptop",
                        "OpenSSH server daemon (sshd) must be enabled and listening on laptop LAN interface",
                    ]
                )
            if "contradict" in lower or "port 22 is open and listening" in lower or "fail2ban" in lower:
                claims.append(
                    "Inbound TCP/22 is already open and sshd is actively accepting packets, but fail2ban drops desktop IP"
                )
            if "ipv6" in lower or "proxycommand" in lower or "vlan" in lower:
                claims.append("Client-side DNS/routing or VLAN boundary isolates the two endpoints")
            if not claims:
                claims.append("Diagnose asymmetric SSH connectivity between desktop and laptop")
            return json.dumps({"core_claims": claims}), False

        # 7. Piece 4: Super judge alignment comparison
        if "alignment_judge" in lower or "compare claim-by-claim" in lower:
            parts = lower.split("branch b (")
            part_a = parts[0]
            part_b = parts[1] if len(parts) > 1 else ""

            a_ingress = any(k in part_a for k in ("firewall", "sshd", "port 22", "hypothesis a", "hypothesis b"))
            b_ingress = any(k in part_b for k in ("firewall", "sshd", "port 22", "hypothesis a", "hypothesis b"))
            has_contradict = (
                "contradict:" in lower
                or "contradict_mode=true" in lower
                or ("fail2ban" in part_a and b_ingress)
                or ("fail2ban" in part_b and a_ingress)
            )

            if has_contradict:
                return (
                    json.dumps(
                        {
                            "verdict": "contradictory",
                            "overlapping_claims": [],
                            "contradictory_claims": [
                                "Branch A claims inbound TCP/22 is blocked/unlistening while Branch B claims sshd is open and fail2ban drops after handshake"
                            ],
                            "disjoint_claims": [],
                            "rationale": "Direct clash on whether laptop TCP/22 is listening vs actively banning after connection.",
                        }
                    ),
                    False,
                )
            if (a_ingress and b_ingress) or "overlap_mode=true" in lower:
                return (
                    json.dumps(
                        {
                            "verdict": "overlap",
                            "overlapping_claims": [
                                "Inbound TCP/22 from desktop to laptop requires both active sshd listener and open host firewall rule on laptop"
                            ],
                            "contradictory_claims": [],
                            "disjoint_claims": [
                                "Check `ss -tlnp | grep :22` and `ufw status` on laptop"
                            ],
                            "rationale": "Complementary host-level diagnostics on laptop TCP/22 ingress; compatible and mergeable.",
                        }
                    ),
                    False,
                )
            return (
                json.dumps(
                    {
                        "verdict": "unrelated",
                        "overlapping_claims": [],
                        "contradictory_claims": [],
                        "disjoint_claims": ["Orthogonal network/client hypotheses with no shared claims"],
                        "rationale": "Claims address disjoint layers without overlap or contradiction.",
                    }
                ),
                False,
            )

        # 8. Piece 5: Merge synthesizer (Super call unifying aligned branches)
        if "merge_synthesizer" in lower or "unify compatible branches" in lower:
            return (
                json.dumps(
                    {
                        "unified_fork": "Unified host-ingress remediation: enable laptop sshd + allow inbound TCP/22 on firewall",
                        "unified_reasoning": (
                            "Merged synthesis: Laptop->desktop works via outbound stateful SYN, while desktop->laptop "
                            "requires both (1) starting/enabling `sshd` on the laptop LAN interface and (2) opening "
                            "inbound TCP/22 in the laptop host firewall (`ufw allow 22/tcp` / Windows Defender rule)."
                        ),
                        "kept_claims": [
                            "Start and enable OpenSSH server daemon (`systemctl enable --now ssh`) on laptop",
                            "Allow inbound TCP/22 in laptop stateful host firewall and verify with `ss -tlnp | grep :22`",
                        ],
                        "dropped_redundant": [
                            "Duplicate diagnosis that outbound laptop SYN succeeds while inbound desktop SYN fails",
                        ],
                        "provenance": {
                            "parent_a": ["Contributed stateful firewall inbound TCP/22 rule analysis"],
                            "parent_b": ["Contributed OpenSSH server daemon (`sshd`) listener verification"],
                        },
                    }
                ),
                False,
            )

        # 9. Piece 6: Fail judge (falsification attack + appeal with different judge)
        if "fail_judge" in lower or "falsify_branch" in lower:
            is_appeal = "appeal_judge" in lower or "appeal_review=true" in lower
            if is_appeal:
                if "overturn_appeal=true" in lower or "salvageable" in lower:
                    return (
                        json.dumps(
                            {
                                "survived": True,
                                "attack_angle": "appeal_re_evaluation",
                                "objection": "",
                                "rationale": "Appeal judge (Super-Appeal-B) overturned kill: branch hypothesis holds under bounded LAN scope.",
                            }
                        ),
                        False,
                    )
                return (
                    json.dumps(
                        {
                            "survived": False,
                            "attack_angle": "appeal_re_evaluation",
                            "objection": "Contradiction with established LAN reachability remains fatal under independent review.",
                            "rationale": "Appeal judge confirmed fatal contradiction.",
                        }
                    ),
                    False,
                )

            # Determine if the target branch has a fatal flaw / contradiction vs solid proof
            if (
                "force_fail=true" in lower
                or "contradict:" in lower
                or "fail2ban" in lower
                or "proxycommand" in lower
                or "flawed_assumption" in lower
            ):
                return (
                    json.dumps(
                        {
                            "survived": False,
                            "objection": "Falsified: contradicts established main-line observation (laptop->desktop SSH already succeeds without fail2ban/proxy block).",
                            "rationale": "Branch hit the verified line and failed falsification.",
                        }
                    ),
                    False,
                )

            return (
                json.dumps(
                    {
                        "survived": True,
                        "objection": "",
                        "rationale": "Survived falsification attack: host firewall + sshd listener state cleanly explains asymmetric TCP/22 behavior.",
                    }
                ),
                False,
            )

        # 10. Piece 8: Self-improvement loop (self-assessment, anti-model critic, proposition synthesis)
        if "self_assessment" in lower or "post-run summary" in lower:
            return (
                json.dumps(
                    {
                        "what_worked": [
                            "Cross-branch merge of firewall + sshd hypotheses produced highest-certainty winner",
                            "3-round rotated falsification gauntlet filtered out ungrounded NS proxy/fail2ban branches",
                        ],
                        "what_to_improve": [
                            "Increase merges_attracted weight slightly to reward convergent branches earlier",
                            "Tighten similarity_threshold slightly to reduce redundant sibling exploration",
                        ],
                    }
                ),
                False,
            )

        if "anti_model" in lower or "purely negative critic" in lower:
            return (
                json.dumps(
                    {
                        "evidence_grounded_attacks": [
                            {
                                "target": "ns_blind_variation_spend",
                                "evidence_seq": "ns_tick -> fail",
                                "critique": "NS mutations on client ProxyCommand/fail2ban consumed exploration budget before being killed by logic/assumption_audit falsifiers.",
                            },
                            {
                                "target": "unmerged_leaf_branches",
                                "evidence_seq": "branch -> conclude",
                                "critique": "Deep leaf branches that did not attract merges had lower spend-to-certainty efficiency than merged branches.",
                            },
                        ]
                    }
                ),
                False,
            )

        if "synthesize_proposition" in lower or "concrete changes (dials, prompts, thresholds)" in lower:
            return (
                json.dumps(
                    {
                        "summary": "Boost merges_attracted weight (1.5 -> 1.75) and survival weight (3.0 -> 3.25) based on mined merge win rate.",
                        "dial_changes": {
                            "merges_attracted_weight": 1.75,
                            "survived_falsifications_weight": 3.25,
                            "prefilter_threshold": 0.22,
                        },
                        "prompt_changes": {
                            "plaus_probe_hint": "Prioritize host-level service + firewall state checks before exotic client overrides.",
                        },
                    }
                ),
                False,
            )

        # Default reasoning continuation
        fork = _PLAUS_FORK_POOL[(digest + choice_idx) % len(_PLAUS_FORK_POOL)]
        return (
            json.dumps(
                {
                    "fork_id": fork["fork_id"],
                    "step": fork["approach"],
                    "reasoning": fork["reasoning"],
                    "conclusion": fork["conclusion"],
                }
            ),
            temperature >= 0.7,
        )

    def create(
        self,
        messages: Sequence[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 256,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None,
        n: int = 1,
        **kwargs: Any,
    ) -> ChatCompletionResponse:
        with self._count_lock:
            self._call_count += 1
            call_no = self._call_count
        target_model = model or self.default_model
        prompt_text = "\n".join(m.get("content", "") for m in messages)

        choices: List[Choice] = []
        total_out_tokens = 0
        for i in range(max(1, n)):
            content, high_entropy = self._generate_one_content(prompt_text, temperature, i)
            out_tokens = max(16, len(content.split()) * 2)
            total_out_tokens += out_tokens
            lp = self._build_logprobs(high_entropy=high_entropy) if logprobs else None
            choices.append(
                Choice(
                    index=i,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason="stop",
                    logprobs=lp,
                )
            )

        prompt_tokens = max(24, len(prompt_text.split()) * 2)
        # Scale mock token usage so budget accounting is realistic and observable in tests
        cost = estimate_call_cost_usd(prompt_tokens, total_out_tokens, target_model) * self.force_cost_scale

        return ChatCompletionResponse(
            id=f"mockcmpl-{call_no:06d}",
            model=target_model,
            choices=choices,
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=total_out_tokens,
                total_tokens=prompt_tokens + total_out_tokens,
                cost_usd=cost,
            ),
        )


class _MockChatNamespace:
    def __init__(self, completions: _MockChatCompletions):
        self.completions = completions


class MockOpenAIClient:
    """
    Mock backend implementing the exact same OpenAI-compatible interface
    (`client.chat.completions.create(...)`) as `OpenAICompatibleClient`.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        seed: int = 42,
        force_cost_scale: float = 50.0,
    ):
        self.base_url = (base_url or get_base_url()).rstrip("/")
        self.model = model or get_model_name()
        self.is_mock = True
        self.chat = _MockChatNamespace(
            _MockChatCompletions(
                default_model=self.model,
                seed=seed,
                force_cost_scale=force_cost_scale,
            )
        )


def create_client(
    use_mock: Optional[bool] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    seed: int = 42,
    force_cost_scale: float = 50.0,
    max_retries: int = 3,
) -> OpenAICompatibleClient | MockOpenAIClient:
    """
    Factory returning an OpenAI-compatible client.
    - Reads NEBIUS_BASE_URL, NEBIUS_MODEL, and NEBIUS_API_KEY from env by default.
    - If NEBIUS_API_KEY is set (and use_mock is not True), returns OpenAICompatibleClient.
    - Otherwise returns MockOpenAIClient with the exact same interface so the harness
      is testable end-to-end without a key, and plugs into Nebius with zero code changes.
    """
    resolved_key = api_key if api_key is not None else os.environ.get("NEBIUS_API_KEY", "")
    if use_mock is True or (use_mock is None and not resolved_key):
        return MockOpenAIClient(
            base_url=base_url,
            model=model,
            seed=seed,
            force_cost_scale=force_cost_scale,
        )
    return OpenAICompatibleClient(
        base_url=base_url,
        api_key=resolved_key,
        model=model,
        max_retries=max_retries,
    )
