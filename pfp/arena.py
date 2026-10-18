"""
Piece 7: Arena Feed & Tron-Style Renderer (Pass 1: literal spec).

From DESIGN.md:
- The Arena (visualization + product UI):
  Tron-like light-cycle aesthetic. Agents drop from the top, branch in different
  directions, keep moving and splitting. A branch fails by hitting the main
  line (the verified trunk) — contradiction with established truth kills it,
  Tron rules. Survivors keep falling until they exit the bottom of the box into
  a winner's box presenting the solution.
  Colors are cosmetic only (agent distinction; can all be white).
  Scoreboard HUD: tracks every agent — time alive + state. Battle-royale
  spectator feel.
- Piece 7: Arena feed — the single event stream:
  Every component emits events: spawn, branch, merge, fail, pass, revocation,
  NS tick, budget change. The feed is append-only — history never rewrites
  (a revoked Pass is itself an event). Every event carries a timestamp and
  the branch's full state snapshot.
  The Tron visualization, the scoreboard, and the winner's box are dumb
  renderers on this stream. Record once, render anywhere: live UI, demo video,
  post-run replay — same data.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .arena_html import build_arena_html
from .fsutil import atomic_write_text as _atomic_write_text
from .branch_spawner import HarnessEvent


def _event_digest(
    *,
    seq: int,
    event_type: str,
    branch_id: str,
    timestamp: float,
    details: Dict[str, Any],
    snapshot: Dict[str, Any],
    previous_digest: str,
    journal_version: int = 2,
) -> str:
    """Hash the complete event and its predecessor using deterministic JSON."""
    payload = {
        "seq": seq,
        "event_type": event_type,
        "branch_id": branch_id,
        "timestamp": timestamp,
        "details": details,
        "snapshot": snapshot,
        "previous_digest": previous_digest,
        "journal_version": journal_version,
    }
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ArenaEvent payload must be finite, JSON-serializable data: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


CANONICAL_EVENT_TYPES: Tuple[str, ...] = (
    "spawn",
    "branch",
    "merge",
    "fail",
    "pass",
    "revocation",
    "ns_tick",
    "budget_change",
    "falsification_survived",
    "degrade",
    "conclude",
    "winner_exit",
    "kill",
    "falsification_inconclusive",
    "halt",
)

# Events about the arena as a whole (no branch): renderers and miners must not treat them as agents
ARENA_LEVEL_EVENT_TYPES: Tuple[str, ...] = ("halt",)

# Longest problem statement a starter spawn event carries (the replay title, not a dump).
MAX_STREAM_PROBLEM_CHARS = 4000

_HEX_DIGITS = frozenset("0123456789abcdef")


def is_journal_digest(value: Any) -> bool:
    """True for a well-formed journal digest: 64 lowercase hex characters (SHA-256)."""
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX_DIGITS


def journal_anchor(events: Iterable[Any]) -> Dict[str, Any]:
    """
    The journal's anchor: event count + head digest (the last event's chained SHA-256).

    The hash chain proves every event is consistent with its predecessors, but a copy cut
    short at the tail is still a valid chain: drop a trailing `revocation` and a revoked
    winner is back in the box. Kept apart from the JSONL (run summary, proof report,
    a chat message), the anchor detects that: `--verify --expect-head` / `--replay
    --expect-head` refuse a journal whose head differs. It is a pure function of the
    stream, so a live run and its replay report the same anchor.
    """
    count = 0
    head = ""
    for e in events:
        count += 1
        raw = e.to_dict() if isinstance(e, ArenaEvent) else e
        head = str(raw.get("digest", "")) if isinstance(raw, dict) else ""
    return {"events": count, "head_digest": head}


def stream_problem(events: Iterable[Any]) -> str:
    """
    The problem statement recorded on the stream by the first starter spawn, or "".
    A replay recovers the run's title from the hashed journal itself, so a re-render
    and its proof report match the live run without the operator retyping --problem.
    """
    for e in events:
        raw = e.to_dict() if isinstance(e, ArenaEvent) else e
        if not isinstance(raw, dict) or raw.get("event_type") != "spawn":
            continue
        details = raw.get("details") or {}
        problem = details.get("problem") if isinstance(details, dict) else None
        if isinstance(problem, str) and problem.strip():
            return problem[:MAX_STREAM_PROBLEM_CHARS]
    return ""


@dataclass(frozen=True)
class ArenaEvent:
    """
    Immutable event entry in the append-only ArenaFeed stream.
    Carries monotonic sequence ID, timestamp, branch_id, event_type, details,
    and the branch's full state snapshot at the moment of emission.
    """

    seq: int
    event_type: str
    branch_id: str
    timestamp: float
    details: Dict[str, Any]
    snapshot: Dict[str, Any]
    previous_digest: str = ""
    digest: str = ""
    journal_version: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "event_type": self.event_type,
            "branch_id": self.branch_id,
            "timestamp": round(self.timestamp, 4),
            "details": copy.deepcopy(self.details),
            "snapshot": copy.deepcopy(self.snapshot),
            "previous_digest": self.previous_digest,
            "digest": self.digest,
            "journal_version": self.journal_version,
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ArenaEvent":
        return cls(
            seq=int(raw.get("seq", 0)),
            event_type=str(raw["event_type"]),
            branch_id=str(raw["branch_id"]),
            timestamp=float(raw["timestamp"]),
            details=dict(raw.get("details", {})),
            snapshot=dict(raw.get("snapshot", {})),
            previous_digest=str(raw.get("previous_digest", "")),
            digest=str(raw.get("digest", "")),
            journal_version=int(raw.get("journal_version", 1)),
        )

    def detached_copy(self) -> "ArenaEvent":
        """Return a deep copy so nested payloads cannot mutate journal history."""
        return ArenaEvent.from_dict(self.to_dict())


class AppendOnlyViolationError(RuntimeError):
    """Raised if caller attempts to mutate, delete, or overwrite history in ArenaFeed."""


class _AppendOnlyEventList(list):
    """List view that forbids in-place mutation or deletion so history never rewrites."""

    def __setitem__(self, index: Any, value: Any) -> None:
        raise AppendOnlyViolationError("ArenaFeed is strictly append-only; history never rewrites.")

    def __delitem__(self, index: Any) -> None:
        raise AppendOnlyViolationError("ArenaFeed is strictly append-only; events cannot be deleted.")

    def pop(self, index: int = -1) -> Any:
        raise AppendOnlyViolationError("ArenaFeed is strictly append-only; cannot pop events.")

    def clear(self) -> None:
        raise AppendOnlyViolationError("ArenaFeed is strictly append-only; cannot clear history.")

    def remove(self, value: Any) -> None:
        raise AppendOnlyViolationError("ArenaFeed is strictly append-only; cannot remove events.")


class ArenaFeed:
    """
    Piece 7: Append-only event stream.
    - Every spawn, branch, merge, fail, pass, revocation, NS tick, and budget change
      is appended with a timestamp and full branch snapshot.
    - History never rewrites (a revoked Pass is itself an appended `"revocation"` event).
    - Supports JSONL serialization (`to_jsonl`, `save_jsonl`, `load_jsonl`) and live
      subscriber callbacks so live UI, post-run replay, and demo video all render
      from the exact same stream.
    """

    def __init__(self) -> None:
        self._events: _AppendOnlyEventList = _AppendOnlyEventList()
        self._seq: int = 0
        self._lock = threading.RLock()
        self._subscribers: List[Callable[[ArenaEvent], None]] = []
        # True when loaded from a legacy (v1, unchained) journal: its events were hashed on
        # load, so the chain proves nothing about what was on disk.
        self.legacy_unverified: bool = False

    def subscribe(self, callback: Callable[[ArenaEvent], None]) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def append(
        self,
        event_type: str,
        branch_id: str,
        snapshot: Dict[str, Any],
        details: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> ArenaEvent:
        if not snapshot or "id" not in snapshot or "state" not in snapshot:
            raise ValueError("Every ArenaEvent must carry a valid branch state snapshot ('id', 'state').")
        if str(snapshot["id"]) != branch_id:
            raise ValueError("ArenaEvent branch_id must match its snapshot id.")
        if not event_type:
            raise ValueError("ArenaEvent event_type must be non-empty.")
        with self._lock:
            self._seq += 1
            # Persist the exact value that is hashed; JSONL intentionally uses four decimal
            # places for stable, compact replay files.
            event_timestamp = round(timestamp if timestamp is not None else time.time(), 4)
            if not math.isfinite(event_timestamp):
                self._seq -= 1
                raise ValueError("ArenaEvent timestamp must be finite.")
            event_details = copy.deepcopy(details or {})
            event_snapshot = copy.deepcopy(snapshot)
            previous_digest = self._events[-1].digest if self._events else ""
            try:
                digest = _event_digest(
                    seq=self._seq,
                    event_type=event_type,
                    branch_id=branch_id,
                    timestamp=event_timestamp,
                    details=event_details,
                    snapshot=event_snapshot,
                    previous_digest=previous_digest,
                )
            except (TypeError, ValueError):
                self._seq -= 1
                raise
            evt = ArenaEvent(
                seq=self._seq,
                event_type=event_type,
                branch_id=branch_id,
                timestamp=event_timestamp,
                details=event_details,
                snapshot=event_snapshot,
                previous_digest=previous_digest,
                digest=digest,
            )
            list.append(self._events, evt)
            subs = list(self._subscribers)
        for sub in subs:
            sub(evt.detached_copy())
        return evt.detached_copy()

    def ingest_harness_event(self, event: HarnessEvent) -> ArenaEvent:
        return self.append(
            event_type=event.event_type,
            branch_id=event.branch_id,
            snapshot=event.snapshot,
            details=event.details,
            timestamp=event.timestamp,
        )

    @property
    def events(self) -> Sequence[ArenaEvent]:
        with self._lock:
            return _AppendOnlyEventList(event.detached_copy() for event in self._events)

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    @property
    def head_digest(self) -> str:
        """Digest of the newest event ("" for an empty feed): anchors the whole chain."""
        with self._lock:
            return self._events[-1].digest if self._events else ""

    def anchor(self) -> Dict[str, Any]:
        """`journal_anchor` of this feed plus whether its chain was verified on disk."""
        with self._lock:
            out = journal_anchor(self._events)
            out["verified_chain"] = bool(self._events) and not self.legacy_unverified
            return out

    def to_jsonl(self) -> str:
        with self._lock:
            return "\n".join(json.dumps(e.to_dict(), allow_nan=False) for e in self._events)

    def save_jsonl(self, path: str | Path) -> Path:
        return _atomic_write_text(Path(path), self.to_jsonl() + "\n")

    @classmethod
    def from_jsonl(cls, jsonl_text: str, *, allow_legacy: bool = True) -> "ArenaFeed":
        """
        Parse and verify a JSONL journal. A legacy v1 journal (no hash chain) is accepted
        for old recordings unless `allow_legacy=False`, and marks the feed
        `legacy_unverified`: anyone can strip the digests from a chained journal, edit it
        and relabel it v1, so its content is unauthenticated.
        """
        feed = cls()
        expected_seq = 1
        previous_digest = ""
        chain_mode: Optional[bool] = None
        for line_no, line in enumerate(jsonl_text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
                if not isinstance(raw, dict):
                    raise TypeError("record is not a JSON object")
                evt = ArenaEvent.from_dict(raw)
            except (ValueError, KeyError, TypeError) as exc:
                # A truncated or hand-edited log must fail with a pointer, not a bare KeyError
                raise ValueError(f"Malformed arena event on line {line_no}: {exc!r}") from exc
            with feed._lock:
                if evt.seq != expected_seq:
                    raise AppendOnlyViolationError(
                        f"Arena replay sequence must be contiguous: expected {expected_seq}, got {evt.seq}."
                    )
                if not math.isfinite(evt.timestamp):
                    raise ValueError(f"Arena event {evt.seq} has a non-finite timestamp.")
                try:
                    json.dumps(
                        {"details": evt.details, "snapshot": evt.snapshot},
                        allow_nan=False,
                        sort_keys=True,
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Arena event {evt.seq} has a non-finite or non-JSON payload: {exc}"
                    ) from exc
                if not evt.snapshot or "id" not in evt.snapshot or "state" not in evt.snapshot:
                    raise ValueError(
                        f"Arena event {evt.seq} lacks the required full branch snapshot ('id', 'state')."
                    )
                if str(evt.snapshot["id"]) != evt.branch_id:
                    raise ValueError(
                        f"Arena event {evt.seq} branch_id does not match its snapshot id."
                    )
                has_chain = evt.journal_version >= 2
                if has_chain and (len(evt.digest) != 64 or (evt.seq > 1 and len(evt.previous_digest) != 64)):
                    raise AppendOnlyViolationError(
                        f"Arena replay event {evt.seq} is missing its required hash chain."
                    )
                if chain_mode is None:
                    chain_mode = has_chain
                    if not has_chain and not allow_legacy:
                        raise AppendOnlyViolationError(
                            "Arena journal is legacy v1 (no hash chain): its content cannot be verified."
                        )
                    feed.legacy_unverified = not has_chain
                elif has_chain != chain_mode:
                    raise AppendOnlyViolationError("Arena replay mixes chained and legacy events.")
                if has_chain:
                    expected_digest = _event_digest(
                        seq=evt.seq,
                        event_type=evt.event_type,
                        branch_id=evt.branch_id,
                        timestamp=evt.timestamp,
                        details=evt.details,
                        snapshot=evt.snapshot,
                        previous_digest=previous_digest,
                        journal_version=evt.journal_version,
                    )
                    if evt.previous_digest != previous_digest or evt.digest != expected_digest:
                        raise AppendOnlyViolationError(
                            f"Arena replay hash chain is invalid at event {evt.seq}."
                        )
                    previous_digest = evt.digest
                else:
                    # Read v1 logs for compatibility, but upgrade them in memory so a
                    # subsequent save/append cannot create a mixed-trust journal.
                    digest = _event_digest(
                        seq=evt.seq,
                        event_type=evt.event_type,
                        branch_id=evt.branch_id,
                        timestamp=evt.timestamp,
                        details=evt.details,
                        snapshot=evt.snapshot,
                        previous_digest=previous_digest,
                    )
                    evt = ArenaEvent(
                        seq=evt.seq,
                        event_type=evt.event_type,
                        branch_id=evt.branch_id,
                        timestamp=evt.timestamp,
                        details=evt.details,
                        snapshot=evt.snapshot,
                        previous_digest=previous_digest,
                        digest=digest,
                    )
                    previous_digest = digest
                feed._seq = evt.seq
                list.append(feed._events, evt)
                expected_seq += 1
        return feed

    @classmethod
    def load_jsonl(cls, path: str | Path, *, allow_legacy: bool = True) -> "ArenaFeed":
        return cls.from_jsonl(Path(path).read_text(encoding="utf-8"), allow_legacy=allow_legacy)


class TronArenaRenderer:
    """
    Stateless / stream-driven Tron-style Arena Renderer on `ArenaFeed` (Piece 7).
    Reconstructs the entire arena state purely from the event stream (dumb renderer):
    - Top ignition bar: starter agents dropping from the top.
    - VerticalVerified Trunk (the "main line" down the center): branches that contradict
      established truth crash into the verified line (`[X] CRASH -> FAIL`), Tron rules.
    - Branching light-cycle trails (`|`, `/`, `\\`), merges (`==> [MERGE] <==`),
      revocations (`[!] REVOKED`), and survivors falling into the bottom **Winner's Box**.
    - Battle-royale **Scoreboard HUD** tracking every agent's `state`, `time_alive`,
      `certainty_score`, and `budget_remaining`.
    - Also exports a standalone interactive HTML/Canvas Tron light-cycle arena player
      embedding the exact event stream for browser replay / demo video recording.
    """

    def __init__(self, width: int = 88) -> None:
        self.width = max(72, width)

    @staticmethod
    def reconstruct_state_from_stream(
        events: Iterable[ArenaEvent | Dict[str, Any]],
        upto_seq: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Pure reduction over the event stream: reconstructs branches, lineages,
        merges, crashes against the verified line, revocations, scoreboard HUD,
        and the winner's box without touching any external mutable tracker.
        """
        ev_list: List[ArenaEvent] = []
        for e in events:
            evt = e if isinstance(e, ArenaEvent) else ArenaEvent.from_dict(e)
            if upto_seq is not None and evt.seq > upto_seq:
                break
            ev_list.append(evt)

        branches: Dict[str, Dict[str, Any]] = {}
        starters: List[str] = []
        splits: List[Dict[str, Any]] = []
        merges: List[Dict[str, Any]] = []
        crashes: List[Dict[str, Any]] = []
        passes: List[Dict[str, Any]] = []
        revocations: List[Dict[str, Any]] = []
        winners_box: List[Dict[str, Any]] = []
        ns_ticks: int = 0
        budget_changes: int = 0
        halted: Optional[Dict[str, Any]] = None

        for evt in ev_list:
            if evt.event_type in ARENA_LEVEL_EVENT_TYPES:
                if evt.event_type == "halt":
                    halted = {"seq": evt.seq, **copy.deepcopy(evt.details)}
                continue
            snap = copy.deepcopy(evt.snapshot)
            bid = evt.branch_id
            branches[bid] = snap

            if evt.event_type == "spawn":
                if snap.get("parent") is None and not snap.get("merge_parents"):
                    if bid not in starters:
                        starters.append(bid)
            elif evt.event_type == "branch":
                splits.append(
                    {
                        "seq": evt.seq,
                        "parent_id": bid,
                        "trigger": evt.details.get("trigger", "plaus"),
                        "children_ids": evt.details.get("children_ids", []),
                        "forks": evt.details.get("forks", []),
                    }
                )
            elif evt.event_type == "merge":
                merges.append(
                    {
                        "seq": evt.seq,
                        "child_id": bid,
                        "parent_a_id": evt.details.get("parent_a_id", ""),
                        "parent_b_id": evt.details.get("parent_b_id", ""),
                        "transferred_budget_usd": evt.details.get("transferred_budget_usd", 0.0),
                        "unified_fork": evt.details.get("unified_fork", snap.get("fork_under_exploration", "")),
                    }
                )
            elif evt.event_type == "fail":
                crashes.append(
                    {
                        "seq": evt.seq,
                        "branch_id": bid,
                        "angle": evt.details.get("angle", ""),
                        "schedule_trigger": evt.details.get("schedule_trigger", ""),
                        "objection": evt.details.get("objection", "Hit the verified main line"),
                    }
                )
                winners_box = [w for w in winners_box if w.get("id") != bid]
            elif evt.event_type == "pass":
                passes.append(
                    {
                        "seq": evt.seq,
                        "branch_id": bid,
                        "angles_survived": evt.details.get("angles_survived", []),
                    }
                )
            elif evt.event_type == "revocation":
                revocations.append(
                    {
                        "seq": evt.seq,
                        "branch_id": bid,
                        "demoted_to": evt.details.get("demoted_to", "Possible"),
                        "reason": evt.details.get("reason", ""),
                    }
                )
                # Remove revoked branch from winners_box if present
                winners_box = [w for w in winners_box if w.get("id") != bid]
            elif evt.event_type == "winner_exit":
                # A winner is a proof claim, not merely a UI placement. Never let a
                # malformed/custom stream certify Possible/Fail/Absorbed state.
                if snap.get("state") == "Pass" and not any(w.get("id") == bid for w in winners_box):
                    winners_box.append(snap)
            elif evt.event_type == "kill":
                winners_box = [w for w in winners_box if w.get("id") != bid]
            elif evt.event_type == "ns_tick":
                ns_ticks += 1
            elif evt.event_type == "budget_change":
                budget_changes += 1

        # Final state is authoritative. This also handles custom event types carrying a
        # later non-Pass snapshot without relying on a specific eviction event name.
        winners_box = [
            w for w in winners_box
            if branches.get(str(w.get("id", "")), {}).get("state") == "Pass"
        ]

        # Build Scoreboard HUD ordered by state priority (Pass > Possible > Absorbed > Fail) and certainty
        state_rank = {"Pass": 3, "Possible": 2, "Absorbed": 1, "Fail": 0}
        scoreboard = sorted(
            branches.values(),
            key=lambda r: (
                state_rank.get(str(r.get("state", "Possible")), 0),
                float(r.get("certainty_score", 0.0)),
                float(r.get("time_alive", 0.0)),
            ),
            reverse=True,
        )

        # Fallback: if no explicit winner_exit event yet, any Pass branch at the end of the run is a survivor
        if not winners_box:
            winners_box = [copy.deepcopy(r) for r in scoreboard if r.get("state") == "Pass"]

        return {
            "events_processed": len(ev_list),
            "starters": starters,
            "branches": branches,
            "splits": splits,
            "merges": merges,
            "crashes": crashes,
            "passes": passes,
            "revocations": revocations,
            "ns_ticks": ns_ticks,
            "budget_changes": budget_changes,
            "scoreboard": scoreboard,
            "winners_box": winners_box,
            "halted": halted,
        }

    @staticmethod
    def _starter_lanes(count: int, w: int, mid_x: int) -> List[int]:
        """Evenly spaced drop lanes, split across both sides of the verified main line."""
        if count <= 0:
            return [max(4, mid_x - 20)]
        left_n = (count + 1) // 2
        right_n = count - left_n
        lanes: List[int] = []
        for k in range(left_n):
            lanes.append(int(4 + (k + 1) * (mid_x - 8) / (left_n + 1)))
        for k in range(right_n):
            lanes.append(int(mid_x + 4 + (k + 1) * (w - mid_x - 8) / (right_n + 1)))
        return [lx if lx != mid_x else lx + 2 for lx in lanes]

    def render_spatial_grid(
        self,
        state: Dict[str, Any],
        height: int = 16,
    ) -> List[str]:
        """
        Pass 2 refinement: 2D spatial Tron light-cycle arena grid (`width x height`).
        - Center column (`mid_x`) is the glowing Verified Main Line (`║`).
        - Starter agents drop from row 0 at distinct horizontal lanes (`▼`), leaving
          vertical light-cycle trails (`│`).
        - Plaus / NS splits fan out horizontally (`╭──┴──╮`) into child lanes.
        - Failed branches veer into the center Verified Main Line (`═══X╣`) and die (`✖`).
        - Merged branches converge inward (`╰──>[M]<──╯`) into a unified child lane.
        - Passing survivors ride the verified trunk down to the bottom exit (`╚══> [WINNER'S BOX]`).
        """
        w = self.width - 4
        h = max(12, height)
        mid_x = w // 2
        grid = [[" " for _ in range(w)] for _ in range(h)]

        # Draw center Verified Main Line (trunk)
        for r in range(h):
            grid[r][mid_x] = "║"

        starters = state["starters"]
        lane_x: Dict[str, int] = {}
        starter_lanes = self._starter_lanes(len(starters), w, mid_x)
        for idx, sid in enumerate(starters):
            lx = starter_lanes[idx % len(starter_lanes)]
            if lx == mid_x:
                lx += 6
            lane_x[sid] = lx
            if 0 <= lx < w:
                grid[0][lx] = "▼"
                for r in range(1, min(4, h)):
                    grid[r][lx] = "│"

        # Assign lanes for children from splits
        row_cursor = 3
        for sp in state["splits"]:
            pid = sp["parent_id"]
            px = lane_x.get(pid, max(6, mid_x - 20))
            kids = sp["children_ids"]
            if row_cursor < h - 3:
                for idx, cid in enumerate(kids):
                    offset = (idx - (len(kids) // 2)) * 5
                    if offset == 0:
                        offset = 4
                    cx = max(2, min(w - 3, px + offset))
                    if cx == mid_x:
                        cx = min(w - 3, cx + 3)
                    lane_x[cid] = cx
                    lo, hi = min(px, cx), max(px, cx)
                    for x in range(lo, hi + 1):
                        if x != mid_x and grid[row_cursor][x] == " ":
                            grid[row_cursor][x] = "─"
                    grid[row_cursor][cx] = "╮" if cx > px else "╭"
                    if row_cursor + 1 < h - 2:
                        grid[row_cursor + 1][cx] = "│"
                row_cursor = min(h - 4, row_cursor + 2)

        # Draw crashes into the center Verified Main Line (Tron rules: hit main line -> FAIL)
        crash_row = min(h - 5, max(4, row_cursor))
        for cr in state["crashes"][:4]:
            bid = cr["branch_id"]
            bx = lane_x.get(bid, max(4, mid_x - 16))
            if crash_row < h - 2:
                lo, hi = min(bx, mid_x), max(bx, mid_x)
                for x in range(lo, hi):
                    if x != mid_x:
                        grid[crash_row][x] = "═"
                collide_x = mid_x - 1 if bx < mid_x else min(w - 2, mid_x + 1)
                grid[crash_row][collide_x] = "X"
                crash_row = min(h - 3, crash_row + 1)

        # Draw merges converging toward the bottom-center survivor path
        merge_row = min(h - 3, max(6, crash_row))
        for mg in state["merges"][:3]:
            pa, pb, cid = mg["parent_a_id"], mg["parent_b_id"], mg["child_id"]
            xa = lane_x.get(pa, max(6, mid_x - 18))
            xb = lane_x.get(pb, min(w - 6, mid_x + 18))
            xm = min(w - 4, mid_x + 4)
            lane_x[cid] = xm
            lo, hi = min(xa, xb, xm), max(xa, xb, xm)
            for x in range(lo, hi + 1):
                if x != mid_x:
                    grid[merge_row][x] = "═"
            if 0 <= xm < w:
                grid[merge_row][xm] = "╬"
                for r in range(merge_row + 1, h):
                    grid[r][xm] = "║"

        # Bottom exit arrow on Verified Main Line
        grid[h - 1][mid_x] = "▼"

        rendered_rows = [" ╭" + ("─" * w) + "╮"]
        for r_idx, row in enumerate(grid):
            tag = "TOP" if r_idx == 0 else ("BOT" if r_idx == h - 1 else "   ")
            rendered_rows.append(f" │{''.join(row)}│ {tag}")
        rendered_rows.append(" ╰" + ("─" * w) + "╯")
        return rendered_rows

    def render_ascii(
        self,
        events: Iterable[ArenaEvent | Dict[str, Any]],
        problem: str = "",
        upto_seq: Optional[int] = None,
    ) -> str:
        """
        Render the Tron-style light-cycle arena, 2D spatial grid, verified main line,
        merge trails, Scoreboard HUD (per-agent state + time alive), and Winner's Box
        from the stream.
        """
        state = self.reconstruct_state_from_stream(events, upto_seq=upto_seq)
        w = self.width
        border = "=" * w
        thin = "-" * w
        lines: List[str] = []

        # 1. Header & Ignition Bar (Agents drop from top)
        lines.append(border)
        lines.append(" TRON ARENA // PFP DARWINISM — \"Nebius — survival of the fittest.\"".center(w))
        if problem:
            lines.append(f" PROBLEM: {problem[:w - 11]}".ljust(w))
        lines.append(border)

        starters = state["starters"]
        if not starters:
            lines.append(" IGNITION (TOP DROP): [WAITING FOR IGNITION]")
        elif len(starters) <= 3:
            starter_cells = "   |   ".join(f"[{sid}: DROP]" for sid in starters)
            lines.append(" IGNITION (TOP DROP): " + starter_cells)
            lines.append("                      " + "   |   ".join("    |     " for _ in starters))
            lines.append("                      " + "   v   ".join("    v     " for _ in starters))
        else:
            # Real scale (9 agents): wrap the drop list inside the arena width
            lines.append(f" IGNITION (TOP DROP): {len(starters)} agents")
            row = "  "
            for sid in starters:
                cell = f"[{sid} ▼] "
                if len(row) + len(cell) > w:
                    lines.append(row.rstrip())
                    row = "  "
                row += cell
            lines.append(row.rstrip())
        lines.append(thin)

        # 2. 2D Spatial Tron Light-Cycle Grid + Event Callouts
        lines.append(" 2D TRON LIGHT-CYCLE GRID (▼=DROP, │/─=TRAIL, ║=VERIFIED MAIN LINE, X=CRASH, ╬=MERGE):".ljust(w))
        lines.extend(self.render_spatial_grid(state, height=12))
        lines.append(thin)

        lines.append(" ARENA LIGHT-CYCLE FLOW & VERIFIED MAIN LINE (║ = VERIFIED TRUNK):".ljust(w))
        lines.append("   [TOP] ───────────────╔═══ VERIFIED MAIN LINE ═══╗─────────────── [TOP]")

        for sp in state["splits"]:
            trig = str(sp["trigger"]).upper()
            kids = ", ".join(sp["children_ids"])
            lines.append(f"     │   ({sp['parent_id']}) ──[{trig} SPLIT]──> / {kids} \\")

        for cr in state["crashes"]:
            bid = cr["branch_id"]
            ang = cr["angle"] or "falsification"
            lines.append(f"     ├───> ({bid}) ═══X╣ ║ CRASH INTO MAIN LINE [{ang}] -> FAIL (DEAD)")

        for mg in state["merges"]:
            pa, pb, cid = mg["parent_a_id"], mg["parent_b_id"], mg["child_id"]
            bud = float(mg["transferred_budget_usd"])
            lines.append(f"     ╰───> ({pa}) + ({pb}) ══>[MERGE +${bud:.4f}]<══> ({cid}) [POSSIBLE]")

        for rv in state["revocations"]:
            lines.append(
                f"     [!] PASS REVOKED on ({rv['branch_id']}) -> {rv['demoted_to']} ({rv['reason']}) | MAIN LINE REWRITTEN"
            )

        for ps in state["passes"]:
            angs = ",".join(ps["angles_survived"])
            lines.append(f"     ║   ({ps['branch_id']}) SURVIVED GAUNTLET [{angs}] ══> PASS (VERIFIED TRUNK)")

        lines.append("   [EXIT] ──────────────╚═══>  WINNER'S BOX  <═══╝─────────────── [EXIT]")
        lines.append(thin)

        # 3. Scoreboard HUD: tracks every agent — state + time alive + certainty + budget
        lines.append(" SCOREBOARD HUD (BATTLE-ROYALE SPECTATOR FEED — ALL AGENTS):".ljust(w))
        lines.append(
            f" {'AGENT ID':<14} | {'STATE':<10} | {'TIME ALIVE':<12} | {'CERTAINTY':<10} | {'BUDGET':<10} | {'FORK'}"
        )
        lines.append(" " + ("-" * (w - 2)))
        for row in state["scoreboard"]:
            bid = str(row.get("id", ""))[:14]
            st = str(row.get("state", "Possible"))[:10]
            ta = f"{float(row.get('time_alive', 0.0)):.4f}s"
            cert = f"{float(row.get('certainty_score', 0.0)):+.2f}"
            bud = f"${float(row.get('budget_remaining', 0.0)):.4f}"
            fork = str(row.get("fork_under_exploration", ""))[: max(12, w - 70)]
            lines.append(f" {bid:<14} | {st:<10} | {ta:<12} | {cert:<10} | {bud:<10} | {fork}")
        lines.append(thin)

        # 4. Winner's Box (Bottom Exit)
        lines.append(" ╔" + ("═" * (w - 4)) + "╗")
        lines.append(" ║" + " WINNER'S BOX — SURVIVOR SOLUTION & PROOF TRAIL ".center(w - 4) + "║")
        lines.append(" ╠" + ("═" * (w - 4)) + "╣")
        winners = state["winners_box"]
        if not winners:
            lines.append(" ║" + " No verified survivor has exited into the Winner's Box yet. ".ljust(w - 4) + "║")
        else:
            for idx, win in enumerate(winners):
                wid = win.get("id", "unknown")
                wcert = float(win.get("certainty_score", 0.0))
                wsurv = int(win.get("survived_falsifications", 0))
                wfork = str(win.get("fork_under_exploration", ""))
                mparents = win.get("merge_parents", [])
                prov_angles = [p.get("falsifier_angle", "") for p in win.get("pass_provenance", [])]
                header_line = (
                    f" #{idx + 1} WINNER: {wid} | State: {win.get('state')} | "
                    f"Certainty: {wcert:+.2f} | Survived: {wsurv} ({','.join(prov_angles)})"
                )
                lines.append(" ║" + header_line[: w - 4].ljust(w - 4) + "║")
                if mparents:
                    coop_line = f"    Cooperation lineage (merged parents): {', '.join(mparents)}"
                    lines.append(" ║" + coop_line[: w - 4].ljust(w - 4) + "║")
                solution = " ".join(str(win.get("solution") or wfork).split())
                prefix = "    Verified Solution: "
                body_w = max(20, w - 4 - len(prefix))
                chunks = [solution[i:i + body_w] for i in range(0, min(len(solution), body_w * 3), body_w)] or [""]
                for c_idx, chunk in enumerate(chunks):
                    lead = prefix if c_idx == 0 else " " * len(prefix)
                    lines.append(" ║" + (lead + chunk)[: w - 4].ljust(w - 4) + "║")
        lines.append(" ╚" + ("═" * (w - 4)) + "╝")
        if state.get("halted"):
            h = state["halted"]
            lines.append(f" !! RUN HALTED ({h.get('reason', '')}): {h.get('detail', '')}"[:w])
        lines.append(border)
        return "\n".join(lines)

    def render_replay_frames(
        self,
        events: Sequence[ArenaEvent | Dict[str, Any]],
        problem: str = "",
        step_size: int = 5,
    ) -> List[str]:
        """
        Replay the append-only event stream step-by-step from the exact same data.
        """
        ev_list = [e if isinstance(e, ArenaEvent) else ArenaEvent.from_dict(e) for e in events]
        if not ev_list:
            return [self.render_ascii([], problem=problem)]
        frames: List[str] = []
        for idx in range(step_size, len(ev_list) + step_size, max(1, step_size)):
            cutoff = ev_list[min(idx - 1, len(ev_list) - 1)].seq
            frames.append(self.render_ascii(ev_list, problem=problem, upto_seq=cutoff))
        return frames

    def export_html_arena(
        self,
        events: Iterable[ArenaEvent | Dict[str, Any]],
        output_path: str | Path,
        problem: str = "",
    ) -> Path:
        """
        Export a self-contained Tron light-cycle HTML5/Canvas arena player and live HUD
        driven directly by the embedded JSON event stream (`events`), featuring animated
        playback controls (Play/Pause/Scrub) so live UI, replay, and demo video all render
        from the exact same stream.
        """
        ev_dicts = [
            (e.to_dict() if isinstance(e, ArenaEvent) else dict(e))
            for e in events
        ]
        problem = problem or stream_problem(ev_dicts)
        ascii_snapshot = self.render_ascii(ev_dicts, problem=problem)
        html = build_arena_html(ev_dicts, problem=problem, ascii_snapshot=ascii_snapshot)
        return _atomic_write_text(Path(output_path), html)
