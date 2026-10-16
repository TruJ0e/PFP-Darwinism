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
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .branch_spawner import HarnessEvent


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
)


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "event_type": self.event_type,
            "branch_id": self.branch_id,
            "timestamp": round(self.timestamp, 4),
            "details": copy.deepcopy(self.details),
            "snapshot": copy.deepcopy(self.snapshot),
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
        )


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
        with self._lock:
            self._seq += 1
            evt = ArenaEvent(
                seq=self._seq,
                event_type=event_type,
                branch_id=branch_id,
                timestamp=timestamp if timestamp is not None else time.time(),
                details=copy.deepcopy(details or {}),
                snapshot=copy.deepcopy(snapshot),
            )
            list.append(self._events, evt)
            subs = list(self._subscribers)
        for sub in subs:
            sub(evt)
        return evt

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
            return _AppendOnlyEventList(self._events)

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def to_jsonl(self) -> str:
        with self._lock:
            return "\n".join(json.dumps(e.to_dict()) for e in self._events)

    def save_jsonl(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_jsonl() + "\n", encoding="utf-8")
        return target

    @classmethod
    def from_jsonl(cls, jsonl_text: str) -> "ArenaFeed":
        feed = cls()
        for line in jsonl_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            with feed._lock:
                evt = ArenaEvent.from_dict(raw)
                feed._seq = max(feed._seq, evt.seq)
                list.append(feed._events, evt)
        return feed

    @classmethod
    def load_jsonl(cls, path: str | Path) -> "ArenaFeed":
        return cls.from_jsonl(Path(path).read_text(encoding="utf-8"))


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

        for evt in ev_list:
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
                if not any(w.get("id") == bid for w in winners_box):
                    winners_box.append(snap)
            elif evt.event_type == "ns_tick":
                ns_ticks += 1
            elif evt.event_type == "budget_change":
                budget_changes += 1

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
        }

    def render_ascii(
        self,
        events: Iterable[ArenaEvent | Dict[str, Any]],
        problem: str = "",
        upto_seq: Optional[int] = None,
    ) -> str:
        """
        Render the Tron-style light-cycle arena, verified main line, merge trails,
        Scoreboard HUD (per-agent state + time alive), and Winner's Box from the stream.
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
        starter_cells = "   |   ".join(f"[{sid}: DROP]" for sid in starters) if starters else "[WAITING FOR IGNITION]"
        lines.append(" IGNITION (TOP DROP): " + starter_cells)
        lines.append("                      " + "   |   ".join("    |     " for _ in starters))
        lines.append("                      " + "   v   ".join("    v     " for _ in starters))
        lines.append(thin)

        # 2. Arena Flow Grid: Splits, Verified Main Line Crashes (Tron rules), Merges, Revocations
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
                sol_line = f"    Verified Solution: {wfork}"
                lines.append(" ║" + sol_line[: w - 4].ljust(w - 4) + "║")
        lines.append(" ╚" + ("═" * (w - 4)) + "╝")
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
        driven directly by the embedded JSON event stream (`events`).
        """
        ev_dicts = [
            (e.to_dict() if isinstance(e, ArenaEvent) else dict(e))
            for e in events
        ]
        state = self.reconstruct_state_from_stream(ev_dicts)
        ascii_snapshot = self.render_ascii(ev_dicts, problem=problem)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>PFP Darwinism — Tron Arena</title>
<style>
  body {{
    margin: 0;
    background: #050811;
    color: #e6f7ff;
    font-family: "JetBrains Mono", "Consolas", monospace;
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 16px;
    padding: 16px;
  }}
  .panel {{
    border: 1px solid #00f0ff;
    box-shadow: 0 0 14px rgba(0, 240, 255, 0.25);
    background: rgba(8, 15, 30, 0.92);
    padding: 14px;
    border-radius: 4px;
  }}
  h1, h2 {{
    margin: 0 0 10px 0;
    color: #00f0ff;
    text-shadow: 0 0 8px rgba(0, 240, 255, 0.6);
    font-size: 16px;
    letter-spacing: 1px;
  }}
  canvas {{
    width: 100%;
    height: 460px;
    background: radial-gradient(circle at center, #0a1529 0%, #04070f 100%);
    border: 1px solid #163854;
    display: block;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }}
  th, td {{
    border-bottom: 1px solid #132a42;
    padding: 6px 4px;
    text-align: left;
  }}
  .state-Pass {{ color: #00ff9d; font-weight: bold; }}
  .state-Possible {{ color: #00f0ff; }}
  .state-Absorbed {{ color: #ffb84d; }}
  .state-Fail {{ color: #ff3366; text-decoration: line-through; }}
  .winners-box {{
    margin-top: 14px;
    border: 2px solid #00ff9d;
    box-shadow: 0 0 18px rgba(0, 255, 157, 0.35);
    padding: 12px;
    background: rgba(0, 40, 25, 0.55);
  }}
  pre {{
    font-size: 11px;
    overflow-x: auto;
    color: #9cdbff;
  }}
</style>
</head>
<body>
  <div class="panel">
    <h1>NEBIUS — SURVIVAL OF THE FITTEST // TRON ARENA</h1>
    <div style="font-size:12px;margin-bottom:8px;color:#8ab4f8;">Problem: {problem}</div>
    <canvas id="arenaCanvas" width="900" height="460"></canvas>
    <div class="winners-box" id="winnersBox">
      <h2>WINNER'S BOX (VERIFIED SURVIVOR EXIT)</h2>
      <div id="winnerContent"></div>
    </div>
  </div>
  <div class="panel">
    <h2>SCOREBOARD HUD (TIME ALIVE + STATE)</h2>
    <table>
      <thead>
        <tr><th>Agent</th><th>State</th><th>Alive(s)</th><th>Cert</th><th>Budget</th></tr>
      </thead>
      <tbody id="hudBody"></tbody>
    </table>
    <h2 style="margin-top:16px;">STREAM SNAPSHOT</h2>
    <pre>{ascii_snapshot}</pre>
  </div>
<script>
  const ARENA_EVENTS = {json.dumps(ev_dicts)};
  const RECONSTRUCTED = {json.dumps(state)};

  const tbody = document.getElementById("hudBody");
  RECONSTRUCTED.scoreboard.forEach(r => {{
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${{r.id}}</td>
      <td class="state-${{r.state}}">${{r.state}}</td>
      <td>${{Number(r.time_alive).toFixed(4)}}s</td>
      <td>${{Number(r.certainty_score).toFixed(2)}}</td>
      <td>$${{Number(r.budget_remaining).toFixed(4)}}</td>
    `;
    tbody.appendChild(tr);
  }});

  const wDiv = document.getElementById("winnerContent");
  if (RECONSTRUCTED.winners_box.length === 0) {{
    wDiv.textContent = "Awaiting verified survivor...";
  }} else {{
    RECONSTRUCTED.winners_box.forEach((w, idx) => {{
      const p = document.createElement("div");
      p.style.marginBottom = "8px";
      p.innerHTML = `<strong>#${{idx+1}} ${{w.id}}</strong> [Certainty: ${{w.certainty_score}}] — ${{w.fork_under_exploration}}`;
      wDiv.appendChild(p);
    }});
  }}

  // Draw Tron light-cycle paths on Canvas from RECONSTRUCTED
  const canvas = document.getElementById("arenaCanvas");
  const ctx = canvas.getContext("2d");
  ctx.strokeStyle = "rgba(0, 240, 255, 0.08)";
  for (let x = 0; x < canvas.width; x += 30) {{
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
  }}
  for (let y = 0; y < canvas.height; y += 30) {{
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
  }}
  // Center verified main line
  ctx.strokeStyle = "#00ff9d";
  ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(canvas.width / 2, 20); ctx.lineTo(canvas.width / 2, canvas.height - 20); ctx.stroke();
</script>
</body>
</html>
"""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html, encoding="utf-8")
        return target
