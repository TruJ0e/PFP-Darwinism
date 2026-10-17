"""
Proof-trail report: what exited into the winner's box, and why it deserves to be there.

The arena player shows the run; this writes the audit. Like the renderers it is a
pure function of the event stream (plus an optional run summary for spend), so a
live run and a `--replay` of its JSONL produce the same report, and no model is
called to write it.

For every winner it lists the claim it carried out, every falsification it
survived (angle + judge), its full ancestry back to the starter(s) — merge
parents included, with per-parent provenance — and the crashes its ancestors
caused or suffered along the way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .fsutil import atomic_write_text


def _md(text: Any, limit: int = 600) -> str:
    """One line of stream text, safe inside Markdown (no headings, tables or raw HTML)."""
    s = " ".join(str(text or "").split())
    if len(s) > limit:
        s = s[: limit - 1].rstrip() + "…"
    for ch in ("\\", "`", "*", "_", "[", "]", "<", ">", "|", "#"):
        s = s.replace(ch, "\\" + ch)
    return s


def _code(text: Any, limit: int = 80) -> str:
    """An inline code span: backslash escapes are literal inside backticks, so strip backticks instead."""
    s = " ".join(str(text or "").split()).replace("`", "'")[:limit]
    return f"`{s}`" if s else "`?`"


def _ancestry(branch_id: str, branches: Dict[str, Dict[str, Any]]) -> List[str]:
    """Every ancestor of `branch_id` (split parents and merge parents), nearest first, cycle-safe."""
    out: List[str] = []
    seen = {branch_id}
    frontier = [branch_id]
    while frontier:
        nxt: List[str] = []
        for bid in frontier:
            snap = branches.get(bid, {})
            parents = list(snap.get("merge_parents") or [])
            if snap.get("parent"):
                parents.insert(0, snap["parent"])
            for p in parents:
                p = str(p)
                if p not in seen:
                    seen.add(p)
                    out.append(p)
                    nxt.append(p)
        frontier = nxt
    return out


def build_proof_report(
    events: Iterable[Any],
    problem: str = "",
    summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Structured proof trail reduced from the stream (JSON-serialisable)."""
    from .arena import TronArenaRenderer, stream_problem

    events = list(events)
    state = TronArenaRenderer.reconstruct_state_from_stream(events)
    branches: Dict[str, Dict[str, Any]] = state["branches"]
    crashes_by_branch: Dict[str, List[Dict[str, Any]]] = {}
    for c in state["crashes"]:
        crashes_by_branch.setdefault(str(c.get("branch_id", "")), []).append(c)
    revoked = {str(r.get("branch_id")) for r in state["revocations"]}

    winners: List[Dict[str, Any]] = []
    for w in state["winners_box"]:
        bid = str(w.get("id", ""))
        latest = branches.get(bid, w)
        ancestors = _ancestry(bid, branches)
        winners.append(
            {
                "id": bid,
                "state": latest.get("state", w.get("state", "")),
                "origin": latest.get("origin", ""),
                "depth": latest.get("depth", 0),
                "certainty_score": latest.get("certainty_score", 0.0),
                "solution": latest.get("solution") or w.get("solution") or latest.get("fork_under_exploration", ""),
                "survived": list(latest.get("pass_provenance") or w.get("pass_provenance") or []),
                "merge_provenance": dict(latest.get("merge_provenance") or {}),
                "ancestors": [
                    {
                        "id": a,
                        "origin": branches.get(a, {}).get("origin", ""),
                        "state": branches.get(a, {}).get("state", ""),
                        "fork": branches.get(a, {}).get("fork_under_exploration", ""),
                    }
                    for a in ancestors
                ],
                "lineage_crashes": [
                    dict(c, branch_id=a) for a in [bid, *ancestors] for c in crashes_by_branch.get(a, [])
                ],
                "previously_revoked": bid in revoked,
            }
        )

    counts: Dict[str, int] = {}
    for snap in branches.values():
        st = str(snap.get("state", "Possible"))
        counts[st] = counts.get(st, 0) + 1

    report: Dict[str, Any] = {
        "format": "pfp-proof-report/1",
        "problem": problem or (summary or {}).get("problem", "") or stream_problem(events),
        "events_processed": state["events_processed"],
        "branches_total": len(branches),
        "branches_by_state": dict(sorted(counts.items())),
        "starters": list(state["starters"]),
        "splits": len(state["splits"]),
        "merges": len(state["merges"]),
        "crashes": len(state["crashes"]),
        "revocations": len(state["revocations"]),
        "halted": state["halted"],
        "winners": winners,
    }
    if summary:
        b = summary.get("budget", {}) or {}
        report["spend"] = {
            "total_api_spend_usd": summary.get("total_api_spend_usd", b.get("api_spend_usd")),
            "arena_api_spend_usd": b.get("api_spend_usd"),
            "self_improvement_spend_usd": summary.get("self_improvement_spend_usd", 0.0),
            "api_calls": b.get("api_calls"),
            "hard_cap_usd": b.get("hard_cap_usd"),
            "overdraft_usd": b.get("overdraft_usd"),
            "is_mock": summary.get("is_mock"),
            "active_version_id": summary.get("active_version_id"),
        }
    return report


def _render_survived(survived: Sequence[Dict[str, Any]]) -> List[str]:
    if not survived:
        return ["- *(no recorded falsifications survived)*"]
    lines = []
    for i, p in enumerate(survived, 1):
        judge = f" — judge {_code(p.get('judge_id'), 80)}" if p.get("judge_id") else ""
        lines.append(
            f"{i}. **{_md(p.get('falsifier_angle', '?'), 80)}**: survived {_md(p.get('survived_what', ''), 200)}{judge}"
        )
    return lines


def render_proof_markdown(report: Dict[str, Any]) -> str:
    """Human-readable proof trail. Every stream string is escaped: model text cannot inject Markdown/HTML."""
    out: List[str] = ["# PFP Darwinism — proof trail", ""]
    if report.get("problem"):
        out += [f"**Problem:** {_md(report['problem'], 1200)}", ""]
    by_state = ", ".join(f"{k} {v}" for k, v in report.get("branches_by_state", {}).items()) or "none"
    out += [
        f"- Events: {report.get('events_processed', 0)} · branches: {report.get('branches_total', 0)} ({by_state})",
        f"- Splits: {report.get('splits', 0)} · merges: {report.get('merges', 0)} · "
        f"crashes into the main line: {report.get('crashes', 0)} · Pass revocations: {report.get('revocations', 0)}",
    ]
    spend = report.get("spend")
    if spend:
        def money(v: Any) -> str:
            return f"${float(v):.4f}" if isinstance(v, (int, float)) else "n/a"

        out.append(
            f"- Spend: {money(spend.get('total_api_spend_usd'))} total "
            f"(arena {money(spend.get('arena_api_spend_usd'))} over {spend.get('api_calls', 'n/a')} calls, "
            f"improve {money(spend.get('self_improvement_spend_usd'))}; cap {money(spend.get('hard_cap_usd'))}, "
            f"overdraft {money(spend.get('overdraft_usd'))}) · "
            f"{'MOCK' if spend.get('is_mock') else 'LIVE'} · version {_code(spend.get('active_version_id'), 80)}"
        )
    halted = report.get("halted")
    if halted:
        out += [
            "",
            f"> **HALTED: {_md(halted.get('reason', 'unknown'), 80)}** at event {halted.get('seq', '?')} — "
            f"{_md(halted.get('detail', ''), 300)}. Results below are partial.",
        ]
    out.append("")

    winners = report.get("winners", [])
    out += [f"## Winner's box ({len(winners)})", ""]
    if not winners:
        out += ["Nothing survived repeated, rotated falsification. No claim is certified by this run.", ""]
    for w in winners:
        out += [
            f"### {_code(w['id'], 80)} — certainty {float(w.get('certainty_score') or 0.0):.3f}",
            "",
            f"- Origin: {_md(w.get('origin') or '?', 40)} · depth {w.get('depth', 0)} · state {_md(w.get('state'), 20)}"
            + (" · *survived a Pass revocation and re-earned it*" if w.get("previously_revoked") else ""),
            "",
            f"**Claim:** {_md(w.get('solution'), 1200)}",
            "",
            "**Survived:**",
            "",
            *_render_survived(w.get("survived", [])),
            "",
        ]
        if w.get("merge_provenance"):
            out += ["**Merge provenance:**", ""]
            for parent, claims in w["merge_provenance"].items():
                joined = "; ".join(_md(c, 200) for c in (claims or [])[:6]) or "(no claims listed)"
                out.append(f"- from {_code(parent, 80)}: {joined}")
            out.append("")
        if w.get("ancestors"):
            out += ["**Ancestry** (nearest first):", ""]
            for a in w["ancestors"]:
                out.append(
                    f"- {_code(a['id'], 80)} ({_md(a.get('origin') or '?', 40)}, {_md(a.get('state') or '?', 20)}): "
                    f"{_md(a.get('fork'), 200)}"
                )
            out.append("")
        if w.get("lineage_crashes"):
            out += ["**Crashes in this lineage:**", ""]
            for c in w["lineage_crashes"]:
                out.append(
                    f"- event {c.get('seq', '?')}: {_code(c.get('branch_id'), 80)} hit the main line via "
                    f"{_md(c.get('angle') or '?', 60)} — {_md(c.get('objection'), 240)}"
                )
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def write_proof_report(
    events: Iterable[Any],
    output_path: str | Path,
    problem: str = "",
    summary: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write the Markdown proof trail (`.md`) or the structured report (`.json`), atomically."""
    import json

    report = build_proof_report(events, problem=problem, summary=summary)
    target = Path(output_path)
    if target.suffix.lower() == ".json":
        text = json.dumps(report, indent=2, default=str) + "\n"
    else:
        text = render_proof_markdown(report)
    return atomic_write_text(target, text)
