#!/usr/bin/env python3
"""
PFP Darwinism — Full Arena Runner (Pieces 1–8: Split, Spawn, Track, Align, Merge,
Falsify, Tron Arena Renderer, and Self-Improvement Loop + Version Registry).

Uses NEBIUS_BASE_URL, NEBIUS_MODEL, and NEBIUS_API_KEY from the environment.
When NEBIUS_API_KEY is unset (or --mock is passed), runs against MockOpenAIClient
which implements the exact same OpenAI-compatible interface so the entire run
from ignition through the winner's box renders live in the arena without a key.

With --state-dir, the version registry (failsafe) and the case bank (recall)
persist between runs: run N+1 starts seeded with run N's wins, on whichever
version the registry has active (or pinned).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from pfp import (
    AppendOnlyViolationError,
    ArenaFeed,
    FAST_JUDGE_MODEL,
    BackendError,
    HardCapExceededError,
    JudgeLadder,
    MainHarness,
    MeteredClient,
    SelfImprovementLoop,
    TronArenaRenderer,
    VersionRegistry,
    create_client,
)
from pfp.arena import is_journal_digest, stream_problem
from pfp.fsutil import StateLockError, atomic_write_text, exclusive_lock
from pfp.proof_report import write_proof_report

DEFAULT_PROBLEM = (
    "A small business's SSH works laptop->desktop but not desktop->laptop. "
    "Diagnose the root causes and verify the solution."
)


def _positive_int(text: str) -> int:
    val = int(text)
    if val < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return val


def _non_negative_int(text: str) -> int:
    val = int(text)
    if val < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return val


def _positive_float(text: str) -> float:
    val = float(text)
    # float() accepts "nan" and "inf": a NaN cap admits every call, an infinite one never binds.
    if not math.isfinite(val) or val <= 0:
        raise argparse.ArgumentTypeError("must be a finite number > 0")
    return val


def _expected_head(value: str) -> str:
    """
    --expect-head: a 64-hex journal head digest, or a JSON file carrying one (a run summary
    or a `.json` proof report, both of which record `journal.head_digest`).
    """
    text = value.strip().lower()
    if is_journal_digest(text):
        return text
    try:
        raw = json.loads(Path(value).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"--expect-head is neither a 64-hex digest nor a readable JSON file: {exc}") from None
    head = raw.get("journal", {}).get("head_digest") if isinstance(raw, dict) and isinstance(raw.get("journal"), dict) else None
    if not is_journal_digest(head):
        raise ValueError(f"--expect-head file {value} has no valid journal.head_digest")
    return str(head)


def _load_anchored_feed(path: str, expect_head: str, *, allow_legacy: bool) -> ArenaFeed:
    """Load and verify a journal; with --expect-head its head must match (a truncated copy cannot)."""
    expected = _expected_head(expect_head) if expect_head else ""
    # A legacy v1 journal is re-hashed on load exactly as the live run hashed it, so a
    # matching anchor authenticates it even though the file carries no digests.
    feed = ArenaFeed.load_jsonl(Path(path), allow_legacy=allow_legacy or bool(expected))
    if expected:
        if feed.head_digest != expected:
            anchor = feed.anchor()
            raise AppendOnlyViolationError(
                f"journal head {anchor['head_digest'] or '(empty)'} after {anchor['events']} events does not match "
                f"the expected head {expected}: the journal was truncated, extended or replaced"
            )
        feed.legacy_unverified = False
    return feed


def _verify(args: argparse.Namespace) -> int:
    """--verify: check a journal's hash chain (and its anchor) without rendering or calling a model."""
    try:
        feed = _load_anchored_feed(args.verify, args.expect_head, allow_legacy=False)
    except (OSError, ValueError, AppendOnlyViolationError) as exc:
        print(f"[pfp] VERIFY FAILED {args.verify}: {exc}", file=sys.stderr)
        return 1
    result = {"journal": str(args.verify), **feed.anchor(), "expected_head_matched": bool(args.expect_head) or None}
    print(json.dumps(result, indent=2))
    if not args.expect_head:
        print(
            "[pfp] chain OK — without --expect-head a copy cut short at the tail would also pass; "
            "anchor it with the head digest from the run summary or proof report",
            file=sys.stderr,
        )
    return 0


def _replay(args: argparse.Namespace) -> int:
    """Re-render a saved arena stream: no model calls unless --state-dir triggers post-replay improvement."""
    try:
        feed = _load_anchored_feed(args.replay, args.expect_head, allow_legacy=True)
    except (OSError, ValueError, AppendOnlyViolationError) as exc:
        print(f"[pfp] cannot replay {args.replay}: {exc}", file=sys.stderr)
        return 1
    if feed.legacy_unverified:
        print(
            f"[pfp] WARNING: {args.replay} is a legacy v1 journal with no hash chain; its content is unverified",
            file=sys.stderr,
        )
    # The journal records its own problem; --problem only overrides the title.
    problem = args.problem or stream_problem(feed.events)
    renderer = TronArenaRenderer()
    if not args.quiet:
        print(renderer.render_ascii(feed.events, problem=problem))
    if args.html_out:
        renderer.export_html_arena(feed.events, Path(args.html_out), problem=problem)
    if args.report_out:
        write_proof_report(feed.events, Path(args.report_out), problem=problem)

    state_dir = Path(args.state_dir) if args.state_dir else None
    improvement_result = None
    improvement_spend_usd = 0.0
    exit_code = 0
    if state_dir is not None and not args.no_improve:
        try:
            with exclusive_lock(state_dir / ".pfp.lock"):
                rc, improvement_result, improvement_spend_usd = _replay_improve(args, state_dir, feed, problem)
                if rc != 0:
                    return rc
                if improvement_result.get("interrupted"):
                    exit_code = 2
        except StateLockError as exc:
            print(f"[pfp] refusing to run: {exc}", file=sys.stderr)
            return 1

    if args.summary_out:
        state = renderer.reconstruct_state_from_stream(feed.events)
        mined = SelfImprovementLoop(client=None).miner.mine(feed.events).to_dict()
        replay_summary = {
            "replay_source": str(args.replay),
            "problem": problem,
            "events_count": state["events_processed"],
            "total_branches_spawned": len(state["branches"]),
            "halted_reason": state["halted"].get("reason") if state["halted"] else None,
            "winners_box": state["winners_box"],
            "scoreboard": state["scoreboard"],
            "mined_metrics": mined,
            "journal": feed.anchor(),
        }
        if improvement_result is not None:
            replay_summary["self_improvement"] = improvement_result
            replay_summary["self_improvement_spend_usd"] = improvement_spend_usd
        atomic_write_text(Path(args.summary_out), json.dumps(replay_summary, indent=2, default=str))
    print(f"[pfp] REPLAY | {len(feed)} events from {args.replay}", file=sys.stderr)
    return exit_code


def _improvement_budget(args: argparse.Namespace, remaining: float) -> float:
    """Piece 8's own ceiling: what --hard-cap left, further bounded by --improve-cap if given."""
    cap = getattr(args, "improve_cap", None)
    return round(remaining if cap is None else min(remaining, cap), 8)


def _replay_improve(
    args: argparse.Namespace,
    state_dir: Path,
    feed: ArenaFeed,
    problem: str,
) -> tuple[int, dict, float]:
    """Run Piece 8 self-improvement on a replayed stream under the exclusive state-dir lock and hard cap."""
    registry_path = state_dir / "registry.json"
    cases_path = state_dir / "case_bank.json"
    try:
        registry = VersionRegistry.load(registry_path) if registry_path.exists() else VersionRegistry()
    except (OSError, ValueError) as exc:
        print(f"[pfp] refusing to run: cannot load registry {registry_path}: {exc}", file=sys.stderr)
        return 1, {}, 0.0
    loop = SelfImprovementLoop(client=None, registry=registry)
    if cases_path.exists():
        try:
            loop.load_case_bank(cases_path)
        except (OSError, ValueError) as exc:
            print(f"[pfp] refusing to run: cannot load case bank {cases_path}: {exc}", file=sys.stderr)
            return 1, {}, 0.0
    client = create_client(use_mock=True if args.mock else None, max_retries=args.max_retries)
    loop.client = client
    budget = _improvement_budget(args, args.hard_cap)
    meter = MeteredClient(client, hard_cap_usd=budget, max_concurrent_calls=args.max_concurrent)
    loop.client = meter
    loop.sandbox_hard_cap_usd = min(loop.sandbox_hard_cap_usd, budget / (2 if loop.paired_sandbox else 1))
    try:
        res = loop.run_post_run_improvement(
            feed.events,
            problem=problem or DEFAULT_PROBLEM,
            run_sandbox=True,
        )
    except (HardCapExceededError, BackendError) as exc:
        res = {"skipped": f"{type(exc).__name__}: {str(exc)[:300]}"}
    except KeyboardInterrupt:
        # Nothing was registered yet (that is the cycle's last step); keep what was paid for
        res = {"skipped": "interrupted by the operator (Ctrl-C)", "interrupted": True}
    finally:
        loop.client = client
    registry.save(registry_path)
    loop.save_case_bank(cases_path)
    return 0, res, meter.stats()["total_cost_usd"]


def _run_improvement(
    args: argparse.Namespace,
    client: object,
    loop: SelfImprovementLoop,
    harness: MainHarness,
    summary: dict,
) -> None:
    """
    Piece 8 spends real money too (assessment, anti-model, proposition, sandbox run), so it is
    metered against whatever the arena run left under --hard-cap rather than the raw client.
    """
    remaining = round(args.hard_cap - summary["budget"]["api_spend_usd"], 8)
    if remaining <= 0:
        summary["self_improvement"] = {"skipped": "hard cap exhausted by the arena run"}
        return
    budget = _improvement_budget(args, remaining)
    summary["self_improvement_budget_usd"] = budget
    meter = MeteredClient(client, hard_cap_usd=budget, max_concurrent_calls=args.max_concurrent)
    loop.client = meter
    # A paired A/B runs two sandboxes (candidate + current version): split what's left
    loop.sandbox_hard_cap_usd = min(loop.sandbox_hard_cap_usd, budget / (2 if loop.paired_sandbox else 1))
    try:
        summary["self_improvement"] = loop.run_post_run_improvement(
            harness.arena_feed.events,
            problem=summary["problem"],
            run_sandbox=True,
        )
    except (HardCapExceededError, BackendError) as exc:
        summary["self_improvement"] = {"skipped": f"{type(exc).__name__}: {str(exc)[:300]}"}
    except KeyboardInterrupt:
        # Ctrl-C during Piece 8 must not lose the arena run: its stream, HTML, report and
        # summary are still written, and nothing half-done reaches the registry (registering
        # the new version is the cycle's last step).
        summary["self_improvement"] = {"skipped": "interrupted by the operator (Ctrl-C)", "interrupted": True}
    finally:
        loop.client = client
    summary["self_improvement_spend_usd"] = meter.stats()["total_cost_usd"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PFP Darwinism — Full Arena Runner (Pieces 1-8)")
    parser.add_argument("--problem", type=str, default=None, help="Problem statement (default: the SSH demo)")
    parser.add_argument("--starters", type=_positive_int, default=3, help="Number of starter agents (3 demo, 9 real)")
    parser.add_argument("--hard-cap", type=_positive_float, default=2.00, help="Hard budget cap in USD (default $2.00)")
    parser.add_argument("--depth-decay", type=_positive_float, default=0.5, help="Child budget inheritance per generation (default 0.5)")
    parser.add_argument("--finish-reserve-ratio", type=_positive_float, default=0.125, help="Finish reserve share of root (default 0.125)")
    parser.add_argument("--root-budget-usd", type=_positive_float, default=None, help="Operator root budget override in USD (default: sized from difficulty estimate)")
    parser.add_argument("--depth-dial", type=_positive_int, default=3, help="Depth dial bounding fan-out")
    parser.add_argument("--k-probes", type=_positive_int, default=3, help="Number of blind probe continuations per step")
    parser.add_argument("--gauntlet-rounds", type=_positive_int, default=3, help="Pass-readiness gauntlet rounds (default 3)")
    parser.add_argument(
        "--judge-ladder",
        choices=("tiered", "super"),
        default="tiered",
        help="tiered (default): fast first-pass judge, Super for appeals / final gauntlet round / pre-exit; "
        "super: every verdict on Super (pre-v0.41 routing)",
    )
    parser.add_argument(
        "--first-pass-judge",
        type=str,
        default=FAST_JUDGE_MODEL,
        help=f"First-pass judge model for --judge-ladder tiered (default {FAST_JUDGE_MODEL})",
    )
    parser.add_argument("--selection-rounds", type=_positive_int, default=3, help="Max align/merge/contradiction rounds")
    parser.add_argument("--stream-out", type=str, default="", help="Optional JSONL path to save append-only ArenaFeed")
    parser.add_argument("--html-out", type=str, default="", help="Optional HTML path to export Tron Arena player")
    parser.add_argument("--state-dir", type=str, default="", help="Persist registry + case bank here between runs")
    parser.add_argument("--pin-version", type=str, default="", help="Pin a registry version in --state-dir before running")
    parser.add_argument("--unpin", action="store_true", help="Clear any manual version pin in --state-dir before running")
    parser.add_argument("--restore-version", type=str, default="", help="Restore a registry version in --state-dir before running")
    parser.add_argument("--list-versions", action="store_true", help="List versions in --state-dir and exit (no model calls)")
    parser.add_argument("--no-improve", action="store_true", help="Skip the Piece 8 post-run improvement loop")
    parser.add_argument(
        "--improve-cap",
        type=_positive_float,
        default=None,
        help="USD ceiling for the Piece 8 loop (default: whatever --hard-cap leaves after the arena)",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON summary in addition to Tron Arena view")
    parser.add_argument("--quiet", action="store_true", help="Do not print the ASCII arena")
    parser.add_argument("--mock", action="store_true", help="Force mock backend even if NEBIUS_API_KEY is set")
    parser.add_argument("--max-retries", type=_non_negative_int, default=3, help="Live backend retries on 429/5xx/network errors")
    parser.add_argument("--max-concurrent", type=_positive_int, default=32, help="Max in-flight model calls")
    parser.add_argument("--summary-out", type=str, default="", help="Optional path to write the run summary JSON")
    parser.add_argument(
        "--report-out",
        type=str,
        default="",
        help="Optional proof-trail report path (.md Markdown, .json structured); works with --replay",
    )
    parser.add_argument("--replay", type=str, default="", help="Re-render a saved --stream-out JSONL (no model calls)")
    parser.add_argument(
        "--verify",
        type=str,
        default="",
        help="Verify a saved --stream-out JSONL's hash chain and print its anchor (no model calls), then exit",
    )
    parser.add_argument(
        "--expect-head",
        type=str,
        default="",
        help="With --verify/--replay: required journal head digest (64 hex, or a summary/proof .json carrying it)",
    )
    args = parser.parse_args(argv)

    if args.verify and args.replay:
        parser.error("--verify and --replay are mutually exclusive")
    if args.expect_head and not (args.verify or args.replay):
        parser.error("--expect-head requires --verify or --replay")
    if args.verify:
        return _verify(args)
    if args.replay:
        return _replay(args)
    if (args.pin_version or args.unpin or args.restore_version or args.list_versions) and not args.state_dir:
        parser.error("--pin-version, --unpin, --restore-version, and --list-versions require --state-dir")
    if sum(bool(x) for x in (args.pin_version, args.unpin, args.restore_version)) > 1:
        parser.error("--pin-version, --unpin, and --restore-version are mutually exclusive")
    if args.problem is None:
        args.problem = DEFAULT_PROBLEM
    if not args.problem.strip():
        parser.error("--problem must be non-empty")

    state_dir = Path(args.state_dir) if args.state_dir else None
    if state_dir is None:
        return _run_live(args, None)
    # One run per state dir: two concurrent runs would each load the failsafe and the
    # case bank, and the later save would silently drop the other run's versions and wins.
    try:
        with exclusive_lock(state_dir / ".pfp.lock"):
            return _run_live(args, state_dir)
    except StateLockError as exc:
        print(f"[pfp] refusing to run: {exc}", file=sys.stderr)
        return 1


def _apply_registry_cli_actions(args: argparse.Namespace, registry: VersionRegistry, registry_path: Path | None) -> int | None:
    """Apply operator failsafe commands (--unpin, --pin-version, --restore-version, --list-versions) at the run boundary."""
    mutated = False
    if args.unpin:
        registry.unpin()
        mutated = True
    if args.restore_version:
        try:
            registry.restore_version(args.restore_version)
            mutated = True
        except KeyError as exc:
            print(f"[pfp] refusing to run: unknown version {args.restore_version!r} ({exc})", file=sys.stderr)
            return 1
    if args.pin_version:
        try:
            registry.pin_version(args.pin_version)
            mutated = True
        except KeyError as exc:
            print(f"[pfp] refusing to run: unknown version {args.pin_version!r} ({exc})", file=sys.stderr)
            return 1
    if mutated and registry_path is not None:
        registry.save(registry_path)
    if args.list_versions:
        listing = {
            "active_version_id": registry.active_version_id,
            "pinned_version_id": registry.pinned_version_id,
            "versions": [v.to_dict() for v in registry.all_versions()],
            "failed_experiments": registry.failed_experiments,
        }
        print(json.dumps(listing, indent=2))
        return 0
    return None


def _run_live(args: argparse.Namespace, state_dir: Path | None) -> int:
    registry_path = state_dir / "registry.json" if state_dir else None
    cases_path = state_dir / "case_bank.json" if state_dir else None
    # Load persisted state before any client exists: a corrupt or tampered failsafe must stop
    # the command before a cent is spent, and must never be silently replaced by a fresh one
    # (that would erase the rollback history the next save would then overwrite).
    try:
        registry = (
            VersionRegistry.load(registry_path)
            if registry_path is not None and registry_path.exists()
            else VersionRegistry()
        )
    except (OSError, ValueError) as exc:
        print(f"[pfp] refusing to run: cannot load registry {registry_path}: {exc}", file=sys.stderr)
        return 1

    action_rc = _apply_registry_cli_actions(args, registry, registry_path)
    if action_rc is not None:
        return action_rc

    # Recall text becomes prompt material. Validate it before constructing a provider
    # client, just like the registry above, so corrupt state cannot reach the network.
    loop = SelfImprovementLoop(client=None, registry=registry)
    if cases_path is not None and cases_path.exists():
        try:
            loop.load_case_bank(cases_path)
        except (OSError, ValueError) as exc:
            print(f"[pfp] refusing to run: cannot load case bank {cases_path}: {exc}", file=sys.stderr)
            return 1

    client = create_client(use_mock=True if args.mock else None, max_retries=args.max_retries)
    loop.client = client

    harness = MainHarness(
        client=client,
        starter_agents=args.starters,
        hard_cap_usd=args.hard_cap,
        depth_dial=args.depth_dial,
        k_probes=args.k_probes,
        gauntlet_rounds=args.gauntlet_rounds,
        selection_rounds=args.selection_rounds,
        max_concurrent_calls=args.max_concurrent,
        depth_decay=args.depth_decay,
        finish_reserve_ratio=args.finish_reserve_ratio,
        root_budget_override_usd=args.root_budget_usd,
        judge_ladder=(
            JudgeLadder.super_only() if args.judge_ladder == "super" else JudgeLadder(first_pass_model=args.first_pass_judge)
        ),
        enable_selection=True,
        version_registry=registry,
        seed_cases=loop.recall(args.problem, k=args.starters),
    )
    summary = harness.run(args.problem)

    if not args.quiet:
        print(harness.render_arena(problem=args.problem))

    if not args.no_improve and summary["halted_reason"] is None:
        _run_improvement(args, client, loop, harness, summary)
    summary["total_api_spend_usd"] = round(
        summary["budget"]["api_spend_usd"] + summary.get("self_improvement_spend_usd", 0.0), 6
    )
    # Anchor the journal outside the JSONL so a copy cut short at the tail is detectable
    summary["journal"] = harness.arena_feed.anchor()

    if args.stream_out:
        harness.arena_feed.save_jsonl(Path(args.stream_out))
    if args.html_out:
        harness.arena_renderer.export_html_arena(
            harness.arena_feed.events,
            Path(args.html_out),
            problem=args.problem,
        )
    if args.report_out:
        write_proof_report(harness.arena_feed.events, Path(args.report_out), problem=args.problem, summary=summary)
    if registry_path is not None and cases_path is not None:
        registry.save(registry_path)
        loop.save_case_bank(cases_path)
    if args.summary_out:
        atomic_write_text(Path(args.summary_out), json.dumps(summary, indent=2, default=str))

    b = summary["budget"]
    print(
        f"\n[pfp] {'MOCK' if summary['is_mock'] else 'LIVE'} | branches={summary['total_branches_spawned']} "
        f"winners={len(summary['winners_box'])} seeded={summary['seeded_starters_count']} "
        f"| spend=${summary['total_api_spend_usd']:.4f} (arena ${b['api_spend_usd']:.4f} over {b['api_calls']} calls, "
        f"improve ${summary.get('self_improvement_spend_usd', 0.0):.4f}; root ${b['root_budget_usd']:.4f}, "
        f"overdraft ${b['overdraft_usd']:.4f}, cap ${b['hard_cap_usd']:.2f}) "
        f"| version={summary['active_version_id']}"
        + (f" | HALTED: {summary['halted_reason']}" if summary["halted_reason"] else "")
        + f" | journal {summary['journal']['events']} events head {summary['journal']['head_digest'][:16]}",
        file=sys.stderr,
    )
    if args.json:
        print("\n--- RUN SUMMARY JSON ---")
        print(json.dumps(summary, indent=2))
    interrupted = bool(summary.get("self_improvement", {}).get("interrupted"))
    return 0 if summary["halted_reason"] is None and not interrupted else 2


if __name__ == "__main__":
    sys.exit(main())
