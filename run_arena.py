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
import sys
from pathlib import Path

from pfp import (
    AppendOnlyViolationError,
    ArenaFeed,
    BackendError,
    HardCapExceededError,
    MainHarness,
    MeteredClient,
    SelfImprovementLoop,
    TronArenaRenderer,
    VersionRegistry,
    create_client,
)

DEFAULT_PROBLEM = (
    "A small business's SSH works laptop->desktop but not desktop->laptop. "
    "Diagnose the root causes and verify the solution."
)


def _positive_int(text: str) -> int:
    val = int(text)
    if val < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return val


def _positive_float(text: str) -> float:
    val = float(text)
    if val <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return val


def _replay(args: argparse.Namespace) -> int:
    """Re-render a saved arena stream: no model calls, no spend (the renderers are pure)."""
    try:
        feed = ArenaFeed.load_jsonl(Path(args.replay))
    except (OSError, ValueError, AppendOnlyViolationError) as exc:
        print(f"[pfp] cannot replay {args.replay}: {exc}", file=sys.stderr)
        return 1
    problem = args.problem or ""
    renderer = TronArenaRenderer()
    if not args.quiet:
        print(renderer.render_ascii(feed.events, problem=problem))
    if args.html_out:
        renderer.export_html_arena(feed.events, Path(args.html_out), problem=problem)
    print(f"[pfp] REPLAY | {len(feed)} events from {args.replay}", file=sys.stderr)
    return 0


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
    meter = MeteredClient(client, hard_cap_usd=remaining, max_concurrent_calls=args.max_concurrent)
    loop.client = meter
    loop.sandbox_hard_cap_usd = min(loop.sandbox_hard_cap_usd, remaining)
    try:
        summary["self_improvement"] = loop.run_post_run_improvement(
            harness.arena_feed.events,
            problem=summary["problem"],
            run_sandbox=True,
        )
    except (HardCapExceededError, BackendError) as exc:
        summary["self_improvement"] = {"skipped": f"{type(exc).__name__}: {str(exc)[:300]}"}
    finally:
        loop.client = client
    summary["self_improvement_spend_usd"] = meter.stats()["total_cost_usd"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PFP Darwinism — Full Arena Runner (Pieces 1-8)")
    parser.add_argument("--problem", type=str, default=None, help="Problem statement (default: the SSH demo)")
    parser.add_argument("--starters", type=_positive_int, default=3, help="Number of starter agents (3 demo, 9 real)")
    parser.add_argument("--hard-cap", type=_positive_float, default=2.00, help="Hard budget cap in USD (default $2.00)")
    parser.add_argument("--depth-dial", type=_positive_int, default=3, help="Depth dial bounding fan-out")
    parser.add_argument("--k-probes", type=_positive_int, default=3, help="Number of blind probe continuations per step")
    parser.add_argument("--gauntlet-rounds", type=_positive_int, default=3, help="Pass-readiness gauntlet rounds (default 3)")
    parser.add_argument("--selection-rounds", type=_positive_int, default=3, help="Max align/merge/contradiction rounds")
    parser.add_argument("--stream-out", type=str, default="", help="Optional JSONL path to save append-only ArenaFeed")
    parser.add_argument("--html-out", type=str, default="", help="Optional HTML path to export Tron Arena player")
    parser.add_argument("--state-dir", type=str, default="", help="Persist registry + case bank here between runs")
    parser.add_argument("--no-improve", action="store_true", help="Skip the Piece 8 post-run improvement loop")
    parser.add_argument("--json", action="store_true", help="Print JSON summary in addition to Tron Arena view")
    parser.add_argument("--quiet", action="store_true", help="Do not print the ASCII arena")
    parser.add_argument("--mock", action="store_true", help="Force mock backend even if NEBIUS_API_KEY is set")
    parser.add_argument("--max-retries", type=int, default=3, help="Live backend retries on 429/5xx/network errors")
    parser.add_argument("--max-concurrent", type=_positive_int, default=32, help="Max in-flight model calls")
    parser.add_argument("--summary-out", type=str, default="", help="Optional path to write the run summary JSON")
    parser.add_argument("--replay", type=str, default="", help="Re-render a saved --stream-out JSONL (no model calls)")
    args = parser.parse_args(argv)

    if args.replay:
        return _replay(args)
    if args.problem is None:
        args.problem = DEFAULT_PROBLEM
    if not args.problem.strip():
        parser.error("--problem must be non-empty")

    client = create_client(use_mock=True if args.mock else None, max_retries=args.max_retries)

    state_dir = Path(args.state_dir) if args.state_dir else None
    registry_path = state_dir / "registry.json" if state_dir else None
    cases_path = state_dir / "case_bank.json" if state_dir else None
    registry = (
        VersionRegistry.load(registry_path)
        if registry_path is not None and registry_path.exists()
        else VersionRegistry()
    )
    loop = SelfImprovementLoop(client=client, registry=registry)
    if cases_path is not None and cases_path.exists():
        loop.load_case_bank(cases_path)

    harness = MainHarness(
        client=client,
        starter_agents=args.starters,
        hard_cap_usd=args.hard_cap,
        depth_dial=args.depth_dial,
        k_probes=args.k_probes,
        gauntlet_rounds=args.gauntlet_rounds,
        selection_rounds=args.selection_rounds,
        max_concurrent_calls=args.max_concurrent,
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

    if args.stream_out:
        harness.arena_feed.save_jsonl(Path(args.stream_out))
    if args.html_out:
        harness.arena_renderer.export_html_arena(
            harness.arena_feed.events,
            Path(args.html_out),
            problem=args.problem,
        )
    if registry_path is not None and cases_path is not None:
        registry.save(registry_path)
        loop.save_case_bank(cases_path)
    if args.summary_out:
        out = Path(args.summary_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    b = summary["budget"]
    print(
        f"\n[pfp] {'MOCK' if summary['is_mock'] else 'LIVE'} | branches={summary['total_branches_spawned']} "
        f"winners={len(summary['winners_box'])} seeded={summary['seeded_starters_count']} "
        f"| spend=${summary['total_api_spend_usd']:.4f} (arena ${b['api_spend_usd']:.4f} over {b['api_calls']} calls, "
        f"improve ${summary.get('self_improvement_spend_usd', 0.0):.4f}; root ${b['root_budget_usd']:.4f}, "
        f"overdraft ${b['overdraft_usd']:.4f}, cap ${b['hard_cap_usd']:.2f}) "
        f"| version={summary['active_version_id']}"
        + (f" | HALTED: {summary['halted_reason']}" if summary["halted_reason"] else ""),
        file=sys.stderr,
    )
    if args.json:
        print("\n--- RUN SUMMARY JSON ---")
        print(json.dumps(summary, indent=2))
    return 0 if summary["halted_reason"] is None else 2


if __name__ == "__main__":
    sys.exit(main())
