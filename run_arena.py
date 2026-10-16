#!/usr/bin/env python3
"""
PFP Darwinism — Full Arena Runner (Pieces 1–8: Split, Spawn, Track, Align, Merge,
Falsify, Tron Arena Renderer, and Self-Improvement Loop + Version Registry).

Uses NEBIUS_BASE_URL, NEBIUS_MODEL, and NEBIUS_API_KEY from the environment.
When NEBIUS_API_KEY is unset (or --mock is passed), runs against MockOpenAIClient
which implements the exact same OpenAI-compatible interface so the entire run
from ignition through the winner's box renders live in the arena without a key.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from pfp import (
    MainHarness,
    SelfImprovementLoop,
    VersionRegistry,
    create_client,
)

DEFAULT_PROBLEM = (
    "A small business's SSH works laptop->desktop but not desktop->laptop. "
    "Diagnose the root causes and verify the solution."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="PFP Darwinism — Full Arena Runner (Pieces 1-8)")
    parser.add_argument("--problem", type=str, default=DEFAULT_PROBLEM)
    parser.add_argument("--starters", type=int, default=3, help="Number of starter agents (3 demo, 9 real)")
    parser.add_argument("--hard-cap", type=float, default=2.00, help="Hard budget cap in USD (default $2.00)")
    parser.add_argument("--depth-dial", type=int, default=3, help="Depth dial bounding fan-out")
    parser.add_argument("--k-probes", type=int, default=3, help="Number of blind probe continuations per step")
    parser.add_argument("--gauntlet-rounds", type=int, default=3, help="Pass-readiness gauntlet rounds (default 3)")
    parser.add_argument("--stream-out", type=str, default="", help="Optional JSONL path to save append-only ArenaFeed")
    parser.add_argument("--html-out", type=str, default="", help="Optional HTML path to export Tron Arena player")
    parser.add_argument("--json", action="store_true", help="Print JSON summary in addition to Tron Arena view")
    parser.add_argument("--mock", action="store_true", help="Force mock backend even if NEBIUS_API_KEY is set")
    args = parser.parse_args()

    client = create_client(use_mock=True if args.mock else None)
    registry = VersionRegistry()
    loop = SelfImprovementLoop(client=client, registry=registry)

    harness = MainHarness(
        client=client,
        starter_agents=args.starters,
        hard_cap_usd=args.hard_cap,
        depth_dial=args.depth_dial,
        k_probes=args.k_probes,
        gauntlet_rounds=args.gauntlet_rounds,
        enable_selection=True,
        version_registry=registry,
        seed_cases=loop.case_bank,
    )
    summary = harness.run(args.problem)

    # Render live Tron Arena from the append-only event stream
    arena_view = harness.render_arena(problem=args.problem)
    print(arena_view)

    # Run Piece 8 post-run self-improvement loop (mining + proposition + sandbox + version registry)
    improvement = loop.run_post_run_improvement(
        harness.arena_feed.events,
        problem=args.problem,
        run_sandbox=True,
    )
    summary["self_improvement"] = improvement

    if args.stream_out:
        harness.arena_feed.save_jsonl(Path(args.stream_out))
    if args.html_out:
        harness.arena_renderer.export_html_arena(
            harness.arena_feed.events,
            Path(args.html_out),
            problem=args.problem,
        )
    if args.json:
        print("\n--- RUN SUMMARY JSON ---")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
