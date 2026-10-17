#!/usr/bin/env python3
"""
PFP Darwinism — Phase 1 Runner (Pieces 1–3: Issue Detector, Branch Spawner, PFP Tracker).

Uses NEBIUS_BASE_URL, NEBIUS_MODEL, and NEBIUS_API_KEY from the environment.
When NEBIUS_API_KEY is unset (or --mock is passed), runs against MockOpenAIClient
which implements the exact same OpenAI-compatible interface so the full harness
logic (branching, tracking, budgeting) runs end-to-end without a key.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pfp import MainHarness, create_client

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
    if not math.isfinite(val) or val <= 0:
        raise argparse.ArgumentTypeError("must be a finite number > 0")
    return val


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PFP Darwinism — Phase 1 Harness (Pieces 1-3)")
    parser.add_argument("--problem", type=str, default=DEFAULT_PROBLEM)
    parser.add_argument("--starters", type=_positive_int, default=3, help="Number of starter agents (3 demo, 9 real)")
    parser.add_argument("--hard-cap", type=_positive_float, default=2.00, help="Hard budget cap in USD (default $2.00)")
    parser.add_argument("--depth-dial", type=_positive_int, default=3, help="Depth dial bounding fan-out")
    parser.add_argument("--k-probes", type=_positive_int, default=3, help="Number of blind probe continuations per step")
    parser.add_argument("--mock", action="store_true", help="Force mock backend even if NEBIUS_API_KEY is set")
    args = parser.parse_args(argv)
    if not args.problem or not args.problem.strip():
        parser.error("--problem must be non-empty")

    client = create_client(use_mock=True if args.mock else None)
    harness = MainHarness(
        client=client,
        starter_agents=args.starters,
        hard_cap_usd=args.hard_cap,
        depth_dial=args.depth_dial,
        k_probes=args.k_probes,
        enable_selection=False,
    )
    summary = harness.run(args.problem)
    print(json.dumps(summary, indent=2))
    return 0 if summary["halted_reason"] is None else 2


if __name__ == "__main__":
    sys.exit(main())
