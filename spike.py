#!/usr/bin/env python3
"""
PFP Darwinism — SPIKE 1: parallel Nano fan-out through Nebius Token Factory.

Proves: N parallel inference calls work, measures latency + cost economics
for PFP branch fan-out. No key = no run; the harness waits for NEBIUS_API_KEY.

    export NEBIUS_API_KEY='...'
    python3 spike.py --calls 50 --workers 20

Endpoint default is the Token Factory OpenAI-compatible base; override with
NEBIUS_BASE_URL if the region-specific endpoint differs at signup time.
"""
import argparse, json, os, statistics, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
BASE_URL = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")
# Live catalog 2026-10-13: Nano $0.06 / $0.24 per 1M tokens in/out.
IN_PER_1M, OUT_PER_1M = 0.06, 0.24

PROMPT = (
    "You are one reasoning branch in a parallel exploration. "
    "Given the problem below, produce exactly one distinct approach as JSON: "
    '{"approach": "<name>", "first_step": "<concrete first step>", "risk": "<biggest risk>"} '
    "Problem: A small business's SSH works laptop->desktop but not desktop->laptop. "
    "List one approach. Keep values under 20 words each."
)

def one_call(i):
    key = os.environ.get("NEBIUS_API_KEY", "")
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 200, "temperature": 0.9,  # high temp => diverse branches
    }).encode()
    req = urllib.request.Request(
        BASE_URL + "/chat/completions", data=body,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.load(r)
        dt = time.time() - t0
        u = d.get("usage", {})
        return {"ok": True, "s": round(dt, 2),
                "in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "s": round(time.time() - t0, 2), "err": str(e)[:120]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=50)
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    if not os.environ.get("NEBIUS_API_KEY"):
        sys.exit("NEBIUS_API_KEY not set — sign up at Token Factory + Builder Program first.")

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(one_call, i) for i in range(a.calls)]
        for f in as_completed(futs):
            results.append(f.result())
    wall = time.time() - t0

    ok = [r for r in results if r["ok"]]
    lat = sorted(r["s"] for r in ok)
    tin = sum(r.get("in", 0) for r in ok)
    tout = sum(r.get("out", 0) for r in ok)
    cost = tin / 1e6 * IN_PER_1M + tout / 1e6 * OUT_PER_1M
    print(json.dumps({
        "model": MODEL,
        "calls": a.calls, "workers": a.workers,
        "succeeded": len(ok), "failed": len(results) - len(ok),
        "wall_s": round(wall, 1),
        "latency_s": {"p50": round(statistics.median(lat), 2) if lat else None,
                      "p95": round(lat[int(0.95 * len(lat))], 2) if lat else None},
        "tokens_in": tin, "tokens_out": tout,
        "cost_usd": round(cost, 4),
        "calls_per_sec": round(len(ok) / wall, 1) if wall else 0,
        "proj_100k_calls_usd": round(cost / max(len(ok), 1) * 100_000, 2),
    }, indent=1))

if __name__ == "__main__":
    main()
