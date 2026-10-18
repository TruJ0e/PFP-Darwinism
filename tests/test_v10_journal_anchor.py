"""v0.10: journal anchoring (truncation / legacy-downgrade detection) and completion-limit cap bounds."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import run_arena
from pfp import (
    AppendOnlyViolationError,
    ArenaFeed,
    HardCapExceededError,
    MeteredClient,
    MockOpenAIClient,
    build_proof_report,
    is_journal_digest,
    journal_anchor,
)
from pfp.backend import (
    UNBOUNDED_COMPLETION_TOKENS,
    completion_token_bound,
    estimate_call_cost_usd,
    prompt_token_upper_bound,
)
from test_v08_live_resilience import MESSAGES, MODEL, _RawServer, _live, _ok


@pytest.fixture(scope="module")
def live_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("v10")
    code = run_arena.main([
        "--mock", "--quiet", "--no-improve",
        "--stream-out", str(out / "run.jsonl"),
        "--summary-out", str(out / "summary.json"),
        "--report-out", str(out / "proof.json"),
    ])
    assert code == 0
    return out


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _as_legacy(lines: list[str], forge: bool = False) -> list[str]:
    out = []
    for line in lines:
        raw = json.loads(line)
        raw.pop("digest")
        raw.pop("previous_digest")
        raw["journal_version"] = 1
        if forge and raw["event_type"] == "pass":
            raw["details"]["forged"] = True
        out.append(json.dumps(raw))
    return out


def test_summary_and_report_carry_the_same_anchor_as_the_journal(live_run):
    feed = ArenaFeed.load_jsonl(live_run / "run.jsonl")
    summary = json.loads((live_run / "summary.json").read_text(encoding="utf-8"))
    report = json.loads((live_run / "proof.json").read_text(encoding="utf-8"))
    assert summary["journal"] == {"events": len(feed), "head_digest": feed.head_digest, "verified_chain": True}
    assert report["journal"] == journal_anchor(feed.events)
    assert is_journal_digest(feed.head_digest)
    # Pure reduction: a replay's report anchors identically
    assert build_proof_report(feed.events)["journal"] == report["journal"]


def test_verify_accepts_an_intact_journal_and_prints_its_anchor(live_run, capsys):
    assert run_arena.main(["--verify", str(live_run / "run.jsonl"),
                           "--expect-head", str(live_run / "summary.json")]) == 0
    printed = json.loads(capsys.readouterr().out)
    summary = json.loads((live_run / "summary.json").read_text(encoding="utf-8"))
    assert printed["head_digest"] == summary["journal"]["head_digest"]
    assert printed["expected_head_matched"] is True
    # A proof report .json and a bare (any-case) digest are equally good anchors
    assert run_arena.main(["--verify", str(live_run / "run.jsonl"), "--expect-head", str(live_run / "proof.json")]) == 0
    assert run_arena.main(["--verify", str(live_run / "run.jsonl"),
                           "--expect-head", summary["journal"]["head_digest"].upper()]) == 0


def test_tail_truncation_passes_the_chain_but_not_the_anchor(live_run, tmp_path):
    cut = _write(tmp_path / "cut.jsonl", _lines(live_run / "run.jsonl")[:-3])
    ArenaFeed.load_jsonl(cut)  # the chain alone cannot see a missing tail
    anchor = str(live_run / "summary.json")
    assert run_arena.main(["--verify", str(cut), "--expect-head", anchor]) == 1
    assert run_arena.main(["--replay", str(cut), "--quiet", "--expect-head", anchor,
                           "--report-out", str(tmp_path / "r.md")]) == 1
    assert not (tmp_path / "r.md").exists()


def test_legacy_downgrade_is_flagged_and_cannot_forge_an_anchored_journal(live_run, tmp_path, capsys):
    lines = _lines(live_run / "run.jsonl")
    forged = _write(tmp_path / "forged.jsonl", _as_legacy(lines, forge=True))
    stripped = _write(tmp_path / "stripped.jsonl", _as_legacy(lines))
    anchor = str(live_run / "summary.json")

    feed = ArenaFeed.load_jsonl(forged)
    assert feed.legacy_unverified and feed.anchor()["verified_chain"] is False
    with pytest.raises(AppendOnlyViolationError):
        ArenaFeed.load_jsonl(forged, allow_legacy=False)

    assert run_arena.main(["--verify", str(forged)]) == 1
    assert run_arena.main(["--replay", str(forged), "--quiet"]) == 0
    assert "legacy v1 journal" in capsys.readouterr().err
    # The anchor authenticates content: the edit changes the re-hashed head ...
    assert run_arena.main(["--verify", str(forged), "--expect-head", anchor]) == 1
    # ... while an unedited legacy copy re-hashes to exactly the live head
    assert run_arena.main(["--verify", str(stripped), "--expect-head", anchor]) == 0


def test_bad_expect_head_values_fail_closed(live_run, tmp_path):
    journal = str(live_run / "run.jsonl")
    assert run_arena.main(["--verify", journal, "--expect-head", "ab" * 32]) == 1
    assert run_arena.main(["--verify", journal, "--expect-head", str(tmp_path / "missing.json")]) == 1
    junk = tmp_path / "junk.json"
    junk.write_text('{"journal": {"head_digest": "nope"}}', encoding="utf-8")
    assert run_arena.main(["--verify", journal, "--expect-head", str(junk)]) == 1
    with pytest.raises(SystemExit):
        run_arena.main(["--mock", "--expect-head", "ab" * 32])
    with pytest.raises(SystemExit):
        run_arena.main(["--verify", journal, "--replay", journal])


def test_completion_bound_honors_both_limit_spellings_and_none():
    assert completion_token_bound(256, 1) == 256
    assert completion_token_bound(64, 2, max_completion_tokens=4096) == 8192
    assert completion_token_bound(None, 1) == UNBOUNDED_COMPLETION_TOKENS
    assert completion_token_bound(None, 1, max_completion_tokens=100) == 100
    assert completion_token_bound("garbage", 1) == UNBOUNDED_COMPLETION_TOKENS
    assert completion_token_bound(10**9, 1) == UNBOUNDED_COMPLETION_TOKENS


def test_meter_reserves_max_completion_tokens_not_the_cheap_default():
    small = estimate_call_cost_usd(prompt_token_upper_bound(MESSAGES), 256, MODEL) * 50.0
    meter = MeteredClient(MockOpenAIClient(), hard_cap_usd=small * 2)
    with pytest.raises(HardCapExceededError):
        meter.chat.completions.create(messages=MESSAGES, max_completion_tokens=100_000)
    with pytest.raises(HardCapExceededError):
        meter.chat.completions.create(messages=MESSAGES, max_tokens=None)
    assert meter.inner.chat.completions._call_count == 0
    assert meter.stats()["in_flight_reserved_usd"] == 0.0


def test_live_client_omits_null_max_tokens_and_bounds_estimated_usage():
    with _RawServer([_ok({"choices": [{"message": {"content": "ok"}}]})]) as srv:
        resp = _live(srv.url, retries=0).chat.completions.create(
            messages=MESSAGES, max_tokens=None, max_completion_tokens=500
        )
    assert "max_tokens" not in srv.bodies[0]
    assert srv.bodies[0]["max_completion_tokens"] == 500
    # Usage omitted: charged at the honored limit, not the 256 default
    assert resp.usage.estimated and resp.usage.completion_tokens == 500
