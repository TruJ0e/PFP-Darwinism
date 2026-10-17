"""
v0.7: complete stream-to-ledger spend accounting parity, registry dial & prompt
validation on both live rewrites and disk loads, stale local PID lock reclamation,
operator failsafe CLI commands (--pin-version, --unpin, --restore-version,
--list-versions), replay summary/self-improvement, HTML player winner eviction parity,
and coerce_bool / run_phase1 CLI hardening.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import socket

import pytest

import run_arena
import run_phase1
from pfp import (
    EventLogMiner,
    MainHarness,
    MockOpenAIClient,
    StateLockError,
    VersionRegistry,
    build_arena_html,
    coerce_bool,
    exclusive_lock,
)


def _run_arena_io(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = run_arena.main(argv)
    return code, out.getvalue(), err.getvalue()


# ---------------------------------------------------------------------------
# 1. coerce_bool rejects non-finite floats (NaN, +inf, -inf)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
def test_coerce_bool_rejects_non_finite_floats(bad_val):
    assert coerce_bool(bad_val, default=False) is False
    assert coerce_bool(bad_val, default=True) is True
    assert coerce_bool(bad_val) is None


def test_coerce_bool_preserves_valid_numeric_and_string_coercions():
    assert coerce_bool(1, default=False) is True
    assert coerce_bool(0, default=True) is False
    assert coerce_bool(0.0, default=True) is False
    assert coerce_bool(2.5, default=False) is True
    assert coerce_bool("yes", default=False) is True
    assert coerce_bool("falsified", default=True) is False


# ---------------------------------------------------------------------------
# 2. Stream-mined spend matches ledger spend (problem_sizing_probe + merge_synthesis)
# ---------------------------------------------------------------------------
def test_stream_mined_spend_matches_ledger_and_api_spend():
    harness = MainHarness(client=MockOpenAIClient(), hard_cap_usd=0.05)
    summary = harness.run("Why is the database query slow on Monday mornings?")
    events = [ev.to_dict() for ev in harness.arena_feed.events]

    mining = EventLogMiner().mine(events)
    mined_total = mining.spend_efficiency["total_spent_usd"]
    ledger_total = summary["budget"]["total_spent_usd"]
    api_total = summary["budget"]["api_spend_usd"]

    assert mined_total == pytest.approx(ledger_total, abs=1e-6)
    assert mined_total == pytest.approx(api_total, abs=1e-6)
    assert mining.spend_efficiency["spend_by_reason"].get("problem_sizing_probe", 0.0) > 0.0

    for m in summary.get("merges", []):
        assert m["synthesis_cost_usd"] > 0.0
    if summary.get("merges"):
        expected_merge_spend = sum(m["synthesis_cost_usd"] for m in summary["merges"])
        assert mining.spend_efficiency["spend_by_reason"].get("merge_synthesis", 0.0) == pytest.approx(
            expected_merge_spend, abs=1e-6
        )


# ---------------------------------------------------------------------------
# 3. VersionRegistry validates dials and prompt lengths on register & load
# ---------------------------------------------------------------------------
def test_version_registry_register_rewrite_rejects_invalid_dials_and_oversized_prompt():
    reg = VersionRegistry()
    cur = reg.get_version()
    with pytest.raises(ValueError, match="invalid config_dials"):
        reg.register_rewrite(
            prompts=cur.prompts,
            config_dials={"unknown_dial": 1},
            proposition={"summary": "inject unknown dial"},
            sandbox_result={"coherent": True},
        )
    with pytest.raises(ValueError, match="invalid config_dials"):
        reg.register_rewrite(
            prompts=cur.prompts,
            config_dials={**cur.config_dials, "certainty_threshold": 99.0},
            proposition={"summary": "out of range threshold"},
            sandbox_result={"coherent": True},
        )
    with pytest.raises(ValueError, match="exceeds maximum length"):
        reg.register_rewrite(
            prompts={"eval_system_suffix": "x" * 4001},
            config_dials=cur.config_dials,
            proposition={"summary": "oversized prompt"},
            sandbox_result={"coherent": True},
        )


def test_version_registry_load_rejects_tampered_dials_even_with_valid_sha256(tmp_path):
    reg = VersionRegistry()
    path = tmp_path / "registry.json"
    reg.save(path)

    raw = json.loads(path.read_text(encoding="utf-8"))
    v0 = raw["versions"][0]
    v0["config_dials"]["certainty_threshold"] = 999.0
    # Recompute both snapshot sha256_hash and registry integrity_sha256 so hash checks alone pass
    v0["sha256_hash"] = VersionRegistry._hash_payload(
        v0["config_dials"],
        v0["prompts"],
        v0["code_snapshot_hashes"],
        v0["proposition"],
        v0["sandbox_result"],
    )
    raw["integrity_sha256"] = VersionRegistry._registry_digest(raw)
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid config_dials"):
        VersionRegistry.load(path)


# ---------------------------------------------------------------------------
# 4. exclusive_lock reclaims stale local PID locks while protecting live ones
# ---------------------------------------------------------------------------
def test_exclusive_lock_reclaims_stale_local_pid_lock(tmp_path):
    lock_file = tmp_path / ".pfp.lock"
    # Dead PID 99999999 on local host using the JSON lock payload format
    lock_file.write_text(
        json.dumps({"pid": 99999999, "host": socket.gethostname(), "started_epoch": 1.0}),
        encoding="utf-8",
    )

    with exclusive_lock(lock_file) as acquired:
        assert acquired == lock_file
        meta = json.loads(lock_file.read_text(encoding="utf-8"))
        assert meta["pid"] == os.getpid()
    assert not lock_file.exists()


def test_exclusive_lock_refuses_foreign_host_or_live_pid_lock(tmp_path):
    lock_file = tmp_path / ".pfp.lock"
    lock_file.write_text(
        json.dumps({"pid": 99999999, "host": "other-remote-host.internal", "started_epoch": 1.0}),
        encoding="utf-8",
    )
    with pytest.raises(StateLockError, match="other-remote-host.internal"):
        with exclusive_lock(lock_file):
            pass

    lock_file.write_text(
        json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "started_epoch": 1.0}),
        encoding="utf-8",
    )
    with pytest.raises(StateLockError):
        with exclusive_lock(lock_file):
            pass


# ---------------------------------------------------------------------------
# 5. Operator failsafe CLI flags (--pin-version, --unpin, --restore-version, --list-versions)
#    plus --replay --summary-out and --replay --state-dir self-improvement
# ---------------------------------------------------------------------------
def test_cli_operator_failsafe_commands_and_replay_extensions(tmp_path):
    state_dir = tmp_path / "state"
    stream_path = tmp_path / "stream.jsonl"
    replay_summary = tmp_path / "replay_summary.json"

    # 1. Run a live mock session that saves stream, then run --replay with --state-dir to evolve v000 -> v002
    code, _, err = _run_arena_io(
        ["--mock", "--quiet", "--no-improve", "--stream-out", str(stream_path)]
    )
    assert code == 0, err

    code, _, err = _run_arena_io(
        [
            "--replay", str(stream_path),
            "--state-dir", str(state_dir),
            "--summary-out", str(replay_summary),
            "--mock",
            "--quiet",
        ]
    )
    assert code == 0, err
    assert replay_summary.exists()
    summary_data = json.loads(replay_summary.read_text(encoding="utf-8"))
    assert summary_data["replay_source"] == str(stream_path)
    assert summary_data["self_improvement"] is not None

    # 2. --list-versions shows all registered versions
    code, out, err = _run_arena_io(["--state-dir", str(state_dir), "--list-versions", "--quiet"])
    assert code == 0, err
    listing = json.loads(out)
    vids = [v["version_id"] for v in listing["versions"]]
    assert len(vids) >= 2
    v_initial, v_latest = vids[0], vids[-1]
    assert listing["active_version_id"] == v_latest

    # 3. --pin-version pins the registry to v_initial (with --list-versions so no arena run is needed)
    code, _, err = _run_arena_io(
        ["--state-dir", str(state_dir), "--pin-version", v_initial, "--list-versions", "--quiet"]
    )
    assert code == 0, err
    reg = VersionRegistry.load(state_dir / "registry.json")
    assert reg.pinned_version_id == v_initial and reg.active_version_id == v_initial

    # 4. --unpin clears the pin
    code, _, err = _run_arena_io(
        ["--state-dir", str(state_dir), "--unpin", "--list-versions", "--quiet"]
    )
    assert code == 0, err
    assert VersionRegistry.load(state_dir / "registry.json").pinned_version_id is None

    # 5. --restore-version switches active version back to v_latest
    code, _, err = _run_arena_io(
        ["--state-dir", str(state_dir), "--restore-version", v_latest, "--list-versions", "--quiet"]
    )
    assert code == 0, err
    assert VersionRegistry.load(state_dir / "registry.json").active_version_id == v_latest

    # 6. Unknown version returns non-zero exit code
    code, _, err = _run_arena_io(["--state-dir", str(state_dir), "--pin-version", "v999", "--quiet"])
    assert code == 1
    assert "unknown version" in err


def test_cli_operator_flags_require_state_dir_and_reject_conflicts(tmp_path):
    with pytest.raises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
        run_arena.main(["--list-versions"])
    with pytest.raises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
        run_arena.main(["--state-dir", str(tmp_path), "--pin-version", "v0", "--unpin"])


# ---------------------------------------------------------------------------
# 6. HTML player stateAt(upto) winner eviction & fallback semantics
# ---------------------------------------------------------------------------
def test_html_player_includes_winner_eviction_and_pass_fallback_logic():
    html = build_arena_html([], problem="Test problem", ascii_snapshot="arena")
    assert 'if (e.event_type === "winner_exit" && s.state === "Pass"' in html
    assert "winners.splice(k, 1);" in html
    assert "if (winners.length === 0 && upto === N)" in html


# ---------------------------------------------------------------------------
# 7. run_phase1.py CLI hardening
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad_argv",
    [
        ["--mock", "--starters", "0"],
        ["--mock", "--hard-cap", "nan"],
        ["--mock", "--hard-cap", "-1"],
        ["--mock", "--hard-cap", "inf"],
        ["--mock", "--problem", "   "],
    ],
)
def test_run_phase1_rejects_invalid_cli_arguments(bad_argv):
    with pytest.raises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
        run_phase1.main(bad_argv)


def test_run_phase1_mock_smoke():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        code = run_phase1.main(["--mock", "--starters", "2", "--depth-dial", "1", "--k-probes", "2"])
    assert code == 0
