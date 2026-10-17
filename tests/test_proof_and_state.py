"""
v0.4: proof-trail report (pure function of the stream, so live == replay), durable
writes for every persisted artifact, an untrusted case bank that cannot poison the
next run's budget split, and a failsafe registry that refuses to half-load.
"""

from __future__ import annotations

import contextlib
import io
import json
import math

import pytest

import run_arena
from pfp import (
    ArenaFeed,
    BudgetLedger,
    MainHarness,
    MockOpenAIClient,
    SelfImprovementLoop,
    VersionRegistry,
    atomic_write_text,
    build_proof_report,
    render_proof_markdown,
)


def _quiet_main(argv):
    err = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
        code = run_arena.main(argv)
    return code, err.getvalue()


@pytest.fixture(scope="module")
def mock_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    code, _ = _quiet_main(
        [
            "--mock", "--quiet", "--no-improve",
            "--stream-out", str(out / "run.jsonl"),
            "--summary-out", str(out / "summary.json"),
            "--report-out", str(out / "proof.md"),
        ]
    )
    assert code == 0
    return out


def test_report_lists_every_winner_with_its_proof_trail(mock_run):
    summary = json.loads((mock_run / "summary.json").read_text(encoding="utf-8"))
    feed = ArenaFeed.load_jsonl(mock_run / "run.jsonl")
    report = build_proof_report(feed.events, summary=summary)

    assert report["format"] == "pfp-proof-report/1"
    assert report["winners"], "the mock run is expected to certify at least one winner"
    assert {w["id"] for w in report["winners"]} == {w["id"] for w in summary["winners_box"]}
    for w in report["winners"]:
        assert w["state"] == "Pass"
        assert w["survived"], "a winner with no survived falsification is not a winner"
        assert w["origin"] == "starter" or any(a["origin"] == "starter" for a in w["ancestors"])
    assert report["spend"]["arena_api_spend_usd"] == summary["budget"]["api_spend_usd"]

    md = (mock_run / "proof.md").read_text(encoding="utf-8")
    assert md.startswith("# PFP Darwinism — proof trail")
    assert "- Spend: $" in md
    for w in report["winners"]:
        assert f"`{w['id']}`" in md


def test_replay_report_matches_live_report(mock_run, tmp_path):
    replayed = tmp_path / "replayed.json"
    code, _ = _quiet_main(["--replay", str(mock_run / "run.jsonl"), "--quiet", "--report-out", str(replayed)])
    assert code == 0
    live = build_proof_report(ArenaFeed.load_jsonl(mock_run / "run.jsonl").events)
    assert json.loads(replayed.read_text(encoding="utf-8")) == json.loads(json.dumps(live))


def _ev(seq, event_type, bid, snapshot, details=None):
    return {"seq": seq, "event_type": event_type, "branch_id": bid, "timestamp": float(seq + 1),
            "details": details or {}, "snapshot": snapshot}


def test_report_escapes_hostile_model_text():
    hostile = "<script>alert(1)</script> | \n# heading [x](javascript:1) `tick`"
    events = [
        _ev(0, "spawn", "b`1", {"id": "b`1", "parent": None, "state": "Possible", "origin": "starter",
                                "fork_under_exploration": hostile}),
        _ev(1, "winner_exit", "b`1", {"id": "b`1", "parent": None, "state": "Pass", "origin": "starter",
                                      "solution": hostile,
                                      "pass_provenance": [{"survived_what": hostile, "falsifier_angle": "logic",
                                                           "judge_id": "j`x"}]}),
    ]
    md = render_proof_markdown(build_proof_report(events, problem=hostile))
    assert "<script>" not in md
    assert "[x](javascript" not in md and r"\[x\](javascript" in md
    assert "\n# heading" not in md
    assert "`b'1`" in md and "`j'x`" in md  # a backtick in an id cannot close the code span early


def test_report_handles_empty_and_halted_streams():
    md = render_proof_markdown(build_proof_report([]))
    assert "Winner's box (0)" in md and "No claim is certified" in md

    halted = [_ev(0, "halt", "arena", {"id": "arena", "state": "Halted"}, {"reason": "hard_cap", "detail": "cap hit"})]
    report = build_proof_report(halted)
    assert report["halted"]["reason"] == "hard_cap"
    assert "HALTED: hard\\_cap" in render_proof_markdown(report)


def test_ancestry_is_cycle_safe():
    events = [
        _ev(0, "spawn", "a", {"id": "a", "parent": "b", "state": "Possible"}),
        _ev(1, "winner_exit", "b", {"id": "b", "parent": "a", "state": "Pass"}),
    ]
    (winner,) = build_proof_report(events)["winners"]
    assert [a["id"] for a in winner["ancestors"]] == ["a"]


def test_atomic_write_replaces_without_leaving_temp_files(tmp_path):
    target = tmp_path / "nested" / "out.json"
    atomic_write_text(target, "one")
    atomic_write_text(target, "two")
    assert target.read_text(encoding="utf-8") == "two"
    assert sorted(p.name for p in target.parent.iterdir()) == ["out.json"]


def test_budget_split_ignores_non_finite_scores():
    ledger = BudgetLedger(hard_cap_usd=1.0)
    shares = ledger.split_among_siblings(1.0, [float("nan"), float("inf"), 1.0], apply_depth_decay=False)
    assert all(math.isfinite(s) and s > 0 for s in shares)
    assert round(sum(shares), 6) == 1.0


@pytest.mark.parametrize("certainty", [float("inf"), float("nan"), -5.0, "garbage", 1e308])
def test_poisoned_case_bank_cannot_break_the_next_run(tmp_path, certainty):
    poisoned = {"problem": "ssh broken one way", "fork_shape": "check sshd", "certainty_score": certainty, "wins": "x"}
    bank = tmp_path / "case_bank.json"
    bank.write_text(
        json.dumps([poisoned, {"problem": "ssh", "fork_shape": "", "certainty_score": 1}, "not a case"]),
        encoding="utf-8",
    )
    loop = SelfImprovementLoop(client=MockOpenAIClient())
    assert loop.load_case_bank(bank) == 1
    case = loop.case_bank[0]
    assert math.isfinite(case["certainty_score"]) and case["certainty_score"] >= 0
    assert case["wins"] == 1

    # Even a raw, unsanitised seed (e.g. handed in by a library caller) must not break ignition.
    harness = MainHarness(client=MockOpenAIClient(), starter_agents=3, hard_cap_usd=1.0,
                          enable_selection=False, seed_cases=[poisoned])
    summary = harness.run("ssh broken one way")
    assert summary["halted_reason"] is None
    assert math.isfinite(summary["budget"]["total_spent_usd"])


def test_case_bank_is_bounded_and_keeps_the_most_proven(tmp_path):
    loop = SelfImprovementLoop(client=MockOpenAIClient(), max_cases=3)
    loop.case_bank = [
        {"problem": "p", "fork_shape": f"s{i}", "certainty_score": float(i), "wins": 1} for i in range(5)
    ] + [{"problem": "p", "fork_shape": "veteran", "certainty_score": 0.0, "wins": 9}]
    path = tmp_path / "bank.json"
    loop.save_case_bank(path)
    assert loop.load_case_bank(path) == 3
    assert [c["fork_shape"] for c in loop.case_bank] == ["s3", "s4", "veteran"]
    with pytest.raises(ValueError):
        SelfImprovementLoop(client=MockOpenAIClient(), max_cases=0)


@pytest.mark.parametrize(
    "payload",
    ["[]", '{"versions": {}}', '{"versions": [{"sha256_hash": "x"}]}', "{not json"],
)
def test_malformed_registry_refuses_to_load(tmp_path, payload):
    path = tmp_path / "registry.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        VersionRegistry.load(path)


def test_cli_refuses_corrupt_registry_before_spending(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    corrupt = '{"versions": [{"sha256_hash": "x"}]}'
    (state / "registry.json").write_text(corrupt, encoding="utf-8")

    def must_not_be_called(*args, **kwargs):
        raise AssertionError("client created despite a corrupt failsafe registry")

    monkeypatch.setattr(run_arena, "create_client", must_not_be_called)
    code, err = _quiet_main(["--mock", "--quiet", "--state-dir", str(state)])
    assert code == 1 and "refusing to run" in err
    # Left for the operator to inspect — never silently replaced by a fresh registry.
    assert (state / "registry.json").read_text(encoding="utf-8") == corrupt


def test_cli_refuses_corrupt_case_bank(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "case_bank.json").write_text('{"oops": 1}', encoding="utf-8")
    code, err = _quiet_main(["--mock", "--quiet", "--state-dir", str(state)])
    assert code == 1 and "case bank" in err


def test_cli_rejects_negative_retries():
    with pytest.raises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
        run_arena.main(["--mock", "--max-retries", "-1"])


def test_state_dir_round_trip_seeds_next_run(tmp_path):
    state = tmp_path / "state"
    assert _quiet_main(["--mock", "--quiet", "--state-dir", str(state)])[0] == 0
    bank = json.loads((state / "case_bank.json").read_text(encoding="utf-8"))
    assert bank and all(SelfImprovementLoop.sanitize_case(c) == c for c in bank)
    out = tmp_path / "s2.json"
    code, _ = _quiet_main(["--mock", "--quiet", "--no-improve", "--state-dir", str(state), "--summary-out", str(out)])
    assert code == 0
    assert json.loads(out.read_text(encoding="utf-8"))["seeded_starters_count"] >= 1
    assert not [p for p in state.iterdir() if p.name.endswith(".tmp")]
