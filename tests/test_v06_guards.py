"""
v0.6: a hard cap that cannot bind is refused (NaN admits every call, inf never stops one),
the journal names its own problem so a replay reproduces the live proof trail exactly,
and one --state-dir serves one run at a time (no lost update on the failsafe).
"""

from __future__ import annotations

import contextlib
import io
import json
import threading

import pytest

import run_arena
from pfp import (
    AppendOnlyViolationError,
    ArenaFeed,
    BudgetLedger,
    MainHarness,
    MeteredClient,
    MockOpenAIClient,
    SelfImprovementLoop,
    StateLockError,
    build_proof_report,
    exclusive_lock,
    stream_problem,
)


def _quiet_main(argv):
    err = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
        code = run_arena.main(argv)
    return code, err.getvalue()


# ---------------------------------------------------------------------------
# A cap that cannot bind is refused everywhere a cap enters the system
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cap", ["nan", "inf", "-inf", "NaN", "0", "-1"])
def test_cli_rejects_a_hard_cap_that_cannot_bind(cap):
    with pytest.raises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
        run_arena.main(["--mock", "--quiet", "--no-improve", "--hard-cap", cap])


@pytest.mark.parametrize("cap", [float("nan"), float("inf"), -0.5])
def test_metered_client_refuses_non_binding_caps(cap):
    with pytest.raises(ValueError):
        MeteredClient(MockOpenAIClient(), hard_cap_usd=cap)


def test_metered_client_still_allows_an_uncapped_meter():
    meter = MeteredClient(MockOpenAIClient(), hard_cap_usd=None)
    assert meter.hard_cap_usd is None and not meter.cap_reached


@pytest.mark.parametrize("cap", [float("nan"), float("inf"), 0.0])
def test_harness_refuses_non_binding_caps(cap):
    with pytest.raises(ValueError):
        MainHarness(client=MockOpenAIClient(), hard_cap_usd=cap)


@pytest.mark.parametrize("cap", [float("nan"), float("inf")])
def test_ledger_and_sandbox_refuse_non_finite_caps(cap):
    with pytest.raises(ValueError):
        BudgetLedger(hard_cap_usd=cap)
    with pytest.raises(ValueError):
        SelfImprovementLoop(client=MockOpenAIClient(), sandbox_hard_cap_usd=cap)


# ---------------------------------------------------------------------------
# The journal names its own problem
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def custom_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("v06")
    problem = "Why does `git push` hang only on the office VPN? <b>verify</b>"
    code, _ = _quiet_main(
        [
            "--mock", "--quiet", "--no-improve", "--problem", problem,
            "--stream-out", str(out / "run.jsonl"),
            "--report-out", str(out / "proof.json"),
        ]
    )
    assert code == 0
    return out, problem


def test_stream_records_the_problem_on_starter_spawns(custom_run):
    out, problem = custom_run
    events = ArenaFeed.load_jsonl(out / "run.jsonl").events
    assert stream_problem(events) == problem
    starters = [e for e in events if e.event_type == "spawn" and e.snapshot.get("parent") is None
                and e.snapshot.get("origin") == "starter"]
    assert starters and all(e.details.get("problem") == problem for e in starters)
    # Children never repeat it: the stream stays compact.
    assert not [e for e in events if e.event_type == "spawn" and e.snapshot.get("parent") and "problem" in e.details]


def test_replay_reproduces_the_live_report_including_problem(custom_run, tmp_path):
    out, problem = custom_run
    replayed = tmp_path / "replay.json"
    html = tmp_path / "replay.html"
    code, _ = _quiet_main(["--replay", str(out / "run.jsonl"), "--quiet",
                           "--report-out", str(replayed), "--html-out", str(html)])
    assert code == 0
    live = json.loads((out / "proof.json").read_text(encoding="utf-8"))
    rep = json.loads(replayed.read_text(encoding="utf-8"))
    assert rep["problem"] == problem
    live.pop("spend", None)
    assert rep == live
    # The HTML title is recovered too, still escaped as text.
    page = html.read_text(encoding="utf-8")
    assert "office VPN" in page and "<b>verify</b>" not in page


def test_problem_override_still_wins_on_replay(custom_run, tmp_path):
    out, _ = custom_run
    replayed = tmp_path / "r.json"
    code, _ = _quiet_main(["--replay", str(out / "run.jsonl"), "--quiet", "--problem", "renamed",
                           "--report-out", str(replayed)])
    assert code == 0
    assert json.loads(replayed.read_text(encoding="utf-8"))["problem"] == "renamed"


def test_recorded_problem_is_covered_by_the_hash_chain(custom_run):
    out, problem = custom_run
    text = (out / "run.jsonl").read_text(encoding="utf-8")
    assert json.dumps(problem)[1:-1] in text
    forged = text.replace("office VPN", "office WAN", 1)
    with pytest.raises(AppendOnlyViolationError):
        ArenaFeed.from_jsonl(forged)


def test_stream_problem_is_empty_for_legacy_or_foreign_streams():
    assert stream_problem([]) == ""
    legacy = [{"seq": 1, "event_type": "spawn", "branch_id": "b", "timestamp": 1.0,
               "details": {"problem": 7}, "snapshot": {"id": "b", "state": "Possible"}}]
    assert stream_problem(legacy) == ""
    assert build_proof_report(legacy)["problem"] == ""


# ---------------------------------------------------------------------------
# One run per state dir
# ---------------------------------------------------------------------------
def test_held_state_dir_refuses_before_spending(tmp_path, monkeypatch):
    state = tmp_path / "state"
    assert _quiet_main(["--mock", "--quiet", "--no-improve", "--state-dir", str(state)])[0] == 0
    registry_before = (state / "registry.json").read_text(encoding="utf-8")
    assert not (state / ".pfp.lock").exists(), "a finished run must release its lock"

    def must_not_be_called(*args, **kwargs):
        raise AssertionError("client created while another run holds the state dir")

    monkeypatch.setattr(run_arena, "create_client", must_not_be_called)
    with exclusive_lock(state / ".pfp.lock"):
        code, err = _quiet_main(["--mock", "--quiet", "--state-dir", str(state)])
        assert code == 1 and "held by another run" in err and "pid" in err
    assert (state / "registry.json").read_text(encoding="utf-8") == registry_before
    assert not (state / ".pfp.lock").exists()


def test_lock_is_released_when_the_run_refuses(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "registry.json").write_text("{not json", encoding="utf-8")
    code, _ = _quiet_main(["--mock", "--quiet", "--state-dir", str(state)])
    assert code == 1
    assert not (state / ".pfp.lock").exists()


def test_exclusive_lock_admits_exactly_one_concurrent_holder(tmp_path):
    lock = tmp_path / "s" / ".pfp.lock"
    barrier = threading.Barrier(8)
    release = threading.Event()
    results: list = []

    def contend():
        barrier.wait()
        try:
            with exclusive_lock(lock):
                results.append("won")
                release.wait(5)
        except StateLockError:
            results.append("lost")

    threads = [threading.Thread(target=contend) for _ in range(8)]
    for t in threads:
        t.start()
    while len(results) < 8 and any(t.is_alive() for t in threads):
        if results.count("lost") == 7:
            break
        threading.Event().wait(0.01)
    release.set()
    for t in threads:
        t.join(5)
    assert results.count("won") == 1 and results.count("lost") == 7
    assert not lock.exists()
