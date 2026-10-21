"""v0.41: the judge ladder — first-pass verdicts on a fast instruct model, Super for appeals,
the final gauntlet round and the pre-exit check.

Every judge was Super through v0.40. Live, a Super verdict took ~10-50 s (it reasons before it
answers), judging was 55-68% of spend, and run 3 lost 3 of 4 attacks to a JSON verdict buried
under the reasoning. The ladder moves first-pass judging to a fast tier; a kill there still gets
its appeal heard by Super, and nothing is crowned Pass or exits without surviving Super.
"""

from __future__ import annotations

import json
import types
from typing import List, Tuple

import run_arena
from pfp import (
    FAST_JUDGE_MODEL,
    SUPER_MODEL,
    BranchState,
    FailJudge,
    JudgeLadder,
    MainHarness,
    MockOpenAIClient,
    PFPTracker,
)
from pfp.alignment_detector import AlignmentDetector
from pfp.backend import (
    MODEL_PRICING_PER_1M,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    CompletionUsage,
    estimate_call_cost_usd,
)

SANDBOX_PROBLEM = (
    "A small business's SSH works laptop->desktop but not desktop->laptop. "
    "Diagnose the root causes and verify the solution."
)


def _response(content: str) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="scripted",
        model="scripted",
        choices=[Choice(index=0, message=ChatMessage(role="assistant", content=content))],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20, cost_usd=0.001),
    )


class VerdictClient:
    """Answers every falsification with `survived` per model; records (model, user prompt)."""

    def __init__(self, survives: dict):
        self.survives = survives
        self.calls: List[Tuple[str, str]] = []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, messages, model=None, **kwargs):
        self.calls.append((model, messages[-1]["content"]))
        survived = self.survives.get(model, True)
        return _response(json.dumps({"survived": survived, "objection": "" if survived else "hole", "rationale": "r"}))


def _judge(survives: dict, rounds: int = 3) -> Tuple[FailJudge, PFPTracker, VerdictClient]:
    tracker = PFPTracker()
    tracker.register_branch("b", parent=None, budget_remaining=1.0, fork_under_exploration="sshd off on laptop")
    client = VerdictClient(survives)
    return FailJudge(client, tracker, gauntlet_rounds=rounds), tracker, client


def test_fast_tier_is_priced_in_the_catalog():
    assert MODEL_PRICING_PER_1M[FAST_JUDGE_MODEL] == (0.20, 0.60)
    # Cheaper than Super, so the meter's unknown-model fallback (the max rates) stays Super's
    assert estimate_call_cost_usd(1000, 1000, FAST_JUDGE_MODEL) < estimate_call_cost_usd(1000, 1000, SUPER_MODEL)


def test_ladder_routes_each_trigger_to_its_tier():
    ladder = JudgeLadder()
    first, final = ladder.first_pass_model, ladder.final_model
    assert (first, final) == (FAST_JUDGE_MODEL, SUPER_MODEL)
    for trigger in ("background_sample", "post_merge", "contradiction", "dependent_reverify"):
        assert ladder.falsifier_model(trigger, 3) == first
    assert [ladder.falsifier_model(f"pass_gauntlet_round_{r}", 3) for r in (1, 2, 3)] == [first, first, final]
    assert ladder.falsifier_model("pre_exit", 3) == final
    assert ladder.falsifier_model("background_sample", 3, is_appeal=True) == final
    # A one-round gauntlet is all crowning round; a malformed round number fails up to Super
    assert ladder.falsifier_model("pass_gauntlet_round_1", 1) == final
    assert ladder.falsifier_model("pass_gauntlet_round_x", 3) == final
    assert JudgeLadder.super_only().falsifier_model("background_sample", 3) == SUPER_MODEL


def test_gauntlet_is_a_climb_ending_on_super():
    judge, tracker, client = _judge({})
    outcomes = judge.run_pass_readiness_gauntlet("b")
    assert [o.survived for o in outcomes] == [True, True, True]
    assert [m for m, _ in client.calls] == [FAST_JUDGE_MODEL, FAST_JUDGE_MODEL, SUPER_MODEL]
    assert [o.judge_model for o in outcomes] == [FAST_JUDGE_MODEL, FAST_JUDGE_MODEL, SUPER_MODEL]
    assert tracker.get("b").state == BranchState.PASS
    assert [p.judge_id for p in tracker.get("b").pass_provenance] == [
        "fast_judge_primary", "fast_judge_primary", "super_judge_primary",
    ]


def test_crowning_requires_super():
    # The fast tier passes everything; Super rejects (primary and appeal): no Pass
    judge, tracker, client = _judge({FAST_JUDGE_MODEL: True, SUPER_MODEL: False})
    outcomes = judge.run_pass_readiness_gauntlet("b")
    assert [o.survived for o in outcomes] == [True, True, False]
    assert tracker.get("b").state == BranchState.FAIL


def test_first_pass_kill_is_appealed_to_super():
    judge, tracker, client = _judge({FAST_JUDGE_MODEL: False, SUPER_MODEL: True})
    out = judge.falsify_branch("b", schedule_trigger="background_sample")
    assert out.survived and out.appeal_overturned
    assert out.judge_model == FAST_JUDGE_MODEL
    assert out.appeal_judge_id == "super_judge_appeal_secondary"
    assert [m for m, _ in client.calls] == [FAST_JUDGE_MODEL, SUPER_MODEL]
    assert "appeal_review=true" in client.calls[1][1]


def test_super_only_ladder_restores_the_old_routing():
    tracker = PFPTracker()
    tracker.register_branch("b", parent=None, budget_remaining=1.0, fork_under_exploration="x")
    client = VerdictClient({})
    FailJudge(client, tracker, ladder=JudgeLadder.super_only()).run_pass_readiness_gauntlet("b")
    assert {m for m, _ in client.calls} == {SUPER_MODEL}


def test_alignment_judging_is_first_pass():
    tracker = PFPTracker()
    for bid in ("a", "b"):
        tracker.register_branch(bid, parent=None, budget_remaining=0.1, fork_under_exploration="sshd tcp 22")
    seen: List[str] = []

    def create(messages, model=None, **kwargs):
        seen.append(model)
        if messages[-1]["content"].startswith("extract_claims"):
            return _response(json.dumps({"core_claims": ["c"]}))
        return _response(json.dumps({"verdict": "unrelated"}))

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    AlignmentDetector(client).judge_pair(tracker.get("a"), tracker.get("b"))
    assert seen and set(seen) == {FAST_JUDGE_MODEL}
    # The pre-ladder keyword still pins a model
    seen.clear()
    AlignmentDetector(client, super_model=SUPER_MODEL).judge_pair(tracker.get("a"), tracker.get("b"))
    assert set(seen) == {SUPER_MODEL}


def test_mock_arena_bills_both_tiers_and_reports_the_ladder():
    harness = MainHarness(client=MockOpenAIClient(), starter_agents=3, hard_cap_usd=2.0)
    summary = harness.run(SANDBOX_PROBLEM)
    assert summary["halted_reason"] is None
    assert summary["winners_box"]
    by_model = summary["budget"]["api_spend_by_model"]
    assert by_model.get(FAST_JUDGE_MODEL, 0.0) > 0.0
    assert by_model.get(SUPER_MODEL, 0.0) > 0.0
    assert summary["judge_ladder"] == JudgeLadder().to_dict()
    # Every winner's last gauntlet survival was ruled on by Super
    for w in summary["winners_box"]:
        gauntlet = [p for p in w["pass_provenance"] if p["survived_what"].startswith("pass_gauntlet_round_3")]
        assert gauntlet and all(p["judge_id"].startswith("super_judge") for p in gauntlet)


def test_cli_judge_ladder_flags(monkeypatch):
    captured = {}
    real = run_arena.MainHarness

    def spy(**kwargs):
        captured["ladder"] = kwargs.get("judge_ladder")
        return real(**kwargs)

    monkeypatch.setattr(run_arena, "MainHarness", spy)
    assert run_arena.main(["--mock", "--quiet", "--no-improve", "--judge-ladder", "super"]) == 0
    assert captured["ladder"] == JudgeLadder.super_only()
    assert run_arena.main(["--mock", "--quiet", "--no-improve", "--first-pass-judge", "some/model"]) == 0
    assert captured["ladder"] == JudgeLadder(first_pass_model="some/model")
