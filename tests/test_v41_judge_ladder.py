"""v0.41: the judge ladder — first-pass judging on a fast model, Super where verdicts become final."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from pfp import FAST_JUDGE_MODEL, SUPER_MODEL, BranchState, BudgetLedger, FailJudge, MainHarness, MockOpenAIClient, PFPTracker
from pfp.backend import MODEL_PRICING_PER_1M

from test_hardening import _response

PROBLEM = "Sandbox verification: SSH works laptop->desktop but not desktop->laptop."
_TRIGGER = re.compile(r"Schedule trigger: (\S+)")
_JUDGE = re.compile(r"Judge ID: (\S+)")


class _RecordingMock(MockOpenAIClient):
    """The mock arena, with every call's (model, judge id, schedule trigger, prompt tag) recorded."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: List[Tuple[str, str, str, str]] = []
        inner = self.chat.completions
        outer = self

        class _Rec:
            def create(self, messages, model: Optional[str] = None, **kwargs):
                text = "\n".join(m.get("content", "") for m in messages)
                trig = _TRIGGER.search(text)
                judge = _JUDGE.search(text)
                user = messages[-1].get("content", "")
                outer.calls.append(
                    (model or outer.model, judge.group(1) if judge else "", trig.group(1) if trig else "", user.split(" ", 1)[0])
                )
                return inner.create(messages, model=model, **kwargs)

        self.chat.completions = _Rec()


def _run(**kwargs) -> Tuple[MainHarness, dict, _RecordingMock]:
    client = _RecordingMock()
    harness = MainHarness(client=client, starter_agents=3, hard_cap_usd=2.0, **kwargs)
    return harness, harness.run(PROBLEM), client


def test_fast_judge_is_priced_in_the_catalog() -> None:
    # An unpriced model falls back to the most expensive rates: the meter would reserve Super
    # prices for every fast-rung call and the hard-cap preflight would refuse cheap work
    assert MODEL_PRICING_PER_1M[FAST_JUDGE_MODEL] == (0.20, 0.60)
    assert FAST_JUDGE_MODEL != SUPER_MODEL


def test_arena_routes_first_pass_judging_to_the_fast_rung() -> None:
    harness, summary, client = _run()
    assert summary["fast_judge_model"] == FAST_JUDGE_MODEL
    assert summary["winners_box"], "ladder arena crowned nobody"
    by_model = summary["budget"]["api_spend_by_model"]
    assert by_model.get(FAST_JUDGE_MODEL, 0.0) > 0.0

    alignment = [c for c in client.calls if c[3] in ("alignment_judge", "extract_claims")]
    assert alignment and all(c[0] == FAST_JUDGE_MODEL for c in alignment)
    merges = [c for c in client.calls if c[3] == "merge_synthesizer"]
    assert merges and all(c[0] == SUPER_MODEL for c in merges)

    attacks = [c for c in client.calls if c[2]]
    for model, judge, trigger, _ in attacks:
        if trigger == "pre_exit" or trigger == "pass_gauntlet_round_3" or "appeal" in judge or "escalation" in judge:
            assert model == SUPER_MODEL, (model, judge, trigger)
        else:
            assert model == FAST_JUDGE_MODEL, (model, judge, trigger)
    assert any(t == "pass_gauntlet_round_1" for _, _, t, _ in attacks)
    assert any(t == "pre_exit" for _, _, t, _ in attacks)


def test_ladder_off_puts_every_judge_back_on_super() -> None:
    harness, summary, client = _run(fast_judge_model=None)
    assert summary["fast_judge_model"] is None
    judged = [c for c in client.calls if c[2] or c[3] in ("alignment_judge", "extract_claims", "merge_synthesizer")]
    assert judged and all(c[0] == SUPER_MODEL for c in judged)
    assert FAST_JUDGE_MODEL not in summary["budget"]["api_spend_by_model"]


def test_winner_provenance_shows_which_rung_judged_each_round() -> None:
    _, summary, _ = _run()
    winner = summary["winners_box"][0]
    judges = {p["survived_what"].split(":", 1)[0]: p["judge_id"] for p in winner["pass_provenance"]}
    assert judges["pass_gauntlet_round_1"] == "fast_judge_primary"
    assert judges["pass_gauntlet_round_3"] == "super_judge_primary"


class _ModelClient:
    """Scripted replies per model; records which model each call went to."""

    def __init__(self, replies: dict) -> None:
        self.replies = {k: list(v) for k, v in replies.items()}
        self.models: List[str] = []
        outer = self

        class _C:
            def create(self, messages, model=None, **kwargs):
                outer.models.append(model)
                queue = outer.replies.get(model, [])
                return _response(queue.pop(0) if queue else "{}")

        self.chat = type("Chat", (), {})()
        self.chat.completions = _C()


def _judge(replies: dict):
    tracker = PFPTracker()
    tracker.register_branch("b1", None, 0.05, "Host firewall blocks inbound TCP/22")
    client = _ModelClient(replies)
    judge = FailJudge(client=client, tracker=tracker, budget_ledger=BudgetLedger(), fast_model=FAST_JUDGE_MODEL)
    return judge, tracker, client


KILL = '{"survived": false, "objection": "x"}'
LIVE = '{"survived": true}'


def test_fast_kill_is_appealed_to_super() -> None:
    judge, tracker, client = _judge({FAST_JUDGE_MODEL: [KILL], SUPER_MODEL: [LIVE]})
    out = judge.falsify_branch("b1", schedule_trigger="background_sample")
    assert client.models == [FAST_JUDGE_MODEL, SUPER_MODEL]
    assert out.survived and out.appeal_overturned
    assert tracker.get("b1").state == BranchState.POSSIBLE


def test_fast_kill_with_appeal_spent_escalates_to_super_before_it_stands() -> None:
    # Without the escalation, a branch whose one appeal was used earlier would die on the fast
    # judge's word alone — the ladder would kill what Super alone might have spared
    judge, tracker, client = _judge({FAST_JUDGE_MODEL: [KILL], SUPER_MODEL: [LIVE]})
    tracker.get("b1").appeal_used = True
    out = judge.falsify_branch("b1", schedule_trigger="background_sample")
    assert client.models == [FAST_JUDGE_MODEL, SUPER_MODEL]
    assert out.survived and out.appeal_judge_id == "super_judge_escalation"
    assert tracker.get("b1").pass_provenance[-1].judge_id == "super_judge_escalation"

    judge, tracker, client = _judge({FAST_JUDGE_MODEL: [KILL], SUPER_MODEL: [KILL]})
    tracker.get("b1").appeal_used = True
    out = judge.falsify_branch("b1", schedule_trigger="background_sample")
    assert not out.survived and tracker.get("b1").state == BranchState.FAIL


def test_verdictless_escalation_kills_nothing() -> None:
    judge, tracker, client = _judge({FAST_JUDGE_MODEL: [KILL], SUPER_MODEL: ["garbled"]})
    tracker.get("b1").appeal_used = True
    out = judge.falsify_branch("b1", schedule_trigger="background_sample")
    assert out.inconclusive and tracker.get("b1").state == BranchState.POSSIBLE


def test_super_kill_with_appeal_spent_is_final_without_escalation() -> None:
    judge, tracker, client = _judge({SUPER_MODEL: [KILL]})
    tracker.get("b1").appeal_used = True
    out = judge.falsify_branch("b1", schedule_trigger="pre_exit")
    assert client.models == [SUPER_MODEL]
    assert not out.survived and tracker.get("b1").state == BranchState.FAIL


def test_gauntlet_climbs_from_fast_to_super() -> None:
    judge, tracker, client = _judge({FAST_JUDGE_MODEL: [LIVE, LIVE], SUPER_MODEL: [LIVE]})
    judge.run_pass_readiness_gauntlet("b1", rounds=3)
    assert client.models == [FAST_JUDGE_MODEL, FAST_JUDGE_MODEL, SUPER_MODEL]
    assert tracker.get("b1").state == BranchState.PASS


def test_judge_without_fast_model_is_all_super() -> None:
    tracker = PFPTracker()
    tracker.register_branch("b1", None, 0.05, "fork")
    client = _ModelClient({SUPER_MODEL: [LIVE, LIVE, LIVE]})
    FailJudge(client=client, tracker=tracker).run_pass_readiness_gauntlet("b1", rounds=3)
    assert client.models == [SUPER_MODEL] * 3
