"""v0.34: the sandbox yardstick counts survived attacks, not merges a dial can manufacture.

`prefilter_threshold` gates the alignment judge for both verdicts it can return, so it gates the
contradiction check too. Raising it to 0.95 on the mock sandbox skipped a 0.71-similarity clash:
the branch that clash would have killed merged into the winner instead, and the candidate scored
22.5 against the incumbent's 19.5 on `dials.evidence_score` (5 survivals each; merges 5 vs 3).
The reverse rewrite, restoring the clash check, read as a 13% degradation, two points from being
auto-reverted. Merges still feed a winner's certainty and ranking; they no longer move the A/B.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from pfp import MockOpenAIClient
from pfp.dials import DEFAULT_DIALS, evidence_score
from pfp.self_improvement import SCORE_YARDSTICK, SelfImprovementLoop, VersionRegistry

from test_v31_sandbox_yardstick import PROBLEM, _events, _scripted_loop


@pytest.mark.parametrize("merges", [0, 1, 5, 40])
def test_merges_attracted_do_not_move_the_yardstick(merges):
    rec = {"survived_falsifications": 5, "merges_attracted": merges, "open_critic_objections": 1}
    assert evidence_score(rec) == 13.75


def test_yardstick_id_marks_the_incomparable_v1_scores():
    assert SCORE_YARDSTICK == "reference_evidence_v3"


@pytest.fixture(scope="module")
def prefilter_runs() -> Dict[float, Dict[str, Any]]:
    loop = SelfImprovementLoop(client=MockOpenAIClient(), registry=VersionRegistry())
    out: Dict[float, Dict[str, Any]] = {}
    for threshold in (float(DEFAULT_DIALS["prefilter_threshold"]), 0.5, 0.95):
        out[threshold] = loop.run_sandbox_test({"prefilter_threshold": threshold}).to_dict()
    return out


def test_hiding_clashes_cannot_buy_a_higher_score(prefilter_runs):
    base = prefilter_runs[float(DEFAULT_DIALS["prefilter_threshold"])]
    blind = prefilter_runs[0.95]
    assert base["coherent"] and blind["coherent"]
    # The run shapes really differ: the blinded candidate's winner absorbed the doomed branch
    assert blind["certainty_score"] > base["certainty_score"]
    assert blind["score"] == base["score"] > 0


def test_restoring_the_clash_check_is_not_scored_as_degradation(prefilter_runs):
    scores = {t: r["score"] for t, r in prefilter_runs.items()}
    assert len(set(scores.values())) == 1, scores


def test_paired_ab_ties_a_clash_hiding_candidate():
    loop, registry = _scripted_loop({"prefilter_threshold": 0.95})
    res = loop.run_post_run_improvement(_events(), problem=PROBLEM, run_sandbox=True)
    sb = res["sandbox_result"]
    assert sb["baseline_source"] == "paired_sandbox"
    assert sb["score_yardstick"] == SCORE_YARDSTICK
    assert sb["score"] == sb["baseline_score"] == sb["baseline_sandbox"]["score"]
    assert res["auto_revert_check"]["reverted"] is False


def test_mined_live_baseline_ignores_merges_too():
    loop, _ = _scripted_loop({})
    metrics = loop.miner.mine(_events())
    for shape in metrics.winning_branch_shapes:
        assert shape["evidence_score"] == evidence_score({**shape, "merges_attracted": 0})
