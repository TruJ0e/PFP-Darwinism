"""v0.13: recall precision — token-bounded concept anchors, reload-stable case dedup,
similarity-weighted seeding, and an honest k=0 recall."""

from __future__ import annotations

from pfp import MainHarness, MockOpenAIClient
from pfp.alignment_detector import compute_branch_embedding, cosine_similarity
from pfp.branch_spawner import _seed_weight
from pfp.self_improvement import CASE_PROBLEM_LIMIT, MinedArenaMetrics, SelfImprovementLoop


def _metrics(fork: str, certainty: float = 5.0, solution: str = "fix") -> MinedArenaMetrics:
    return MinedArenaMetrics(
        total_events=0,
        total_branches=1,
        winning_branch_shapes=[{
            "fork_shape": fork,
            "solution": solution,
            "is_merged_branch": False,
            "certainty_score": certainty,
        }],
        merge_patterns={},
        falsifier_angles={},
        spend_efficiency={},
    )


def test_concept_anchors_do_not_fire_on_substrings():
    unrelated = compute_branch_embedding("monkey report from the ghost author")
    ssh = compute_branch_embedding("open port on the ssh host key auth")
    # Old substring anchors scored this pair ~0.59 — far past the 0.20 judge threshold
    assert cosine_similarity(unrelated, ssh) < 0.20


def test_concept_anchors_still_link_whole_word_matches():
    a = compute_branch_embedding("firewall drops inbound tcp")
    b = compute_branch_embedding("sshd listening but ufw blocks it")
    assert cosine_similarity(a, b) >= 0.20


def test_repeat_win_after_reload_counts_instead_of_duplicating(tmp_path):
    problem = "x" * (CASE_PROBLEM_LIMIT + 50)
    fork = "inspect   the\n inbound firewall"
    loop = SelfImprovementLoop(client=MockOpenAIClient())
    loop.record_wins_for_recall(_metrics(fork, 4.0), problem)
    path = loop.save_case_bank(tmp_path / "case_bank.json")

    reloaded = SelfImprovementLoop(client=MockOpenAIClient())
    reloaded.load_case_bank(path)
    added = reloaded.record_wins_for_recall(_metrics(fork, 6.0, solution="allow 22"), problem)

    assert added == []
    assert len(reloaded.case_bank) == 1
    case = reloaded.case_bank[0]
    assert case["wins"] == 2
    assert case["certainty_score"] == 6.0
    assert case["solution"] == "allow 22"


def test_weaker_repeat_win_keeps_the_better_solution():
    loop = SelfImprovementLoop(client=MockOpenAIClient())
    loop.record_wins_for_recall(_metrics("fork", 6.0, solution="best"), "p")
    loop.record_wins_for_recall(_metrics("fork", 2.0, solution="worse"), "p")
    assert loop.case_bank[0]["solution"] == "best"
    assert loop.case_bank[0]["wins"] == 2


def test_recalled_case_is_sanitized_before_it_enters_the_bank():
    loop = SelfImprovementLoop(client=MockOpenAIClient())
    loop.record_wins_for_recall(_metrics("fork", float("nan")), "p")
    loop.record_wins_for_recall(_metrics("   ", 3.0), "p")
    assert len(loop.case_bank) == 1
    assert loop.case_bank[0]["certainty_score"] == 0.0


def test_recall_with_k_zero_returns_nothing():
    loop = SelfImprovementLoop(client=MockOpenAIClient())
    loop.case_bank = [{"problem": "p", "fork_shape": "a b", "certainty_score": 1.0, "wins": 1}]
    assert loop.recall("p", k=0) == []
    assert len(loop.recall("p", k=1)) == 1


def test_seed_weight_scales_certainty_by_similarity():
    assert _seed_weight({"certainty_score": 8.0, "similarity": 0.25}) == 2.0
    assert _seed_weight({"certainty_score": 8.0}) == 8.0  # seeded directly: exact match
    for bad in (float("nan"), "x", True, None):
        assert _seed_weight({"certainty_score": 8.0, "similarity": bad}) == 0.0
    assert _seed_weight({"certainty_score": 8.0, "similarity": 7.0}) == 8.0


def test_close_recall_gets_a_bigger_starter_budget_than_a_distant_one():
    seeds = [
        {"fork_shape": "near match", "certainty_score": 40.0, "similarity": 1.0},
        {"fork_shape": "distant match", "certainty_score": 40.0, "similarity": 0.15},
    ]
    harness = MainHarness(
        client=MockOpenAIClient(), starter_agents=2, hard_cap_usd=1.0, seed_cases=seeds,
    )
    harness.budget_ledger.exploration_pool_usd = 0.5
    harness._spawn_starters("problem")
    near, distant = (s.directive.budget for s in harness.starters)
    assert near > distant
