"""v0.33: survival credit belongs to the text that survived; a merged child earns its own.

Before v0.33 a merge child inherited max(parent survivals). Its text is a new synthesis no judge
had seen, yet it started with the parents' credit: +3 certainty per inherited survival, skipped by
the background falsifier's "untested first" pass, and the mock demo's winner showed "Survived: 8"
over a proof trail listing 5 tests (certainty 33 instead of 24). Each round of merging compounded it.
"""

from __future__ import annotations

from pfp import BranchState, BudgetLedger, MergeSynthesizer, PFPTracker
from pfp.backend import MockOpenAIClient
from pfp.branch_spawner import MainHarness
from pfp.dials import evidence_score
from pfp.tracker import PassProvenanceRecord

from test_v26_merge_custody import _CountingClient

SANDBOX_PROBLEM = (
    "A small business's SSH works laptop->desktop but not desktop->laptop. "
    "Diagnose the root causes and verify the solution."
)


def _tested(tracker: PFPTracker, branch_id: str, n: int) -> None:
    rec = tracker.get(branch_id)
    for i in range(n):
        rec.pass_provenance.append(
            PassProvenanceRecord(
                survived_what=f"background_sample:logic#{i}",
                timestamp=0.0,
                falsifier_angle="logic",
                judge_id="super_judge_primary",
            )
        )
    tracker.update_branch(branch_id, survived_falsifications=n, merges_attracted=1, tested=True)


def test_merge_child_starts_with_no_survival_credit() -> None:
    tracker = PFPTracker()
    tracker.register_branch("p1", parent="root", budget_remaining=0.05, fork_under_exploration="firewall TCP/22")
    tracker.register_branch("p2", parent="root", budget_remaining=0.05, fork_under_exploration="sshd listening")
    _tested(tracker, "p1", 4)
    _tested(tracker, "p2", 2)
    synth = MergeSynthesizer(client=_CountingClient(), tracker=tracker, budget_ledger=BudgetLedger())

    synth.merge_branches("p1", "p2", "m1")

    child = tracker.get("m1")
    assert child.state == BranchState.POSSIBLE
    assert child.survived_falsifications == 0
    assert child.pass_provenance == []
    # Cooperation is still credited, and the parents keep the survivals they earned
    assert child.merges_attracted == 3
    assert tracker.get("p1").survived_falsifications == 4
    assert tracker.get("p2").survived_falsifications == 2
    assert evidence_score(child.to_dict()) == evidence_score(
        {"survived_falsifications": 0, "merges_attracted": 3, "open_critic_objections": 0}
    )


def test_every_branch_survival_count_matches_its_own_proof_trail() -> None:
    harness = MainHarness(
        client=MockOpenAIClient(),
        starter_agents=3,
        hard_cap_usd=2.0,
        depth_dial=3,
        k_probes=3,
        gauntlet_rounds=3,
        enable_selection=True,
    )
    summary = harness.run(SANDBOX_PROBLEM)
    records = harness.tracker.all_records()
    assert any(r.origin == "merge" for r in records), "mock arena produced no merge"
    for rec in records:
        assert rec.survived_falsifications == len(rec.pass_provenance), rec.id
    assert summary["winners_box"], "mock arena produced no winner"
    for w in summary["winners_box"]:
        assert w["survived_falsifications"] == len(w["pass_provenance"]), w["id"]
    merged = [w for w in summary["winners_box"] if w.get("merge_parents")]
    assert merged, "expected a merged winner on the mock demo"
    # Gauntlet (3) + post-merge check + pre-exit check: tests on the merged text itself
    assert all(w["survived_falsifications"] == 5 for w in merged)
