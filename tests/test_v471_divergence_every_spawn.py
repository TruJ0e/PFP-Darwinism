"""
v0.47.1 amendment — divergence at EVERY spawn (Truman, 2026-10-21).

A descendant inherits the MAP (what the lineage covered, what died, which objections
landed), never the ROUTE. Run 7 briefed a child with its parent's own hypothesis copied
back. These regressions pin:
(a) no spawn in any mock run carries a child brief identical to its parent's brief;
(b) a failed mutation yields a brief-less child whose journal carries the parent's brief
    only as an exclusion, and whose probes present it as covered ground;
(c) the mutation prompt for a second sibling differs from the first's — the sibling
    briefs are visibly in the generation context, as are the parent's brief, claims and
    objections.
"""

from __future__ import annotations

import json

from pfp import (
    ExplorationBrief,
    MainHarness,
    MockOpenAIClient,
    compute_brief_id,
    mutate_brief_for_fork,
)


PROBLEM = "A small business's SSH works laptop->desktop but not desktop->laptop. Diagnose the root causes and verify the solution."
ALT_PROBLEM = "Database transactions deadlock intermittently under peak load. Diagnose the root causes and verify the fix."

PARENT = ExplorationBrief(
    brief_id=compute_brief_id("Investigate stateful filtering on the return path."),
    brief="Investigate stateful filtering on the return path.",
)


def _recording_client(mutate_reply=None):
    """MockOpenAIClient recording every mutation prompt; `mutate_reply(prompt)` overrides replies."""
    client = MockOpenAIClient()
    comps = client.chat.completions
    stock_generate = comps._generate_one_content
    client.mutation_prompts = []

    def generate(prompt_text, temperature, choice_idx):
        if "mutate_exploration_brief" in prompt_text.lower():
            client.mutation_prompts.append(prompt_text)
            if mutate_reply is not None:
                return mutate_reply(prompt_text), False
        return stock_generate(prompt_text, temperature, choice_idx)

    comps._generate_one_content = generate
    return client


def _run(client, problem=PROBLEM, starters=3):
    harness = MainHarness(
        client=client,
        starter_agents=starters,
        hard_cap_usd=2.0,
        enable_briefs=True,
        enable_selection=True,
    )
    harness.run(problem)
    return harness


def _spawns(harness):
    return [e for e in harness.arena_feed.events if e.event_type == "spawn"]


def _assert_no_child_restates_parent(harness):
    spawns = _spawns(harness)
    brief_by_branch = {e.branch_id: e.details.get("brief", "") for e in spawns}
    checked = 0
    for e in spawns:
        parent = e.snapshot.get("parent")
        if parent is None or not e.details.get("brief"):
            continue
        checked += 1
        assert e.details["brief"] != brief_by_branch.get(parent), e.details
        assert e.details["brief_id"] != e.details.get("parent_brief_id"), e.details
    return checked


# ---------------------------------------------------------------------------
# (a) No child brief identical to its parent's brief, in any mock run
# ---------------------------------------------------------------------------


def test_no_spawn_carries_its_parents_brief_stock_mock():
    for problem, starters in ((PROBLEM, 3), (PROBLEM, 5), (ALT_PROBLEM, 4)):
        harness = _run(MockOpenAIClient(), problem=problem, starters=starters)
        assert _assert_no_child_restates_parent(harness) > 0, "no brief-carrying children to check"
        assert harness.arena_feed.anchor()["verified_chain"] is True


def test_model_that_echoes_parent_brief_never_yields_an_identical_child_brief():
    """A model restating the parent's brief is rejected, journaled, and the child runs brief-less."""

    def echo_parent(prompt):
        line = next(ln for ln in prompt.splitlines() if ln.startswith("Parent brief ["))
        return json.dumps({"brief": line.split(": ", 1)[1], "focus_area": "echo"})

    harness = _run(_recording_client(mutate_reply=echo_parent))
    _assert_no_child_restates_parent(harness)
    children = [e for e in _spawns(harness) if e.snapshot.get("parent") is not None]
    assert children and all("brief" not in e.details for e in children)
    reasons = {
        e.details["reason"]
        for e in harness.arena_feed.events
        if e.event_type == "brief_generation_failed" and e.details["stage"] == "fork_mutation"
    }
    assert reasons == {"duplicates_parent_brief"}


def test_sibling_restating_a_sibling_brief_is_rejected():
    first = json.dumps({"brief": "Map conntrack state on the desktop side.", "focus_area": "f"})
    client = _recording_client(mutate_reply=lambda p: first)
    failures = []
    c1, _ = mutate_brief_for_fork(client, PROBLEM, PARENT, "fork A", "r", depth=1, on_failure=failures.append)
    assert c1 is not None and not failures
    c2, _ = mutate_brief_for_fork(
        client, PROBLEM, PARENT, "fork B", "r", prior_sibling_briefs=[c1], depth=1, on_failure=failures.append
    )
    assert c2 is None
    assert [f["reason"] for f in failures] == ["duplicates_sibling_brief"] * 2


# ---------------------------------------------------------------------------
# (b) Exclusion path: failed mutation -> brief-less child, parent brief only as exclusion
# ---------------------------------------------------------------------------


def test_failed_mutation_yields_briefless_child_with_parent_brief_only_as_exclusion():
    client = _recording_client(mutate_reply=lambda p: "")
    harness = _run(client)

    starters = {s.branch_id: s for s in harness.starters}
    direct = [e for e in _spawns(harness) if e.snapshot.get("parent") in starters]
    assert direct, "no starter forked"
    for e in direct:
        parent = starters[e.snapshot["parent"]]
        # No brief of its own anywhere in the journal entry
        assert "brief" not in e.details and "brief_id" not in e.details
        assert "brief" not in e.details["directive"]
        # The parent's brief appears ONLY under the exclusion keys
        assert e.details["excluded_route"] == parent.directive.brief
        assert e.details["excluded_route_brief_id"] == parent.directive.brief_id
        assert e.details["directive"]["excluded_routes"] == [
            {"brief_id": parent.directive.brief_id, "brief": parent.directive.brief}
        ]

    # In memory too: the child's assignment is empty, the parent's route is covered ground
    children = [c for s in harness.starters for c in s.children if c.directive.excluded_routes]
    assert children
    for child in children:
        assert child.directive.brief == ""
        assert child.directive.brief_id == ""
    assert harness.arena_feed.anchor()["verified_chain"] is True


def test_briefless_child_probe_context_presents_parent_route_as_covered():
    seen = []
    client = MockOpenAIClient()
    comps = client.chat.completions
    stock_generate = comps._generate_one_content

    def generate(prompt_text, temperature, choice_idx):
        lower = prompt_text.lower()
        if "mutate_exploration_brief" in lower:
            return "", False
        if lower.startswith("probe_continuation") or lower.startswith("ns_fanout"):
            seen.append(prompt_text)
        return stock_generate(prompt_text, temperature, choice_idx)

    comps._generate_one_content = generate
    harness = _run(client)
    starter_briefs = {s.directive.brief for s in harness.starters}

    covered = [p for p in seen if "Covered by parent — do not repeat this route: " in p]
    assert covered, "no brief-less child probed with its parent's route as covered ground"
    for p in covered:
        route = next(
            ln.split(": ", 1)[1] for ln in p.splitlines() if ln.startswith("Covered by parent — do not repeat this route: ")
        )
        assert route in starter_briefs
        # Never presented as the child's own assignment
        assert f"Exploration brief: {route}" not in p


# ---------------------------------------------------------------------------
# (c) Divergence inputs: sibling briefs, parent brief, claims and objections in context
# ---------------------------------------------------------------------------


def test_second_sibling_mutation_prompt_includes_first_siblings_brief():
    client = _recording_client()
    c1, _ = mutate_brief_for_fork(
        client, PROBLEM, PARENT, "fork A", "r", depth=1, parent_conclusion="Return traffic is dropped by conntrack."
    )
    c2, _ = mutate_brief_for_fork(
        client,
        PROBLEM,
        PARENT,
        "fork A",
        "r",
        prior_sibling_briefs=[c1],
        depth=1,
        parent_conclusion="Return traffic is dropped by conntrack.",
    )
    assert c1 is not None and c2 is not None and c1.brief != c2.brief
    p1, p2 = client.mutation_prompts
    assert p1 != p2
    assert c1.brief not in p1 and c1.brief in p2
    assert f"- Sibling [{c1.brief_id}]: {c1.brief}" in p2
    for p in (p1, p2):
        assert f"Parent brief [{PARENT.brief_id}] (route already taken by the parent): {PARENT.brief}" in p
        assert "Parent conclusion / surviving claims: Return traffic is dropped by conntrack." in p
        assert "Divergence obligation: assign a route the parent did NOT take" in p


def test_mutation_prompt_carries_parent_objections_and_upstream_exclusions():
    client = _recording_client()
    child, _ = mutate_brief_for_fork(
        client,
        PROBLEM,
        PARENT,
        "fork A",
        "r",
        survived_objection="Ignores asymmetric routing.",
        parent_objections=["Ignores asymmetric routing.", "No evidence the daemon listens."],
        excluded_routes=["Audit the SSH daemon config."],
        depth=2,
    )
    assert child is not None and child.trigger == "falsification_objection"
    assert child.parent_brief_id == PARENT.brief_id
    (prompt,) = client.mutation_prompts
    assert prompt.count("- Ignores asymmetric routing.") == 1
    assert "- No evidence the daemon listens." in prompt
    assert "Routes already covered upstream in this lineage (do not repeat):\n- Audit the SSH daemon config." in prompt


def test_harness_sibling_mutation_prompts_see_earlier_siblings():
    client = _recording_client()
    harness = _run(client)
    with_siblings = [p for p in client.mutation_prompts if "- Sibling [brief_" in p]
    assert with_siblings, "no second-or-later sibling mutation observed"
    # Every parent's mutation prompt carries the parent's own brief as the route already taken
    starter_briefs = {s.directive.brief for s in harness.starters}
    first_gen = [p for p in client.mutation_prompts if "depth=1" in p.splitlines()[0]]
    assert first_gen
    for p in first_gen:
        line = next(ln for ln in p.splitlines() if ln.startswith("Parent brief ["))
        assert line.split("(route already taken by the parent): ", 1)[1] in starter_briefs
