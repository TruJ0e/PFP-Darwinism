"""
v0.47.1 — briefs fail closed (live run 7 regressions).

Run 7 showed three brief defects, each reproduced here with a mock client that overrides
only the brief calls:
1. Empty content (a reasoning model's 200-token budget eaten by reasoning) silently became
   the boilerplate "Explore distinct hypothesis space N" — now: reasoning-class budgets, a
   journaled `brief_generation_failed` event, one retry, then a brief-less branch.
2. Probe-shaped JSON leaked through as brief text — now: rejected; a fork child whose
   mutation fails runs brief-less, carrying its parent's brief only as an excluded route
   (amendment: divergence at every spawn — never verbatim inheritance).
3. Fork children got template echoes ("Specialized exploration of ... from brief_...") — now:
   a valid model brief is used verbatim and no template exists.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pfp import (
    BRIEF_FAILURE_EVENT,
    BRIEF_MAX_TOKENS,
    BRIEF_RETRY_MAX_TOKENS,
    ExplorationBrief,
    MainHarness,
    MockOpenAIClient,
    compute_brief_id,
    generate_starter_briefs,
    mutate_brief_for_fork,
    revise_brief_for_objection,
    validate_brief_text,
)


PROBLEM = "A small business's SSH works laptop->desktop but not desktop->laptop. Diagnose the root causes and verify the solution."

PROBE_PAYLOAD = json.dumps(
    {
        "fork_id": "diagnostic_ssh_desktop_01",
        "approach": "Check the desktop firewall",
        "reasoning": "Hypothesis 1: Desktop firewall blocks inbound SSH traffic.",
    }
)
# Every shape a probe payload arrived in: bare, as the brief field, fenced, and as a list.
PROBE_SHAPES = [
    PROBE_PAYLOAD,
    json.dumps({"brief": PROBE_PAYLOAD, "focus_area": "x"}),
    json.dumps({"brief": 'Look at "fork_id": "diagnostic_ssh_desktop_01" first.'}),
    "```json\n" + PROBE_PAYLOAD + "\n```",
    json.dumps({"brief": [PROBE_PAYLOAD]}),
]

BOILERPLATE_MARKERS = ("Explore distinct hypothesis space", "Specialized exploration of", "Address objection (")


def _is_brief_prompt(prompt_text: str) -> bool:
    lower = prompt_text.lower()
    return "generate_exploration_brief" in lower or "mutate_exploration_brief" in lower


def _brief_client(starter_reply=None, mutate_reply=None):
    """
    A MockOpenAIClient whose brief calls are answered by `starter_reply` / `mutate_reply`
    (callables taking (prompt, attempt_no) -> content; None keeps the stock mock reply).
    Records the max_tokens of every brief call.
    """
    client = MockOpenAIClient()
    comps = client.chat.completions
    stock_generate = comps._generate_one_content
    stock_create = comps.create
    client.brief_calls = []  # (stage, max_tokens)
    attempts = {}

    def generate(prompt_text, temperature, choice_idx):
        if _is_brief_prompt(prompt_text):
            starter = "generate_exploration_brief" in prompt_text.lower()
            reply = starter_reply if starter else mutate_reply
            if reply is not None:
                attempts[prompt_text] = attempts.get(prompt_text, 0) + 1
                return reply(prompt_text, attempts[prompt_text]), False
        return stock_generate(prompt_text, temperature, choice_idx)

    def create(messages, *args, **kwargs):
        prompt_text = "\n".join(m.get("content", "") for m in messages)
        if _is_brief_prompt(prompt_text):
            stage = "starter" if "generate_exploration_brief" in prompt_text.lower() else "mutate"
            client.brief_calls.append((stage, kwargs.get("max_tokens")))
        return stock_create(messages, *args, **kwargs)

    comps._generate_one_content = generate
    comps.create = create
    return client


def _run(client, starters=3):
    harness = MainHarness(
        client=client,
        starter_agents=starters,
        hard_cap_usd=2.0,
        enable_briefs=True,
        enable_selection=True,
    )
    harness.run(PROBLEM)
    return harness


def _spawns(harness):
    return [e for e in harness.arena_feed.events if e.event_type == "spawn"]


def _failures(harness):
    return [e for e in harness.arena_feed.events if e.event_type == BRIEF_FAILURE_EVENT]


def _assert_no_boilerplate(harness):
    for e in harness.arena_feed.events:
        blob = json.dumps(e.to_dict() if hasattr(e, "to_dict") else e.details, default=str)
        for marker in BOILERPLATE_MARKERS:
            assert marker not in blob, f"boilerplate {marker!r} in {e.event_type} event"


# ---------------------------------------------------------------------------
# A. Budgets
# ---------------------------------------------------------------------------


def test_brief_budgets_are_reasoning_class():
    assert 768 <= BRIEF_MAX_TOKENS <= 1024
    assert BRIEF_MAX_TOKENS <= BRIEF_RETRY_MAX_TOKENS <= 1024
    client = _brief_client()
    _run(client)
    assert client.brief_calls, "no brief calls observed"
    assert all(mt is not None and mt >= 768 for _, mt in client.brief_calls), client.brief_calls


def test_fallback_templates_are_gone_from_the_codebase():
    src = "\n".join(p.read_text(encoding="utf-8") for p in Path(__file__).resolve().parents[1].joinpath("pfp").glob("*.py"))
    for marker in BOILERPLATE_MARKERS:
        assert marker not in src


# ---------------------------------------------------------------------------
# B. Empty content -> journaled failure, one retry, brief-less branch
# ---------------------------------------------------------------------------


def test_empty_starter_briefs_fail_closed_unit():
    client = _brief_client(starter_reply=lambda p, n: "")
    failures = []
    briefs, costs = generate_starter_briefs(client, PROBLEM, count=3, on_failure=failures.append)
    assert briefs == [None, None, None]
    assert len(costs) == 3 and all(c > 0 for c in costs)  # both attempts were paid for
    assert len(failures) == 6
    assert [f["max_tokens"] for f in failures] == [BRIEF_MAX_TOKENS, BRIEF_RETRY_MAX_TOKENS] * 3
    assert [f["will_retry"] for f in failures] == [True, False] * 3
    assert {f["reason"] for f in failures} == {"empty_content"}
    assert {f["stage"] for f in failures} == {"starter"}


def test_empty_starter_briefs_journal_failure_and_spawn_briefless():
    client = _brief_client(starter_reply=lambda p, n: "")
    harness = _run(client)

    for starter in harness.starters:
        assert starter.directive.brief == ""
        assert starter.directive.brief_id == ""
    for e in _spawns(harness):
        assert "brief" not in e.details
        assert "brief" not in e.details.get("directive", {})

    failures = _failures(harness)
    starter_ids = {s.branch_id for s in harness.starters}
    assert len(failures) == 2 * len(harness.starters)
    assert {e.branch_id for e in failures} == starter_ids
    for e in failures:
        assert e.details["stage"] == "starter"
        assert e.details["outcome_on_failure"] == "no_brief"
        assert e.details["reason"] == "empty_content"
    # Every brief call was at a reasoning-class budget, retry at the larger one
    assert sorted({mt for _, mt in client.brief_calls}) == sorted({BRIEF_MAX_TOKENS, BRIEF_RETRY_MAX_TOKENS})
    _assert_no_boilerplate(harness)
    assert harness.arena_feed.anchor()["verified_chain"] is True


def test_retry_recovers_after_one_empty_reply():
    good = json.dumps({"brief": "Map which side initiates each TCP session and where state is kept.", "focus_area": "f"})
    client = _brief_client(starter_reply=lambda p, n: "" if n == 1 else good)
    failures = []
    briefs, _ = generate_starter_briefs(client, PROBLEM, count=1, on_failure=failures.append)
    assert briefs[0] is not None
    assert briefs[0].brief == "Map which side initiates each TCP session and where state is kept."
    assert len(failures) == 1 and failures[0]["will_retry"] is True
    assert [mt for _, mt in client.brief_calls] == [BRIEF_MAX_TOKENS, BRIEF_RETRY_MAX_TOKENS]


def test_empty_fork_mutation_spawns_briefless_child_with_parent_route_excluded():
    client = _brief_client(mutate_reply=lambda p, n: "")
    harness = _run(client)

    starter_briefs = {s.directive.brief_id: s.directive.brief for s in harness.starters}
    child_spawns = [e for e in _spawns(harness) if e.snapshot.get("parent") is not None]
    assert child_spawns
    excluded = [e for e in child_spawns if "excluded_route" in e.details]
    assert excluded, "no brief-less child carried its parent's route as an exclusion"
    for e in child_spawns:
        # Never verbatim inheritance: no child carries a brief of its own here
        assert "brief" not in e.details
        assert "brief" not in e.details["directive"]
    for e in excluded:
        assert e.details["excluded_route"] == starter_briefs[e.details["excluded_route_brief_id"]]

    fork_failures = [e for e in _failures(harness) if e.details["stage"] == "fork_mutation"]
    assert fork_failures
    assert {e.details["outcome_on_failure"] for e in fork_failures} == {"briefless_parent_route_excluded"}
    _assert_no_boilerplate(harness)
    assert harness.arena_feed.anchor()["verified_chain"] is True


# ---------------------------------------------------------------------------
# C. Probe-shaped JSON never becomes brief text
# ---------------------------------------------------------------------------


def test_validate_brief_text_rejects_payloads_and_caps_length():
    for shape in (PROBE_PAYLOAD, "  {\"x\": 1}", "[1, 2]", "```json {}```", "", "   ", None, 42):
        text, reason = validate_brief_text(shape)
        assert text is None and reason
    text, reason = validate_brief_text('Check "fork_id": "abc" in the trace.')
    assert text is None and reason == "payload_markers_in_brief"
    text, _ = validate_brief_text("word " * 400)
    assert text is not None and len(text) <= 600 and not text.endswith(" ")
    text, _ = validate_brief_text("  Inspect   the return path.\n")
    assert text == "Inspect the return path."


def test_probe_shaped_mutation_replies_never_become_briefs_unit():
    parent = ExplorationBrief(
        brief_id=compute_brief_id("Investigate stateful filtering on the return path."),
        brief="Investigate stateful filtering on the return path.",
    )
    for shape in PROBE_SHAPES:
        client = _brief_client(mutate_reply=lambda p, n, s=shape: s)
        failures = []
        child, _ = mutate_brief_for_fork(
            client=client,
            problem=PROBLEM,
            parent_brief=parent,
            fork_description="Hypothesis 1: Desktop firewall blocks inbound SSH traffic.",
            fork_reasoning="r",
            depth=1,
            on_failure=failures.append,
        )
        assert child is None, shape
        assert len(failures) == 2


def test_probe_shaped_replies_never_appear_in_spawn_briefs():
    client = _brief_client(
        starter_reply=lambda p, n: PROBE_PAYLOAD,
        mutate_reply=lambda p, n: PROBE_SHAPES[1],
    )
    harness = _run(client)
    for e in _spawns(harness):
        brief = e.details.get("brief", "")
        assert "fork_id" not in brief and not brief.startswith("{")
    assert all(s.directive.brief == "" for s in harness.starters)
    assert _failures(harness)
    _assert_no_boilerplate(harness)

    # Valid starters, probe-shaped mutations: children run brief-less, parent route excluded
    client = _brief_client(mutate_reply=lambda p, n: PROBE_SHAPES[0])
    harness = _run(client)
    starter_texts = {s.directive.brief for s in harness.starters}
    child_spawns = [e for e in _spawns(harness) if e.snapshot.get("parent") is not None]
    assert child_spawns
    for e in child_spawns:
        assert "brief" not in e.details
    excluded = [e.details["excluded_route"] for e in child_spawns if "excluded_route" in e.details]
    assert excluded and set(excluded) <= starter_texts


# ---------------------------------------------------------------------------
# Valid brief JSON is used verbatim
# ---------------------------------------------------------------------------


def test_valid_mutation_brief_is_used_verbatim():
    def reply(prompt, n):
        # Distinct per depth and sibling so content addressing keeps generations apart
        depth = re.search(r"depth=(\d+)", prompt).group(1)
        sib = prompt.count("- Sibling [")
        return json.dumps({"brief": f"Trace the asymmetric return path, angle {depth}.{sib}.", "focus_area": "f"})

    client = _brief_client(mutate_reply=reply)
    harness = _run(client)
    child_spawns = [e for e in _spawns(harness) if e.snapshot.get("parent") is not None and "brief" in e.details]
    assert child_spawns
    for e in child_spawns:
        assert e.details["brief"].startswith("Trace the asymmetric return path, angle ")
        assert e.details["brief_id"] == compute_brief_id(e.details["brief"])
        assert e.details["brief_id"] != e.details["parent_brief_id"]
    assert not _failures(harness)

    parent = ExplorationBrief(brief_id=compute_brief_id("P."), brief="P.")
    client = _brief_client(mutate_reply=lambda p, n: json.dumps({"brief": "  Probe conntrack timeouts.  "}))
    revised, _ = revise_brief_for_objection(client, PROBLEM, parent, objection="o", depth=1)
    assert revised.brief == "Probe conntrack timeouts."
    assert revised.parent_brief_id == parent.brief_id


def test_failed_objection_revision_keeps_current_brief():
    parent = ExplorationBrief(brief_id=compute_brief_id("P."), brief="P.")
    client = _brief_client(mutate_reply=lambda p, n: "")
    failures = []
    revised, _ = revise_brief_for_objection(client, PROBLEM, parent, objection="o", on_failure=failures.append)
    assert revised is parent
    assert [f["stage"] for f in failures] == ["objection_revision"] * 2
