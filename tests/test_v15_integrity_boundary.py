"""v0.15: fail-closed model parsing and finite journal/ledger custody."""

from __future__ import annotations

import json
import math

import pytest

from pfp import ArenaFeed, PFPTracker
from pfp.jsonutil import extract_json


def test_unterminated_think_block_never_becomes_a_visible_verdict() -> None:
    assert extract_json('<think>Considering {"survived": true}') is None
    assert extract_json('visible preface <think>{"survived": true}') is None


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_tracker_rejects_non_finite_metrics_at_registration_and_update(bad: float) -> None:
    tracker = PFPTracker()
    with pytest.raises(ValueError, match="finite"):
        tracker.register_branch("bad-budget", None, bad, "fork")
    with pytest.raises(ValueError, match="finite"):
        tracker.register_branch("bad-certainty", None, 1.0, "fork", certainty_score=bad)

    tracker.register_branch("good", None, 1.0, "fork")
    with pytest.raises(ValueError, match="finite"):
        tracker.update_branch("good", certainty_score=bad)
    with pytest.raises(ValueError, match="finite"):
        tracker.update_branch("good", budget_remaining=bad)


@pytest.mark.parametrize("container", ["snapshot", "details"])
def test_arena_rejects_non_finite_nested_payloads_without_advancing_sequence(container: str) -> None:
    feed = ArenaFeed()
    snapshot = {"id": "b1", "state": "Possible"}
    details = {}
    (snapshot if container == "snapshot" else details)["nested"] = {"poison": math.nan}

    with pytest.raises(ValueError, match="finite"):
        feed.append("spawn", "b1", snapshot, details, timestamp=1.0)
    assert len(feed) == 0

    event = feed.append("spawn", "b1", {"id": "b1", "state": "Possible"}, timestamp=1.0)
    assert event.seq == 1
    assert "NaN" not in feed.to_jsonl()


def test_replay_rejects_non_standard_non_finite_json_even_for_legacy_stream() -> None:
    raw = {
        "seq": 1,
        "event_type": "spawn",
        "branch_id": "b1",
        "timestamp": 1.0,
        "details": {},
        "snapshot": {"id": "b1", "state": "Possible", "certainty_score": math.inf},
    }
    with pytest.raises(ValueError, match="non-finite"):
        ArenaFeed.from_jsonl(json.dumps(raw))
