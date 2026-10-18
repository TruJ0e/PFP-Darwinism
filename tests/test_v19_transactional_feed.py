"""v0.19: journal appends are transactional and projections cannot break custody."""

from __future__ import annotations

import json

import pytest

from pfp import ArenaFeed


class _Uncopyable:
    def __deepcopy__(self, memo: object) -> object:
        raise RuntimeError("cannot copy")


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"timestamp": "not-a-time"}, ValueError),
        ({"details": {"bad": _Uncopyable()}}, RuntimeError),
        ({"snapshot": {"id": "b1", "state": "Possible", "bad": _Uncopyable()}}, RuntimeError),
    ],
)
def test_failed_append_never_consumes_a_sequence(kwargs: dict, error: type[Exception]) -> None:
    feed = ArenaFeed()
    snapshot = kwargs.pop("snapshot", {"id": "b1", "state": "Possible"})

    with pytest.raises(error):
        feed.append("spawn", "b1", snapshot, **kwargs)

    event = feed.append("spawn", "b1", {"id": "b1", "state": "Possible"}, timestamp=1)
    assert event.seq == 1
    assert len(feed) == 1
    assert ArenaFeed.from_jsonl(feed.to_jsonl()).events[0].seq == 1


def test_subscriber_failure_does_not_roll_back_or_escape_committed_event() -> None:
    feed = ArenaFeed()
    delivered: list[int] = []

    def broken(_event: object) -> None:
        raise RuntimeError("display offline")

    feed.subscribe(broken)
    feed.subscribe(lambda event: delivered.append(event.seq))

    first = feed.append("spawn", "b1", {"id": "b1", "state": "Possible"}, timestamp=1)
    second = feed.append("conclude", "b1", {"id": "b1", "state": "Possible"}, timestamp=2)

    assert (first.seq, second.seq) == (1, 2)
    assert delivered == [1, 2]
    assert [failure["seq"] for failure in feed.subscriber_failures] == [1, 2]
    assert all(failure["error_type"] == "RuntimeError" for failure in feed.subscriber_failures)
    assert len(ArenaFeed.from_jsonl(feed.to_jsonl()).events) == 2


def test_subscriber_diagnostics_are_detached_and_subscribers_must_be_callable() -> None:
    feed = ArenaFeed()
    with pytest.raises(TypeError):
        feed.subscribe(None)  # type: ignore[arg-type]

    feed.subscribe(lambda _event: (_ for _ in ()).throw(ValueError("boom")))
    feed.append("spawn", "b1", {"id": "b1", "state": "Possible"}, timestamp=1)
    diagnostics = feed.subscriber_failures
    diagnostics[0]["error"] = "changed"
    assert feed.subscriber_failures[0]["error"] == "boom"


@pytest.mark.parametrize("field", ["event_type", "branch_id"])
def test_replay_rejects_empty_identity_fields_even_with_a_valid_chain(field: str) -> None:
    feed = ArenaFeed()
    event = feed.append("spawn", "b1", {"id": "b1", "state": "Possible"}, timestamp=1).to_dict()
    event[field] = ""
    # Downgrade solely to exercise schema custody independently of the v2 digest check.
    event.update({"journal_version": 1, "previous_digest": "", "digest": ""})

    with pytest.raises(ValueError, match=f"empty {field}"):
        ArenaFeed.from_jsonl(json.dumps(event))
