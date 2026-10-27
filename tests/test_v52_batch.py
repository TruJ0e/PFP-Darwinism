"""v0.52 batch regression tests: brief prose fallback, calibration coverage."""

from __future__ import annotations

from pfp.briefs import _parse_brief_reply
from pfp.fail_judge import _is_possibility_only_objection


def test_prose_brief_reply_falls_back_to_sentences() -> None:
    text, reason = _parse_brief_reply(
        "Explore peer-led token exchanges for transition cues. "
        "Use a shared visual board to signal start and end times."
    )
    assert text is not None and "peer-led token exchanges" in text
    assert reason == ""


def test_truncated_json_brief_never_falls_back() -> None:
    text, reason = _parse_brief_reply('{"brief": "Explore peer cueing via a cust')
    assert text is None
    assert reason == "unparseable_json"


def test_empty_brief_still_fails() -> None:
    text, reason = _parse_brief_reply("")
    assert text is None
    assert reason == "empty_content"


def test_run14_style_possibility_phrasings_convert() -> None:
    assert _is_possibility_only_objection(
        "Students often misestimate time needed for tasks, leading to unrealistic micro-deadlines.",
        "",
    )
    assert _is_possibility_only_objection(
        "assumes a universal benefit without accounting for scenarios where added input exacerbates anxiety",
        "",
    )
    assert _is_possibility_only_objection(
        "assumes a low-stimulus workspace will uniformly reduce sensory overload", ""
    )


def test_formal_logical_errors_still_kill() -> None:
    assert not _is_possibility_only_objection(
        "the conclusion does not logically follow from its stated premise; it commits a non sequitur.",
        "",
    )
    assert not _is_possibility_only_objection(
        "the reasoning commits the logical fallacy of affirming the consequent", ""
    )
    assert not _is_possibility_only_objection(
        "the branch equivocates between perceived control and genuine autonomy", ""
    )
