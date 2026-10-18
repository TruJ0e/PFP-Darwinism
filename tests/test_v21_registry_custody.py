"""v0.21: version-registry inputs remain finite, typed, and persistable."""

from __future__ import annotations

import copy
import json
import math

import pytest

from pfp import VersionRegistry


@pytest.mark.parametrize("bad", [math.nan, math.inf, -0.1, 1.01, True, "broken"])
def test_registry_rejects_invalid_degradation_threshold(bad: object) -> None:
    with pytest.raises(ValueError):
        VersionRegistry(degradation_threshold=bad)


@pytest.mark.parametrize(
    "field,value",
    [
        ("proposition", {"score": math.nan}),
        ("proposition", {"items": {1, 2}}),
        ("sandbox_result", {"coherent": "false"}),
        ("sandbox_result", {"coherent": True, "score": math.inf}),
        ("prompts", {"hint": 42}),
        ("prompts", {1: "hint"}),
    ],
)
def test_rewrite_rejects_unpersistable_control_data_transactionally(
    field: str, value: object
) -> None:
    registry = VersionRegistry()
    before = registry.to_dict()
    kwargs = {
        "config_dials": registry.get_version().config_dials,
        "prompts": {"hint": "safe"},
        "proposition": {"summary": "safe"},
        "sandbox_result": {"coherent": True, "score": 1.0},
    }
    kwargs[field] = value

    with pytest.raises(ValueError):
        registry.register_rewrite(**kwargs)

    assert registry.to_dict() == before


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0, True, "broken"])
def test_auto_revert_rejects_poison_scores_without_switching(bad: object) -> None:
    registry = VersionRegistry()
    registry.register_rewrite(
        config_dials=registry.get_version().config_dials,
        prompts=registry.get_version().prompts,
        proposition={"summary": "candidate"},
        sandbox_result={"coherent": True, "score": 5.0, "baseline_score": 10.0},
    )
    active = registry.active_version_id
    failures = copy.deepcopy(registry.failed_experiments)

    with pytest.raises(ValueError):
        registry.evaluate_and_auto_revert(bad, 10.0)
    with pytest.raises(ValueError):
        registry.evaluate_and_auto_revert(5.0, bad)

    assert registry.active_version_id == active
    assert registry.failed_experiments == failures


def test_registered_snapshot_is_immediately_serializable_and_detached() -> None:
    registry = VersionRegistry()
    proposition = {"summary": "candidate", "nested": [1, 2]}
    sandbox = {"coherent": True, "score": 7.0, "baseline_score": 7.0}
    snap = registry.register_rewrite(
        config_dials=registry.get_version().config_dials,
        prompts=registry.get_version().prompts,
        proposition=proposition,
        sandbox_result=sandbox,
    )
    proposition["nested"].append(3)
    sandbox["score"] = 0.0

    payload = registry.to_dict()
    assert snap.proposition["nested"] == [1, 2]
    assert snap.sandbox_result["score"] == 7.0
    assert "NaN" not in json.dumps(payload, allow_nan=False)


def test_loader_rejects_semantically_invalid_but_correctly_rehashed_registry(tmp_path) -> None:
    registry = VersionRegistry()
    raw = registry.to_dict()
    raw["versions"][0]["status"] = "invented"
    snap = raw["versions"][0]
    snap["sha256_hash"] = VersionRegistry._hash_payload(
        snap["config_dials"],
        snap["prompts"],
        snap["code_snapshot_hashes"],
        snap["proposition"],
        snap["sandbox_result"],
    )
    raw["integrity_sha256"] = VersionRegistry._registry_digest(raw)
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown status"):
        VersionRegistry.load(path)
