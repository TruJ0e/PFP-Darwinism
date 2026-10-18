import json

import pytest

import run_arena
from pfp.backend import MockOpenAIClient
from pfp.self_improvement import SelfImprovementLoop, VersionRegistry


def _bank_with_one_case(directory):
    path = directory / "case_bank.json"
    loop = SelfImprovementLoop(client=MockOpenAIClient())
    loop.case_bank = [{
        "problem": "one-way SSH failure",
        "fork_shape": "inspect the inbound firewall",
        "solution": "allow TCP/22",
        "is_merged_branch": False,
        "certainty_score": 7.5,
        "wins": 2,
    }]
    loop.save_case_bank(path)
    return path


@pytest.mark.parametrize("field,value", [
    ("case_count", 0), ("cases", []), ("format", "pfp-case-bank/999"),
])
def test_v2_case_bank_detects_control_plane_tampering(tmp_path, field, value):
    path = _bank_with_one_case(tmp_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw[field] = value
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        SelfImprovementLoop(client=MockOpenAIClient()).load_case_bank(path)


def test_case_bank_count_is_validated_even_with_recomputed_digest(tmp_path):
    path = _bank_with_one_case(tmp_path)
    loop = SelfImprovementLoop(client=MockOpenAIClient())
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["case_count"] = 99
    raw["integrity_sha256"] = loop._case_bank_digest(raw)
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="case_count"):
        loop.load_case_bank(path)


def test_legacy_case_bank_loads_and_upgrades_on_save(tmp_path):
    path = tmp_path / "case_bank.json"
    path.write_text(json.dumps([{"problem": "p", "fork_shape": "f", "wins": 1}]), encoding="utf-8")
    loop = SelfImprovementLoop(client=MockOpenAIClient())
    assert loop.load_case_bank(path) == 1
    loop.save_case_bank(path)
    assert json.loads(path.read_text(encoding="utf-8"))["format"] == "pfp-case-bank/2"


def test_cli_rejects_tampered_recall_before_client_creation(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    VersionRegistry().save(state / "registry.json")
    bank = _bank_with_one_case(state)
    raw = json.loads(bank.read_text(encoding="utf-8"))
    raw["cases"][0]["solution"] = "injected future prompt"
    bank.write_text(json.dumps(raw), encoding="utf-8")

    def must_not_be_called(*args, **kwargs):
        raise AssertionError("client created despite a corrupt recall bank")

    monkeypatch.setattr(run_arena, "create_client", must_not_be_called)
    assert run_arena.main(["--mock", "--quiet", "--state-dir", str(state)]) == 1


def test_new_registry_uses_full_code_snapshot_hashes():
    hashes = VersionRegistry().get_version().code_snapshot_hashes
    assert hashes and all(len(digest) == 64 for digest in hashes.values())
