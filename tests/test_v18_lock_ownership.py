"""v0.18: state locks are atomic, token-owned, and safe against stale cleanup races."""

from __future__ import annotations

import json
import os
import socket

import pytest

from pfp.fsutil import StateLockError, exclusive_lock


def _owner(lock):
    return json.loads((lock / "owner.json").read_text(encoding="utf-8"))


def test_lock_is_a_token_owned_directory_and_cleans_up(tmp_path) -> None:
    lock = tmp_path / ".pfp.lock"
    with exclusive_lock(lock) as acquired:
        assert acquired == lock and lock.is_dir()
        meta = _owner(lock)
        assert meta["pid"] == os.getpid()
        assert meta["host"] == socket.gethostname()
        assert len(meta["token"]) == 32
    assert not lock.exists()


def test_cleanup_never_removes_replacement_owner(tmp_path) -> None:
    lock = tmp_path / ".pfp.lock"
    with exclusive_lock(lock):
        owner = lock / "owner.json"
        replacement = _owner(lock)
        replacement["token"] = "f" * 32
        owner.write_text(json.dumps(replacement), encoding="utf-8")
    assert lock.is_dir()
    assert _owner(lock)["token"] == "f" * 32


def test_stale_directory_is_reclaimed_but_live_and_foreign_are_not(tmp_path) -> None:
    lock = tmp_path / ".pfp.lock"
    lock.mkdir()
    (lock / "owner.json").write_text(json.dumps({
        "pid": 99999999, "host": socket.gethostname(), "token": "old"
    }), encoding="utf-8")
    with exclusive_lock(lock):
        assert _owner(lock)["token"] != "old"

    for host, pid in ((socket.gethostname(), os.getpid()), ("remote.example", 99999999)):
        lock.mkdir()
        (lock / "owner.json").write_text(json.dumps({"pid": pid, "host": host, "token": "held"}), encoding="utf-8")
        with pytest.raises(StateLockError):
            with exclusive_lock(lock):
                pass
        assert _owner(lock)["token"] == "held"
        (lock / "owner.json").unlink()
        lock.rmdir()


def test_legacy_stale_file_is_migrated(tmp_path) -> None:
    lock = tmp_path / ".pfp.lock"
    lock.write_text(json.dumps({"pid": 99999999, "host": socket.gethostname()}), encoding="utf-8")
    with exclusive_lock(lock):
        assert lock.is_dir()


def test_malformed_lock_fails_closed(tmp_path) -> None:
    lock = tmp_path / ".pfp.lock"
    lock.mkdir()
    (lock / "owner.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(StateLockError, match="held by another run"):
        with exclusive_lock(lock):
            pass
