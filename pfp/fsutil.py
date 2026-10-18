"""
Durable file writes shared by every artifact PFP persists.

Replay streams, the HTML player, the version registry (the failsafe), the case
bank (recall) and run summaries are all written here: temp file in the target
directory → flush + fsync → atomic replace → directory fsync. A crash or power
loss mid-write leaves the previous file intact, never a truncated one.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def atomic_write_text(target: str | Path, text: str) -> Path:
    """Durably replace ``target`` with ``text`` (UTF-8, ``\n`` newlines) without exposing a partial file."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
        # Persist the directory entry on platforms that support directory fsync.
        try:
            dir_fd = os.open(target.parent, os.O_RDONLY)
        except OSError:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            except OSError:
                pass
            finally:
                os.close(dir_fd)
    finally:
        tmp.unlink(missing_ok=True)
    return target


class StateLockError(RuntimeError):
    """Another live run already holds the state directory."""


def _pid_is_alive(pid: int) -> bool:
    """Return True if a local process with ``pid`` currently exists."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            # Access denied (5) means the process exists under another security context
            return int(ctypes.GetLastError()) == 5  # type: ignore[attr-defined]
        try:
            exit_code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return int(exit_code.value) == STILL_ACTIVE
            return True
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _is_stale_local_lock(holder_text: str) -> bool:
    """True only when ``holder_text`` parses as a lock on this host whose PID is dead."""
    try:
        meta = json.loads(holder_text)
    except (ValueError, TypeError):
        return False
    if not isinstance(meta, dict):
        return False
    host = meta.get("host")
    pid = meta.get("pid")
    if not isinstance(host, str) or host != socket.gethostname():
        return False
    if not isinstance(pid, int) or isinstance(pid, bool):
        return False
    return not _pid_is_alive(pid)


_LOCK_OWNER_FILE = "owner.json"


def _lock_holder(path: Path) -> str:
    """Read either a v0.18 lock directory or the legacy lock-file payload."""
    try:
        source = path / _LOCK_OWNER_FILE if path.is_dir() else path
        return source.read_text(encoding="utf-8").strip()[:1000]
    except OSError:
        return ""


def _remove_stale_lock(path: Path, holder: str) -> bool:
    """Remove a known-stale lock without ever deleting a differently owned lock."""
    if not holder or not _is_stale_local_lock(holder):
        return False
    try:
        if path.is_dir():
            owner = path / _LOCK_OWNER_FILE
            # Re-read immediately before removal. A token/PID change means ownership changed.
            if owner.read_text(encoding="utf-8").strip()[:1000] != holder:
                return False
            owner.unlink()
            path.rmdir()
        else:
            # Legacy O_EXCL lock files have no ownership token. Removing the stale inode
            # creates the contention point; the following mkdir decides exactly one winner.
            if path.read_text(encoding="utf-8").strip()[:1000] != holder:
                return False
            path.unlink()
        return True
    except OSError:
        return False


@contextmanager
def exclusive_lock(path: str | Path) -> Iterator[Path]:
    """
    Hold ``path`` as an exclusive, token-owned lock directory for the duration.

    Two runs sharing one ``--state-dir`` would each load the registry and case bank, then
    the later save would silently discard the earlier run's versions and wins (a lost
    update on the failsafe). Directory creation is atomic, so exactly one holder wins;
    the loser fails before any client exists. ``owner.json`` names the holder and carries
    an unguessable ownership token. Cleanup verifies that token, so a late ``finally``
    can never remove a successor's lock. A dead local holder is reclaimed automatically.
    Legacy JSON lock files from v0.17 and earlier are recognized and safely migrated.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    metadata = json.dumps(
        {"pid": os.getpid(), "host": socket.gethostname(), "acquired": round(time.time(), 3), "token": token}
    )
    acquired = False
    for attempt in range(2):
        try:
            path.mkdir()
            try:
                # The directory already excludes contenders. Write ownership durably before yielding.
                atomic_write_text(path / _LOCK_OWNER_FILE, metadata)
            except BaseException:
                try:
                    (path / _LOCK_OWNER_FILE).unlink(missing_ok=True)
                    path.rmdir()
                finally:
                    raise
            acquired = True
            break
        except FileExistsError:
            holder = _lock_holder(path)
            if attempt == 0 and _remove_stale_lock(path, holder):
                continue
            raise StateLockError(
                f"{path} is held by another run ({holder or 'unknown holder'}); "
                "if no run is active, remove the lock and retry"
            ) from None
    if not acquired:
        raise StateLockError(f"Could not acquire {path}")
    try:
        yield path
    finally:
        # Only the owner may release. If the path was externally replaced, leave it alone.
        owner = path / _LOCK_OWNER_FILE
        try:
            current = json.loads(owner.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            current = {}
        if isinstance(current, dict) and current.get("token") == token:
            try:
                owner.unlink()
                path.rmdir()
            except OSError:
                # Unknown files or a racing external mutation make fail-closed retention
                # safer than deleting a lock we can no longer prove we wholly own.
                pass
