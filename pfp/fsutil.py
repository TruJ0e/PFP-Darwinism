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


@contextmanager
def exclusive_lock(path: str | Path) -> Iterator[Path]:
    """
    Hold ``path`` as an exclusive lock file for the duration of the block.

    Two runs sharing one ``--state-dir`` would each load the registry and case bank, then
    the later save would silently discard the earlier run's versions and wins (a lost
    update on the failsafe). Creation is atomic (``O_CREAT | O_EXCL``), so exactly one
    holder wins; the loser fails before any client exists. The file names its holder,
    and a lock left behind by a killed local process is reclaimed automatically (or can
    be cleared by deleting it).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = None
    for attempt in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            break
        except FileExistsError:
            try:
                holder = path.read_text(encoding="utf-8").strip()[:300]
            except OSError:
                holder = ""
            if attempt == 0 and holder and _is_stale_local_lock(holder):
                try:
                    path.unlink(missing_ok=True)
                    continue
                except OSError:
                    pass
            raise StateLockError(
                f"{path} is held by another run ({holder or 'unknown holder'}); "
                "if no run is active, delete the lock file and retry"
            ) from None
    assert fd is not None
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"pid": os.getpid(), "host": socket.gethostname(), "acquired": round(time.time(), 3)}
                )
            )
            handle.flush()
            os.fsync(handle.fileno())
        yield path
    finally:
        path.unlink(missing_ok=True)
