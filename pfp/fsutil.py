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


@contextmanager
def exclusive_lock(path: str | Path) -> Iterator[Path]:
    """
    Hold ``path`` as an exclusive lock file for the duration of the block.

    Two runs sharing one ``--state-dir`` would each load the registry and case bank, then
    the later save would silently discard the earlier run's versions and wins (a lost
    update on the failsafe). Creation is atomic (``O_CREAT | O_EXCL``), so exactly one
    holder wins; the loser fails before any client exists. The file names its holder,
    and a lock left behind by a killed process is cleared by deleting it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        try:
            holder = path.read_text(encoding="utf-8").strip()[:300]
        except OSError:
            holder = ""
        raise StateLockError(
            f"{path} is held by another run ({holder or 'unknown holder'}); "
            "if no run is active, delete the lock file and retry"
        ) from None
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
