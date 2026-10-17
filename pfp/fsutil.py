"""
Durable file writes shared by every artifact PFP persists.

Replay streams, the HTML player, the version registry (the failsafe), the case
bank (recall) and run summaries are all written here: temp file in the target
directory → flush + fsync → atomic replace → directory fsync. A crash or power
loss mid-write leaves the previous file intact, never a truncated one.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


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
