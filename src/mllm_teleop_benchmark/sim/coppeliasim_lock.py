from __future__ import annotations

import errno
import fcntl
import os
from pathlib import Path


COPPELIASIM_LOCK_PATH_ENV = "MLLM_TELEOP_COPPELIASIM_LOCK_PATH"
DEFAULT_COPPELIASIM_LOCK_PATH = Path(
    "/tmp/mllm-teleop-coppeliasim-v4.1.lock"
)


class CoppeliaSimRuntimeBusyError(RuntimeError):
    pass


class CoppeliaSimRuntimeLock:
    def __init__(self, owner: str) -> None:
        self.owner = owner
        self.path = Path(
            os.environ.get(COPPELIASIM_LOCK_PATH_ENV)
            or DEFAULT_COPPELIASIM_LOCK_PATH
        )
        self._fd: int | None = None

    def acquire(self) -> None:
        if self._fd is not None:
            return

        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        fd = os.open(self.path, flags, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise CoppeliaSimRuntimeBusyError(
                    f"Cannot launch {self.owner}: CoppeliaSim v4.1 is already "
                    "in use by another RLBench or Colosseum process on this "
                    f"host (advisory lock: {self.path}). Close the running "
                    "adapter before retrying."
                ) from None
            raise
        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return

        fd = self._fd
        self._fd = None
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
